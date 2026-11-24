# Reload earlier stages' results, so this stage also works in a fresh kernel (after 00_setup.py).
A = load_json("A_detour.json") if have("A_detour.json") else None
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
summary = load_json("B_summary.json")["per_sentence"] if have("B_summary.json") else {}


assert C_FITS, 'run C2_spectrum.py in this same kernel first (it keeps the fitted bases in GPU memory)'

def decomposer(lang, l):
    F_ = C_FITS[(lang, l)]; fit = F_["fit"]
    idx = F_["shared"]
    A_r = fit["A"][:, idx].float(); D_r = (fit["Cxx_r"] @ fit["A"][:, idx]).float(); mx = fit["mx"].float()
    def split(h):
        xc = h.float().to(A_r.device) - mx; sh = (xc @ A_r) @ D_r.T
        return sh, xc - sh
    test = torch.randn(4, D_MODEL, device=A_r.device) * 3 + mx
    sh, _ = split(test); sh2, _ = split(sh + mx)
    print(f"  layer {l}: shared dims {len(idx)} of {D_MODEL}; oblique projection idempotence error {float((sh2 - sh).norm() / sh.norm()):.1e}")
    return split, mx

def ablate_fn(split, mx, which):
    def fn(h):
        sh, pr = split(h); dev = h.device
        if which == "private": return (mx + sh).to(dev)
        if which == "shared": return (mx + pr).to(dev)
        return mx.expand_as(sh).to(dev)
    return fn

def interchange_fn(split, mx, donor, which):
    def fn(h):
        sh, pr = split(h); dsh, dpr = split(donor); xc = sh + pr
        if which == "private": new = xc - pr + dpr
        elif which == "shared": new = xc - sh + dsh
        else: new = dsh + dpr
        return (new + mx).to(h.device)
    return fn

if RUN_D and "es" in C_RES and ladder and not have("D_mediation.json"):
    D_out = {}
    med_layers = sorted({l for (lg, l) in C_FITS if lg == "es"})
    for l in med_layers:
        with Timer(f"Part D: layer {l}"):
            split, mx = decomposer("es", l)
            # --- 1. ablation dissociation ---
            es_sents = [S for S in SENTENCES if S["lang"] == "es"]
            targets = []
            for S in es_sents:
                hi = hot_word_index(S)
                for k, (w, a, b) in enumerate(words_with_spans(S["text"])):
                    if k in hi or (len(w) >= 4 and S["cat"] == "control"): targets.append((S, w, k in hi, (a, b)))
            ask = [chat_prompt(f'{S["text"]}\n\nIn the Spanish sentence above, what does the word "{w}" mean here? Answer in one line.')
                   for S, w, _, _ in targets]
            expl = generate(ask, max_new_tokens=40)
            P, Cn, E, meta = [], [], [], []
            for (S, w, is_hot, span), ap, ex in zip(targets, ask, expl):
                tp = chat_prompt(rung_user("B1", S["text"], "es"))
                for task, prompt, cont in [("translate", tp, ladder[S["id"]]["B1"]), ("explain", ap, ex)]:
                    pos = positions_in_prompt(prompt, S["text"], span=span)
                    if task == "explain":                      # also the word quoted in the question
                        q = prompt.rfind(f'"{w}"') + 1; pos = sorted(set(pos) | set(positions_at(prompt, q, q + len(w))))
                    for cond in ["none", "private", "shared", "full"]:
                        P.append(prompt); Cn.append(cont)
                        E.append(None if cond == "none" else [(l, pos, ablate_fn(split, mx, cond))])
                        meta.append((S["id"], w, is_hot, task, cond))
            sc = score(P, Cn, edits=E)
            lp = {m: float(r["lp"].mean()) for m, r in zip(meta, sc)}
            abl = []
            for (S, w, is_hot, _), ex in zip(targets, expl):
                row = dict(sid=S["id"], word=w, hot=is_hot, explanation=ex)
                for task in ["translate", "explain"]:
                    base = lp[(S["id"], w, is_hot, task, "none")]
                    for cond in ["private", "shared", "full"]:
                        row[f"{task}_{cond}"] = base - lp[(S["id"], w, is_hot, task, cond)]   # drop in log-prob per token
                abl.append(row)
            # --- 2. interchange on sense pairs ---
            inter = []
            for SP in SENSE_PAIRS:
                runs = {}
                for me, other in [("a", "b"), ("b", "a")]:
                    pm = chat_prompt(rung_user("B1", SP[me], "es")); po = chat_prompt(rung_user("B1", SP[other], "es"))
                    pos_m = positions_in_prompt(pm, SP[me], SP["word"]); pos_o = positions_in_prompt(po, SP[other], SP["word"])
                    if len(pos_m) != len(pos_o): print(f"  skip {SP['id']}: token counts differ"); runs = None; break
                    donor = capture_prompt_states(po, l)[pos_o]
                    tmpl = SP[f"en_{me}"]; cands = [tmpl.format(SP["sense_a"]), tmpl.format(SP["sense_b"])]
                    conds = {"none": None, "private": interchange_fn(split, mx, donor, "private"),
                             "shared": interchange_fn(split, mx, donor, "shared"), "full": interchange_fn(split, mx, donor, "full")}
                    Ps, Cs, Es, keys = [], [], [], []
                    for cname, fn in conds.items():
                        for ci, c in enumerate(cands):
                            Ps.append(pm); Cs.append(c); Es.append(None if fn is None else [(l, pos_m, fn)]); keys.append((cname, ci))
                    r = score(Ps, Cs, edits=Es); tot = {k: float(x["lp"].sum()) for k, x in zip(keys, r)}
                    lo = {c: tot[(c, 0)] - tot[(c, 1)] for c in conds}          # log-odds sense_a vs sense_b
                    gen = {c: generate_with_edits(pm, [] if fn is None else [(l, pos_m, fn)]) for c, fn in conds.items()}
                    runs[me] = dict(logodds=lo, generations=gen)
                if runs is None: continue
                base_a, base_b = runs["a"]["logodds"]["none"], runs["b"]["logodds"]["none"]
                gap = base_b - base_a
                flip = {}
                for c in ["private", "shared", "full"]:
                    fa = (runs["a"]["logodds"][c] - base_a) / gap if abs(gap) > 1e-6 else float("nan")
                    fb = (runs["b"]["logodds"][c] - base_b) / (-gap) if abs(gap) > 1e-6 else float("nan")
                    flip[c] = float(np.nanmean([fa, fb]))
                inter.append(dict(id=SP["id"], base_gap=gap, flip=flip, runs=runs))
                print(f"  {SP['id']:11s} base log-odds a={base_a:+.1f} b={base_b:+.1f}  flip private={flip['private']:+.2f} "
                      f"shared={flip['shared']:+.2f} full={flip['full']:+.2f}")
            D_out[l] = dict(ablation=abl, interchange=inter)
    save_json("D_mediation.json", D_out)

D = load_json("D_mediation.json") if have("D_mediation.json") else None
if D:
    for l, d in D.items():
        ab = d["ablation"]
        def ratio(r, task, cond): f = r[f"{task}_full"]; return r[f"{task}_{cond}"] / f if abs(f) > 1e-6 else np.nan
        print(f"\nlayer {l}: effect of removing the PRIVATE part, as a share of removing the whole state (mean, 95% CI)")
        for lab, grp in [("hot words", [r for r in ab if r["hot"]]), ("control words", [r for r in ab if not r["hot"]])]:
            t = [ratio(r, "translate", "private") for r in grp]; e = [ratio(r, "explain", "private") for r in grp]
            ct, ce = boot_ci(t), boot_ci(e)
            print(f"  {lab:14s} n={len(grp):3d}  on translation {np.nanmean(t):.2f} [{ct[0]:.2f}, {ct[1]:.2f}] (median {np.nanmedian(t):.2f})"
                  f"   on explanation {np.nanmean(e):.2f} [{ce[0]:.2f}, {ce[1]:.2f}] (median {np.nanmedian(e):.2f})")
        hot = [r for r in ab if r["hot"]]
        gap = [ratio(r, "explain", "private") - ratio(r, "translate", "private") for r in hot]
        cg = boot_ci(gap)
        print(f"  hot words: explanation minus translation = {np.nanmean(gap):+.2f} [{cg[0]:+.2f}, {cg[1]:+.2f}] (median {np.nanmedian(gap):+.2f}); positive = understood but not translated")
    l = list(D)[0]; ab = D[l]["ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for grp, col, lab in [([r for r in ab if not r["hot"]], EASY, "ordinary words"), ([r for r in ab if r["hot"]], HARD, "hard words")]:
        x = [r["translate_private"] / r["translate_full"] if abs(r["translate_full"]) > 1e-6 else np.nan for r in grp]
        y = [r["explain_private"] / r["explain_full"] if abs(r["explain_full"]) > 1e-6 else np.nan for r in grp]
        x, y = np.clip(x, -0.5, 1.5), np.clip(y, -0.5, 1.5)     # extreme ratios (tiny denominators) pinned to the edge
        ax.scatter(x, y, color=col, s=28, alpha=0.8, label=lab)
        for r, xi, yi in zip(grp, x, y):
            if r["hot"]: ax.text(xi, yi, " " + r["word"], fontsize=7, style="italic", color=col)
    ax.plot([-1, 2], [-1, 2], color="#bbb", lw=0.8); ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("harm to the TRANSLATION\n(removing the language-private part, share of removing everything)")
    ax.set_ylabel("harm to EXPLAINING the word")
    ax.set_title("Above the diagonal: the model understands more than it translates", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]; it = D[l]["interchange"]; w = 0.26
    for k, (c, col) in enumerate([("private", HARD), ("shared", EASY), ("full", GREY)]):
        ax.bar(np.arange(len(it)) + (k - 1) * w, [x["flip"][c] for x in it], w, color=col, label=f"swap the {c} part" if c != "full" else "swap everything")
    ax.set_xticks(np.arange(len(it))); ax.set_xticklabels([x["id"] for x in it], rotation=30, ha="right", style="italic")
    ax.axhline(0, color="#888", lw=0.8); ax.axhline(1, color="#888", lw=0.8, ls=":")
    ax.set_ylabel("how far the English moves to the other sense"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Swapping part of one word's state between two senses", loc="left", fontsize=9)
    savefig(fig, f"D_mediation_L{l}")
    print("\nExample generations under interchange (frame a, patched from frame b):")
    for x in D[l]["interchange"][:8]:
        g = x["runs"]["a"]["generations"]; print(f"  {x['id']:11s} none: {g['none'][:60]!r} | private: {g['private'][:60]!r} | full: {g['full'][:60]!r}")

pack()
