# Reload earlier stages' results, so this stage also works in a fresh kernel (after 00_setup.py).
A = load_json("A_detour.json") if have("A_detour.json") else None
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
summary = load_json("B_summary.json")["per_sentence"] if have("B_summary.json") else {}


if RUN_B and not have("B_ladder.json"):
    with Timer("Part B: generating the ladder"):
        ladder = {S["id"]: {} for S in SENTENCES}
        for r in RUNGS:
            outs = generate([chat_prompt(rung_user(r, S["text"], S["lang"])) for S in SENTENCES], max_new_tokens=MAXTOK[r])
            for S, o in zip(SENTENCES, outs): ladder[S["id"]][r] = o
        save_json("B_ladder.json", ladder)
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
if ladder:
    for S in SENTENCES[:3] + [S for S in SENTENCES if S["id"] == "tren"]:
        print(f"\n{S['text']}"); [print(f"  {r:3s} {RUNG_NAME[r]:18s} {ladder[S['id']][r][:150]!r}") for r in RUNGS]

def mask_source_words(t, s):
    """Replace words in t that copy a source word of >= 4 letters (ignoring accents and a plural s) with '[·]'.
    Proper nouns are left alone. Genuine cognate translations (moment / momento) are NOT masked: they are translations."""
    import unicodedata
    def norm(w): return "".join(c for c in unicodedata.normalize("NFD", w.lower()) if unicodedata.category(c) != "Mn")
    src = set()
    for k, (w, a, b) in enumerate(words_with_spans(s)):
        if len(w) >= 4 and not (w[0].isupper() and k > 0): src |= {norm(w), norm(w) + "s", norm(w).rstrip("s")}
    n = [0]
    def rep(m):
        if norm(m.group(0)) in src: n[0] += 1; return "[·]"
        return m.group(0)
    return WORD_RE.sub(rep, t), n[0]

def bt_user(t, L): return f"Translate this English text into {LANG_NAME[L]}. Reply with the translation only.\n\n{t}"
def copy_user(s, L): return f"Translate this {LANG_NAME[L]} text into {LANG_NAME[L]}. Reply with the translation only.\n\n{s}"
def sense_user(ctx, span, L):
    return (f"Here is a text:\n\n{ctx}\n\nIn the original {LANG_NAME[L]} sentence, what did \"{span}\" mean in this context? "
            f"Answer in one line.")
EMPTY, DELTA = "(no text)", 0.5

def token_word_map(offsets, words):
    return [next((k for k, (_, wa, wb) in enumerate(words) if y > x and y > wa and x < wb), None) for (x, y) in offsets]

def drt_from(lp_t, lp_0, lp_s, offsets, words):
    R, Rs = lp_t - lp_0, lp_s - lp_0
    valid = Rs > DELTA
    w = np.where(valid, Rs, 0.0)
    ell = np.where(valid, 1 - np.clip(R / np.where(valid, Rs, 1.0), 0, 1), 0.0)
    d = float((w * ell).sum() / w.sum()) if w.sum() > 0 else float("nan")
    W, WL = np.zeros(len(words)), np.zeros(len(words))
    for i, k in enumerate(token_word_map(offsets, words)):
        if k is not None: W[k] += w[i]; WL[k] += w[i] * ell[i]
    return d, np.where(W > 0, WL / np.maximum(W, 1e-12), np.nan), W

if RUN_B and ladder and not have("B_roundtrip.json"):
    with Timer("Part B: round-trip scoring"):
        items, meta = [], []
        for S in SENTENCES:
            s, L = S["text"], S["lang"]
            items += [(chat_prompt(bt_user(EMPTY, L)), s), (chat_prompt(copy_user(s, L)), s)]
            meta += [(S["id"], "empty", None), (S["id"], "oracle", None)]
            for r in RUNGS:
                t = ladder[S["id"]][r]; tm, nm = mask_source_words(t, s)
                items += [(chat_prompt(bt_user(tm, L)), s), (chat_prompt(bt_user(t, L)), s)]
                meta += [(S["id"], r, "masked"), (S["id"], r, "raw")]
        sc = score([p for p, _ in items], [c for _, c in items])
        got = {(m[0], m[1], m[2]): r for m, r in zip(meta, sc)}

        # sense recovery
        s_items, s_meta = [], []
        for S in SENTENCES:
            s, L = S["text"], S["lang"]
            for span, gloss in S["hot"].items():
                ctxs = {"source": s, "empty": EMPTY}
                for r in RUNGS:
                    t = ladder[S["id"]][r]
                    ctxs[r] = t if r == "B2" else mask_source_words(t, s)[0]
                for k, ctx in ctxs.items():
                    s_items.append((chat_prompt(sense_user(ctx, span, L)), gloss)); s_meta.append((S["id"], span, k))
        ssc = score([p for p, _ in s_items], [c for _, c in s_items])
        sense_lp = defaultdict(dict)
        for (sid, span, k), r in zip(s_meta, ssc): sense_lp[(sid, span)][k] = float(r["lp"].mean())

        out = {}
        for S in SENTENCES:
            sid, s = S["id"], S["text"]; words = words_with_spans(s)
            lp0, lps = got[(sid, "empty", None)]["lp"], got[(sid, "oracle", None)]["lp"]
            offs = got[(sid, "empty", None)]["offsets"]
            rec = dict(words=[w for w, _, _ in words], rungs={})
            for r in RUNGS:
                t = ladder[sid][r]
                dm, wl_m, W = drt_from(got[(sid, r, "masked")]["lp"], lp0, lps, offs, words)
                du, wl_u, _ = drt_from(got[(sid, r, "raw")]["lp"], lp0, lps, offs, words)
                rec["rungs"][r] = dict(len=ntok(t), d_masked=dm, d_raw=du, word_left_masked=wl_m, word_left_raw=wl_u,
                                       n_masked=mask_source_words(t, s)[1])
            rec["word_weight"] = W
            sr = {}
            for span in S["hot"]:
                v = sense_lp[(sid, span)]; denom = v["source"] - v["empty"]
                sr[span] = dict(raw=v, norm={r: (v[r] - v["empty"]) / denom if abs(denom) > 1e-6 else float("nan") for r in RUNGS})
            rec["sense"] = sr
            out[sid] = rec
        save_json("B_roundtrip.json", out)

B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
if B:
    ctrl = [B[S["id"]]["rungs"]["B1"]["d_masked"] for S in SENTENCES if S["cat"] == "control"]
    EPS = float(np.nanpercentile(ctrl, 90))
    def curve_stats(rec, rungs):
        L0 = rec["rungs"]["B1"]["len"]
        cands = [(rec["rungs"][r]["len"], rec["rungs"][r]["d_raw" if r == "B2" else "d_masked"]) for r in rungs]
        ok = [l for l, d in cands if d <= EPS]
        U = math.log2(max(min(ok), 1) / L0) if ok else math.log2(max(l for l, _ in cands) / L0)
        budgets = L0 * 2 ** np.linspace(0, 4, 9)
        Dl = [min([d for l, d in cands if l <= bgt] or [np.nan]) for bgt in budgets]
        return dict(U=max(U, 0.0), censored=not ok, A=float(np.nanmean(Dl)))
    summary = {}
    for sid, rec in B.items():
        summary[sid] = dict(cat=SID[sid]["cat"], lang=SID[sid]["lang"], **curve_stats(rec, NON_BORROW),
                            U_with_borrowing=curve_stats(rec, NON_BORROW + ["B2"])["U"],
                            d_B1=rec["rungs"]["B1"]["d_masked"],
                            borrow_gain=rec["rungs"]["B1"]["d_masked"] - rec["rungs"]["B2"]["d_raw"],
                            borrow_gain_masked=rec["rungs"]["B1"]["d_masked"] - rec["rungs"]["B2"]["d_masked"])
    # hot words vs other content words (standard translation, masked)
    hot_v, oth_v = [], []
    for S in SENTENCES:
        rec = B[S["id"]]; wl = np.array(rec["rungs"]["B1"]["word_left_masked"], float); W = np.array(rec["word_weight"])
        hot_idx = set()
        for span in S["hot"]:
            sp = find_phrase(S["text"], span)
            if sp: hot_idx |= {k for k, (_, a, b) in enumerate(words_with_spans(S["text"])) if b > sp[0] and a < sp[1]}
        for k, w in enumerate(rec["words"]):
            if W[k] <= 0 or np.isnan(wl[k]): continue
            if k in hot_idx: hot_v.append(wl[k])
            elif len(w) >= 4: oth_v.append(wl[k])
    B_auroc = auroc(hot_v, oth_v)
    print(f"tolerance ε (90th pct of control d at 'standard') = {EPS:.3f}")
    print(f"hot words vs other content words, leftover AUROC = {B_auroc:.3f} ({len(hot_v)} hot, {len(oth_v)} other)\n")
    print(f"{'category':15s} {'n':>2s} {'U (doublings)':>13s} {'cens':>4s} {'A':>6s} {'d std':>6s} {'borrow buys':>11s} {'…masked':>8s}")
    cats = ["untranslatable", "construction", "idiom", "control"]
    for c in cats:
        v = [x for x in summary.values() if x["cat"] == c]
        print(f"{c:15s} {len(v):2d} {np.mean([x['U'] for x in v]):13.2f} {sum(x['censored'] for x in v):4d} "
              f"{np.mean([x['A'] for x in v]):6.3f} {np.nanmean([x['d_B1'] for x in v]):6.3f} "
              f"{np.nanmean([x['borrow_gain'] for x in v]):11.3f} {np.nanmean([x['borrow_gain_masked'] for x in v]):8.3f}")
    fa = B["duende"]; wl = dict(zip(fa["words"], fa["rungs"]["B1"]["word_left_masked"]))
    print(f"\nMe gusta mucho el duende (standard translation: {ladder['duende']['B1']!r})")
    print("  leftover: " + ", ".join(f"{w} {wl.get(w, float('nan')):.2f}" for w in ["duende", "gusta", "mucho"])
          + "   [Llama-3.1-8B last time: duende 0.50, gusta 0.23, mucho 0.01]")
    print("  sense recovered (0 = none, 1 = as good as the source):")
    for span, v in fa["sense"].items():
        print(f"    {span:8s} " + "  ".join(f"{RUNG_NAME[r]} {v['norm'][r]:.2f}" for r in RUNGS))
    save_json("B_summary.json", dict(epsilon=EPS, auroc_hot_words=B_auroc, per_sentence=summary))

if B:
    def heat_row(ax, y, words, vals, title, sub):
        """Words set in a row, each shaded by its leftover; positions measured from the rendered text."""
        fig_ = ax.figure; r = fig_.canvas.get_renderer(); inv = ax.transData.inverted(); x = 0.0
        ax.text(0, y + 0.45, title, fontsize=9, color="#666", ha="left")
        for w, v in zip(words, vals):
            bad = v is None or np.isnan(v)
            c = GREY if bad else plt.cm.Oranges(0.08 + 0.85 * float(np.clip(v, 0, 1)))
            t = ax.text(x, y, w, fontsize=17, style="italic", family="serif", va="center", ha="left",
                        bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none"))
            bb = t.get_window_extent(renderer=r); (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            ax.text((x0 + x1) / 2, y - 0.36, "" if bad else f"{v:.2f}", fontsize=7.5, color="#555", ha="center")
            x = x1 + 0.03
        ax.text(0, y - 0.66, sub, fontsize=9, color=INK, ha="left")
    show = [s for s in ["duende", "olvido", "pelo", "estrenar", "tren"] if s in B]
    fig, ax = plt.subplots(figsize=(9, 1.45 * len(show))); ax.axis("off")
    ax.set_xlim(-0.02, 1.25); ax.set_ylim(-1.5 * len(show) + 0.6, 0.9)
    for k, sid in enumerate(show):
        rec = B[sid]; S = SID[sid]
        wds = [w for w, _, _ in words_with_spans(S["text"])]
        heat_row(ax, -1.5 * k, wds, rec["rungs"]["B1"]["word_left_masked"], S["cat"], "→ " + ladder[sid]["B1"][:110])
    ax.set_xlim(-0.02, 1.25); ax.set_ylim(-1.5 * len(show) + 0.6, 0.9)
    ax.set_title("Where each sentence resists translation: darker words are harder to recover from the English\n"
                 f"(share of each word's information lost after a standard translation; {MODEL_ID.split('/')[-1]})", loc="left", fontsize=10)
    savefig(fig, "B1_heatmaps")

    show = [s for s in ["duende", "sobremesa", "pelo", "tren"] if s in B]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for sid in show:
        rec = B[sid]; col = EASY if SID[sid]["cat"] == "control" else HARD
        pts = sorted((rec["rungs"][r]["len"], rec["rungs"][r]["d_masked"], r) for r in NON_BORROW)
        env, best = [], 1.0
        for l, d, r in pts: best = min(best, d); env.append((l, best))
        ax.step([l for l, _ in env], [d for _, d in env], where="post", color=col, lw=2, alpha=0.9)
        ax.scatter([l for l, _, _ in pts], [d for _, d, _ in pts], color=col, s=18, zorder=3)
        ax.text(env[-1][0] * 1.05, env[-1][1], SID[sid]["text"][:28] + "…", fontsize=8, color=col, va="center")
    ax.axhline(EPS, color="#888", ls="--", lw=1); ax.text(ax.get_xlim()[0], EPS, "  gets across", fontsize=8, color="#888", va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("length of the English (tokens, log scale): word for word → standard → gloss → footnote → thick")
    ax.set_ylabel("share of meaning not carried"); ax.set_title("How much English does it take?", loc="left")
    savefig(fig, "B2_distortion_vs_length")

    spans = [(sid, sp) for sid in B for sp in B[sid]["sense"]][:12]
    if spans:
        fig, ax = plt.subplots(figsize=(9, 3.6)); w = 0.15
        rr = ["B1", "B2", "B3", "B5", "B6"]; cols = [GREY, LOAN, "#e3a587", "#d07a55", HARD]
        for k, r in enumerate(rr):
            ax.bar(np.arange(len(spans)) + (k - 2) * w, np.clip([B[s]["sense"][sp]["norm"][r] for s, sp in spans], -0.5, 1.5), w, color=cols[k], label=RUNG_NAME[r])
        ax.set_xticks(np.arange(len(spans))); ax.set_xticklabels([sp for _, sp in spans], rotation=30, ha="right", style="italic")
        ax.axhline(1, color="#888", lw=0.8, ls=":"); ax.axhline(0, color="#888", lw=0.8)
        ax.set_ylabel("intended sense recovered\n(0 = none, 1 = as from the original)"); ax.set_ylim(-0.55, 1.55)
        ax.set_title("Does the English carry the sense the sentence means?", loc="left"); ax.legend(frameon=False, fontsize=8, ncol=5)
        savefig(fig, "B3_sense_recovery")

pack()
