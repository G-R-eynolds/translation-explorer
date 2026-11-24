# Reload earlier stages' results, so this stage also works in a fresh kernel (after 00_setup.py).
A = load_json("A_detour.json") if have("A_detour.json") else None
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
summary = load_json("B_summary.json")["per_sentence"] if have("B_summary.json") else {}


CDEV = "cuda" if torch.cuda.is_available() else "cpu"

def inv_sqrt(C):
    e, V = torch.linalg.eigh(C); return (V * e.clamp_min(1e-12).rsqrt()) @ V.T

def cca_fit(cov, lam):
    Cxx, Cyy, Cxy = (cov[k].to(CDEV) for k in ["Cxx", "Cyy", "Cxy"]); d = Cxx.shape[0]
    I = torch.eye(d, dtype=Cxx.dtype, device=CDEV)
    Cxx_r = Cxx + lam * torch.trace(Cxx) / d * I; Cyy_r = Cyy + lam * torch.trace(Cyy) / d * I
    Wx, Wy = inv_sqrt(Cxx_r), inv_sqrt(Cyy_r)
    U, S, Vh = torch.linalg.svd(Wx @ Cxy @ Wy)
    return dict(lam=lam, rho_in=S, U=U, Wx=Wx, Wy=Wy, A=Wx @ U, B=Wy @ Vh.T, Cxx_r=Cxx_r, Cyy_r=Cyy_r,
                Cxx=Cxx, Cyy=Cyy, Cxy=Cxy, mx=cov["mx"].to(CDEV), my=cov["my"].to(CDEV))

def colcorr(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    return (P * Q).sum(0) / (P.norm(dim=0) * Q.norm(dim=0) + 1e-12)

def heldout_rho(fit, X, Y):
    return colcorr((X - fit["mx"]) @ fit["A"], (Y - fit["my"]) @ fit["B"])

def r2(Yc, Yp):
    raw = 1 - ((Yc - Yp) ** 2).sum() / (Yc ** 2).sum()
    per = 1 - ((Yc - Yp) ** 2).sum(0) / ((Yc - Yc.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return float(raw), float(per.mean())

def analyse_layer(lang, l):
    cov = torch.load(os.path.join(CACHE, f"C_{lang}_cov_L{l}.pt"), weights_only=False)
    held = torch.load(os.path.join(CACHE, f"C_{lang}_held.pt"), weights_only=False)
    X, Y = held["X"][l].to(CDEV).double(), held["Y"][l].to(CDEV).double()
    sent = torch.tensor([m["sent"] for m in held["meta"]])
    half = (sent % 2 == 0).to(CDEV)                       # split held-out by sentence
    best = None
    for lam in LAMBDAS:
        fit = cca_fit(cov, lam); rho = heldout_rho(fit, X[half], Y[half])
        s50 = float(rho[:50].sum())
        if best is None or s50 > best[0]: best = (s50, lam)
        del fit
    fit = cca_fit(cov, best[1])
    rho_h = heldout_rho(fit, X[~half], Y[~half])          # reported on the other half
    perm = torch.randperm(int((~half).sum()), generator=torch.Generator().manual_seed(SEED)).to(CDEV)
    rho_null = heldout_rho(fit, X[~half], Y[~half][perm])
    thr = float(torch.quantile(rho_null.abs().float(), 0.99))
    shared = torch.nonzero(rho_h > thr)[:, 0]
    # English bias: held-out share of each language's variance predictable from the other
    Xc, Yc = X[~half] - fit["mx"], Y[~half] - fit["my"]
    Gxy = torch.linalg.solve(fit["Cxx_r"], fit["Cxy"]); Gyx = torch.linalg.solve(fit["Cyy_r"], fit["Cxy"].T)
    E_en, E_en_std = r2(Yc, Xc @ Gxy)          # English predicted from the other language
    E_x, E_x_std = r2(Xc, Yc @ Gyx)            # other language predicted from English
    spec_x = float(torch.trace(fit["Cxy"] @ Gyx) / torch.trace(fit["Cxx"]))
    res = dict(layer=l, lam=best[1], n_fit_pairs=int(cov["n"]), n_held_pairs=int((~half).sum()),
               rho_in=fit["rho_in"][:1024].cpu().numpy(), rho_held=rho_h[:1024].cpu().numpy(),
               null_thr=thr, n_shared=int(len(shared)),
               n_strong=int((rho_h > 0.5 ** 0.5).sum()),   # directions that keep more than half per round trip (rho^2 > 0.5)
               n_strong_top=int((fit["rho_in"] > 0.5 ** 0.5).sum()), top50_held=float(rho_h[:50].sum()), top50_in=float(fit["rho_in"][:50].sum()),
               E_en_from_x=E_en, E_x_from_en=E_x, E_en_from_x_std=E_en_std, E_x_from_en_std=E_x_std,
               bias=E_en - E_x, bias_std=E_en_std - E_x_std, E_x_spectral_in_sample=spec_x)
    return res, fit, rho_h, shared, Gxy, Gyx

C_RES, C_FITS, BEST_LAYER = {}, {}, {}
if RUN_C:
    for lang in PAIRS:
        with Timer(f"Part C [{lang}]: CCA at {len(C_LAYERS)} layers"):
            rows = []
            for l in C_LAYERS:
                out = analyse_layer(lang, l); res = out[0]; rows.append(res); del out
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print(f"  L{l:3d}  λ={res['lam']:.0e}  shared dirs={res['n_shared']:5d} (rho>0.71: {res['n_strong']:4d})  top-50 ρ held/in-sample="
                      f"{res['top50_held']:.1f}/{res['top50_in']:.1f}  E_en←{lang}={res['E_en_from_x']:.3f}  "
                      f"E_{lang}←en={res['E_x_from_en']:.3f}  bias={res['bias']:+.3f}")
            save_json(f"C_spectrum_{lang}.json", rows); C_RES[lang] = rows
            BEST_LAYER[lang] = max(rows, key=lambda r: r["top50_held"])["layer"]
    print("best (most shared) layer:", BEST_LAYER)
    # refit and keep only what later cells need: the best layer per language, plus the detour layer for Spanish (Part D)
    keep = {(lang, BEST_LAYER[lang]) for lang in C_RES}
    if A and "es" in C_RES: keep.add(("es", min(C_LAYERS, key=lambda l: abs(l - A["detour_peak_layer"]))))
    for (lang, l) in sorted(keep):
        res, fit, rho_h, shared, Gxy, Gyx = analyse_layer(lang, l)
        C_FITS[(lang, l)] = dict(fit={k: fit[k] for k in ["Wx", "U", "A", "Cxx_r", "mx"]}, rho_h=rho_h, shared=shared, Gxy=Gxy, Gyx=Gyx)
        del fit; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def span_scores(lang, l, sentences):
    """U (whitened), U_raw, half-life and #tokens for every word of the given test sentences."""
    F_ = C_FITS[(lang, l)]; fit = F_["fit"]; rho2 = F_["rho_h"].clamp(0, 1).double() ** 2
    texts = [S["text"] for S in sentences]
    wl = [[(w, a, b) for w, a, b in words_with_spans(t)] for t in texts]
    st = word_states(texts, wl, [l])
    ns = np.linspace(0, 50, 501); out = {}
    for S, words, (rec, valid) in zip(sentences, wl, st):
        X = rec[l].double().to(CDEV) - fit["mx"]
        c2 = ((X @ fit["Wx"]) @ fit["U"]) ** 2               # canonical coordinates squared
        Rn = torch.stack([(c2 * rho2 ** n).sum(1) / c2.sum(1) for n in ns], 1).cpu().numpy()
        Xrt = (X @ F_["Gxy"]) @ F_["Gyx"]
        U_raw = (((X - Xrt) ** 2).sum(1) / (X ** 2).sum(1)).cpu().numpy()
        half = [float(ns[np.argmax(r <= 0.5)]) if (r <= 0.5).any() else 50.0 for r in Rn]
        out[S["id"]] = dict(words=[w for w, _, _ in words], U=(1 - Rn[:, 10]).tolist(), U_raw=U_raw.tolist(),
                            half_life=half, R_curve=Rn[:, ::10][:, :11].tolist(),
                            ntok=[len(enc(w)["input_ids"]) for w, _, _ in words], valid=valid.tolist())
    return out

def hot_word_index(S):
    idx = set()
    for span in S["hot"]:
        sp = find_phrase(S["text"], span)
        if sp: idx |= {k for k, (_, a, b) in enumerate(words_with_spans(S["text"])) if b > sp[0] and a < sp[1]}
    return idx

C_SPANS = {}
if RUN_C and C_RES:
    for lang in C_RES:
        test = [S for S in SENTENCES if S["lang"] == lang]
        if not test: continue
        rows = {}
        for (lg, l) in C_FITS:
            if lg != lang: continue
            sc = span_scores(lang, l, test)
            hot, oth = defaultdict(list), defaultdict(list)
            for S in test:
                r = sc[S["id"]]; hi = hot_word_index(S)
                for k, w in enumerate(r["words"]):
                    if not r["valid"][k]: continue
                    tgt = hot if k in hi else (oth if len(w) >= 4 else None)
                    if tgt is None: continue
                    for m in ["U", "U_raw", "half_life", "ntok"]: tgt[m].append(r[m][k])
            au = {m: auroc(hot[m], oth[m]) for m in ["U", "U_raw", "half_life", "ntok"]}
            # agreement with Part B (sentence-level, content words weighted by round-trip information weight)
            agree = None
            if B:
                ui, ub = [], []
                for S in test:
                    r = sc[S["id"]]; W = np.array(B[S["id"]]["word_weight"]); u = np.array(r["U"]); v = np.array(r["valid"])
                    if (W[v] > 0).any(): ui.append(float((W[v] * u[v]).sum() / W[v].sum())); ub.append(summary[S["id"]]["A"])
                agree = spearman(ui, ub)
            rows[l] = dict(scores=sc, auroc=au, spearman_with_roundtrip_A=agree, n_hot=len(hot["U"]), n_other=len(oth["U"]))
            print(f"[{lang}] layer {l}: AUROC hot vs other content words — U {au['U']:.2f}, U_raw {au['U_raw']:.2f}, "
                  f"half-life {1-au['half_life']:.2f} (shorter = harder), #tokens baseline {au['ntok']:.2f}; "
                  f"agreement with round trip ρ_s={agree if agree is None else round(agree, 2)}")
            if lang == "es" and "duende" in sc:
                r = sc["duende"]; d = dict(zip(r["words"], zip(r["U"], r["half_life"])))
                print("   Me gusta mucho el duende: " + ", ".join(f"{w} U={d[w][0]:.3f} (half-life {d[w][1]:.1f})" for w in ["duende", "gusta", "mucho"] if w in d))
        C_SPANS[lang] = rows
        save_json(f"C_spans_{lang}.json", rows)

if RUN_C and "es" in C_RES and C_SPANS.get("es"):
    held = torch.load(os.path.join(CACHE, "C_es_held.pt"), weights_only=False); meta = held["meta"]
    l = BEST_LAYER["es"]; F_ = C_FITS[("es", l)]; fit = F_["fit"]
    if any(m["morph"] for m in meta):
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import cross_val_score
        Z = (((held["X"][l].to(CDEV).double() - fit["mx"]) @ fit["Wx"]) @ fit["U"]).float().cpu().numpy()
        Z /= Z.std(0, keepdims=True) + 1e-8
        sh = np.zeros(Z.shape[1], bool); sh[F_["shared"].cpu().numpy()] = True
        def feat(m, key): return dict(kv.split("=") for kv in m["morph"].split("|") if "=" in kv).get(key)
        tasks = {
            "grammatical gender": [(i, feat(m, "Gender")) for i, m in enumerate(meta) if m["pos"] in ("NOUN", "ADJ") and feat(m, "Gender") in ("Masc", "Fem")],
            "number": [(i, feat(m, "Number")) for i, m in enumerate(meta) if m["pos"] in ("NOUN", "ADJ", "VERB") and feat(m, "Number") in ("Sing", "Plur")],
            "subjunctive vs indicative": [(i, feat(m, "Mood")) for i, m in enumerate(meta) if m["pos"] in ("VERB", "AUX") and feat(m, "Mood") in ("Sub", "Ind")],
            "imperfect vs preterite": [(i, feat(m, "Tense")) for i, m in enumerate(meta) if m["pos"] in ("VERB", "AUX") and feat(m, "Mood") == "Ind" and feat(m, "Tense") in ("Imp", "Past")],
            "noun vs verb (sanity)": [(i, m["pos"]) for i, m in enumerate(meta) if m["pos"] in ("NOUN", "VERB")],
        }
        probe = {}
        for name, items in tasks.items():
            labs = [b for _, b in items]
            if len(items) < 60 or min(labs.count(v) for v in set(labs)) < 20: print(f"  {name}: too few examples"); continue
            ii = np.array([i for i, _ in items]); y = np.array([b == sorted(set(labs))[0] for b in labs])
            clf = RidgeClassifier(alpha=float(len(ii)), class_weight="balanced")
            acc = {k: float(cross_val_score(clf, Z[ii][:, cols], y, cv=5, scoring="balanced_accuracy").mean())
                   for k, cols in [("all", np.ones_like(sh)), ("shared only", sh), ("private only", ~sh)] if cols.any()}
            clf.fit(Z[ii], y); wgt = np.ravel(clf.coef_) ** 2      # coef_ is (1, d) or (d,) depending on the sklearn version
            probe[name] = dict(n=len(ii), shared_share=float(wgt[sh].sum() / wgt.sum()), expected=float(sh.mean()), acc=acc)
            print(f"  {name:28s} n={len(ii):5d}  weight on shared directions {probe[name]['shared_share']:.2f} "
                  f"(chance {sh.mean():.2f})  accuracy all/shared/private = " + "/".join(f"{v:.2f}" for v in acc.values()))
        save_json("C_probes_es.json", dict(layer=l, probes=probe))

if RUN_C and C_RES:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for lang, col in zip(C_RES, [HARD, LOAN, EASY]):
        r = next(x for x in C_RES[lang] if x["layer"] == BEST_LAYER[lang]); k = np.arange(1, len(r["rho_in"]) + 1)
        ax.plot(k, r["rho_in"], color=col, lw=1, alpha=0.5, ls="--")
        ax.plot(k, r["rho_held"], color=col, lw=1.6, label=f"{lang}–en, layer {r['layer']} (held-out)")
        ax.axhline(r["null_thr"], color=col, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("canonical direction k"); ax.set_ylabel("ρ_k")
    ax.set_title("Round-trip spectrum (dashed: in-sample; dotted: shuffled-pair null)", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for lang, col in zip(C_RES, [HARD, LOAN, EASY]):
        ax.plot([r["layer"] / N_LAYERS for r in C_RES[lang]], [r["top50_held"] / 50 for r in C_RES[lang]], "o-", color=col, label=f"{lang}–en")
    if A: ax.axvline(A["detour_peak_layer"] / N_LAYERS, color="#888", ls="--", lw=1); ax.text(A["detour_peak_layer"] / N_LAYERS, ax.get_ylim()[0], " English detour peak", fontsize=8, color="#888")
    ax.set_xlabel("depth"); ax.set_ylabel("mean held-out ρ, top 50 directions"); ax.legend(frameon=False)
    ax.set_title("Where the two languages share most", loc="left", fontsize=9)
    savefig(fig, "C1_spectrum")

    if C_SPANS.get("es"):
        sc = C_SPANS["es"][BEST_LAYER["es"]]["scores"]
        pick = [("duende", "duende"), ("duende", "gusta"), ("sobremesa", "sobremesa"), ("madrugada", "madrugada"),
                ("tren", "tren"), ("hermano", "hermano"), ("gatos", "gatos")]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for sid, w in pick:
            if sid not in sc or w not in sc[sid]["words"]: continue
            k = sc[sid]["words"].index(w); col = EASY if SID[sid]["cat"] == "control" else HARD
            ax.plot(range(11), sc[sid]["R_curve"][k], color=col, lw=2, alpha=0.85)
            ax.text(10.2, sc[sid]["R_curve"][k][-1], f"{w} (half-life {sc[sid]['half_life'][k]:.1f})", fontsize=8, style="italic", color=col, va="center")
        ax.axhline(0.5, color="#888", ls=":", lw=1)
        ax.set_xlabel("number of round trips Spanish → English → Spanish (inside the model)"); ax.set_ylabel("share of the word that survives")
        ax.set_title("What survives repeated round trips", loc="left"); ax.set_xlim(0, 13.5)
        savefig(fig, "C2_retention")

    fig, ax = plt.subplots(figsize=(5.5, 3.4)); xs = np.arange(len(C_RES)); w = 0.35
    r_ = [next(x for x in C_RES[lang] if x["layer"] == BEST_LAYER[lang]) for lang in C_RES]
    ax.bar(xs - w / 2, [r["E_en_from_x"] for r in r_], w, color=GREY, label="English, predicted from the other language")
    ax.bar(xs + w / 2, [r["E_x_from_en"] for r in r_], w, color=HARD, label="other language, predicted from English")
    ax.set_xticks(xs); ax.set_xticklabels([f"{LANG_NAME[l]}–English" for l in C_RES]); ax.set_ylabel("share of variation predictable")
    ax.set_title("How much of each language survives the crossing", loc="left"); ax.legend(frameon=False, fontsize=8)
    savefig(fig, "C3_english_bias")

pack()
