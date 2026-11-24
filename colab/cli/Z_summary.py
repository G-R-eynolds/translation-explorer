# Reload earlier stages' results, so this stage also works in a fresh kernel (after 00_setup.py).
A = load_json("A_detour.json") if have("A_detour.json") else None
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
summary = load_json("B_summary.json")["per_sentence"] if have("B_summary.json") else {}
D = load_json("D_mediation.json") if have("D_mediation.json") else None
C_RES = {lang: load_json(f"C_spectrum_{lang}.json") for lang in PAIRS if have(f"C_spectrum_{lang}.json")}
BEST_LAYER = {lang: max(rows, key=lambda r: r["top50_held"])["layer"] for lang, rows in C_RES.items()}
C_SPANS = {lang: {int(k): v for k, v in load_json(f"C_spans_{lang}.json").items()} for lang in C_RES if have(f"C_spans_{lang}.json")}


lines = [f"# Summary: {MODEL_ID}", "", f"{N_LAYERS} layers, d = {D_MODEL}. Every number is inside this one model.", ""]
if A:
    lines += ["## A. English detour", f"- English peaks at layer {A['detour_peak_layer']}/{N_LAYERS}; Italian overtakes at {A['italian_overtakes_at']}.",
              f"- Peak mean P(English) = {np.array(A['P_en']).mean(0).max():.3f} over {len(A['P_en'])} words.", ""]
if have("B_summary.json"):
    bs = load_json("B_summary.json"); ps = bs["per_sentence"]
    lines += ["## B. Round trip (masked)", f"- Tolerance ε = {bs['epsilon']:.3f}; hot-word AUROC = {bs['auroc_hot_words']:.3f}."]
    for c in ["untranslatable", "construction", "idiom", "control"]:
        v = [x for x in ps.values() if x["cat"] == c]
        if v: lines.append(f"- {c}: U = {np.mean([x['U'] for x in v]):.2f} doublings, borrowing buys {np.nanmean([x['borrow_gain'] for x in v]):.3f} (masked: {np.nanmean([x['borrow_gain_masked'] for x in v]):.3f}).")
    fa = B["duende"]; wl = dict(zip(fa["words"], fa["rungs"]["B1"]["word_left_masked"]))
    lines += [f"- *Me gusta mucho el duende* → {ladder['duende']['B1']!r}; leftover duende {wl.get('duende', np.nan):.2f}, gusta {wl.get('gusta', np.nan):.2f}, mucho {wl.get('mucho', np.nan):.2f}.",
              "- Sense recovered for *duende* (standard / keep / gloss / footnote / thick): " + " / ".join(f"{fa['sense']['duende']['norm'][r]:.2f}" for r in ["B1", "B2", "B3", "B5", "B6"]), ""]
for lang in C_RES:
    r = next(x for x in C_RES[lang] if x["layer"] == BEST_LAYER[lang])
    lines += [f"## C. Spectrum {lang}–en (layer {r['layer']})",
              f"- {r['n_shared']} shared directions above the null (of {D_MODEL}); fit on {r['n_fit_pairs']} word pairs.",
              f"- English predicted from {lang}: {r['E_en_from_x']:.3f}; {lang} predicted from English: {r['E_x_from_en']:.3f}; bias {r['bias']:+.3f} (standardised {r['bias_std']:+.3f})."]
    if C_SPANS.get(lang):
        au = C_SPANS[lang][BEST_LAYER[lang]]["auroc"]; ag = C_SPANS[lang][BEST_LAYER[lang]]["spearman_with_roundtrip_A"]
        lines.append(f"- AUROC hot vs other words: U {au['U']:.2f}, U_raw {au['U_raw']:.2f}, #tokens baseline {au['ntok']:.2f}; agreement with round trip {ag}.")
    lines.append("")
if D:
    for l, d in D.items():
        ab = d["ablation"]; hot = [r for r in ab if r["hot"]]
        f = lambda r, t: r[f"{t}_private"] / r[f"{t}_full"] if abs(r[f"{t}_full"]) > 1e-6 else np.nan
        lines += [f"## D. Mediation (layer {l})",
                  f"- Hot words, private-part share of harm: translation {np.nanmean([f(r, 'translate') for r in hot]):.2f}, explanation {np.nanmean([f(r, 'explain') for r in hot]):.2f}.",
                  "- Flip fractions (private / shared / full): " + "; ".join(f"{x['id']} {x['flip']['private']:+.2f}/{x['flip']['shared']:+.2f}/{x['flip']['full']:+.2f}" for x in d["interchange"]), ""]
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(lines))
print("\n".join(lines))

pack()
