"""results/internals/report.md and COLAB_RECOMMENDATION.md, written only from the computed JSON (HANDOFF_internals §8, §10).

Each pre-registered prediction (PREDICTIONS.md, P1-P16) is marked confirmed / refuted / unclear with the statistic it
rests on.  "unclear" covers: inputs missing, a confidence interval straddling the threshold, or too few items.
"""
import json
from glob import glob
from pathlib import Path

import numpy as np

from internals.common import MAIN, RESULTS, stage
from internals.figures import MODELS_N1, NAME

R2_MODELS = ["8b", "gemma12b", "qwen14b", "nemo12b"]


def J(*p):
    f = RESULTS.joinpath(*p)
    return json.loads(f.read_text()) if f.exists() else None


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return f"{x[0]:+.{nd}f} (95% CI {x[1]:+.{nd}f} to {x[2]:+.{nd}f})"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def verdict(ok, unclear=False):
    return "unclear" if unclear or ok is None else ("confirmed" if ok else "refuted")


def peak(m, pair):
    q = J(m, f"n1_quick_{pair}.json")
    return q["peak_layer"] if q else None


def spectrum(m, pair, layer=None):
    layer = layer or peak(m, pair)
    return J(m, f"n1_spectrum_{pair}_L{layer}.json") if layer else None


def predictions():
    P = []
    # P1 detour
    res = {}
    for m in ("gemma12b", "qwen14b", "nemo12b", "aya8b", "8b"):
        r = J(m, "replication_R1.json")
        if r:
            res[m] = {k: max(x[0] for x in v["english_share"]) for k, v in r["pairs"].items()}
    ok = all(max(v.values()) > 0.5 for v in res.values()) if res else None
    P.append(("P1", "English-detour peak share > 0.5 on every new model", verdict(ok), json.dumps(
        {m: {k: round(x, 2) for k, x in v.items()} for m, v in res.items()})))
    # P2-P5 R2
    st2, st3, st4, st5 = {}, {}, {}, {}
    for m in ("gemma12b", "qwen14b"):
        V = J(m, "replication_R2", "validation.json")
        if not V:
            continue
        c = V["categories"]
        if "control" in c and "untranslatable_in_context" in c:
            st2[m] = (c["control"]["rt"]["U_mean"], c["untranslatable_in_context"]["rt"]["U_mean"])
        b = V["borrowing"]
        if "untranslatable_in_context" in b and "control" in b:
            st3[m] = (b["untranslatable_in_context"]["gain_unmasked"], b["control"]["gain_unmasked"])
        sr = V.get("sense_recovery")
        if sr:
            st4[m] = (sr.get("auroc_probe_vs_hot_B1"), sr.get("B6_minus_B1_hot"))
            st5[m] = sr.get("duende_rank_among_hot")
    P.append(("P2", "masked R2: U(control) < U(untranslatable) on gemma12b and qwen14b",
              verdict(all(a < b for a, b in st2.values()) if st2 else None), json.dumps(st2)))
    P.append(("P3", "borrowing gain (unmasked) positive for untranslatables and larger than for controls",
              verdict(all(u[1] > 0 and u[0] > c[0] for u, c in st3.values()) if st3 else None),
              json.dumps({m: dict(untranslatable=u, control=c) for m, (u, c) in st3.items()})))
    ok4 = None
    if st4:
        ok4 = all(a is not None and a[1] > 0.5 and b is not None and b[1] > 0 for a, b in st4.values())
    P.append(("P4", "sense recovery of B1 worse for hot spans than control probes (AUROC probe>hot, CI > 0.5) and B6 > B1",
              verdict(ok4), json.dumps(st4)))
    ok5 = all(r is not None and r[0] < r[1] / 2 for r in st5.values()) if st5 else None
    P.append(("P5", "duende's B1 sense recovery below the median of hot spans", verdict(ok5),
              json.dumps({m: f"{r[0]} of {r[1]} hot spans recover less" if r else None for m, r in st5.items()})))
    # P6 shared peak near detour
    st6 = {}
    for m in MODELS_N1:
        q = J(m, "n1_quick_es-en.json"); r = J(m, "replication_R1.json")
        if q and r:
            dp = int(np.median([v["peak_layer"] for v in r["pairs"].values()]))
            st6[m] = dict(shared_peak=q["peak_layer"], detour_peak=dp, n_layers=max(int(k) for k in q["layers"]))
    P.append(("P6", "shared-direction peak in the middle layers, within 4 layers of the detour peak",
              verdict(all(abs(v["shared_peak"] - v["detour_peak"]) <= 4 and 0.2 < v["shared_peak"] / v["n_layers"] < 0.85
                          for v in st6.values()) if st6 else None), json.dumps(st6)))
    # P7 U validation (main)
    S = J(MAIN, "n1_spans_summary.json")
    if S and "es-en" in S:
        a = S["es-en"]["at_peak"]
        au = a.get("auroc_hot_vs_control_U")
        fs = a.get("fathers_sentence", {})
        cat = a.get("span_U_by_category", {})
        sp = a.get("spearman_with_R2_U")
        parts = dict(auroc=au, fathers_order=fs.get("order_holds_U"),
                     untranslatable_gt_control=(cat.get("untranslatable_in_context", {}).get("mean", np.nan) >
                                                cat.get("control", {}).get("mean", np.nan)) if cat else None,
                     spearman_R2=sp)
        ok = bool(au and au[0] > 0.6 and fs.get("order_holds_U") and parts["untranslatable_gt_control"] and sp and sp["rho"] > 0)
        unclear = bool(au and au[1] < 0.6 < au[0])
        P.append(("P7", "U separates es hot words from control words (AUROC > 0.6), duende > gusta > mucho, "
                  "U(untranslatable) > U(control), positive Spearman with R2", verdict(ok, unclear), json.dumps(parts, default=float)))
    else:
        P.append(("P7", "U validation", "unclear", "no N1 span summary"))
    # P8 English bias
    st8 = {}
    for m in MODELS_N1:
        es, hi = spectrum(m, "es-en"), spectrum(m, "hi-en")
        st8[m] = dict(es=es["bias"]["raw"]["bias"] if es else None, hi=hi["bias"]["raw"]["bias"] if hi else None)
    have = {m: v for m, v in st8.items() if v["es"] is not None or v["hi"] is not None}
    ok = all((v["es"] is None or v["es"] > 0) and (v["hi"] is None or v["hi"] > 0) for v in have.values()) if have else None
    larger = [v["hi"] > v["es"] for v in have.values() if v["es"] is not None and v["hi"] is not None]
    aya = have.get("aya8b")
    others = [v["hi"] for m, v in have.items() if m in ("gemma12b", "8b", "qwen14b") and v["hi"] is not None]
    aya_smaller = aya and aya["hi"] is not None and others and aya["hi"] < min(others)
    P.append(("P8a", "E_en - E_es > 0 and E_en - E_hi > 0 for every model (raw)", verdict(ok), json.dumps(have)))
    P.append(("P8b", "the hi gap is larger than the es gap", verdict(all(larger) if larger else None,
                                                                   unclear=bool(larger) and 0 < np.mean(larger) < 1),
              f"{sum(larger)} of {len(larger)} models"))
    P.append(("P8c", "the hi gap is smaller for Aya than for Gemma, Llama and Qwen", verdict(bool(aya_smaller) if aya else None),
              json.dumps(dict(aya=aya, others=others))))
    # P9 hub
    H = J(MAIN, "n1_hub.json")
    if H:
        L = sorted(H["layers"], key=int)[0]
        h = H["layers"][L]
        if all(k in h for k in ("es-en", "hi-en", "es-hi")):
            ok = h["es-hi"]["n_shared"] < min(h["es-en"]["n_shared"], h["hi-en"]["n_shared"]) and \
                 h["es-hi"]["mean_rho2_top20"] < min(h["es-en"]["mean_rho2_top20"], h["hi-en"]["mean_rho2_top20"])
            P.append(("P9", "English is a hub: es-hi has fewer shared directions and lower rho^2 than es-en and hi-en (FLORES)",
                      verdict(ok), json.dumps({k: dict(n_shared=h[k]["n_shared"], rho2_top20=round(h[k]["mean_rho2_top20"], 3))
                                               for k in ("es-en", "hi-en", "es-hi")})))
        b = h.get("bhojpuri_minus_hindi_U")
        P.append(("P10", "Bhojpuri words have higher U than Hindi on the same sentences (hi-en maps)",
                  verdict(b["hindi_centring"][1] > 0 if b else None, unclear=bool(b) and b["hindi_centring"][1] <= 0 < b["hindi_centring"][0]),
                  fmt(b["hindi_centring"]) if b else "n/a"))
    else:
        P += [("P9", "English as hub", "unclear", "no FLORES hub result"), ("P10", "Bhojpuri", "unclear", "no FLORES hub result")]
    # P11 probes
    Pr = J(MAIN, "n1_probes.json")
    if Pr:
        exp = {"es-en": dict(gender="private", number="shared", imperfect_vs_preterite="shared", subjunctive="private",
                             ser_vs_estar="private", dative_experiencer="private"),
               "hi-en": dict(gender="private", ne_ergative="private", honorific_aap_vs_tum="private", dative_subject="private")}
        detail, hits, n = {}, 0, 0
        for pair, e in exp.items():
            for f, want in e.items():
                r = Pr.get(pair, {}).get("features", {}).get(f)
                if r and "shared_share" in r:
                    got = "shared" if r["shared_share"] > r["random_expectation"] else "private"
                    detail[f"{pair}:{f}"] = f"{got} ({r['shared_share']:.3f} vs r/d {r['random_expectation']:.3f}; predicted {want})"
                    hits += got == want; n += 1
                else:
                    detail[f"{pair}:{f}"] = r.get("skipped", "missing") if r else "missing"
        P.append(("P11", "probe predictions (gender private; number, tense shared; subjunctive, ser/estar, dative experiencer "
                  "private; Hindi gender, ne, honorific, dative subject private)",
                  verdict(hits == n if n else None, unclear=0 < hits < n), f"{hits}/{n} as predicted; " + json.dumps(detail, ensure_ascii=False)))
    else:
        P.append(("P11", "probes", "unclear", "no probe results"))
    # P12 MLP
    M = J(MAIN, "n1_mlp.json")
    if M:
        gains = {f"{p}:{d}": v[d]["mlp_r2"] - v[d]["linear_r2"] for p, v in M.items() for d in ("X_to_Y", "Y_to_X") if d in v}
        P.append(("P12", "MLP held-out R^2 gain over linear < 0.05", verdict(all(g < 0.05 for g in gains.values())),
                  json.dumps({k: round(g, 3) for k, g in gains.items()})))
    else:
        P.append(("P12", "MLP check", "unclear", "no MLP results"))
    # P13-P14 N2
    N2 = J(MAIN, "n2_summary.json")
    if N2:
        sps = {k: v["spearman_rhobar_tau_star"] for k, v in N2["per_layer"].items()}
        P.append(("P13", "Spearman(rho-bar, tau*) > 0 across features", verdict(all(s is not None and s > 0 for s in sps.values())),
                  json.dumps(sps)))
        au = {k: v.get("U_SAE_auroc_hot_vs_probe_es") for k, v in N2["per_layer"].items()}
        P.append(("P14", "U_SAE separates hot spans from control probes (AUROC > 0.6); duende art features vanish (see cards)",
                  verdict(all(a and a[0] > 0.6 for a in au.values()), unclear=True), json.dumps(au) +
                  " (the art-vs-folklore part needs reading the cards by eye: marked unclear until then)"))
    else:
        P += [("P13", "N1-N2 agreement", "unclear", "no N2 results"), ("P14", "U_SAE / duende features", "unclear", "no N2 results")]
    # P15-P16 N3
    N3 = J(MAIN, "n3_summary.json")
    if N3:
        g = N3["groups"]
        key = next((k for k in g if k.startswith("es-en") and k.endswith("_hot")), None)
        ck = key.replace("_hot", "_control") if key else None
        dh, dc = (g[key].get("dissociation_private") if key else None), (g[ck].get("dissociation_private") if ck in g else None)
        ok = bool(dh and dc and dh["ci"][1] > dc["ci"][2])
        P.append(("P15", "private part: E_mono_sense - E_trans larger for hot spans than control", verdict(ok, unclear=not (dh and dc)),
                  f"hot {fmt(dh['ci']) if dh else 'n/a'}; control {fmt(dc['ci']) if dc else 'n/a'}"))
    elif J(MAIN, "n3_skipped.json"):
        P.append(("P15", "N3 dissociation", "unclear", "N3 not run: " + J(MAIN, "n3_skipped.json")["reason"]))
    else:
        P.append(("P15", "N3 dissociation", "unclear", "no N3 results"))
    ip = RESULTS / MAIN / "n3_interchange.jsonl"
    if ip.exists():
        rows = [json.loads(l) for l in open(ip)]
        addr = [r for r in rows if r["type"] == "address"]
        moved_src = [r["logodds_src"]["patch_private"] - r["logodds_src"]["clean"] for r in addr]
        moved_en = [r["logodds_en"]["patch_private"] - r["logodds_en"]["clean"] for r in addr]
        ok = bool(addr) and np.mean(moved_src) > 0 and abs(np.mean(moved_en)) < abs(np.mean(moved_src))
        gender = [r for r in rows if r["type"] == "gender"]
        flips = sum(r["logodds_en"]["patch_private"] > 0 > r["logodds_en"]["clean"] for r in gender)
        P.append(("P16", "tu/usted, aap/tum: private patch moves the source side more than the English (dropped); gender pairs "
                  "reported either way", verdict(ok, unclear=not addr),
                  f"address: mean dLO src {np.mean(moved_src) if addr else float('nan'):+.2f}, en {np.mean(moved_en) if addr else float('nan'):+.2f}; "
                  f"gender: English flipped to the other sense in {flips} of {len(gender)} patched runs"))
    else:
        P.append(("P16", "interchange", "unclear", "no interchange results"))
    return P


def colab():
    out = ["# What this run says about Colab (HANDOFF_internals §10)", "",
           "Written by `python -m internals report` from the computed results; every number is linked to a file in "
           "`results/internals/`.", ""]
    S = J(MAIN, "n1_spans_summary.json")
    g = (S or {}).get("gate_N1", {})
    out += ["## 1. N1", f"- Validation gate (AUROC hot vs control > 0.6, es, peak layer): **{'passed' if g.get('passed') else 'failed' if g else 'not run'}**"
            f" (AUROC {fmt(g.get('auroc'))}).", "- Spectrum and bias by model size (8B -> 12B -> 14B), at each model's peak-sharing layer:", "",
            "| model | pair | peak layer | shared directions | mean rho^2 (shared) | E_en - E_other (raw) | held-out E_en - E_other |",
            "|---|---|---|---|---|---|---|"]
    for m in MODELS_N1:
        for pair in ("es-en", "hi-en"):
            sp = spectrum(m, pair)
            if sp:
                out.append(f"| {NAME[m]} | {pair} | {sp['layer']} | {sp['n_shared']} | {sp['mean_rho2_shared']:.3f} | "
                           f"{sp['bias']['raw']['bias']:+.3f} | {sp['bias_heldout']['E_Y'] - sp['bias_heldout']['E_X']:+.3f} |")
    N2 = J(MAIN, "n2_summary.json")
    out += ["", "## 2. N2", f"- Spearman(rho-bar, tau*) per SAE layer: {json.dumps({k: v['spearman_rhobar_tau_star'] for k, v in N2['per_layer'].items()}) if N2 else 'not run'}.",
            f"- Quantisation used for N2: {N2['used'] if N2 else 'n/a'}."]
    N3 = J(MAIN, "n3_summary.json")
    out += ["", "## 3. N3", "- Dissociation groups (mean, 95% CI) are in `gemma12b/n3_summary.json`." if N3 else
            "- N3 not run or skipped (see `gemma12b/n3_skipped.json`)."]
    out += ["", "## 4. What failed or was flat", "- See the 'refuted' and 'unclear' rows of `report.md`; they are the things not "
            "to scale without a change of design.", "", "## 5. Cost on this desktop (RTX 4070, nf4), from the stage timings", "",
            "| model | stage | minutes |", "|---|---|---|"]
    for f in sorted(glob(str(RESULTS / "*" / ".done" / "*.json"))):
        d = json.loads(Path(f).read_text())
        m = Path(f).parent.parent.name
        out.append(f"| {m} | {Path(f).stem} | {d.get('minutes', 'n/a')} |")
    out += ["", "Rough scaling to Colab: an A100 80 GB runs a 70B model in 4-bit at roughly the same tokens/s as the 4070 runs a "
            "12B in nf4 (both memory-bandwidth bound: 70B/12B ~ 6x the bytes, 2 TB/s vs 0.5 TB/s ~ 4x the bandwidth), so "
            "multiply the GPU stages above by ~1.5 for a 70B on an A100 and ~0.8 on an H100. The CPU stages (cca, spans, "
            "probes) grow with d^3: d = 8192 for 70B-class models is ~8-10x the 12B's cost.",
            "", "Reusable artifacts (compute once): the corpora and alignments (`$UT_CORPUS`), the streamed covariances "
            "`acc_L*.pt` and held-out pairs per model (all of N1 is recomputable from them), and the canonical bases "
            "`basis_L*.npz` (reused by N2 and N3)."]
    return out


@stage("report")
def run(cfg):
    P = predictions()
    lines = ["# Untranslatability inside the model: report (internals run)", "",
             "Every number is **model-relative and corpus-relative**: inside model M, relative to the conventional "
             "translations in corpus C. 'Private' means 'not linearly predictable from the aligned English word'; it does "
             "not by itself mean meaningless or untranslatable (dropped, converted nonlinearly, or carried elsewhere in the "
             "sentence are kept apart). Definitions: `DEFINITIONS.md`. The Hindi materials are drafts until a speaker checks "
             "them. Aya Expanse 8B was not accessible (licence), so Aya 23 8B stands in for it.", "",
             "## Pre-registered predictions", "", "| id | prediction | verdict | statistic |", "|---|---|---|---|"]
    for pid, text, v, stat in P:
        lines.append(f"| {pid} | {text} | **{v}** | {stat.replace('|', '/')} |")
    S = J(MAIN, "n1_spans_summary.json")
    if S:
        ha = S.get("hot_alignment") or {}
        lines += ["", "## Alignment coverage of hot words (§13)", "",
                  "Share of hot-span words with a 1-to-1 link to the model's own standard translation (unaligned hot words are "
                  "themselves evidence):", "", "```", json.dumps(ha.get("by_category", ha), ensure_ascii=False, indent=1), "```"]
    for m in MODELS_N1:
        mt = J(m, "n1_quick_es-en.json") or J(m, "n1_quick_hi-en.json")
        if not mt:
            continue
        lines += ["", f"## {NAME[m]}", ""]
        for pair in ("es-en", "hi-en"):
            q = J(m, f"n1_quick_{pair}.json")
            if q:
                sp = spectrum(m, pair)
                lines.append(f"- {pair}: quick-pass peak layer {q['peak_layer']}; at that layer "
                             + (f"{sp['n_shared']} shared directions (null q99 {sp['rho_null_heldout_q99']:.3f}), lambda {sp['lam']}, "
                                f"E_X {sp['bias']['raw']['E_X']:.3f}, E_Y {sp['bias']['raw']['E_Y']:.3f}; robustness "
                                f"{json.dumps(sp.get('robustness', {}))}" if sp else "no full spectrum"))
        Sm = J(m, "n1_spans_summary.json")
        if Sm:
            for pair in ("es-en", "hi-en"):
                if pair in Sm:
                    a = Sm[pair]["at_peak"]
                    lines.append(f"- {pair} spans at the peak: AUROC hot vs control U {fmt(a.get('auroc_hot_vs_control_U'))}, "
                                 f"U_raw {fmt(a.get('auroc_hot_vs_control_U_raw'))}; father's sentence {a.get('fathers_sentence')}; "
                                 f"Spearman with R2 {a.get('spearman_with_R2_U')}")
    for m in R2_MODELS:
        V = J(m, "replication_R2", "validation.json")
        if V:
            lines += ["", f"## R2 replication: {NAME.get(m, m)}", "",
                      f"- stop flags: {V['stop'] or 'none'}", f"- ladder (masked): {fmt(V['ladder']['rt']['B6_minus_B1'])}; "
                      f"heat-map AUROC {fmt(V['heatmap']['rt']['auroc_hot_vs_other_content'])}",
                      f"- father's sentence (masked): {V.get('fathers_sentence', {}).get('rt')}",
                      f"- categories: " + json.dumps({k: round(v['rt']['U_mean'], 2) for k, v in V['categories'].items()}),
                      f"- what borrowing buys (B1 - B2_borrow, unmasked / masked): " + json.dumps(
                          {k: (round(v['gain_unmasked'][0], 3), round(v['gain_masked'][0], 3)) for k, v in V['borrowing'].items()}),
                      f"- sense recovery: {json.dumps(V.get('sense_recovery', {}).get('by_kind_and_rung', {}), default=float)[:600]}"]
    cap = J("figures", "captions.json")
    if cap:
        lines += ["", "## Figures", ""]
        for k, v in cap.items():
            if isinstance(v, dict) and "png" in v:
                lines += [f"**{v['title']}** ({v['audience']})  ", f"![{k}](figures/{v['png']})  ", v["how_to_read"], ""]
    p = RESULTS / "report.md"
    p.write_text("\n".join(lines))
    q = RESULTS / "COLAB_RECOMMENDATION.md"
    q.write_text("\n".join(colab()))
    for pid, text, v, stat in P:
        print(f"  {pid:4s} {v:10s} {text[:70]}")
    return [p, q]
