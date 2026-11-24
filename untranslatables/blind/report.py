"""Draft report (results/blind/report.md) from the stage outputs (§6).  Every line quotes its number; each section
opens with positive / null / mixed, decided by the pre-registered rules in PREDICTIONS.md.  The draft is meant to be
read and rewritten by hand (Claude/the user) before it goes to anyone; it never invents interpretations.
"""
import json

import numpy as np

from blind.common import EPS_HEAD, RESULTS, headline_layer, read_jsonl, stage

MODELS = ["g3_4b", "3b", "aya8b", "g3_12b"]


def J(m, n):
    p = RESULTS / m / n
    return json.loads(p.read_text()) if p.exists() else None


def verdict(passes):
    passes = [p for p in passes if p is not None]
    if not passes:
        return "not run"
    return "positive" if all(passes) else ("null" if not any(passes) else "mixed")


def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def e1_section(m):
    V = J(m, "validation.json")
    if not V:
        return [f"## E1 ({m}): not run", ""]
    h = V["headline"]
    out = [f"## E1 validation ({m}, layer {V['headline_layer']}, ε {V['eps']:g}): "
           f"{verdict([h['t1_pass'], h['t2_pass'], h['t3_pass'], V['t4_auto_pass']])}", ""]
    if "e0" in V:
        e = V["e0"]
        out.append(f"- E0: source states across channels max |Δh|/rms = {e['identical_h_max_rel']:.1e} "
                   f"({'pass' if e['identical_h_pass'] else 'FAIL'}); gradient vs finite differences, best correlation "
                   f"per layer {', '.join(f'{k}: {v:.2f}' for k, v in e['grad_fd_best_corr'].items())} "
                   f"({'pass' if e['grad_fd_pass'] else 'FAIL'}).")
    s2 = h["src2"]
    out.append(f"- Spanish-into-Spanish check: {f(100 * s2['share_near_null'] if s2['share_near_null'] is not None else None, 0)}% "
               f"of items near the null (need ≥ 75%); median U_src2 {f(s2['median_U_src2'])} vs median U_en "
               f"{f(s2['median_U_en'])} bits. {'PASS' if h['src2_pass'] else 'FAIL: prompt sensitivity dominates'}.")
    for name, t in h["t1_pooled"].items():
        out.append(f"- Test 1 ({name}, gender + plural you + formal you, n = {t.get('n')}): median difference "
                   f"{f(t.get('median_diff'))} bits, 95% CI [{f(t.get('ci95', [None])[0])}, {f(t.get('ci95', [None, None])[1])}], "
                   f"Wilcoxon p = {f(t.get('p_greater'), 3)}, {t.get('n_positive')} of {t.get('n')} positive.")
    t2 = h["t2"]
    out.append(f"- Test 2 (profession): en − it {f(t2['en>it'].get('median_diff'))} (p {f(t2['en>it'].get('p_greater'), 3)}), "
               f"fr − it {f(t2['fr>it'].get('median_diff'))} (p {f(t2['fr>it'].get('p_greater'), 3)}), fr − en "
               f"{f(t2['fr-en'].get('median_diff'))} CI {t2['fr-en'].get('ci95')}. {'pass' if h['t2_pass'] else 'fail'}.")
    out.append(f"- Test 3 (controls near the null): {f(100 * (h['t3']['share_near_null'] or 0), 0)}% of "
               f"{h['t3']['n']} (item, language) pairs; by language {h['t3']['by_lang']}. {'pass' if h['t3_pass'] else 'fail'}.")
    out.append("- Test 4 (steering flips, automatic pre-count; to be confirmed by reading): "
               + "; ".join(f"{g} {v['auto_flipped_en_unchanged']}/{v['n']}" for g, v in V["t4_auto"].items()) + ".")
    out.append(f"- Copy diagnostic: median share of U_en carried by surface directions {f(V.get('surface_share_median'))}.")
    out.append("- Median U by type (bits): " + "; ".join(
        f"{g}: " + ", ".join(f"{L} {f(v)}" for L, v in d.items()) for g, d in h["median_U_by_type"].items()))
    rob = V["robustness"]
    out.append("- Robustness (test passes at each layer × ε): " + "; ".join(
        f"{k}: " + "".join("✓" if v else "✗" for v in d.values()) for k, d in rob.items()) + " (order t1 t2 t3 src2).")
    return out + [""]


def e2_section(m):
    H = J(m, "heatmaps.json")
    rows = [r for r in read_jsonl(RESULTS / m / "spectra_e2.jsonl") if r["eps"] == EPS_HEAD
            and r["layer"] == headline_layer(m)]
    if not rows:
        return [f"## E2 ({m}): not run", ""]
    W = [r for r in rows if r["scope"] == "whole" and r["target"] == "en"]
    out = [f"## E2 our sentences ({m}): see numbers (verdict written by hand)", ""]
    cats = sorted({r["group"] for r in W})
    out.append("- Median U into English by category: " + "; ".join(
        f"{c} {f(np.median([r['U']['med'] for r in W if r['group'] == c]))}" for c in cats))
    k = str(headline_layer(m))
    for i in ("es_duende", "es_duende_flamenco", "es_duende_cuento"):
        if H and i in H and k in H[i]["layers"]:
            toks, U = H[i]["tokens"], H[i]["layers"][k]["en"]["U"]
            out.append(f"- {i}: per-token U_en " + " ".join(f"{t.strip()}={u:.2f}" for t, u in zip(toks, U)))
    for p in ("es_passage_plain", "es_passage_dense"):
        whole = next((r for r in W if r["item"] == p), None)
        sents = [r for r in W if r["item"].startswith(p + "_s")]
        if whole and sents:
            out.append(f"- {p}: whole-passage U_en {f(whole['U']['med'])} vs sum of its sentences alone "
                       f"{f(sum(r['U']['med'] for r in sents))} (sentences: "
                       + ", ".join(f(r['U']['med']) for r in sorted(sents, key=lambda r: r['item'])) + ")")
    return out + [""]


def e3_section(m):
    P = J(m, "practices.json")
    if not P:
        return [f"## E3 practices ({m}): not run", ""]
    out = [f"## E3 practices ({m}): see numbers", ""]
    for c in ("plain", "p_keep", "p_gloss", "p_foot", "p_thick"):
        v = [r["practices"].get(c, {}) for r in P]
        out.append(f"- {c}: median U {f(np.nanmedian([x.get('U') or np.nan for x in v]))} raw / "
                   f"{f(np.nanmedian([x.get('U_lennorm') or np.nan for x in v]))} length-normalised; share of the plain "
                   f"translation's blind directions seen (ρ ≥ 0.5) {f(np.nanmedian([x.get('share_seen') if x.get('share_seen') is not None else np.nan for x in v]))} raw / "
                   f"{f(np.nanmedian([x.get('share_seen_lennorm') if x.get('share_seen_lennorm') is not None else np.nan for x in v]))} length-normalised")
    return out + [""]


def e5_section():
    out = ["## E5 models: structural vs model-induced", ""]
    rows = {}
    for m in MODELS:
        V = J(m, "validation.json")
        if V:
            rows[m] = V["headline"]["median_U_by_type"]
    if len(rows) < 2:
        return out + ["- fewer than two models run", ""]
    types = list(next(iter(rows.values())))
    out.append("| type | " + " | ".join(rows) + " |")
    out.append("|---|" + "---|" * len(rows))
    for t in types:
        out.append(f"| {t} | " + " | ".join(f"{f(rows[m][t].get('en'))} / {f(rows[m][t].get('it'))}" for m in rows) + " |")
    out.append("")
    out.append("(cells: median U into English / into Italian, bits, each model at its own 50% layer)")
    return out + [""]


@stage("report")
def run(cfg):
    lines = ["# Translation blind spots: draft report (generated; to be rewritten by hand)", "",
             "Definition and method: ../../HANDOFF_blind_spots.md; predictions: ../../blind/PREDICTIONS.md.", "",
             "**Honesty notes (§9).** Local (small changes around each text's representation); model- and "
             "prompt-relative (the Spanish-into-Spanish check bounds the prompt effect, quoted with every headline); "
             "only directions above the split-half null count; 'blind' means this model's translation output is "
             "insensitive, not that no translation could carry it.", ""]
    V = J("g3_4b", "validation.json")
    if V:
        h = V["headline"]
        ok = sum([h["t1_pass"], h["t2_pass"], h["t3_pass"], V["t4_auto_pass"]])
        if ok < 2:
            lines.insert(1, "**The built-in blind spots were mostly NOT found (E1 tests passed: "
                            f"{ok} of 4). The definition is not measuring what we want; the atlas was not run.**\n")
    for m in MODELS:
        lines += e1_section(m)
    lines += e2_section("g3_4b") + e3_section("g3_4b") + e5_section()
    p = RESULTS / "report.md"
    p.write_text("\n".join(lines))
    return [p]
