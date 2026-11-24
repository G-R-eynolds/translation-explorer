"""Step 6 (CPU): results/sent/report.md, written only from the computed JSON (no number is typed by hand).

    python -m sent.report --models 3b 8b        # every model that has curves.json; the first is the lead
"""
import argparse

import numpy as np

from ut.common import ROOT, load_json
from sent.data import sent_dir
from sent.prompts import PRACTICE

CAT_ORDER = ["untranslatable_in_context", "construction", "idiom", "dialect", "control"]


def f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def ci(t, nd=3):
    return f"{t[0]:+.{nd}f} ({t[1]:+.{nd}f} to {t[2]:+.{nd}f})"


def model_section(m):
    d = sent_dir(m)
    C, V = load_json(d / "curves.json"), load_json(d / "validation.json")
    sp_ = load_json(d / "spread.json") if (d / "spread.json").exists() else None
    tg = load_json(d / "targeting.json") if (d / "targeting.json").exists() else None
    out = [f"## {V['model']}", ""]
    if V["stop"]:
        out += ["**Validation stop flags (read these before any figure):**", ""] + [f"- {s}" for s in V["stop"]] + [""]
    else:
        out += ["Validation raised no stop flags.", ""]
    out += [f"*Warning:* {w}" for w in V.get("warnings", [])] + ([""] if V.get("warnings") else [])
    lc = V["layer_choice"]
    sel = lc["per_layer"][lc["layer"] - 1]
    out += ["### Validation (§6)", "",
            f"- **Layer for d_CL:** {lc['layer']} (neighbours {lc['neighbours']}), tau = {lc['tau']}; AUROC matched vs "
            f"mismatched controls {f(sel['auroc'], 3)} (95% CI {f(sel['auroc_ci'][0])}-{f(sel['auroc_ci'][1])}). "
            f"Layers where the spelling control won: "
            f"{[p['layer'] for p in lc['per_layer'] if p['spelling_wins']] or 'none'}.",
            f"- **Tolerance epsilon** (90th percentile of d(control, B1)): round trip {f(V['epsilon']['rt'], 3)}, "
            f"concept layer {f(V['epsilon']['cl'], 3)}."]
    for sp, name in (("rt", "round trip"), ("cl", "concept layer")):
        lad = V["ladder"][sp]
        out.append(f"- **Ladder ({name}):** best B6 minus best B1 = {ci(lad['B6_minus_B1'])}; "
                   f"{'passed' if lad['passed'] else 'FAILED'}. Spearman of rung length vs mean best d: "
                   f"{f(lad['length_vs_d_spearman'])}.")
        hm = V["heatmap"][sp]
        out.append(f"- **Heat map ({name}):** AUROC of word leftover, annotated hot words vs other content words: "
                   f"{f(hm['auroc_hot_vs_other_content'][0], 3)} (CI {f(hm['auroc_hot_vs_other_content'][1])}-"
                   f"{f(hm['auroc_hot_vs_other_content'][2])}; {hm['n_hot']} hot, {hm['n_other']} other).")
    if "fathers_sentence" in V:
        for sp in ("rt", "cl"):
            fs = V["fathers_sentence"][sp]
            out.append(f"- **Me gusta mucho el duende ({sp}):** leftover duende {f(fs['leftover']['duende'])}, gusta "
                       f"{f(fs['leftover']['gusta'])}, mucho {f(fs['leftover']['mucho'])} (n/a = the word carries no "
                       f"information weight); expected order duende > gusta > mucho "
                       f"{'holds' if fs['order_holds'] else 'does not hold' if fs['order_holds'] is False else 'cannot be checked'}.")
    ag = V["agreement"]
    out.append(f"- **Agreement of the two spaces:** Spearman across sentences, A {f(ag['A']['spearman'])} "
               f"(n={ag['A']['n']}), U {f(ag['U']['spearman'])}; per-token leftover on B1 {f(ag['token_leftover_B1']['spearman'])} "
               f"(n={ag['token_leftover_B1']['n']}).")
    for bt, v in V.get("second_backtranslators", {}).items():
        out.append(f"- **Second back-translator {v['model']}:** Spearman of A with the main round trip "
                   f"{f(v['A_vs_main']['spearman'])} (n={v['A_vs_main']['n']}).")
    out += ["", "### English detour (§6.5)", "",
            f"d_CL of the greedy standard translation at the concept layer vs the detour layers {V['detour']['layers']} "
            "(negative = the English looks closer in the detour layers).", "",
            "| category | n | concept layer | detour | detour - concept (95% CI) |", "|---|---|---|---|---|"]
    for cat in CAT_ORDER:
        if cat in V["detour"]["by_category"]:
            x = V["detour"]["by_category"][cat]
            out.append(f"| {cat} | {x['n']} | {f(x['concept_layer'], 3)} | {f(x['detour'], 3)} | {ci(x['detour_minus_concept'])} |")
    if V.get("spelling_check"):
        out += ["", "### Spelling / copying check (§2.1, §2.2)", "",
                "| space | gain of B2 (keep the word) | B2 gain with source words masked | share surviving | gain of B3 (gloss) |",
                "|---|---|---|---|---|"]
        for sp, x in V["spelling_check"].items():
            out.append(f"| {sp} | {f(x['B2_gain'], 3)} | {f(x['B2_gain_masked'], 3)} | {f(x['share_surviving_masking'])} | "
                       f"{f(x['B3_gain'], 3)} |")
    out += ["", "### Untranslatability by category", "",
            "U = doublings of length beyond the standard translation before d <= epsilon (censored = never reached; "
            "the bound is used). A = mean distortion over log-spaced lengths from L0 to L0 x 2^"
            f"{C['K']}.", "", "| category | n | U round trip | censored | U concept | censored | A round trip | A concept |",
            "|---|---|---|---|---|---|---|---|"]
    S = C["sentences"]
    for cat in CAT_ORDER:
        ids = [s for s, e in S.items() if e["record"]["category"] == cat]
        if not ids:
            continue
        row = [cat, str(len(ids))]
        for sp in ("rt", "cl"):
            cv = [S[s]["spaces"][sp]["whole"] for s in ids if S[s]["spaces"][sp]["whole"]]
            row += [f(np.mean([c["U"] for c in cv])), f"{sum(c['censored'] for c in cv)}/{len(cv)}"]
        for sp in ("rt", "cl"):
            cv = [S[s]["spaces"][sp]["whole"] for s in ids if S[s]["spaces"][sp]["whole"]]
            row.append(f(np.mean([c["A"] for c in cv]), 3))
        out.append("| " + " | ".join(row) + " |")
    out += ["", "### Per sentence", "", "| sentence | category | standard translation | U rt | U cl | hot spans: U rt (construction gap) |",
            "|---|---|---|---|---|---|"]
    for sid, e in sorted(S.items(), key=lambda t: CAT_ORDER.index(t[1]["record"]["category"])):
        u = lambda sp: (lambda c: "n/a" if not c else (">" if c["censored"] else "") + f(c["U"]))(e["spaces"][sp]["whole"])
        spans = []
        for x in e["spaces"]["rt"]["spans"]:
            if x["span"]["kind"] != "hot" or not x["curve"]:
                continue
            g = x["construction_gap"]
            spans.append(f"*{x['span']['text']}* {'>' if x['curve']['censored'] else ''}{f(x['curve']['U'])}"
                         + (f" (gap {g['U']:+.2f})" if g else ""))
        out.append(f"| *{e['record']['text']}* | {e['record']['category']} | {e['text_B1'].splitlines()[0]} | {u('rt')} | "
                   f"{u('cl')} | {'; '.join(spans)} |")
    if sp_:
        out += ["", "### Spread over standard translations (§1)", "",
                f"{sp_['n']} standard translations at T={sp_['temperature']}; H in bits over NLI meaning clusters.", "",
                "| sentence | span | kind | H | clusters | usable | top renderings |", "|---|---|---|---|---|---|---|"]
        for r in sp_["spans"]:
            top = ", ".join(f"{c['rep'] or '(left out)'} {c['share']:.0%}" for c in r["clusters"][:3])
            out.append(f"| {r['sid']} | *{r['span']['text']}* | {r['span']['kind']} | {f(r['H_bits'])} | {r['n_clusters']} | "
                       f"{r['n_used']}/{len(r['renderings'])} | {top} |")
        hot = [r["H_bits"] for r in sp_["spans"] if r["span"]["kind"] == "hot" and r["H_bits"] is not None]
        probe = [r["H_bits"] for r in sp_["spans"] if r["span"]["kind"] == "probe" and r["H_bits"] is not None]
        out += ["", f"Mean H: hot spans {f(np.mean(hot) if hot else None)} bits (n={len(hot)}), control probe words "
                f"{f(np.mean(probe) if probe else None)} bits (n={len(probe)})."]
    if tg:
        out += ["", "### Targeting test (§3)", "",
                f"Glosses chosen by the {tg['chooser']} heat map. Cut = distortion of the standard translation minus "
                "distortion after the glosses (positive = the glosses helped).", "",
                "| space | cut, targeted | cut, random | targeted - random (95% CI) | sentences where targeted won |",
                "|---|---|---|---|---|"]
        for sp, x in tg["summary"].items():
            out.append(f"| {sp} | {ci(x['cut_targeted'])} | {ci(x['cut_random'])} | {ci(x['targeted_minus_random'])} | "
                       f"{x['share_sentences_targeted_better']:.0%} |")
        out.append(f"\nTokens added: targeted {f(tg['added_length']['targeted'][0], 1)}, random "
                   f"{f(tg['added_length']['random'][0], 1)}.")
    figd = ROOT / "results" / "sent" / "figures" / m
    if (figd / "captions.json").exists():
        cap = load_json(figd / "captions.json")
        out += ["", "### Figures", ""]
        for k, v in cap.items():
            if isinstance(v, dict) and "png" in v:
                out += [f"**{v['title']}**  ", f"![{k}](figures/{m}/{v['png']})  ", v["how_to_read"], ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["3b"])
    a = ap.parse_args()
    models = [m for m in a.models if (sent_dir(m) / "curves.json").exists()]
    out = ["# Untranslatability beyond single words: distortion against length", "",
           "Generated by `python -m sent.report` from `results/sent/<model>/*.json`. Design: "
           "`../HANDOFF_sentence_untranslatability.md`.", "",
           "**How to read every number here.** Each is measured *inside one model*, and both meaning spaces come "
           "from English-leaning models: that is a limitation and also part of the question. D_s(L) is the lower "
           "envelope over a finite set of candidates, so it is an upper bound: better translations at a given length "
           "may exist. Practices on the ladder: " + ", ".join(f"{k} {v}" for k, v in PRACTICE.items()) + ".", ""]
    for m in models:
        out += model_section(m) + [""]
    p = ROOT / "results" / "sent" / "report.md"
    p.write_text("\n".join(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
