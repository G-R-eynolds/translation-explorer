"""results/report.html (handoff §9): plain summary, the figures in order with finding + how-to-read, what this
does and doesn't show, and a technical appendix (methods with arXiv citations, exclusions, prompts, familiarity
table, every Claude-written list).  Images are embedded so the file stands alone.

    python report.py --model 8b
The summary is DRAFTED from the numbers and flagged as such; a person should rewrite it after reading the results.
"""
import argparse, base64, csv, html, json

from ut.common import DATA, MODELS, ROOT, describe, load_json, load_yaml, out_dir
from ut.words import load_words

ORDER = ["fig01_recipe_cards_lead", "fig01_recipe_cards_all", "fig02_how_many_english_words", "fig03_neighbourhood",
         "fig04_leftover_ranking", "fig05_what_the_model_would_say", "fig06_into_english_vs_others",
         "fig07_does_it_think_in_english", "fig08_which_way_does_it_lean", "fig09_lost_through_english",
         "fig10_push_away_from_english", "fig11_uncertainty_map", "fig12_before_and_after_the_note", "fig13_model_vs_humans"]
METHODS = [
    ("Concept-space recipes: non-negative sparse decomposition over another language's vocabulary", ["1601.03764", "2610.08303"]),
    ("Language identity removed by centring, no Procrustes alignment", ["2004.05160", "2605.24310"]),
    ("Chance baselines by permutation", ["2602.14486"]),
    ("Nearest-neighbour (CSLS) gap baseline", ["2605.24310"]),
    ("Patching readout", ["2411.08745"]),
    ("Latent language by logit lens", ["2402.10588", "2303.08112"]),
    ("Steering toward/away from English", ["2505.13141", "2502.15603"]),
    ("Distinction collapse ('language spilling')", ["2601.12549"]),
    ("Meaning clustering and semantic entropy; kernel language entropy", ["2302.09664", "2405.20003"]),
    ("One factor at a time; diagnostics not an additive partition", ["2603.24967", "2209.03302"]),
    ("Clarification ensembling", ["2311.08718"]),
    ("Other ignorance signals: cross-model, small vs large, iterative prompting", ["2604.17112", "2402.03563", "2406.02543"]),
    ("Linear probes for ambiguity / unfamiliarity", ["2506.01205"]),
]
CAVEATS = """<ul>
<li>Everything is <b>inside these models</b>: it says how a model represents a word, not what the word means.</li>
<li>Frequency, token count and familiarity can each fake 'untranslatability'. The baseline comparison shows whether the
leftover beats frequency and token count alone; where it does not, the result is not evidence about meaning.</li>
<li>The two-dimensional pictures are flattened sketches of spaces with thousands of dimensions.</li>
<li>Ignorance and multiplicity are separate diagnostics, not two halves of one total.</li>
<li>Word lists, contexts, glosses and distinction sentences were drafted by Claude and need checking (appendix).</li>
<li>Romanesco items have few contexts and placeholder glosses until Belli's footnotes are added.</li></ul>"""


def img(path):
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">' if path.exists() else ""


def draft_summary(rd):
    s = []
    b = load_json(rd / "baselines.json") if (rd / "baselines.json").exists() else None
    if b:
        sc = b["headline"]["scores"] or b["all_words"]["scores"]
        lv, fq = sc["leftover"]["auroc"][0], sc["zipf"]["auroc"][0]
        s.append(f"Inside {describe(rd.name)}, the part of a word English cannot rebuild separates untranslatables from "
                 f"ordinary words with AUROC {lv:.2f}, against {fq:.2f} for word frequency alone"
                 + (" — so frequency explains it at least as well." if fq >= lv else "."))
    ll = load_json(rd / "latent_lang.json") if (rd / "latent_lang.json").exists() else None
    if ll:
        pk = max(max(c[0] for c in p["english_share"]) for p in ll["pairs"].values())
        s.append(f"Translating between two non-English languages, up to {pk:.0%} of the model's working answer is English midway through.")
    ds = load_json(rd / "distinctions.json") if (rd / "distinctions.json").exists() else None
    if ds:
        ex = [r["excess_error_greedy"] for r in ds["summary"].values() if r["type"] == "english_lacks"
              and r.get("excess_error_greedy") is not None and r["excess_error_greedy"] == r["excess_error_greedy"]]
        if ex:
            s.append(f"Distinctions English lacks are mistranslated {sum(ex) / len(ex):+.0%} more often than ones it keeps.")
    ig = load_json(rd / "ignorance.json") if (rd / "ignorance.json").exists() else None
    if ig:
        s.append(f"{ig['cell_accuracy']:.0%} of test words land in the 'knows it / doesn't' × 'one reading / many' corner their kind predicts.")
    s.append("These are first-pass numbers; read the caveats below before quoting any of them.")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=MODELS)
    a = ap.parse_args()
    rd = out_dir(a.model)
    caps = {}
    for p in [rd / "figures" / "captions.json"]:
        if p.exists():
            caps.update(json.loads(p.read_text()))
    papers = {r["arxiv_id"]: r for r in csv.DictReader(open(ROOT.parent / "literature" / "papers.csv"))}
    cite = lambda ids: ", ".join(f'<a href="{papers[i]["abs_url"]}">[{i}]</a>' if i in papers else f"[{i}]" for i in ids)
    present = [n for n in ORDER if n in caps]
    figs = "".join(f"""<section><h3>Figure {k + 1}</h3>{img(ROOT / 'results' / caps[n]['file'])}
<p class=how><b>How to read this:</b> {html.escape(caps[n]['how_to_read'])}</p></section>""" for k, n in enumerate(present))
    fam = load_json(rd / "familiarity.json")["words"] if (rd / "familiarity.json").exists() else {}
    fam_rows = "".join(f"<tr><td>{html.escape(w)}</td><td>{html.escape(r['definition_own'])}</td><td>{html.escape(r['explanation_en'])}</td>"
                       f"<td>{r['provisional']}</td><td>{r['zipf']}</td><td>{r['n_tokens']}</td></tr>" for w, r in fam.items())
    words = "".join(f"<details><summary>{html.escape(w.id)} — {html.escape(w.category)}</summary><p><i>{html.escape(w.gloss_en)}</i></p>"
                    f"<p>Gloss: {html.escape(w.gloss_src)}</p><ol>{''.join(f'<li>{html.escape(c)}</li>' for c in w.contexts)}</ol>"
                    f"<p>Senses: {html.escape('; '.join(s['label'] + ' — ' + s.get('clarify', '') for s in w.senses))}</p></details>"
                    for w in load_words())
    dist = load_yaml(DATA / "distinctions.yaml")["pairs"]
    dist_html = "".join(f"<details><summary>{p['id']} ({p['type']})</summary><ol>"
                        + "".join(f"<li>{html.escape(i['s'])} — <i>{html.escape(i['en'])}</i> → {i['x']}</li>" for i in p["items"])
                        + "</ol></details>" for p in dist)
    rec = load_json(rd / "recipes.json") if (rd / "recipes.json").exists() else {}
    excl = "".join(f"<li>{html.escape(k)}: {html.escape(', '.join(v))}</li>" for k, v in rec.get("exclusions", {}).items() if v)
    prompts = {}
    for f, keys in (("familiarity.json", ["prompts"]), ("spread_samples.json", ["ask", "again"]),
                    ("distinctions.json", ["prompt"]), ("patching.json", ["example_source_prompt", "target_prompt"])):
        if (rd / f).exists():
            j = load_json(rd / f)
            prompts[f] = {k: j.get(k) for k in keys}
    out = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Untranslatables inside language models</title>
<style>body{{font:17px/1.55 Georgia,serif;max-width:980px;margin:auto;padding:24px 16px;color:#1f1f1e;background:#fcfcfb}}
h1,h2,h3{{font-family:system-ui,sans-serif}}img{{max-width:100%}}.how{{color:#555}}.draft{{background:#fff4d6;padding:10px 14px;border-radius:6px}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #e6e5e0;padding:4px 6px;vertical-align:top;overflow-wrap:anywhere}}
.scroll{{overflow-x:auto}}body *{{overflow-wrap:anywhere}}
section{{margin:36px 0}}details{{margin:4px 0}}pre{{white-space:pre-wrap;font-size:12px;background:#f3f2ee;padding:8px}}</style>
<h1>Untranslatables inside language models</h1>
<p>Main model: {html.escape(describe(a.model))}.</p>
<h2>Summary</h2><p class=draft><b>Draft, generated from the numbers; to be rewritten by a person.</b><br>{' '.join(html.escape(x) for x in draft_summary(rd))}</p>
<h2>Figures</h2>{figs}
<h2>What this does and doesn't show</h2>{CAVEATS}
<h2>Technical appendix</h2>
<h3>Methods and sources</h3><ul>{''.join(f'<li>{html.escape(t)}: {cite(ids)}</li>' for t, ids in METHODS)}</ul>
<p>Concept layer: {rec.get('layer', '–')}; centring: {html.escape(rec.get('centring', ''))}.  Tuned lens not used: no pretrained lens exists for these checkpoints.</p>
<h3>Prompts (verbatim)</h3><pre>{html.escape(json.dumps(prompts, ensure_ascii=False, indent=1))}</pre>
<h3>Familiarity check</h3><div class=scroll><table><tr><th>word</th><th>own-language definition</th><th>English explanation</th><th>flag</th><th>Zipf</th><th>tokens</th></tr>{fam_rows}</table></div>
<h3>Pool exclusions (cognates, the word itself, loans)</h3><ul>{excl}</ul>
<h3>Word list, contexts and glosses (drafted by Claude; to be checked)</h3>{words}
<h3>Distinction items (drafted by Claude; to be checked)</h3>{dist_html}"""
    (ROOT / "results" / "report.html").write_text(out)
    print("wrote results/report.html")


if __name__ == "__main__":
    main()
