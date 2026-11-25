"""Static HTML report (spec §Outputs): per text, reading cards, branch timeline, steering cards.

    python -m explorer report --model g3_4b_pt      -> results/explorer/<model>/report.html
Labels for readings come from explorer/data/readings_labels.yaml if present ({text_id: {reading_id: label}}), written by
hand from the texts; otherwise readings are shown as "reading <id>".
"""
import html
import json
from pathlib import Path

import yaml

from explorer.run import OUT

CSS = """
:root{--bg:#fcfcfb;--ink:#1f1f1e;--muted:#6b6a66;--line:#e6e5e0;--teal:#2f7d6d;--coral:#c4562f;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--ink:#ecebe7;--muted:#a3a29c;
--line:#33322f;--teal:#5fb3a1;--coral:#e3825f;--card:#201f1d}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif;margin:0;padding:24px 16px;
max-width:1100px;margin-inline:auto}
h1{font-size:22px}h2{font-size:18px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
.src{font-family:Charter,Georgia,serif;font-style:italic;font-size:17px}
.muted{color:var(--muted);font-size:13px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tl span{display:inline-block;padding:2px 1px;margin:1px 0;border-radius:3px;font-family:Charter,Georgia,serif}
.tl .cur{outline:2px solid var(--coral)}.ok{color:var(--teal)}.no{color:var(--coral)}
table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
"""


def _labels():
    p = Path(__file__).parent / "data" / "readings_labels.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def run(cfg):
    m = cfg.get("model", "g3_4b_pt")
    out = OUT / m
    labs = _labels()
    summ = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    e = html.escape
    H = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Interpretation explorer</title><style>{CSS}</style></head><body>",
         f"<h1>Interpretation explorer, step 1 ({e(m)})</h1>",
         f"<p class=muted>Decision rule: {e(summ.get('rule', ''))}. Result: <b>{'PASS' if summ.get('pass') else 'not passed'}</b>; "
         f"hard texts with 2+ readings {summ.get('n_hard_with_2plus')}/{summ.get('n_hard')}; steering success "
         f"(natural, cursor) {summ.get('steer_success_rate')}; control readings {summ.get('control_readings')}.</p>"]
    if summ.get("control_warning"):
        H.append("<p class=no><b>Warning: the plain control split into more than one reading; clusters may follow "
                 "wording, not meaning.</b></p>")
    for sp in sorted((out / "samples").glob("*.json")):
        S = json.loads(sp.read_text())
        tid = S["id"]
        lab = labs.get(tid, {})
        shares = {int(k): v for k, v in S["shares"].items()}
        H.append(f"<h2>{e(tid)}</h2><p class=src>{e(S['text'])}</p>"
                 f"<p class=muted>{len(S['readings'])} reading(s); silhouette {S['silhouette']:.2f}; entropy "
                 f"{S['entropy_bits']:.2f} bits over {S['k']} cluster(s); {len(S['ys'])} samples.</p><div class=cards>")
        for r in S["readings"]:
            idx = [i for i, l in enumerate(S["labels"]) if l == r]
            top = sorted(idx, key=lambda i: -S["logp"][i])[:3]
            H.append(f"<div class=card><b>{e(lab.get(r, f'reading {r}'))}</b> <span class=muted>{100 * shares[r]:.0f}%</span>"
                     + "".join(f"<p>{e(S['texts'][i])}</p>" for i in top) + "</div>")
        H.append("</div>")
        bp = out / "branches" / f"{tid}.json"
        if bp.exists():
            B = json.loads(bp.read_text())
            mx = max(B["fisher_top"] or [1]) or 1
            spans = "".join(f"<span class='{'cur' if j == B['t'] else ''}' title='entropy {B['entropy'][j]:.2f}, Fisher {B['fisher_top'][j]:.3g}' "
                            f"style='background:color-mix(in srgb, var(--coral) {int(70 * B['fisher_top'][j] / mx)}%, transparent)'>"
                            f"{e(tok)}</span>" for j, tok in enumerate(B["ref_tokens"]))
            H.append(f"<p class=muted>Where it could go differently (darker = the next word is more open; outlined = the "
                     f"cursor used for steering, {e(B['why'])}):</p><p class=tl>{spans}</p>")
        stp = out / "steering" / f"{tid}.jsonl"
        if stp.exists() and stp.read_text().strip():
            rows = [json.loads(l) for l in stp.read_text().splitlines()]
            H.append("<table><tr><th>towards</th><th>away from</th><th>KL</th><th>steered translation (greedy)</th>"
                     "<th>landed</th><th>success</th><th>baseline</th></tr>")
            for r in rows:
                if r["variant"] != "natural" or r["mode"] != "cursor" or r["kl"] not in (0.1, 0.25, 0.5, 1.0):
                    continue
                na, nb = lab.get(r["A"], f"reading {r['A']}"), lab.get(r["B"], f"reading {r['B']}")
                H.append(f"<tr><td>{e(na)}</td><td>{e(nb)}</td>"
                         f"<td>{r['kl']}</td><td class=src>{e(r['texts'][0])}</td>"
                         f"<td class={'ok' if r['greedy_success'] else 'no'}>{'yes' if r['greedy_success'] else 'no'}</td>"
                         f"<td>{100 * r['success_rate']:.0f}%</td><td>{100 * r['baseline_share_A']:.0f}%</td></tr>")
            H.append("</table>")
    H.append("</body></html>")
    p = out / "report.html"
    p.write_text("\n".join(H))
    return p
