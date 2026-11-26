"""Static HTML report (spec §Outputs): per text, reading cards, branch timeline, steering cards.

    python -m explorer report --model g3_4b_pt      -> results/explorer/<model>/report.html
Labels for readings come from explorer/data/readings_labels.yaml (multi-model schema under
`models.<model>.<text_id>.<reading_id>` or legacy flat `{text_id: {reading_id: label}}`),
with a contrastive keyword auto-labeler fallback (never bare "reading 0").
"""
import collections
import html
import json
import re
from pathlib import Path

import yaml

from explorer.mapgrid import _STOPWORDS, _WORD_RE, phrases_by_reading
from explorer.readings import normalize_text
from explorer.run import OUT

CSS = """
:root{--bg:#fcfcfb;--ink:#1f1f1e;--muted:#6b6a66;--line:#e6e5e0;--teal:#2f7d6d;--coral:#c4562f;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--ink:#ecebe7;--muted:#a3a29c;
--line:#33322f;--teal:#5fb3a1;--coral:#e3825f;--card:#201f1d}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif;margin:0;padding:24px 16px;
max-width:1100px;margin-inline:auto}
h1{font-size:22px}h2{font-size:18px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
.src{font-family:Charter,Georgia,serif;font-style:italic;font-size:17px;white-space:pre-wrap}
.muted{color:var(--muted);font-size:13px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tl span{display:inline-block;padding:2px 1px;margin:1px 0;border-radius:3px;font-family:Charter,Georgia,serif}
.tl .cur{outline:2px solid var(--coral)}.ok{color:var(--teal)}.no{color:var(--coral)}
table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
"""


def load_reading_labels(path=None) -> dict:
    """Load curated multi-model reading labels and text metadata from YAML."""
    p = Path(path) if path is not None else (Path(__file__).parent / "data" / "readings_labels.yaml")
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _labels():
    return load_reading_labels()


def text_metadata(labels_data: dict | None, tid: str) -> dict[str, str]:
    """Return {'title': ..., 'description': ...} for text `tid` with clean fallback."""
    texts_map = (labels_data or {}).get("texts", {})
    entry = texts_map.get(tid, {}) if isinstance(texts_map, dict) else {}
    if not isinstance(entry, dict):
        entry = {}
    title = str(entry.get("title") or tid.replace("_", " ").title()).strip()
    desc = str(entry.get("description") or "").strip()
    return {"title": title, "description": desc}


def _format_phrase(p: str) -> str:
    s = p.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _short_snippet(text: str, max_len: int = 36) -> str:
    s = normalize_text(text, collapse_newlines=True)
    if not s:
        return ""
    if len(s) <= max_len:
        return f'"{s}"'
    cut = s[: max_len - 1].rsplit(" ", 1)[0].strip(" ,;:-")
    return f'"{(cut or s[: max_len - 1]).strip()}…"'


def auto_label_readings(texts, labels, reading_ids, logp=None, top=3) -> dict[int, str]:
    """Generate clean, human-readable fallback labels for readings (never bare 'reading N')."""
    rd = [int(r) for r in reading_ids]
    if not rd:
        return {}

    S_tmp = {"texts": list(texts or []), "labels": list(labels or [])}
    cands = phrases_by_reading(S_tmp, rd, top=max(int(top), 1))

    out: dict[int, str] = {}
    for r in rd:
        kws = [k for k in cands.get(r, []) if k.strip() and not re.fullmatch(r"reading\s+\d+", k.strip(), flags=re.I)]
        if kws:
            out[r] = " / ".join(_format_phrase(k) for k in kws[: max(1, min(int(top), 2))])
            continue

        idxs = [i for i, l in enumerate(S_tmp["labels"]) if int(l) == r and i < len(S_tmp["texts"])]
        if idxs:
            freq = collections.Counter()
            for i in idxs:
                for w in _WORD_RE.findall(normalize_text(S_tmp["texts"][i], collapse_newlines=True)):
                    if w.lower() not in _STOPWORDS and (len(w) > 1 or not w.isascii()):
                        freq[w.lower()] += 1
            if freq:
                top_words = [_format_phrase(w) for w, _ in freq.most_common(2)]
                out[r] = " / ".join(top_words)
                continue

            if logp is not None and len(logp) > max(idxs):
                best_i = max(idxs, key=lambda i: float(logp[i]))
            else:
                best_i = idxs[0]
            snip = _short_snippet(S_tmp["texts"][best_i])
            if snip:
                out[r] = snip
                continue

        out[r] = f"Cluster #{r + 1}"
    return out


def _lookup_curated_map(labels_data: dict | None, model: str, tid: str) -> dict[int, str]:
    if not isinstance(labels_data, dict):
        return {}
    raw = None
    models = labels_data.get("models")
    if isinstance(models, dict) and isinstance(models.get(model), dict):
        raw = models[model].get(tid)
    if raw is None and tid in labels_data and tid not in ("texts", "models"):
        cand = labels_data.get(tid)
        if isinstance(cand, dict):
            raw = cand
    if not isinstance(raw, dict):
        return {}
    curated: dict[int, str] = {}
    for k, v in raw.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            continue
        sv = str(v).strip()
        if sv:
            curated[ki] = sv
    return curated


def resolve_reading_labels(labels_data: dict | None, model: str, tid: str, S: dict, extra_ids=()) -> dict[int, str]:
    """Resolve curated + fallback human-readable labels for all readings of text `tid`.

    Applies a single-reading collapse guard: if the sample collapsed to 1 reading while curated
    metadata defines >1 historical sub-readings, labels it 'Single unanimous reading' rather than
    misapplying a stale multi-cluster label.
    """
    primary_ids = [int(r) for r in S.get("readings", [])]
    all_ids = list(primary_ids)
    for r in extra_ids:
        ri = int(r)
        if ri not in all_ids:
            all_ids.append(ri)
    if not all_ids:
        return {}

    curated = _lookup_curated_map(labels_data, model, tid)

    # Single-reading collapse guard
    if len(primary_ids) == 1 and len(curated) > 1:
        r0 = primary_ids[0]
        resolved = {r0: "Single unanimous reading"}
        if len(all_ids) > 1:
            auto = auto_label_readings(S.get("texts", []), S.get("labels", []), all_ids, logp=S.get("logp"))
            for r in all_ids:
                if r not in resolved:
                    resolved[r] = auto.get(r) or f"Cluster #{r + 1}"
        return resolved

    auto = auto_label_readings(S.get("texts", []), S.get("labels", []), all_ids, logp=S.get("logp"))
    resolved = {}
    for r in all_ids:
        if r in curated:
            resolved[r] = curated[r]
        elif r in auto and auto[r]:
            resolved[r] = auto[r]
        else:
            resolved[r] = f"Cluster #{r + 1}"
    return resolved


def run(cfg):
    m = cfg.get("model", "g3_4b_pt")
    out = Path(cfg["results_dir"]) if cfg.get("results_dir") else (OUT / m)
    labs = load_reading_labels()
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
        meta = text_metadata(labs, tid)
        stp = out / "steering" / f"{tid}.jsonl"
        rows = [json.loads(l) for l in stp.read_text().splitlines()] if (stp.exists() and stp.read_text().strip()) else []
        extra_ids = [r[k] for r in rows for k in ("A", "B") if k in r]
        lab = resolve_reading_labels(labs, m, tid, S, extra_ids=extra_ids)
        shares = {int(k): v for k, v in S["shares"].items()}
        heading = f"{e(meta['title'])} <span class=muted>({e(tid)})</span>" if meta["title"] != tid else e(tid)
        desc_html = f"<p class=muted>{e(meta['description'])}</p>" if meta["description"] else ""
        H.append(f"<h2>{heading}</h2>{desc_html}<p class=src>{e(S['text'])}</p>"
                 f"<p class=muted>{len(S['readings'])} reading(s); silhouette {S['silhouette']:.2f}; entropy "
                 f"{S['entropy_bits']:.2f} bits over {S['k']} cluster(s); {len(S['ys'])} samples.</p><div class=cards>")
        for r in S["readings"]:
            ri = int(r)
            idx = [i for i, l in enumerate(S["labels"]) if int(l) == ri]
            top = sorted(idx, key=lambda i: -S["logp"][i])[:3]
            r_label = lab.get(ri, f"Cluster #{ri + 1}")
            H.append(f"<div class=card><b>{e(r_label)}</b> <span class=muted>{100 * shares[ri]:.0f}%</span>"
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
        if rows:
            H.append("<table><tr><th>towards</th><th>away from</th><th>KL</th><th>steered translation (greedy)</th>"
                     "<th>landed</th><th>success</th><th>baseline</th></tr>")
            for r in rows:
                if r["variant"] != "natural" or r["mode"] != "cursor" or r["kl"] not in (0.1, 0.25, 0.5, 1.0):
                    continue
                ra, rb = int(r["A"]), int(r["B"])
                na = lab.get(ra, f"Cluster #{ra + 1}")
                nb = lab.get(rb, f"Cluster #{rb + 1}")
                H.append(f"<tr><td>{e(na)}</td><td>{e(nb)}</td>"
                         f"<td>{r['kl']}</td><td class=src>{e(r['texts'][0])}</td>"
                         f"<td class={'ok' if r['greedy_success'] else 'no'}>{'yes' if r['greedy_success'] else 'no'}</td>"
                         f"<td>{100 * r['success_rate']:.0f}%</td><td>{100 * r['baseline_share_A']:.0f}%</td></tr>")
            H.append("</table>")
    H.append("</body></html>")
    p = out / "report.html"
    p.write_text("\n".join(H), encoding="utf-8")
    return p
