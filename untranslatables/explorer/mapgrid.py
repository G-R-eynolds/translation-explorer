"""Precomputed map of translation directions at the fork, for the interactive page (24 Nov 2026).

    python -m explorer.mapgrid --text duende_flamenco [--n 15]   -> results/explorer/<model>/map_<text>.json

At the fork (cursor after the reference prefix, layer L = 50%):
  m_r     mean gradient of reading r's continuations (its "direction" in the model's internal state)
  plane   the top-2 principal axes of {m_r - mean_r m_r}; anchors = reading coordinates, scaled so the farthest = 1
  dots    every member translation's own gradient projected on the plane (same centring and scale)
  vector at map point p = α_unit · (p1 u1 + p2 u2), with α_unit calibrated (measured, cursor mode) so that the
          farthest anchor's vector gives a mean per-token KL of 0.25 nats along the reference
At each grid point (n × n over [-1.6, 1.6]²), steering in cursor mode (added at the fork and every later position):
  shares  probability of each candidate phrase after the prefix (phrases taken from the model's own translations,
          "he has <phrase>;"), normalised over the candidates, summed per reading; plus the candidates' total mass
  text    the greedy continuation
"""
import argparse
import collections
import json
import re

import numpy as np
import torch

from explorer.grad_readings import spectral_basis
from explorer.readings import normalize_text
from explorer.run import OUT, _generate_for_style


_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "no", "nor", "not", "of", "on", "one", "or", "our", "out", "she",
    "so", "that", "the", "their", "them", "there", "these", "they", "this",
    "those", "to", "too", "up", "us", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would", "you",
    "your", "el", "la", "los", "las", "un", "una", "de", "del", "en", "y",
    "o", "que", "es", "se", "por", "para", "con", "su", "sus", "al", "lo",
    "le", "les", "le", "les", "du", "des", "au", "aux", "et", "est", "der",
    "die", "das", "ein", "eine", "und", "ist", "im", "il", "di", "che", "da",
})
_CLAUSE_SPLIT_RE = re.compile(r"[\n;.,!?—–:]")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", flags=re.UNICODE)
_TRIM_CHARS = " \t\r\n\"'()[]{}—–-:,;.!?¡¿«»"


def build_map_basis(G_by_reading, rd, whiten=True, eps=1e-2):
    """Build an orthonormal 2D steering plane via Fisher-natural spectral_basis.

    Preserves full rank-2 even when R == 2 (axis 2 comes from orthogonal within-reading sample
    variance) and handles R == 1 or degenerate rank-1 inputs via deterministic Gram-Schmidt padding.
    """
    rd_int = [int(r) for r in rd]
    if not rd_int:
        raise ValueError("readings must be non-empty")

    parts = []
    labs = []
    for r in rd_int:
        Gr = np.asarray(G_by_reading[r], dtype=np.float64)
        if Gr.ndim == 1:
            Gr = Gr[None, :]
        if len(Gr) == 0:
            raise ValueError(f"Reading {r} has no gradient rows")
        parts.append(Gr)
        labs.append(np.full(len(Gr), r, dtype=int))

    G_mat = np.vstack(parts)
    G_labels = np.concatenate(labs)
    d = G_mat.shape[1]

    sb = spectral_basis(G_mat, G_labels, rd_int, n_dims=2, whiten=whiten, eps=eps)
    axes = np.asarray(sb["axes"], dtype=np.float64)
    anchors = {int(r): list(sb["anchors"][int(r)]) for r in rd_int}
    dots = {int(r): [list(pt) for pt in sb["dots"][int(r)]] for r in rd_int}

    k_got = axes.shape[0]
    if k_got < 2:
        if d < 2:
            raise ValueError(f"Cannot build 2D map basis in ambient dimension d={d}")
        axes_list = [axes[i].copy() for i in range(k_got)]
        while len(axes_list) < 2:
            if axes_list:
                A_cur = np.vstack(axes_list)
                proj_sq = (A_cur ** 2).sum(axis=0)
                order = np.argsort(proj_sq)
            else:
                order = np.arange(d)
            added = False
            for j in order:
                v = np.zeros(d, dtype=np.float64)
                v[int(j)] = 1.0
                for u in axes_list:
                    v -= float(np.dot(v, u)) * u
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-12:
                    axes_list.append(v / nrm)
                    added = True
                    break
            if not added:
                break
        axes = np.vstack(axes_list)
        pad_n = axes.shape[0] - k_got
        if pad_n > 0:
            for r in rd_int:
                anchors[r] = anchors[r] + [0.0] * pad_n
                dots[r] = [pt + [0.0] * pad_n for pt in dots[r]]

    return dict(
        axes=axes,
        anchors=anchors,
        dots=dots,
        variance_explained=float(sb["variance_explained"]),
        scale=float(sb["scale"]),
        readings=rd_int,
    )


def _clean_clause(text: str, max_words: int = 4) -> str:
    s = normalize_text(text, collapse_newlines=False)
    head = _CLAUSE_SPLIT_RE.split(s, maxsplit=1)[0].strip(_TRIM_CHARS)
    words = head.split()
    if not words:
        return ""
    if len(words) > max_words:
        words = words[:max_words]
    while len(words) > 1 and words[-1].lower().strip(_TRIM_CHARS) in _STOPWORDS:
        words.pop()
    cand = " ".join(words).strip(_TRIM_CHARS)
    return cand


def _extract_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    norm = normalize_text(text, collapse_newlines=False)
    out: list[str] = []
    for line in norm.split("\n"):
        toks = _WORD_RE.findall(line)
        L = len(toks)
        for n in range(n_min, min(n_max, L) + 1):
            for i in range(L - n + 1):
                span = toks[i : i + n]
                low = [w.lower() for w in span]
                if all(w in _STOPWORDS for w in low):
                    continue
                if n == 1 and len(low[0]) <= 1 and low[0].isascii():
                    continue
                if n >= 2 and (low[0] in _STOPWORDS or low[-1] in _STOPWORDS):
                    continue
                out.append(" ".join(span))
    return out


def _detect_shared_prefix(texts: list[str], min_words: int = 2, min_frac: float = 0.6) -> str:
    if len(texts) < 2:
        return ""
    word_lists = [normalize_text(t, collapse_newlines=True).split() for t in texts if t.strip()]
    if len(word_lists) < 2:
        return ""
    needed = max(2, int(np.ceil(min_frac * len(word_lists))))
    prefix_words: list[str] = []
    max_len = max(len(w) for w in word_lists)
    for idx in range(max_len):
        c = collections.Counter(w[idx].lower() for w in word_lists if len(w) > idx)
        if not c:
            break
        word, cnt = c.most_common(1)[0]
        if cnt < needed:
            break
        # Keep only lists matching the majority prefix so far
        word_lists = [w for w in word_lists if len(w) > idx and w[idx].lower() == word]
        prefix_words.append(word)
    if len(prefix_words) < min_words:
        return ""
    return " ".join(prefix_words)


def _strip_prefix_ci(text: str, prefix: str) -> str | None:
    if not prefix:
        return None
    norm = normalize_text(text, collapse_newlines=False)
    p_norm = normalize_text(prefix, collapse_newlines=True)
    if not p_norm:
        return None
    # Match prefix words allowing flexible whitespace/newlines in the text
    p_words = p_norm.split()
    pat = r"^\s*" + r"\s+".join(re.escape(w) for w in p_words) + r"(?:\b|\s+)"
    m = re.match(pat, norm, flags=re.IGNORECASE)
    if not m:
        return None
    return norm[m.end():].strip()


def _rank_and_dedup(per_r_doc_counts, per_r_n, display_form, rd, top, total_docs, filter_universal=True):
    rd_int = [int(r) for r in rd]
    R = len(rd_int)
    global_counts = collections.Counter()
    for r in rd_int:
        global_counts.update(per_r_doc_counts[r])

    out = {}
    for r in rd_int:
        nr = max(per_r_n[r], 1)
        scored = []
        for key, cnt in per_r_doc_counts[r].items():
            if filter_universal and R >= 2 and total_docs > 0 and (global_counts[key] / total_docs) >= 0.85:
                continue
            p_in = cnt / nr
            if R >= 2:
                other_rates = [
                    per_r_doc_counts[o].get(key, 0) / max(per_r_n[o], 1)
                    for o in rd_int if o != r
                ]
                p_out = float(np.mean(other_rates)) if other_rates else 0.0
                lift = (p_in - p_out) * (0.5 + p_in)
            else:
                lift = p_in
            # Prefer slightly richer (2-3 word) phrases on exact ties
            n_words = len(key.split())
            tie_bonus = 1 if 1 <= n_words <= 3 else 0
            scored.append((lift, p_in, tie_bonus, cnt, key))

        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        chosen_keys: list[str] = []
        chosen_texts: list[str] = []
        for lift, p_in, _, _, key in scored:
            if len(chosen_keys) >= top:
                break
            # Deduplicate strict token-boundary sub/super-phrases when a stronger form is already selected
            key_toks = set(key.split())
            redundant = False
            for ck in chosen_keys:
                ck_toks = set(ck.split())
                if key == ck or key in ck or ck in key:
                    if key_toks <= ck_toks or ck_toks <= key_toks:
                        redundant = True
                        break
            if redundant:
                continue
            chosen_keys.append(key)
            chosen_texts.append(display_form[(r, key)])
        out[r] = chosen_texts
    return out


def phrases_by_reading(S, rd, top=4, prefix=None, cont_texts_by_reading=None):
    """Extract contrastive candidate phrases per reading for any text, poem, or language.

    Mode 1 (when `cont_texts_by_reading` is provided): extracts leading 1-4 word clauses from
    each reading's aligned fork continuations and ranks by contrastive lift.
    Mode 2 (standalone): strips explicit or auto-detected shared prefix and extracts both
    post-prefix clauses and all-line Unicode 1-3-grams ranked by contrastive lift.
    """
    rd_int = [int(r) for r in rd]
    if not rd_int:
        return {}

    # Gather full texts per reading (always preserving ALL lines!)
    texts_by_r: dict[int, list[str]] = {r: [] for r in rd_int}
    for t, l in zip(S.get("texts", []), S.get("labels", [])):
        li = int(l)
        if li in texts_by_r:
            texts_by_r[li].append(str(t))

    # Mode 1: fork continuations supplied directly by mapgrid.main()
    if cont_texts_by_reading is not None:
        per_r_doc_counts = {r: collections.Counter() for r in rd_int}
        per_r_n = {r: 0 for r in rd_int}
        display_counts: dict[tuple[int, str], collections.Counter] = collections.defaultdict(collections.Counter)
        for r in rd_int:
            conts = cont_texts_by_reading.get(r, [])
            per_r_n[r] = len(conts)
            for c_txt in conts:
                cl = _clean_clause(c_txt, max_words=4)
                if not cl:
                    continue
                k = cl.lower()
                per_r_doc_counts[r][k] += 1
                display_counts[(r, k)][cl] += 1
        display_form = {rk: cnts.most_common(1)[0][0] for rk, cnts in display_counts.items()}
        total_docs = sum(per_r_n.values())
        out = _rank_and_dedup(
            per_r_doc_counts, per_r_n, display_form, rd_int, top, total_docs,
            filter_universal=False,
        )
        # Supplement any empty/short reading from Mode 2 below
        if all(len(out[r]) > 0 for r in rd_int):
            return out
    else:
        out = {r: [] for r in rd_int}

    # Mode 2: standalone multi-line / multi-language contrastive extraction
    all_kept = [t for r in rd_int for t in texts_by_r[r]]
    shared_prefix = normalize_text(prefix, collapse_newlines=True) if prefix else _detect_shared_prefix(all_kept)

    per_r_doc_counts = {r: collections.Counter() for r in rd_int}
    per_r_n = {r: len(texts_by_r[r]) for r in rd_int}
    display_counts = collections.defaultdict(collections.Counter)

    for r in rd_int:
        for txt in texts_by_r[r]:
            seen_in_doc: set[str] = set()
            rest = _strip_prefix_ci(txt, shared_prefix) if shared_prefix else None
            if rest:
                cl = _clean_clause(rest, max_words=4)
                if cl:
                    k = cl.lower()
                    if k not in seen_in_doc:
                        seen_in_doc.add(k)
                        per_r_doc_counts[r][k] += 1
                    display_counts[(r, k)][cl] += 1
            for ng in _extract_ngrams(txt, n_min=1, n_max=3):
                k = ng.lower()
                if k not in seen_in_doc:
                    seen_in_doc.add(k)
                    per_r_doc_counts[r][k] += 1
                display_counts[(r, k)][ng] += 1

    display_form = {rk: cnts.most_common(1)[0][0] for rk, cnts in display_counts.items()}
    total_docs = sum(per_r_n.values())
    ranked = _rank_and_dedup(
        per_r_doc_counts, per_r_n, display_form, rd_int, top, total_docs,
        filter_universal=True,
    )

    for r in rd_int:
        if not out[r]:
            out[r] = ranked[r]
        if not out[r] and texts_by_r[r]:
            # Clean non-empty fallback when all candidates were universal or stopwords
            fallback = _rank_and_dedup(
                per_r_doc_counts, per_r_n, display_form, [r], top, per_r_n[r],
                filter_universal=False,
            )[r]
            if fallback:
                out[r] = fallback
            else:
                cl = _clean_clause(texts_by_r[r][0], max_words=4)
                if cl:
                    out[r] = [cl]
                else:
                    s0 = normalize_text(texts_by_r[r][0], collapse_newlines=True)
                    if s0:
                        out[r] = [s0[:32].strip()]
    return out


def main():
    from explorer.branches import align_index
    from explorer.directions import calibrate, grads_at
    from explorer.frame import build_doc_prompt, build_prompt, load
    from explorer.steer import CursorHook
    from blind.fisher import seq_logp
    from ut import arch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g3_4b_pt")
    ap.add_argument("--text", default="duende_flamenco")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--extent", type=float, default=1.6)
    ap.add_argument("--prompt-style", choices=("auto", "legacy", "doc"), default="auto")
    ap.add_argument("--no-whiten", action="store_true")
    a = ap.parse_args()
    out = OUT / a.model
    S = json.loads((out / "samples" / f"{a.text}.json").read_text())
    B = json.loads((out / "branches" / f"{a.text}.json").read_text())
    prompt_style = S.get("prompt_style", "legacy") if a.prompt_style == "auto" else a.prompt_style

    tok, model = load(a.model)
    L = arch.n_layers(model) // 2
    hook = CursorHook(model, L)
    ids, start = build_doc_prompt(tok, S["text"]) if prompt_style == "doc" else build_prompt(tok, S["text"])
    labels = np.array(S["labels"])
    rd = [int(r) for r in S["readings"]]
    ys = S["ys"]
    ref_i = max(np.where(labels == rd[0])[0], key=lambda i: S["logp"][i])
    y_ref, t = ys[ref_i], B["t"]
    prefix = ids + y_ref[:t]
    b = len(prefix) - 1

    members = {r: [i for i in np.where(labels == r)[0]] for r in rd}
    cont_ids_by_r = {r: [ys[i][align_index(y_ref, ys[i], t):] for i in members[r]] for r in rd}
    G = {r: grads_at(model, prefix, cont_ids_by_r[r], L, b) for r in rd}

    basis = build_map_basis(G, rd, whiten=not a.no_whiten)
    axes = basis["axes"]
    anchors = basis["anchors"]
    dots = basis["dots"]
    var_expl = basis["variance_explained"]

    anc_arr = np.array([anchors[r] for r in rd], dtype=np.float64)
    anc_norms = np.linalg.norm(anc_arr, axis=1)
    far = int(np.argmax(anc_norms))
    v_far = anc_arr[far] @ axes
    v_norm = float(np.linalg.norm(v_far))
    if v_norm <= 1e-12:
        v_unit = axes[0]
        v_norm = 1.0
    else:
        v_unit = v_far / v_norm
    a_far = calibrate(model, hook, ids + y_ref, b, v_unit, 0.25, "cursor", 1.0)
    alpha_unit = float(a_far / v_norm)

    prefix_raw = tok.decode(y_ref[:t])
    prefix_text = prefix_raw.strip()
    cont_texts_by_r = {r: [tok.decode(c) for c in cont_ids_by_r[r]] for r in rd}
    cands = phrases_by_reading(S, rd, prefix=prefix_text, cont_texts_by_reading=cont_texts_by_r)

    prefix_ends_ws = (not prefix_raw) or prefix_raw[-1].isspace()
    cand_list = []
    cand_ids = []
    for r in rd:
        for p_txt in cands[r]:
            needs_space = (not prefix_ends_ws) and bool(p_txt) and not p_txt[0].isspace() and p_txt[0] not in ",.;:!?)]}"
            t_ids = tok((" " + p_txt) if needs_space else p_txt, add_special_tokens=False)["input_ids"]
            if t_ids:
                cand_list.append((int(r), p_txt))
                cand_ids.append(t_ids)

    grid = np.linspace(-a.extent, a.extent, a.n)
    pts = []
    max_rest = len(y_ref) - t + 12
    for gy in grid[::-1]:
        for gx in grid:
            vec = torch.tensor(alpha_unit * (np.array([gx, gy]) @ axes), dtype=torch.float32)
            share = collections.defaultdict(float)
            mass = 0.0
            top_cand = ""
            if cand_ids:
                hook.set(vec, b)
                try:
                    with torch.no_grad():
                        lp = seq_logp(model, prefix, cand_ids).float().cpu().numpy()
                finally:
                    hook.set(None)
                lp_max = float(np.max(lp))
                w = np.exp(lp - lp_max)
                w_sum = float(w.sum())
                probs = (w / w_sum) if w_sum > 0.0 else np.full_like(w, 1.0 / len(w))
                for (r, _), pi in zip(cand_list, probs):
                    share[r] += float(pi)
                mass = float(np.exp(np.clip(lp, -700.0, 0.0)).sum())
                top_cand = cand_list[int(np.argmax(lp))][1]
            hook.set(vec, b)
            try:
                y = _generate_for_style(model, tok, prefix, 1, max_rest, prompt_style=prompt_style, temperature=0.0)[0]
            finally:
                hook.set(None)
            text = tok.decode(y_ref[:t] + y).strip()
            pts.append(dict(x=round(float(gx), 3), y=round(float(gy), 3), text=text,
                            shares={str(k): round(v, 4) for k, v in share.items()}, mass=round(mass, 4),
                            top=top_cand))
        print(f"[mapgrid] row {len(pts) // a.n}/{a.n}", flush=True)
    res = dict(text=S["text"], prefix=prefix_text, layer=L, n=a.n, extent=a.extent,
               readings=[int(r) for r in rd],
               anchors={int(r): [round(float(v), 4) for v in anchors[r]] for r in rd},
               dots=dots, candidates={int(r): cands[r] for r in rd}, variance_explained=round(var_expl, 3),
               alpha_unit=alpha_unit, calibration="farthest anchor = mean per-token KL 0.25 nats (cursor mode)",
               grid=pts)
    p = out / f"map_{a.text}.json"
    p.write_text(json.dumps(res, ensure_ascii=False))
    print("wrote", p, "variance explained by the plane:", var_expl, flush=True)


if __name__ == "__main__":
    main()
