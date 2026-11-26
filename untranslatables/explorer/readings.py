"""Readings = clusters of the model's own translations, by meaning (spec §Method 2, revised 24 Nov).

Representation (revised after the first texts): a semantic sentence embedder (all-mpnet-base-v2, mean pooled,
L2-normalised), CENTRED on the text's own samples so that the shared content drops out and what varies (the
interpretive choice) dominates.  The first version (mean last-layer state of the base model itself) merged clearly
different readings (spirit / charm / elf) and split off only a garbage sample: base-model states mostly encode surface
form.  The lowest 10% of samples by per-token log p (degenerate tail) are left out of the clustering (label -1).
`embed` (model states) is kept for reference.  Clustering: agglomerative, cosine distance, average linkage; k in [2, kmax] by silhouette,
k = 1 if the best silhouette < sil_min.  A reading is a cluster with >= min_share of the samples.
"""
import re

import numpy as np
import torch


_ORTHO_TRANS = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u00b4": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
    "\u00a0": " ",
    "\u202f": " ",
    "\ufeff": " ",
})
_UNICODE_SPACES_RE = re.compile(r"[\u2000-\u200b]")
_HORIZ_WS_RE = re.compile(r"[ \t]+")
_ALL_WS_RE = re.compile(r"\s+")


def normalize_text(text: str, *, collapse_newlines: bool = False) -> str:
    """Normalise orthographic surface variants (curly quotes, ellipsis, unicode spaces, whitespace)."""
    s = str(text).translate(_ORTHO_TRANS).replace("\u2026", "...")
    s = _UNICODE_SPACES_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if collapse_newlines:
        return _ALL_WS_RE.sub(" ", s).strip()
    lines = [_HORIZ_WS_RE.sub(" ", ln).strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


def raw_dispersion(E) -> float:
    """Uncentered spherical dispersion 1 - ||mean(E_unit)|| on L2-normalised rows.

    Returns 0.0 for empty/singleton/zero inputs; approaches 0 when all translations share the same
    uncentred semantic meaning (e.g. surface-variant controls like control_museo).
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or len(E) <= 1:
        return 0.0
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    if float(norms.max()) <= 1e-12:
        return 0.0
    E_unit = np.where(norms > 1e-12, E / norms.clip(1e-12), 0.0)
    return float(max(0.0, 1.0 - float(np.linalg.norm(E_unit.mean(axis=0)))))


def embed(model, ids, ys, batch=16):
    from ut import arch
    dev = arch.text_model(model).embed_tokens.weight.device
    X = []
    for c in range(0, len(ys), batch):
        part = [y if y else [ids[-1]] for y in ys[c:c + batch]]
        rows = [ids + y for y in part]
        T = max(map(len, rows))
        x = torch.tensor([r + [0] * (T - len(r)) for r in rows], device=dev)
        att = torch.tensor([[1] * len(r) + [0] * (T - len(r)) for r in rows], device=dev)
        with torch.no_grad():
            H = arch.forward_hidden(model, input_ids=x, attention_mask=att).last_hidden_state.float()
        for j, y in enumerate(part):
            X.append(H[j, len(ids):len(ids) + len(y)].mean(0).cpu().numpy())
    X = np.array(X)
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
_EMB = {}


def embed_texts(texts, batch=64):
    """Semantic embeddings (n, 768), L2-normalised.  Only the first line of each text is used."""
    from transformers import AutoModel, AutoTokenizer
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    if "m" not in _EMB:
        _EMB["t"] = AutoTokenizer.from_pretrained(EMBEDDER)
        _EMB["m"] = AutoModel.from_pretrained(EMBEDDER).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    tk, em = _EMB["t"], _EMB["m"]
    out = []
    for c in range(0, len(texts), batch):
        part = [normalize_text(t.split("\n")[0], collapse_newlines=True) or "-" for t in texts[c:c + batch]]
        b = tk(part, padding=True, truncation=True, return_tensors="pt").to(em.device)
        with torch.no_grad():
            h = em(**b).last_hidden_state
        m = b["attention_mask"][..., None].float()
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy())
    return np.concatenate(out)


def centre(E, mu):
    Z = E - mu
    return Z / np.linalg.norm(Z, axis=-1, keepdims=True).clip(1e-12)


def readings_from(E, logp_per_token, drop_q=0.1, disp_min=0.035, **kw):
    """Cluster the kept samples on centred semantic embeddings.  Returns cluster() output with labels for ALL
    samples (-1 = dropped tail), mu (the centring vector), and uncentred raw_disp.

    If raw_disp < disp_min (all kept translations agree in uncentred embedding space), collapses cleanly to k=1.
    """
    E = np.asarray(E, dtype=np.float64)
    lp = np.asarray(logp_per_token, dtype=np.float64)
    if len(E) == 0:
        return dict(labels=[], k=0, silhouette=0.0, readings=[], shares={}, mu=[], n_dropped=0, raw_disp=0.0)
    keep = lp >= np.quantile(lp, drop_q)
    if not np.any(keep):
        keep = np.ones(len(E), dtype=bool)
    E_keep = E[keep]
    mu = E_keep.mean(0)
    raw_disp = raw_dispersion(E_keep)
    labels = np.full(len(E), -1, dtype=int)
    if len(E_keep) < 2 or (disp_min is not None and raw_disp < float(disp_min)):
        labels[keep] = 0
        return dict(
            labels=labels.tolist(),
            k=1,
            silhouette=0.0,
            readings=[0],
            shares={0: 1.0},
            mu=mu.tolist(),
            n_dropped=int((~keep).sum()),
            raw_disp=float(raw_disp),
        )
    c = cluster(centre(E_keep, mu), **kw)
    labels[keep] = c["labels"]
    return {**c, "labels": labels.tolist(), "mu": mu.tolist(), "n_dropped": int((~keep).sum()),
            "raw_disp": float(raw_disp)}


def cluster(X, min_share=0.05, kmax=6, sil_min=0.1):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    n = len(X)
    if n == 0:
        return dict(labels=[], k=0, silhouette=0.0, readings=[], shares={})
    best = (None, -1.0, np.zeros(n, int))
    for k in range(2, min(kmax, n - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(X, lab, metric="cosine"))
        if s > best[1]:
            best = (k, s, lab)
    k, sil, lab = best
    if k is None:
        k, sil, lab = 1, 0.0, np.zeros(n, int)
    elif sil < sil_min:
        k, lab = 1, np.zeros(n, int)
    sizes = np.bincount(lab, minlength=k)
    readings = [int(c) for c in np.argsort(-sizes) if sizes[c] >= min_share * n]
    return dict(labels=lab.tolist(), k=int(k), silhouette=float(sil), readings=readings,
                shares={int(c): float(sizes[c] / n) for c in range(k)})


def centroids(X, labels, readings):
    """X: the (centred, normalised) vectors the clustering used, one per sample."""
    labels = np.asarray(labels)
    C = np.array([X[labels == c].mean(0) for c in readings])
    return C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)


def assign(x, C):
    """Index (into the readings list) of the nearest centroid by cosine."""
    return int(np.argmax(C @ (x / max(np.linalg.norm(x), 1e-12))))


def entropy(shares):
    p = np.array([v for v in shares.values() if v > 0])
    return float(-(p * np.log2(p)).sum())
