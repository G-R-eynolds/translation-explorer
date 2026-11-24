"""Readings = clusters of the model's own translations, by meaning (spec §Method 2, revised 24 Nov).

Representation (revised after the first texts): a semantic sentence embedder (all-mpnet-base-v2, mean pooled,
L2-normalised), CENTRED on the text's own samples so that the shared content drops out and what varies (the
interpretive choice) dominates.  The first version (mean last-layer state of the base model itself) merged clearly
different readings (spirit / charm / elf) and split off only a garbage sample: base-model states mostly encode surface
form.  The lowest 10% of samples by per-token log p (degenerate tail) are left out of the clustering (label -1).
`embed` (model states) is kept for reference.  Clustering: agglomerative, cosine distance, average linkage; k in [2, kmax] by silhouette,
k = 1 if the best silhouette < sil_min.  A reading is a cluster with >= min_share of the samples.
"""
import numpy as np
import torch


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
    if "m" not in _EMB:
        _EMB["t"] = AutoTokenizer.from_pretrained(EMBEDDER)
        _EMB["m"] = AutoModel.from_pretrained(EMBEDDER).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    tk, em = _EMB["t"], _EMB["m"]
    out = []
    for c in range(0, len(texts), batch):
        part = [(t.split("\n")[0].strip() or "-") for t in texts[c:c + batch]]
        b = tk(part, padding=True, truncation=True, return_tensors="pt").to(em.device)
        with torch.no_grad():
            h = em(**b).last_hidden_state
        m = b["attention_mask"][..., None].float()
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy())
    return np.concatenate(out)


def centre(E, mu):
    Z = E - mu
    return Z / np.linalg.norm(Z, axis=-1, keepdims=True).clip(1e-12)


def readings_from(E, logp_per_token, drop_q=0.1, **kw):
    """Cluster the kept samples on centred semantic embeddings.  Returns cluster() output with labels for ALL
    samples (-1 = dropped tail) plus mu (the centring vector)."""
    keep = logp_per_token >= np.quantile(logp_per_token, drop_q)
    mu = E[keep].mean(0)
    c = cluster(centre(E[keep], mu), **kw)
    labels = np.full(len(E), -1)
    labels[keep] = c["labels"]
    return {**c, "labels": labels.tolist(), "mu": mu.tolist(), "n_dropped": int((~keep).sum())}


def cluster(X, min_share=0.05, kmax=6, sil_min=0.1):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    n = len(X)
    best = (None, -1.0, np.zeros(n, int))
    for k in range(2, min(kmax, n - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(X, lab, metric="cosine"))
        if s > best[1]:
            best = (k, s, lab)
    k, sil, lab = best
    if k is None or sil < sil_min:
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
