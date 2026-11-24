"""Which way does the space lean?  Coverage asymmetry (handoff §6.2).

At the concept layer, centre each pool (its own mean) and take the top-k principal directions of pool A; the
share of pool B's variance they capture is cov(A->B).  Asymmetry(en, xx) = cov(en->xx) - cov(xx->en): positive
means English directions cover the other language better than the reverse (space built around English).
Pools are equal-size and frequency-matched (pools.py).  Bootstrap CIs resample words in both pools.  Because the
spaces are not globally isomorphic [1805.03620], the same is computed within Zipf bands.
References: a random k-dim subspace captures about k/d; a split-half of the SAME language is the ceiling.

    python coverage.py --model 3b      # CPU
"""
import argparse

import numpy as np

from ut.common import MODELS, load_json, out_dir, save_json
from ut.space import pool

KS = [10, 25, 50, 100, 200]
BANDS = [(2.5, 3.5), (3.5, 4.5), (4.5, 6.01)]


def top_dirs(X, k=max(KS)):
    """Top-k principal directions [k, d] via the covariance eigendecomposition (n >> k)."""
    Xc = (X - X.mean(0)).astype(np.float64)
    w, V = np.linalg.eigh(Xc.T @ Xc)
    return V[:, ::-1][:, :k].T


def captured(V, Y):
    Yc = Y - Y.mean(0)
    return float(((Yc @ V.T) ** 2).sum() / (Yc ** 2).sum())


def asym_all(A, B):
    """{k: (asymmetry, A->B, B->A)} with directions computed once for the largest k."""
    VA, VB = top_dirs(A), top_dirs(B)
    out = {}
    for k in KS:
        ab, ba = captured(VA[:k], B), captured(VB[:k], A)
        out[k] = (ab - ba, ab, ba)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    rd = out_dir(a.model)
    layer = a.layer if a.layer is not None else load_json(rd / "layer_choice.json")["concept_layer"]
    from wordfreq import zipf_frequency
    ew, E, _ = pool(a.model, "en", layer)
    d = E.shape[1]
    rng = np.random.default_rng(0)
    res = {}
    for l in ["pt", "de", "it", "fr", "es"]:
        xw, X, _ = pool(a.model, l, layer)
        r = {}
        full = asym_all(E, X)
        boots = [asym_all(E[rng.integers(0, len(E), len(E))], X[rng.integers(0, len(X), len(X))]) for _ in range(a.n_boot)]
        half = rng.permutation(len(X))
        Vh = top_dirs(X[half[: len(X) // 2]])
        for k in KS:
            bk = [b[k][0] for b in boots]
            r[k] = dict(asymmetry=full[k][0], en_to_xx=full[k][1], xx_to_en=full[k][2],
                        ci=[float(np.quantile(bk, .025)), float(np.quantile(bk, .975))], random_subspace=k / d,
                        same_language_ceiling=captured(Vh[:k], X[half[len(X) // 2:]]))
        zx = np.array([zipf_frequency(w.lower(), l) for w in xw])
        ze = np.array([zipf_frequency(w, "en") for w in ew])
        r["bands"] = {f"{lo}-{hi}": asym_all(E[(ze >= lo) & (ze < hi)], X[(zx >= lo) & (zx < hi)])[50][0]
                      for lo, hi in BANDS if ((ze >= lo) & (ze < hi)).sum() > 100 and ((zx >= lo) & (zx < hi)).sum() > 100}
        res[l] = r
        print(l, {k: round(r[k]["asymmetry"], 4) for k in KS})
    save_json(rd / "coverage.json", dict(layer=layer, ks=KS, languages=res, band_k=50))


if __name__ == "__main__":
    main()
