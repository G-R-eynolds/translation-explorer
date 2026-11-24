"""Layer choice by translation retrieval (handoff §5.3).

For the MUSE en<->xx pairs in data/pools/muse_pairs.json (both words in their pools), at every layer: is the
correct English word the nearest neighbour (cosine, centred spaces) of the source word among all English pool
words?  The concept layer is the one with the best mean precision@1; its neighbours are reported too.

    python layers.py --model 3b        # CPU, needs cache/<model>/pool_*.npy  ->  results/<model>/layer_choice.json
"""
import argparse

import numpy as np

from ut.common import DATA, MODELS, cache_dir, load_json, out_dir, save_json
from ut.space import pool


def retrieval(model_key, n_layers):
    pairs = load_json(DATA / "pools" / "muse_pairs.json")["pairs"]
    res = {}
    for l, ps in pairs.items():
        acc = []
        for layer in range(n_layers):
            src_w, S, _ = pool(model_key, l, layer)
            en_w, E, _ = pool(model_key, "en", layer)
            si, ei = {w: i for i, w in enumerate(src_w)}, {w: i for i, w in enumerate(en_w)}
            ok = [(si[x], ei[y]) for x, y in ps if x in si and y in ei]
            if not ok:                                  # only happens with truncated smoke-test pools
                acc.append(float("nan")); continue
            sims = S[[a for a, _ in ok]] @ E.T
            acc.append(float(np.mean(sims.argmax(1) == np.array([b for _, b in ok]))))
            pool.cache_clear()
        res[l] = dict(n_pairs=len(ok) if acc and np.isfinite(acc[-1]) else 0, p_at_1=acc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    a = ap.parse_args()
    n_layers = np.load(cache_dir(a.model) / "pool_en.npy", mmap_mode="r").shape[1]
    res = retrieval(a.model, n_layers)
    mean = np.nanmean([r["p_at_1"] for r in res.values() if r["n_pairs"]] or [[np.nan] * n_layers], 0)
    cand = np.where(np.arange(n_layers) >= 1, mean, np.nan)          # layer 0 = embeddings, not a concept space
    best = int(np.nanargmax(cand)) if np.isfinite(cand).any() else n_layers // 2
    out = dict(per_language=res, mean_p_at_1=mean.tolist(), concept_layer=best,
               neighbours={str(l): float(mean[l]) for l in (best - 1, best + 1) if 0 <= l < n_layers},
               rule="argmax over layers >= 1 of mean precision@1 (cosine NN among all English pool words)")
    save_json(out_dir(a.model) / "layer_choice.json", out)
    print("concept layer", best, "P@1", round(mean[best], 3), "| by layer:", np.round(mean, 2).tolist())


if __name__ == "__main__":
    main()
