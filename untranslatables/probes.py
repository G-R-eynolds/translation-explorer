"""Stretch (handoff §7.6): internal signatures of 'unfamiliar' vs 'many readings'.

Inputs: the word-in-context states of the thread-1 cache (natural contexts), every layer.  Labels from the 2x2:
unfamiliar = dialect_* (optionally also words that failed the familiarity gate: --familiarity-labels);
many = untranslatable, polysemous, dialect_untranslatable [2506.01205], [2402.03563].
Per layer: standardised L2 logistic probes, leave-one-word-out CV (contexts of a word never split), AUROC;
cosine between the two probes' weight vectors (fitted on all data).  At the concept layer, each word's out-of-fold
probe score is correlated (Spearman) with its recipe leftover and recipe variability: the bridge to thread 1.
Caveat recorded in the output: only a handful of dialect words, so the 'unfamiliar' probe is weakly identified.

    python probes.py --model 8b
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ut.common import MODELS, load_json, out_dir, save_json
from ut.represent import load_cached
from ut.stats import auroc
from ut.words import THREAD2, by_id, load_words


def loo_scores(X, y, groups, C=0.1):
    s = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            s[te] = np.nan; continue
        sc = StandardScaler().fit(X[tr])
        s[te] = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
    return s


def weights(X, y, C=0.1):
    sc = StandardScaler().fit(X)
    return LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y).coef_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=MODELS)
    ap.add_argument("--familiarity-labels", action="store_true")
    a = ap.parse_args()
    rd = out_dir(a.model)
    H, meta = load_cached(a.model, "words_t1")
    W = by_id(load_words())
    rows = [i for i, r in enumerate(meta["index"]) if r[1] == "natural" and W[r[0]].category in THREAD2]
    wid = np.array([meta["index"][i][0] for i in rows])
    cats = [W[w].category for w in wid]
    unf = np.array([c.startswith("dialect") for c in cats], int)
    if a.familiarity_labels:
        from familiarity import status
        st = status(a.model)
        unf = np.array([u or st[w][0] == "no" for u, w in zip(unf, wid)], int)
    many = np.array([c in ("untranslatable", "polysemous", "dialect_untranslatable") for c in cats], int)
    layer_c = load_json(rd / "layer_choice.json")["concept_layer"]
    res = dict(n_contexts=len(rows), n_words=len(set(wid)), n_unfamiliar_words=len(set(wid[unf == 1])),
               n_many_words=len(set(wid[many == 1])), layers={},
               caveat="few unfamiliar words: the unfamiliar probe is weakly identified")
    for l in range(1, H.shape[1]):
        X = np.asarray(H[rows, l], dtype=np.float32)
        su, sm = loo_scores(X, unf, wid), loo_scores(X, many, wid)
        wu, wm = weights(X, unf), weights(X, many)
        res["layers"][l] = dict(auroc_unfamiliar=auroc(su[unf == 1], su[unf == 0]), auroc_many=auroc(sm[many == 1], sm[many == 0]),
                                cosine=float(wu @ wm / (np.linalg.norm(wu) * np.linalg.norm(wm))))
        if l == layer_c:
            rec = load_json(rd / "recipes.json")["words"]
            per = {w: (np.nanmean(su[wid == w]), np.nanmean(sm[wid == w])) for w in set(wid)}
            br = {}
            for name, j in (("unfamiliar", 0), ("many", 1)):
                for key in ("leftover", "variability"):
                    pts = [(per[w][j], rec[w]["directions"]["en"][key]) for w in per if w in rec]
                    br[f"{name}~{key}"] = float(spearmanr(*zip(*pts))[0]) if len(pts) > 4 else None
            res["bridge_at_concept_layer"] = br
        print(l, {k: round(v, 2) for k, v in res["layers"][l].items()})
    save_json(rd / "probes.json", res)


if __name__ == "__main__":
    main()
