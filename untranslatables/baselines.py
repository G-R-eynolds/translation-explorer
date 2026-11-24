"""Baselines to beat (handoff §5.5): does the recipe leftover pick out untranslatables better than
  B-NN   CSLS nearest-neighbour similarity to the English pool (replicating [2605.24310]),
  B-freq Zipf frequency alone (rarer = 'more untranslatable'),
  B-tok  token count alone?
Positives: `untranslatable`; negatives: `ordinary` + `emotion_plain` (non-English phase-1 words).
Each score gets AUROC with a bootstrap CI, AUROC on a 1:1 frequency/token-matched subset, and AUROC after
regressing out frequency and token count.  Headline excludes words that failed the familiarity gate.

    python baselines.py --model 3b      # CPU; needs recipes.json, familiarity.json and the pool cache
"""
import argparse

import numpy as np

from familiarity import status
from ut.common import MODELS, load_json, out_dir, save_json
from ut.space import Targets, pool
from ut.stats import auroc_ci, matched_auroc
from ut.words import PHASE1, load_words

POS, NEG = {"untranslatable"}, {"ordinary", "emotion_plain"}


def csls_scores(model_key, layer, words, T, k=10):
    _, E, _ = pool(model_key, "en", layer)
    rS = {}
    out = {}
    for w in words:
        if w.language not in rS:
            _, S, _ = pool(model_key, w.language, layer)
            sims = E @ S.T                                   # en pool vs source pool
            rS[w.language] = np.sort(sims, 1)[:, -k:].mean(1)
        x = T.mean_vector(layer, w.id)
        cos = E @ x
        rT = np.sort(cos)[-k:].mean()
        out[w.id] = float((2 * cos - rT - rS[w.language]).max())
    return out


def evaluate(rows, keys):
    pos = [r for r in rows if r["category"] in POS]
    neg = [r for r in rows if r["category"] in NEG]
    cov = lambda rs: [[r["zipf"], r["n_tokens"]] for r in rs]
    if len(pos) < 2 or len(neg) < 2:
        return dict(n_pos=len(pos), n_neg=len(neg), scores=None, note="too few words after exclusions")
    res = {}
    for name, sign in keys.items():
        p = [sign * r[name] for r in pos]
        n = [sign * r[name] for r in neg]
        res[name] = dict(auroc=auroc_ci(p, n), **matched_auroc(p, n, cov(pos), cov(neg)))
    return dict(n_pos=len(pos), n_neg=len(neg), scores=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    a = ap.parse_args()
    rd = out_dir(a.model)
    rec = load_json(rd / "recipes.json")
    fam = load_json(rd / "familiarity.json")["words"]
    st = status(a.model)
    layer = rec["layer"]
    words = [w for w in load_words(langs=PHASE1) if w.category in POS | NEG]
    nn = csls_scores(a.model, layer, words, Targets(a.model))
    rows = []
    for w in words:
        d = rec["words"][w.id]["directions"]["en"]
        rows.append(dict(id=w.id, category=w.category, leftover=d["leftover"], csls=nn[w.id],
                         zipf=fam[w.id]["zipf"], n_tokens=fam[w.id]["n_tokens"], familiar=st[w.id][0],
                         familiar_source=st[w.id][1]))
    # orientation: higher = predicted untranslatable
    keys = {"leftover": 1, "csls": -1, "zipf": -1, "n_tokens": 1}
    out = dict(layer=layer, positives=sorted(POS), negatives=sorted(NEG), rows=rows,
               headline=evaluate([r for r in rows if r["familiar"] != "no"], keys),
               all_words=evaluate(rows, keys),
               note="headline excludes familiarity 'no' (human ticks if present, else provisional flag)")
    save_json(rd / "baselines.json", out)
    for name, r in (out["headline"]["scores"] or out["all_words"]["scores"]).items():
        print(f"{name:9s} AUROC {r['auroc'][0]:.2f} [{r['auroc'][1]:.2f}, {r['auroc'][2]:.2f}]  "
              f"matched {r['matched']:.2f}  residualised {r['residualised']:.2f}")


if __name__ == "__main__":
    main()
