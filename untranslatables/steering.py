"""Push it toward or away from English (handoff §6.4), after [2505.13141].

Steering vector at layer l = mean last-token state of the English versions of the distinction items minus the
mean over their source sentences (fr, pt, it), plain text, no chat template.  It is added, times gamma, after
every block in a band just BELOW the concept layer (concept_layer-4 .. concept_layer-1), so its downstream
effect is read at the concept layer rather than cancelled by centring there.
For each gamma in {-3..3}:
  (a) distinction-collapse error rates (greedy translations, steered throughout generation);
  (b) recipe leftovers into English for untranslatables and their controls, with the English pool and the
      target words re-extracted under steering.  Targets use their natural+frame contexts, centred by the
      steered mean of those contexts in their language (the same rule at every gamma, gamma=0 included);
  (c) fluency guard: mean token log-prob of the steered translations under the UNSTEERED model, and the share
      of outputs that contain any class word.  A gamma is flagged 'degraded' if log-prob falls > 1 nat/token
      or the on-task share falls > 20 points relative to gamma=0.

    python steering.py --model 3b
"""
import argparse

import numpy as np
import torch

from distinctions import classes_found, load_sets, summarise, translate_all
from recipe import K_CARD, nnomp
from ut.common import MODELS, Progress, cache_dir, load_json, load_model, out_dir, save_json
from ut.lm import token_logprobs
from ut.represent import span_states, steering, unmark
from ut.space import cognate_mask
from ut.words import PHASE1, load_words

GAMMAS = [-3, -2, -1, 0, 1, 2, 3]


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def vectors(model, tok, sets, band):
    src = [(it["s"], None) for p in sets.values() for it in p["items"]]
    en = [(it["en"], None) for p in sets.values() for it in p["items"]]
    Hs, He = span_states(model, tok, src), span_states(model, tok, en)
    return {l: torch.tensor(He[:, l].astype(np.float32).mean(0) - Hs[:, l].astype(np.float32).mean(0)) for l in band}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--pool-size", type=int, default=0, help="subsample the English pool (0 = all)")
    ap.add_argument("--layer", type=int, default=None, help="read layer (default: the concept layer)")
    ap.add_argument("--gammas", type=float, nargs="*", default=GAMMAS,
                    help="multipliers; the handoff's -3..3 degraded the 3B at every gamma != 0")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    a = ap.parse_args()
    rd = out_dir(a.model)
    layer = a.layer if a.layer is not None else load_json(rd / "layer_choice.json")["concept_layer"]
    band = list(range(max(1, layer - 4), layer))
    if not band:
        raise SystemExit(f"concept layer {layer} leaves no layers below it to steer; pass --layer")
    sets = load_sets()
    tok, model = load_model(a.model)
    V = vectors(model, tok, sets, band)
    en_words = load_json(cache_dir(a.model) / "pool_en.json")["words"]
    if a.pool_size:
        en_words = [en_words[i] for i in sorted(np.random.default_rng(0).choice(len(en_words), a.pool_size, replace=False))]
    from pools import templates
    tps = templates("en")
    targets = [w for w in load_words(langs=PHASE1) if w.category in ("untranslatable", "ordinary", "emotion_plain")]
    t_items = [(w, unmark(c)) for w in targets for kind, c in w.t1_contexts() if kind != "template"]
    masks = {w.id: cognate_mask(en_words, [w.word], w.exclude_from_pool) for w in targets}
    G = a.gammas
    res = {"layer": layer, "band": band, "gammas": G, "vector_norms": {l: float(v.norm()) for l, v in V.items()},
           "distinctions": {}, "leftovers": {}, "fluency": {}}
    pr = Progress("steering", len(G))
    for gi, g in enumerate(G):
        with steering(model, V, g):
            trans = translate_all(model, tok, sets, n_samples=0)
            P = np.stack([span_states(model, tok, [unmark(t.format(w=f"[[{w}]]")) for w in en_words], batch=256)[:, layer]
                          for t in tps]).astype(np.float32).mean(0)
            T = span_states(model, tok, [it for _, it in t_items], batch=64)[:, layer].astype(np.float32)
        summ = summarise(sets, trans)
        res["distinctions"][g] = {sid: dict(greedy_error=r["greedy_error"], other=r["greedy_other_rate"],
                                            excess=r.get("excess_error_greedy")) for sid, r in summ.items()}
        D = unit(P - P.mean(0))
        cent = {}
        for l in PHASE1:
            rows = [i for i, (w, _) in enumerate(t_items) if w.language == l]
            cent[l] = T[rows].mean(0)
        lv = {}
        for w in targets:
            rows = [i for i, (x, _) in enumerate(t_items) if x.id == w.id]
            x = unit(unit(T[rows] - cent[w.language]).mean(0))
            _, _, R2 = nnomp(D[~masks[w.id]], x)
            lv[w.id] = 1 - (R2[K_CARD - 1] if len(R2) >= K_CARD else (R2[-1] if R2 else 0))
        res["leftovers"][g] = lv
        outs = [o for sid in trans for o in trans[sid]["greedy"]]
        lp = token_logprobs(model, tok, outs)                              # unsteered: outside the context manager
        on_task = np.mean([bool(classes_found(o, p["classes"])) for sid, p in sets.items() for o in trans[sid]["greedy"]])
        res["fluency"][g] = dict(mean_token_logprob=float(np.mean([l.mean() for _, l in lp if len(l)])),
                                 on_task=float(on_task), examples=outs[:3])
        pr.tick(gi + 1)
    f0 = res["fluency"][0]
    for g in G:
        f = res["fluency"][g]
        f["degraded"] = bool(f0["mean_token_logprob"] - f["mean_token_logprob"] > 1.0 or f0["on_task"] - f["on_task"] > 0.2)
    save_json(rd / f"steering{a.tag}.json", res)
    for g in G:
        u = np.mean([v for k, v in res["leftovers"][g].items() if k in {w.id for w in targets if w.category == "untranslatable"}])
        print(f"gamma {g:+.2f}: untranslatable leftover {u:.3f}  degraded={res['fluency'][g]['degraded']}  "
              + "  ".join(f"{sid.split('_')[1]} {r['greedy_error']:.2f}" for sid, r in res["distinctions"][g].items()))


if __name__ == "__main__":
    main()
