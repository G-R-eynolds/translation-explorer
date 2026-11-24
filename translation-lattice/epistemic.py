"""Phase 5 (§8): genuine multiplicity vs the model not knowing.

Pools this model's samples with a second model's (e.g. Qwen2.5-7B) per line, clusters the pooled
renderings by meaning with the tau-calibrated metrics, and computes

    MI = H(pooled clusters) - mean_model H(model's clusters)     (bits)

i.e. the mutual information between "which model wrote it" and "which reading it takes".  High MI:
the models disagree about the reading (epistemic).  Low MI with high H: both branch the same way
(better candidate for genuine multiplicity).  Many singleton clusters bias MI upward, so each line also
gets a permutation null (model labels shuffled) and the excess MI over that null.

    python epistemic.py --model 3b --other qwen7b      # needs runs/3b/results/L_pool_qwen7b.npz for m1r
"""
import argparse
from collections import Counter

import numpy as np

from common import MODELS, load_json, run_dir, save_json
from metrics import NLI, entropy_bits, load_m1, medoid, nli_D, tau_cut


def mi_bits(lab, model):
    models = sorted(set(model))
    h_tot = entropy_bits(lab)
    h_each = [entropy_bits(lab[model == m]) for m in models]
    w = [np.mean(model == m) for m in models]
    return h_tot - float(np.dot(w, h_each)), h_tot, dict(zip(models, h_each))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--other", default="qwen7b", choices=MODELS)
    ap.add_argument("--n-perm", type=int, default=500)
    a = ap.parse_args()
    rd = run_dir(a.model)
    val = load_json(rd / "results" / "validation.json")["validation"]
    mine = load_json(rd / f"samples_{MODELS[a.model]['family']}.json")["samples"]
    other = load_json(run_dir(a.other) / f"samples_{MODELS[a.other]['family']}.json")["samples"]
    pooled = [(f"{a.model}/{s['id']}", a.model, s) for s in mine] + [(f"{a.other}/{s['id']}", a.other, s) for s in other]
    model_of = np.array([m for _, m, _ in pooled])
    dists = {}
    nli = NLI()
    dists["b2"] = {i: nli_D(nli.matrix([s["lines"][i] for _, _, s in pooled])) for i in range(14)}
    del nli
    if (rd / "results" / f"L_pool_{a.other}.npz").exists():
        m1r = load_m1(a.model, f"pool_{a.other}", "m1r")
        order = [u for u, _, _ in pooled]
        dists["m1r"] = {}
        for i in range(14):
            D, se, uids = m1r[f"pool{i}"]
            idx = [uids.index(f"{u}:{i}") for u in order]
            dists["m1r"][i] = D[np.ix_(idx, idx)]
    rng = np.random.default_rng(0)
    out = {}
    for metric, per in dists.items():
        tau = val[metric]["tau"]
        rows = []
        for i in range(14):
            lab = tau_cut(per[i], tau)
            mi, h, h_each = mi_bits(lab, model_of)
            null = [mi_bits(lab, rng.permutation(model_of))[0] for _ in range(a.n_perm)]
            # shared readings: clusters with members from both models
            cl = []
            for c, n in Counter(lab).most_common():
                mem = list(np.where(lab == c)[0])
                if n < 2:
                    continue
                med = medoid(per[i], mem)
                cl.append(dict(size=n, by_model=dict(Counter(model_of[mem])),
                               medoid=pooled[med][0], text=pooled[med][2]["lines"][i]))
            rows.append(dict(line=i, H_pooled=h, H_by_model=h_each, MI=mi, MI_null_mean=float(np.mean(null)),
                             MI_excess=mi - float(np.mean(null)), p_value=float(np.mean(np.array(null) >= mi)),
                             k=int(len(set(lab))), shared_clusters=cl[:6]))
            print(f"{metric} line {i + 1:2d}: H={h:.2f}  H_each={ {k: round(v, 2) for k, v in h_each.items()} }  "
                  f"MI={mi:.2f} null={np.mean(null):.2f} p={rows[-1]['p_value']:.3f}", flush=True)
        out[metric] = dict(tau=tau, lines=rows)
    save_json(rd / "results" / f"epistemic_{a.other}.json", out)


if __name__ == "__main__":
    main()
