"""M1 step 2: teacher-force every pooled back-translation under every unit's BT prompt.

For each group (a line of the lattice, or a control family) with units j=1..N and pooled
back-translations B (m per unit), writes L[j, b] = log q_j(b) = sum of token log-probs of b
(plus the end-of-turn token) under BT-prompt(r_j), and len[b] = number of tokens scored.  The prompt's KV cache is computed once per j.

    python score.py --model 3b --set lattice
"""
import argparse, time
from collections import defaultdict

import numpy as np

from backtranslate import bt_prompt, set_name
from common import MODELS, chat, load_json, load_model, run_dir, score_continuations, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--set", default="lattice")
    ap.add_argument("--pool-with", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    name = set_name(a.set, a.pool_with) + a.suffix
    rd = run_dir(a.model) / "results"
    bt = load_json(rd / f"bt_{name}.json")
    groups = defaultdict(list)
    for u in bt["units"]:
        groups[u["group"]].append(u)
    tok, model = load_model(a.model)
    out, t0, done = {}, time.time(), 0
    total = sum(len(g) for g in groups.values())
    for gname, us in groups.items():
        B = [b for u in us for b in u["bt"]]
        owner = np.array([j for j, u in enumerate(us) for _ in u["bt"]])
        L = np.zeros((len(us), len(B)), dtype=np.float32)
        for j, u in enumerate(us):
            L[j] = score_continuations(model, tok, chat(tok, bt_prompt(u["prev"], u["text"], u["next"])), B,
                                       batch=a.batch, temperature=bt["temperature"])
        out[f"{gname}__L"], out[f"{gname}__owner"] = L, owner
        out[f"{gname}__uids"] = np.array([u["uid"] for u in us])
        out[f"{gname}__len"] = np.array([len(encode(tok, b)) + 1 for b in B])  # tokens scored, incl. stop
        done += len(us)
        el = time.time() - t0
        print(f"{gname}: N={len(us)} |B|={len(B)}  elapsed {el:.0f}s, est. remaining {el / done * (total - done):.0f}s",
              flush=True)
    np.savez_compressed(rd / f"L_{name}.npz", **out)
    print("wrote", rd / f"L_{name}.npz")


if __name__ == "__main__":
    main()
