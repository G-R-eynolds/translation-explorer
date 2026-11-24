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

from explorer.run import OUT


def phrases_by_reading(S, rd, top=4):
    out = {}
    for r in rd:
        c = collections.Counter()
        for t, l in zip(S["texts"], S["labels"]):
            if l == r:
                m = re.search(r"\bhas (.+?)[;.,]", t.split("\n")[0])
                if m:
                    c[m.group(1).strip()] += 1
        out[r] = [p for p, _ in c.most_common(top)]
    return out


def main():
    from explorer.branches import align_index
    from explorer.directions import calibrate, grads_at
    from explorer.frame import build_prompt, load
    from explorer.sample import generate
    from explorer.steer import CursorHook
    from blind.fisher import seq_logp
    from ut import arch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g3_4b_pt")
    ap.add_argument("--text", default="duende_flamenco")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--extent", type=float, default=1.6)
    a = ap.parse_args()
    out = OUT / a.model
    S = json.loads((out / "samples" / f"{a.text}.json").read_text())
    B = json.loads((out / "branches" / f"{a.text}.json").read_text())
    tok, model = load(a.model)
    L = arch.n_layers(model) // 2
    hook = CursorHook(model, L)
    ids, start = build_prompt(tok, S["text"])
    labels = np.array(S["labels"])
    rd = S["readings"]
    ys = S["ys"]
    ref_i = max(np.where(labels == rd[0])[0], key=lambda i: S["logp"][i])
    y_ref, t = ys[ref_i], B["t"]
    prefix = ids + y_ref[:t]
    b = len(prefix) - 1

    members = {r: [i for i in np.where(labels == r)[0]] for r in rd}
    G = {r: grads_at(model, prefix, [ys[i][align_index(y_ref, ys[i], t):] for i in members[r]], L, b) for r in rd}
    M = np.array([G[r].mean(0) for r in rd])
    c = M.mean(0)
    U, s, Vt = np.linalg.svd(M - c, full_matrices=False)
    axes = Vt[:2]                                         # (2, d) orthonormal
    anchors = (M - c) @ axes.T
    scale = np.linalg.norm(anchors, axis=1).max()
    anchors /= scale
    dots = {int(r): ((G[r] - c) @ axes.T / scale).tolist() for r in rd}
    far = int(np.argmax(np.linalg.norm(anchors, axis=1)))
    v_far = anchors[far] @ axes
    a_far = calibrate(model, hook, ids + y_ref, b, v_far / np.linalg.norm(v_far), 0.25, "cursor", 1.0)
    alpha_unit = a_far / np.linalg.norm(v_far)            # α per map unit (vector = alpha_unit * (p @ axes))
    var_expl = float((s[:2] ** 2).sum() / (s ** 2).sum())

    cands = phrases_by_reading(S, rd)
    cand_list = [(int(r), p) for r in rd for p in cands[r]]
    cand_ids = [tok(" " + p, add_special_tokens=False)["input_ids"] for _, p in cand_list]
    grid = np.linspace(-a.extent, a.extent, a.n)
    pts = []
    max_rest = len(y_ref) - t + 12
    for gy in grid[::-1]:
        for gx in grid:
            vec = torch.tensor(alpha_unit * (np.array([gx, gy]) @ axes), dtype=torch.float32)
            hook.set(vec, b)
            with torch.no_grad():
                lp = seq_logp(model, prefix, cand_ids).float().cpu().numpy()
            hook.set(None)
            p = np.exp(lp)
            share = collections.defaultdict(float)
            for (r, _), pi in zip(cand_list, p / p.sum()):
                share[r] += float(pi)
            hook.set(vec, b)
            try:
                y = generate(model, tok, prefix, 1, max_rest, temperature=0.0)[0]
            finally:
                hook.set(None)
            text = tok.decode(y_ref[:t] + y).strip()
            pts.append(dict(x=round(float(gx), 3), y=round(float(gy), 3), text=text,
                            shares={str(k): round(v, 4) for k, v in share.items()}, mass=round(float(p.sum()), 4),
                            top=cand_list[int(np.argmax(p))][1]))
        print(f"[mapgrid] row {len(pts) // a.n}/{a.n}", flush=True)
    res = dict(text=S["text"], prefix=tok.decode(y_ref[:t]).strip(), layer=L, n=a.n, extent=a.extent,
               readings=[int(r) for r in rd], anchors={int(r): anchors[i].round(4).tolist() for i, r in enumerate(rd)},
               dots=dots, candidates={int(r): cands[r] for r in rd}, variance_explained=round(var_expl, 3),
               alpha_unit=alpha_unit, calibration="farthest anchor = mean per-token KL 0.25 nats (cursor mode)",
               grid=pts)
    p = out / f"map_{a.text}.json"
    p.write_text(json.dumps(res, ensure_ascii=False))
    print("wrote", p, "variance explained by the plane:", var_expl, flush=True)


if __name__ == "__main__":
    main()
