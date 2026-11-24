"""Directions between readings (spec §Method 4).

Cursor convention: to steer output token t, act on the state that predicts it, i.e. absolute position
b = start + t - 1 (the last token of the prefix ids + y_ref[:t]).

grads_at: per-sample gradients of log p(cont | prefix) w.r.t. the block-`layer` output at position b, via an additive
zero leaf (one row per sample).
natural:  v ∝ (Ĝ + λI)⁻¹ (ḡ_A − ḡ_B),  Ĝ = ZᵀZ/n over Z = [G_A; G_B],  λ = ε·tr(Ĝ)/rank.  Since ḡ_A − ḡ_B = Zᵀc,
          v = Zᵀ (K/n + λI)⁻¹ c  with K = ZZᵀ  (exact; no d-dimensional solve).
plain:    v ∝ ḡ_A − ḡ_B.
alpha_for: α with ½ α² vᵀĜv = kl (predicted KL of the steered next-step distribution family).
"""
import numpy as np
import torch

from blind.fisher import seq_logp
from ut import arch


def grads_at(model, prefix, conts, layer, b, batch=8):
    d = arch.hidden_size(model)
    dev = arch.text_model(model).embed_tokens.weight.device
    out = []
    state = {}

    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        if "z" not in state or h.shape[1] <= b:
            return None
        oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
        oh[b] = 1
        new = h + state["z"].to(h.dtype)[:, None, :] * oh[None, :, None]
        return (new, *o[1:]) if isinstance(o, tuple) else new

    hd = arch.layers(model)[layer].register_forward_hook(hook)
    try:
        for c in range(0, len(conts), batch):
            part = [y if y else [prefix[-1]] for y in conts[c:c + batch]]
            state["z"] = torch.zeros(len(part), d, device=dev, dtype=torch.float32, requires_grad=True)
            lp = seq_logp(model, prefix, part)
            (g,) = torch.autograd.grad(lp.sum(), [state["z"]])
            out.append(g.float().cpu().numpy())
            state.pop("z")
    finally:
        hd.remove()
    return np.concatenate(out).astype(np.float64)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-300)


def plain(GA, GB):
    return _unit(GA.mean(0) - GB.mean(0))


def natural(GA, GB, eps=1e-2):
    Z = np.vstack([GA, GB])
    n = len(Z)
    K = Z @ Z.T
    lam = eps * (np.trace(K) / n) / n
    c = np.r_[np.full(len(GA), 1 / len(GA)), np.full(len(GB), -1 / len(GB))]
    w = np.linalg.solve(K / n + lam * np.eye(n), c)
    return _unit(Z.T @ w)


def alpha_for(G_all, v, kl):
    q = float(np.mean((G_all @ v) ** 2))
    return float(np.sqrt(2 * kl / max(q, 1e-300)))


def measured_kl(model, hook, seq, b, vec, mode):
    """Mean KL(unsteered || steered) of the next-token distributions over positions >= b (cursor) or at b (at_b),
    teacher-forced along `seq` (prefix + reference continuation)."""
    dev = arch.text_model(model).embed_tokens.weight.device
    x = torch.tensor([seq], device=dev)
    with torch.no_grad():
        hook.set(None)
        p0 = model(input_ids=x).logits[0, b:].float().log_softmax(-1)
        hook.set(vec, b, only_at_b=mode == "at_b")
        p1 = model(input_ids=x).logits[0, b:].float().log_softmax(-1)
        hook.set(None)
    kl = (p0.exp() * (p0 - p1)).sum(-1)
    return float(kl[0] if mode == "at_b" else kl.mean())


def calibrate(model, hook, seq, b, v, target, mode, a0, iters=14):
    """α (by bisection in log space) such that measured_kl(α v) ≈ target.  Revised 24 Nov: the single-position Fisher
    prediction badly underestimates the cursor mode, where the vector is added at every later position."""
    vt = torch.as_tensor(v, dtype=torch.float32)
    lo, hi = a0 * 1e-3, a0 * 1e2
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if measured_kl(model, hook, seq, b, mid * vt, mode) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
