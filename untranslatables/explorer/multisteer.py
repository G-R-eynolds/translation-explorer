"""Multi-layer cursor steering, simultaneous multi-layer gradients, and fast KL calibration.

Extends single-layer steering (`explorer.steer`, `explorer.directions`) to a band of decoder blocks:
  default_layer_band    mid-network layer band (non-empty even on 2-layer models)
  layer_weights         normalised per-layer weights ("gaussian" or "uniform")
  MultiLayerCursorHook  forward hooks across multiple blocks with per-layer position tracking across
                        prefill (T > 1) and KV-cache decode steps (T == 1), supporting additive
                        local (position >= b or == b) and global (position >= global_start) vectors
  multi_layer_grads_at  exact per-sample gradients at position b for all layers in one backward pass
  measured_kl_multi     mean KL(unsteered || steered) over positions >= b (cursor) or at b (at_b)
  fast_calibrate        log-space secant + bracketed bisection solver for target KL
"""
import math

import numpy as np
import torch

from blind.fisher import seq_logp
from ut import arch


def default_layer_band(model, frac_lo=0.35, frac_hi=0.65):
    """Return a non-empty list of decoder layer indices spanning [frac_lo, frac_hi] of network depth."""
    n = int(arch.n_layers(model))
    if n <= 0:
        raise ValueError(f"Model has no decoder layers (n_layers={n})")
    lo = max(0, min(n - 1, int(math.floor(frac_lo * n))))
    hi = max(lo, min(n - 1, int(math.floor(frac_hi * n))))
    return list(range(lo, hi + 1))


def layer_weights(layers, profile="gaussian"):
    """Normalised dict {layer: weight} summing to 1.0 ('gaussian' peaked at band centre, or 'uniform')."""
    ls = [int(l) for l in layers]
    if not ls:
        raise ValueError("layers must be non-empty")
    if len(ls) == 1:
        return {ls[0]: 1.0}
    if profile == "uniform":
        w = np.full(len(ls), 1.0 / len(ls), dtype=np.float64)
    elif profile == "gaussian":
        pos = np.asarray(ls, dtype=np.float64)
        mid = 0.5 * (pos.min() + pos.max())
        span = max(float(pos.max() - pos.min()), 1.0)
        sigma = 0.35 * span
        w = np.exp(-0.5 * ((pos - mid) / sigma) ** 2)
        w /= w.sum()
    else:
        raise ValueError(f"Unknown layer weight profile: {profile!r}")
    return {l: float(wi) for l, wi in zip(ls, w)}


class MultiLayerCursorHook:
    """Multi-layer forward hook with per-layer absolute position tracking across prefill and KV-cache decode.

    Combines additive local vectors (applied at positions >= b, or == b when only_at_b=True) and global
    vectors (applied at positions >= global_start).  Each vector argument may be None, a single 1D (d,)
    vector (broadcast across `layers` using `weights`, default uniform 1/len(layers)), or a dict {layer: (d,)}.
    """

    def __init__(self, model, layers, weights=None):
        ls = [int(l) for l in layers]
        if not ls:
            raise ValueError("layers must be non-empty")
        self.layers = ls
        self.weights = (
            layer_weights(ls, "uniform")
            if weights is None
            else {int(k): float(v) for k, v in weights.items()}
        )
        self.local_vecs = {}
        self.global_vecs = {}
        self.b = 0
        self.only = False
        self.global_start = 0
        self.pos = {l: 0 for l in self.layers}
        blocks = arch.layers(model)
        self.handles = [blocks[l].register_forward_hook(self._mk(l)) for l in self.layers]

    def _coerce(self, vecs, weights):
        if vecs is None:
            return {}
        if isinstance(vecs, dict):
            return {int(k): torch.as_tensor(v) for k, v in vecs.items() if v is not None}
        t = torch.as_tensor(vecs, dtype=torch.float32)
        if t.ndim != 1:
            raise ValueError(f"Expected 1D steering vector, got shape {tuple(t.shape)}")
        return {l: float(weights[l]) * t for l in self.layers if l in weights}

    def set(self, local_vecs=None, b=0, only_at_b=False, global_vecs=None, global_start=0, weights=None):
        w = self.weights if weights is None else {int(k): float(v) for k, v in weights.items()}
        self.local_vecs = self._coerce(local_vecs, w)
        self.global_vecs = self._coerce(global_vecs, w)
        self.b = int(b)
        self.only = bool(only_at_b)
        self.global_start = int(global_start)
        self.pos = {l: 0 for l in self.layers}

    def clear(self):
        self.set(None, global_vecs=None)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def _mk(self, layer):
        def _hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            T = h.shape[1]
            start = 0 if T > 1 else self.pos[layer]
            if T > 1:
                self.pos[layer] = T
            else:
                self.pos[layer] += 1
            lv = self.local_vecs.get(layer)
            gv = self.global_vecs.get(layer)
            if lv is None and gv is None:
                return None
            idx = torch.arange(start, start + T, device=h.device)
            add = None
            if lv is not None:
                lmask = (idx == self.b) if self.only else (idx >= self.b)
                if bool(lmask.any()):
                    add = lmask[None, :, None].to(h.dtype) * lv.to(h.device, h.dtype)[None, None, :]
            if gv is not None:
                gmask = idx >= self.global_start
                if bool(gmask.any()):
                    gadd = gmask[None, :, None].to(h.dtype) * gv.to(h.device, h.dtype)[None, None, :]
                    add = gadd if add is None else (add + gadd)
            if add is None:
                return None
            new = h + add
            return (new, *out[1:]) if isinstance(out, tuple) else new

        return _hook


def multi_layer_grads_at(model, prefix, conts, layers, b, batch=8):
    """Per-sample gradients of log p(cont | prefix) w.r.t. block output at position b for all `layers`
    simultaneously in a single backward pass per batch.  Returns {layer: ndarray (len(conts), d), float64}."""
    ls = [int(l) for l in layers]
    d = arch.hidden_size(model)
    if not conts:
        return {l: np.zeros((0, d), dtype=np.float64) for l in ls}
    dev = arch.text_model(model).embed_tokens.weight.device
    out = {l: [] for l in ls}
    state = {}

    def _mk(layer):
        def _hook(mod, inp, o):
            h = o[0] if isinstance(o, tuple) else o
            if layer not in state or h.shape[1] <= b:
                return None
            oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
            oh[b] = 1
            new = h + state[layer].to(h.dtype)[:, None, :] * oh[None, :, None]
            return (new, *o[1:]) if isinstance(o, tuple) else new

        return _hook

    blocks = arch.layers(model)
    handles = [blocks[l].register_forward_hook(_mk(l)) for l in ls]
    try:
        for c in range(0, len(conts), batch):
            part = [y if y else [prefix[-1]] for y in conts[c:c + batch]]
            for l in ls:
                state[l] = torch.zeros(len(part), d, device=dev, dtype=torch.float32, requires_grad=True)
            lp = seq_logp(model, prefix, part)
            grads = torch.autograd.grad(lp.sum(), [state[l] for l in ls], allow_unused=True)
            for l, g in zip(ls, grads):
                if g is None:
                    out[l].append(np.zeros((len(part), d), dtype=np.float64))
                else:
                    out[l].append(g.detach().float().cpu().numpy())
            state.clear()
    finally:
        for h in handles:
            h.remove()
    return {l: np.concatenate(out[l], axis=0).astype(np.float64) for l in ls}


def _log_probs_from_b(model, x, b):
    h = arch.forward_hidden(model, input_ids=x).last_hidden_state[:, b:]
    return arch.logits(model, h)[0].float().log_softmax(-1)


def _kl_from_log_probs(p0, p1, mode):
    kl = (p0.exp() * (p0 - p1)).sum(-1)
    val = float(kl[0] if mode == "at_b" else kl.mean())
    return max(val, 0.0)


def measured_kl_multi(model, hook, seq, b, local_vecs, mode="cursor", global_vecs=None, global_start=0):
    """Mean KL(unsteered || steered) of next-token distributions over positions >= b (cursor) or at b (at_b)."""
    dev = arch.text_model(model).embed_tokens.weight.device
    x = torch.tensor([seq], device=dev)
    try:
        with torch.no_grad():
            hook.clear()
            p0 = _log_probs_from_b(model, x, b)
            hook.set(local_vecs, b=b, only_at_b=(mode == "at_b"),
                     global_vecs=global_vecs, global_start=global_start)
            p1 = _log_probs_from_b(model, x, b)
    finally:
        hook.clear()
    return _kl_from_log_probs(p0, p1, mode)


def _scale_vecs(unit_vecs, alpha):
    if isinstance(unit_vecs, dict):
        return {l: alpha * v for l, v in unit_vecs.items()}
    return alpha * unit_vecs


def fast_calibrate(model, hook, seq, b, unit_vecs, target, mode="cursor", a0=1.0, max_evals=6, tol=0.05,
                   global_vecs=None, global_start=0):
    """Fast log-space secant + safeguarded bracketed bisection solver for `alpha` giving
    measured_kl_multi(..., alpha * unit_vecs) ≈ target within `max_evals` steered evaluations.

    Returns (alpha, achieved_kl, n_evals).
    """
    target = float(target)
    if target <= 0.0 or max_evals <= 0:
        return 0.0, 0.0, 0

    if isinstance(unit_vecs, dict):
        u_t = {int(l): torch.as_tensor(v, dtype=torch.float32) for l, v in unit_vecs.items() if v is not None}
    else:
        u_t = torch.as_tensor(unit_vecs, dtype=torch.float32)

    dev = arch.text_model(model).embed_tokens.weight.device
    x = torch.tensor([seq], device=dev)

    with torch.no_grad():
        hook.clear()
        p0 = _log_probs_from_b(model, x, b)

    y_target = math.log(target)
    tol_abs = tol * target

    n_evals = 0
    best_a, best_k, best_err = float(a0), 0.0, float("inf")
    lo_x = lo_y = hi_x = hi_y = None
    hist = []          # list of (log_a, log_kl)
    same_side = 0
    last_side = None

    def _eval(alpha):
        nonlocal n_evals, best_a, best_k, best_err, lo_x, lo_y, hi_x, hi_y, same_side, last_side
        alpha = float(max(alpha, 1e-12))
        with torch.no_grad():
            hook.set(_scale_vecs(u_t, alpha), b=b, only_at_b=(mode == "at_b"),
                     global_vecs=global_vecs, global_start=global_start)
            p1 = _log_probs_from_b(model, x, b)
        k = _kl_from_log_probs(p0, p1, mode)
        n_evals += 1
        err = abs(k - target)
        if err < best_err:
            best_a, best_k, best_err = alpha, k, err
        lx = math.log(alpha)
        ly = math.log(max(k, 1e-30))
        hist.append((lx, ly))
        if k < target:
            if lo_x is None or lx > lo_x:
                lo_x, lo_y = lx, ly
            same_side = same_side + 1 if last_side == "lo" else 1
            last_side = "lo"
        else:
            if hi_x is None or lx < hi_x:
                hi_x, hi_y = lx, ly
            same_side = same_side + 1 if last_side == "hi" else 1
            last_side = "hi"
        return k, lx, ly

    try:
        a = max(float(a0), 1e-8)
        while n_evals < max_evals:
            k, lx, ly = _eval(a)
            if abs(k - target) <= tol_abs or n_evals >= max_evals:
                break

            if n_evals == 1:
                # Local quadratic prior: KL(alpha) ~ c * alpha^2 -> log-log slope ≈ 2
                if k <= 1e-15:
                    nx = lx + math.log(10.0)
                else:
                    step = 0.5 * (y_target - ly)
                    step = max(-math.log(1e3), min(math.log(1e3), step))
                    nx = lx + step
            elif lo_x is not None and hi_x is not None:
                # Bracketed: safeguarded log-space secant + Illinois bisection fallback
                mid = 0.5 * (lo_x + hi_x)
                if same_side >= 2 or (hi_y - lo_y) <= 1e-12:
                    nx = mid
                    same_side = 0
                else:
                    x_sec = lo_x + (y_target - lo_y) * (hi_x - lo_x) / (hi_y - lo_y)
                    margin = 0.15 * (hi_x - lo_x)
                    nx = min(max(x_sec, lo_x + margin), hi_x - margin)
            else:
                # Unbracketed: log-space secant with clamped step, falling back to geometric expansion
                (x0, y0), (x1, y1) = hist[-2], hist[-1]
                dx, dy = x1 - x0, y1 - y0
                slope = dy / dx if abs(dx) > 1e-9 else 0.0
                if slope > 0.1:
                    step = (y_target - y1) / slope
                    step = max(-math.log(100.0), min(math.log(100.0), step))
                    nx = x1 + step
                else:
                    # Saturated or flat region: step toward target
                    nx = x1 + (math.log(4.0) if y1 < y_target else -math.log(4.0))

            # Avoid revisiting an already-evaluated point
            if any(abs(nx - hx) < 1e-6 for hx, _ in hist):
                if lo_x is not None and hi_x is not None:
                    nx = 0.5 * (lo_x + hi_x)
                else:
                    nx = lx + (math.log(2.0) if ly < y_target else -math.log(2.0))
            a = math.exp(max(-25.0, min(25.0, nx)))
    finally:
        hook.clear()

    return float(best_a), float(best_k), int(n_evals)
