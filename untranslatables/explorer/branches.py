"""Branch points along a reference translation (spec §Method 3).

timeline: at each output index t (the state at absolute position start+t-1 predicts y_ref[t]):
  entropy     H(p_t) of the next-token distribution (nats)
  fisher_top  top eigenvalue of the output Fisher at the final (normed) residual, Wᵀ(diag p − ppᵀ)W, by power
              iteration with matrix-vector products only (W = lm_head, never squared into a d×d matrix)
choose_t: the token where the reference commits to its reading (most reading-specific token); see its docstring.  Returned as an output index t; the absolute cursor position is start + t - 1.
"""
import numpy as np
import torch

from ut import arch


def timeline(model, ids, y_ref, iters=20, seed=0):
    dev = arch.text_model(model).embed_tokens.weight.device
    x = torch.tensor([ids + y_ref], device=dev)
    with torch.no_grad():
        H = arch.forward_hidden(model, input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
        W = model.lm_head.weight
        g = torch.Generator(device="cpu").manual_seed(seed)
        ent, top = [], []
        for t in range(len(y_ref)):
            h = H[len(ids) - 1 + t]
            p = arch.logits(model, h[None])[0].softmax(-1)
            ent.append(float(-(p * p.clamp_min(1e-30).log()).sum()))
            v = torch.randn(W.shape[1], generator=g).to(dev)
            v /= v.norm()
            lam = 0.0
            for _ in range(iters):
                Wv = (W @ v.to(W.dtype)).float()
                u = p * Wv - p * (p @ Wv)
                Fv = (W.T @ u.to(W.dtype)).float()
                lam = float(Fv @ v)
                v = Fv / Fv.norm().clamp_min(1e-30)
            top.append(lam)
    return dict(entropy=ent, fisher_top=top)


def align_index(y_ref, m, t):
    """Position in member m that corresponds to position t of y_ref (sequence alignment; the end of the last matched
    block before t when y_ref[t-1] itself has no match)."""
    import difflib
    if t == 0:
        return 0
    j = 0
    for a, b, size in difflib.SequenceMatcher(a=y_ref, b=m, autojunk=False).get_matching_blocks():
        if a >= t:
            break
        j = b + min(size, t - a)
    return j


def choose_t(y_ref, members_by_reading, ref_reading, fisher_top, min_spec=0.2):
    """The token where the reference commits to its reading: argmax over t of
        spec(t) = share of ref-reading members containing y_ref[t]  -  share of other readings' members containing it
    (earliest on ties); fallback argmax fisher_top when max spec < min_spec.  Revised twice on 24 Nov: exact-prefix
    rules either fired on stray first tokens (t = 0) or never fired (punctuation variants break exact prefixes)."""
    own = members_by_reading[ref_reading]
    other = [m for r, ms in members_by_reading.items() if r != ref_reading for m in ms]
    if own and other and y_ref:
        spec = [np.mean([tok in m for m in own]) - np.mean([tok in m for m in other]) for tok in y_ref]
        t = int(np.argmax(spec))
        if spec[t] >= min_spec:
            return t, float(spec[t]), "reference commits to its reading here"
    if len(fisher_top):
        return int(np.argmax(fisher_top)), None, "no reading-specific token: max output Fisher"
    return 0, None, "empty reference"
