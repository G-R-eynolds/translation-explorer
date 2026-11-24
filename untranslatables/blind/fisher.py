"""Model side (GPU): prompts with shared source positions, hooks, sampling, per-sample gradients, KL, steering (§2.1).

Prompt = [chat prefix] + [source text] + [channel instruction + chat tail], each piece tokenised separately and the ids
concatenated, so the source tokens (positions P) are the same ids in every channel and nothing channel-specific comes
before them.  (Cost: the boundary between the text and "\n\n" is not merged into one token, e.g. Llama's ".\n\n";
identical across channels.)

Gradients: at every stored layer k a zero leaf z_k (B, |P|, d) is ADDED to the block-k output at P.  d log p / d z_k is
the total derivative with respect to the residual at (k, P); because z is added (not a detached copy spliced in) the
gradient also flows on to the earlier stored layers.  Each batch row has its own z row, so the gradients are
per-sample, which the Fisher outer products need.
"""
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ut import arch
from ut.common import chat, seed_all

def dev(model):
    """Device of the text stack (model.device follows the first parameter, which is Gemma 3's image encoder once
    that is parked on the CPU)."""
    return arch.text_model(model).embed_tokens.weight.device


SENTINEL = "⁣@@SRC@@⁣"


def stop_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|im_end|>", "<|END_OF_TURN_TOKEN|>", "<|end_of_text|>", "<end_of_turn>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


class Prompter:
    def __init__(self, tok):
        self.tok = tok
        full = chat(tok, SENTINEL)
        pre, tail = full.split(SENTINEL)
        self.pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
        self.tail = tail

    def src_ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def build(self, text, instr):
        """-> (ids, (p0, p1)) with ids[p0:p1] the source tokens."""
        s = self.src_ids(text)
        suf = self.tok(instr + self.tail, add_special_tokens=False)["input_ids"]
        p0 = len(self.pre_ids)
        return self.pre_ids + s + suf, (p0, p0 + len(s))

    def token_spans(self, text, char_spans):
        """char spans [(name, a, b)] -> {name: [token indices within the source]} (tokens overlapping the span)."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        out = {}
        for name, a, b in char_spans:
            out[name] = [j for j, (x, y) in enumerate(enc["offset_mapping"]) if x < b and y > a and y > x]
        return out

    def tokens(self, text):
        ids = self.src_ids(text)
        return [self.tok.decode([i]) for i in ids]


class Hooks:
    """Forward hooks on the stored blocks.  Per call set .P = (p0, p1) and at most one of
      .z[k]      grad leaf (B, |P|, d), added at P            (gradients)
      .delta[k]  fixed tensor (B or 1, |P|, d), added at P     (steering, finite differences)
    and optionally .capture = {k...} to keep the (post-edit) states at P in .cap[k].  A call whose sequence is shorter
    than p1 (a decode step with a KV cache) passes through untouched."""

    def __init__(self, model, layers):
        self.layers, self.P, self.z, self.delta, self.capture, self.cap = list(layers), None, {}, {}, set(), {}
        self.handles = [arch.layers(model)[k].register_forward_hook(self._mk(k)) for k in self.layers]

    def clear(self):
        self.z, self.delta, self.capture, self.cap = {}, {}, set(), {}

    def _mk(self, k):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.P is None or h.shape[1] < self.P[1]:
                return None
            p0, p1 = self.P
            add = self.z.get(k)
            if add is None:
                add = self.delta.get(k)
            new = None
            if add is not None:
                new = torch.cat([h[:, :p0], h[:, p0:p1] + add.to(h.dtype), h[:, p1:]], 1)
            if k in self.capture:
                self.cap[k] = (new if new is not None else h)[:, p0:p1].detach().float().cpu()
            if new is None:
                return None
            return (new, *out[1:]) if isinstance(out, tuple) else new
        return hook


# ------------------------------------------------------------------ sampling
def generate_ids(model, tok, prompt_ids, n, max_new, temperature=1.0, seed=0, batch=64, rows_delta=None, hooks=None):
    """n continuations of one prompt (no padding: every row is the same prompt).  temperature 0 = greedy.
    Pure ancestral sampling at T (top_k=0, top_p=1: the model's true distribution, §2.1).  Returns y id lists, each
    cut after (and including) the first stop token; a capped output has no stop token.
    rows_delta: optional {k: (n, |P|, d)} per-row steering deltas (requires batch >= n)."""
    from transformers import GenerationConfig
    stops = stop_ids(tok)
    gc = GenerationConfig(max_new_tokens=max_new, eos_token_id=stops, pad_token_id=tok.pad_token_id,
                          do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                          top_k=0 if temperature > 0 else None, top_p=1.0 if temperature > 0 else None)
    out = []
    for c in range(0, n, batch):
        m = min(batch, n - c)
        seed_all((seed * 100003 + c) % 2 ** 32)
        x = torch.tensor([prompt_ids] * m, device=dev(model))
        if rows_delta is not None:
            hooks.delta = {k: v[c:c + m].to(dev(model)) for k, v in rows_delta.items()}
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)[:, len(prompt_ids):]
        if rows_delta is not None:
            hooks.delta = {}
        for row in g.tolist():
            cut = next((i + 1 for i, t in enumerate(row) if t in stops), None)
            out.append(row[:cut] if cut else [t for t in row if t != tok.pad_token_id][:max_new])
    return out


def decode(tok, y):
    return tok.decode(y, skip_special_tokens=True).strip()


# ------------------------------------------------------------------ teacher forcing
def _token_logp(model, h, t):
    return arch.logits(model, h).log_softmax(-1).gather(1, t[:, None])[:, 0]


def seq_logp(model, prompt_ids, ys, chunk=256):
    """Sum of log p(y_b | prompt) for each row (right padding; prompt shared).  Differentiable through hooks."""
    L = len(prompt_ids)
    rows = [prompt_ids + y for y in ys]
    T = max(len(r) for r in rows)
    pad = 0
    ids = torch.tensor([r + [pad] * (T - len(r)) for r in rows], device=dev(model))
    att = torch.tensor([[1] * len(r) + [0] * (T - len(r)) for r in rows], device=dev(model))
    H = arch.forward_hidden(model, input_ids=ids, attention_mask=att).last_hidden_state
    bi = [b for b, y in enumerate(ys) for _ in y]
    ti = [L - 1 + j for y in ys for j in range(len(y))]
    tg = torch.tensor([t for y in ys for t in y], device=dev(model))
    hs = H[bi, ti]
    lp = torch.cat([checkpoint(_token_logp, model, hs[i:i + chunk], tg[i:i + chunk], use_reentrant=False)
                    if hs.requires_grad else _token_logp(model, hs[i:i + chunk], tg[i:i + chunk])
                    for i in range(0, len(tg), chunk)])
    return torch.zeros(len(ys), device=lp.device, dtype=lp.dtype).index_add(0, torch.tensor(bi, device=lp.device), lp)


def per_sample_grads(model, hooks, prompt_ids, P, ys, layers, batch=8):
    """-> ({k: (n, |P|, d) float32 numpy}, logp (n,)).  Gradient of log p(y|h) w.r.t. the residual at (k, P)."""
    d = arch.hidden_size(model)
    G = {k: [] for k in layers}
    LP = []
    hooks.P = P
    for c in range(0, len(ys), batch):
        yb = ys[c:c + batch]
        hooks.z = {k: torch.zeros(len(yb), P[1] - P[0], d, device=dev(model), dtype=torch.float32,
                                  requires_grad=True) for k in layers}
        lp = seq_logp(model, prompt_ids, yb)
        gr = torch.autograd.grad(lp.sum(), [hooks.z[k] for k in layers])
        for k, g in zip(layers, gr):
            G[k].append(g.detach().float().cpu().numpy())
        LP.append(lp.detach().float().cpu().numpy())
        hooks.z = {}
    return {k: np.concatenate(v) for k, v in G.items()}, np.concatenate(LP)


def logp_shifted(model, hooks, prompt_ids, P, ys, delta=None, batch=16):
    """log p(y | h + delta) per sample, no grad; delta {k: (|P|, d)} or None."""
    hooks.P = P
    out = []
    for c in range(0, len(ys), batch):
        yb = ys[c:c + batch]
        hooks.delta = {k: v[None].to(dev(model)) for k, v in (delta or {}).items()}
        with torch.no_grad():
            out.append(seq_logp(model, prompt_ids, yb).float().cpu().numpy())
        hooks.delta = {}
    return np.concatenate(out)


def states_at(model, hooks, prompt_ids, P, layers):
    """Residual states at (k, P) for one prompt: {k: (|P|, d) float32 tensor (cpu)}."""
    hooks.P, hooks.capture = P, set(layers)
    x = torch.tensor([prompt_ids], device=dev(model))
    with torch.no_grad():
        arch.forward_hidden(model, input_ids=x, attention_mask=torch.ones_like(x))
    cap = {k: hooks.cap[k][0] for k in layers}
    hooks.capture, hooks.cap = set(), {}
    return cap


def steer_greedy(model, tok, hooks, prompt_ids, P, layer, deltas, max_new):
    """Greedy continuation for each delta (S, |P|, d) added at (layer, P); one row per delta."""
    hooks.P = P
    ys = generate_ids(model, tok, prompt_ids, len(deltas), max_new, temperature=0, batch=len(deltas),
                      rows_delta={layer: deltas}, hooks=hooks)
    return [decode(tok, y) for y in ys]


# ------------------------------------------------------------------ storage
def save_grads(path, G, meta):
    """Scaled float16: g / scale with scale = max|g| per layer (float16 alone underflows: residual gradients of a
    large-norm residual stream are ~1e-5 per component)."""
    arrs = {}
    for k, g in G.items():
        s = float(np.abs(g).max()) or 1.0
        arrs[f"g{k}"] = (g / s).astype(np.float16)
        arrs[f"s{k}"] = np.float32(s)
    import json
    arrs["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode(), dtype=np.uint8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    tmp.rename(path)


def load_grads(path, layers=None):
    import json
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes().decode())
    ks = [int(n[1:]) for n in z.files if n.startswith("g")]
    G = {k: z[f"g{k}"].astype(np.float32) * float(z[f"s{k}"]) for k in ks if layers is None or k in layers}
    return G, meta


class Clock:
    def __init__(self, label, total):
        self.label, self.total, self.t0, self.n = label, total, time.time(), 0

    def tick(self, what=""):
        self.n += 1
        el = time.time() - self.t0
        eta = el / self.n * (self.total - self.n)
        print(f"[{self.label}] {self.n}/{self.total} {what}  {el / 60:.1f} min, ~{eta / 60:.1f} min left", flush=True)
