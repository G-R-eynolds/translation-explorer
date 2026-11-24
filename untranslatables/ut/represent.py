"""Residual-stream representations: span-mean states at every layer, cached to disk; steering and
patching hooks.

Layer convention (used everywhere): layer l = hidden_states[l], i.e. 0 is the embedding output and l>=1 is
the output of decoder block l-1.  Steering or patching "at layer l" modifies the output of block l-1, so
everything computed after block l-1 sees it.  Quirk (transformers 5, verified in tests/test_core.py): the
RECORDED hidden_states[l] at the hooked layer itself still shows the pre-hook value, because the library's own
output-capturing hook runs first; hidden_states[l+1:] and the logits reflect the change.  No experiment reads
the hooked layer itself (steering reads above its band; patching acts through generation).
"""
import contextlib, re

import numpy as np

from ut.common import Progress, cache_dir, load_json, save_json, stable_hash

MARK = re.compile(r"\[\[(.+?)\]\]")


def unmark(text):
    """'Sinto [[saudade]] de casa' -> ('Sinto saudade de casa', (6, 13)).  Exactly one marked span."""
    m = list(MARK.finditer(text))
    if len(m) != 1:
        raise ValueError(f"need exactly one [[span]]: {text!r}")
    s = m[0].start()
    clean = text[:s] + m[0].group(1) + text[m[0].end():]
    return clean, (s, s + len(m[0].group(1)))


def blocks(model):
    from ut.arch import layers
    return layers(model)


def trunk(model):
    from ut.arch import text_model
    return text_model(model)


def _tok_span(offsets, span):
    s, e = span
    idx = [k for k, (a, b) in enumerate(offsets) if b > s and a < e and b > a]
    if not idx:
        raise ValueError(f"span {span} covers no token")
    return idx


def span_states(model, tok, items, batch=64, dtype=np.float16, out=None, progress=None):
    """items: list of (text, (char_start, char_end)) or (text, None) for the last token.
    Returns array [n, L+1, d]: mean residual state over the span's tokens at every layer.
    If `out` is given (e.g. a memmap of that shape) it is filled in place and returned."""
    import torch
    tok.padding_side = "right"
    from ut.arch import hidden_size, n_layers
    L1, d = n_layers(model) + 1, hidden_size(model)
    if out is None:
        out = np.zeros((len(items), L1, d), dtype=dtype)
    order = sorted(range(len(items)), key=lambda k: len(items[k][0]))     # length-sorted batches: less padding
    for c in range(0, len(order), batch):
        ks = order[c:c + batch]
        texts = [items[k][0] for k in ks]
        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping").tolist()
        m = torch.zeros(enc["input_ids"].shape, dtype=torch.float32)
        for b, k in enumerate(ks):
            if items[k][1] is None:
                m[b, int(enc["attention_mask"][b].sum()) - 1] = 1
            else:
                m[b, _tok_span(offs[b], items[k][1])] = 1
        with torch.no_grad():
            hs = trunk(model)(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
        mm = m.to(model.device)[:, :, None]
        pooled = torch.stack([(h.float() * mm).sum(1) / mm.sum(1) for h in hs], 1)
        out[ks] = pooled.cpu().numpy().astype(dtype)
        if progress:
            progress.tick(min(c + batch, len(order)))
    return out


def cached_span_states(model, tok, name, items, batch=64, extra_meta=None):
    """span_states with a disk cache at cache/<model>/<name>.npy (memmap) + .json sidecar (items, hash).
    Recomputed only if the item list changed."""
    cd = cache_dir(model.key)
    npy, meta = cd / f"{name}.npy", cd / f"{name}.json"
    h = stable_hash(items, model.key)
    from ut.arch import hidden_size, n_layers
    shape = (len(items), n_layers(model) + 1, hidden_size(model))
    if npy.exists() and meta.exists() and load_json(meta).get("hash") == h:
        return np.load(npy, mmap_mode="r")
    mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=shape)
    span_states(model, tok, items, batch=batch, out=mm, progress=Progress(f"extract {name}", len(items)))
    mm.flush()
    save_json(meta, dict(hash=h, n=len(items), shape=shape, items=items, **(extra_meta or {})))
    return np.load(npy, mmap_mode="r")


def load_cached(model_key, name):
    cd = cache_dir(model_key)
    return np.load(cd / f"{name}.npy", mmap_mode="r"), load_json(cd / f"{name}.json")


# ------------------------------------------------------------------ hooks
def _replace_output(output, new):
    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def steering(model, vectors, gamma):
    """Add gamma * vectors[l] to the residual stream after block l-1, at every position, for each l in
    `vectors` (dict layer -> tensor[d]).  gamma = 0 is a no-op."""
    hooks = []
    try:
        for l, v in vectors.items():
            def hook(mod, inp, output, v=v):
                h = _hidden(output)
                return _replace_output(output, h + gamma * v.to(h.dtype).to(h.device))
            hooks.append(blocks(model)[l - 1].register_forward_hook(hook))
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def patching(model, latents, position=-1):
    """Overwrite the residual stream at `position` of the PROMPT (prefill pass only) after block l-1 with
    latents[l] (dict layer -> tensor[d] for every row, or tensor[batch, d] for one latent per row).  Decoding
    steps (sequence length 1, KV cache) are left alone, so generation continues from the patched state."""
    hooks = []
    try:
        for l, v in latents.items():
            def hook(mod, inp, output, v=v):
                h = _hidden(output)
                if h.shape[1] == 1:
                    return output
                h = h.clone()
                h[:, position] = v.to(h.dtype).to(h.device)       # [d] broadcasts; [B, d] goes row by row
                return _replace_output(output, h)
            hooks.append(blocks(model)[l - 1].register_forward_hook(hook))
        yield
    finally:
        for h in hooks:
            h.remove()
