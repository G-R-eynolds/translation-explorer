"""Sampling, scoring and steered continuation (spec §Method 1, 5)."""
import numpy as np
import torch

from blind.fisher import seq_logp
from explorer.frame import STOPS, cut
from ut.common import seed_all


def _dev(model):
    from ut import arch
    return arch.text_model(model).embed_tokens.weight.device


def generate(model, tok, ids, n, max_new, temperature=1.0, seed=0, batch=64):
    """n continuations of one prompt (rows share it: no padding).  temperature 0 = greedy; otherwise pure ancestral
    sampling (top_k=0, top_p=1).  Stops at a blank line / 'Spanish:' (cut applied) or EOS."""
    from transformers import GenerationConfig
    gc = GenerationConfig(max_new_tokens=max_new, do_sample=temperature > 0, eos_token_id=tok.eos_token_id,
                          pad_token_id=tok.pad_token_id, stop_strings=list(STOPS),
                          temperature=temperature if temperature > 0 else None,
                          top_k=0 if temperature > 0 else None, top_p=1.0 if temperature > 0 else None)
    out = []
    for c in range(0, n, batch):
        m = min(batch, n - c)
        seed_all((seed * 100003 + c) % 2 ** 32)
        x = torch.tensor([ids] * m, device=_dev(model))
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc,
                               tokenizer=tok)[:, len(ids):].tolist()
        for row in g:
            row = [t for t in row if t not in (tok.pad_token_id, tok.eos_token_id)]
            out.append(cut(tok, row))
    return out


def sample(model, tok, ids, n, max_new, seed=0, temperature=1.0):
    return generate(model, tok, ids, n, max_new, temperature, seed)


def score(model, ids, ys, batch=16):
    """Summed log p(y | ids) per continuation (no grad).  Empty continuations score 0."""
    out = np.zeros(len(ys))
    idx = [i for i, y in enumerate(ys) if y]
    for c in range(0, len(idx), batch):
        part = idx[c:c + batch]
        with torch.no_grad():
            lp = seq_logp(model, ids, [ys[i] for i in part]).float().cpu().numpy()
        out[part] = lp
    return out


def steered(model, tok, hook, ids, vec, b, max_new, n=1, temperature=0.0, only_at_b=False, seed=0):
    """Continue the prompt `ids` with vec added at absolute positions >= b (or == b) of the hooked layer."""
    hook.set(vec, b, only_at_b)
    try:
        return generate(model, tok, ids, n, max_new, temperature, seed)
    finally:
        hook.set(None)
