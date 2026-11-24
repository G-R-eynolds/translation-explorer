"""Token log-probabilities of plain texts (no chat template), with character offsets."""
import numpy as np


def token_logprobs(model, tok, texts, batch=64):
    """For each text: (offsets [(start, end)] per token after the first, logprobs [n_tokens - 1]).
    The first token (BOS or first word piece) has no prediction and is dropped."""
    import torch
    tok.padding_side = "right"
    out = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda k: len(texts[k]))
    for c in range(0, len(order), batch):
        ks = order[c:c + batch]
        enc = tok([texts[k] for k in ks], return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping").tolist()
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            from ut.arch import forward_hidden, logits as _logits
            h = forward_hidden(model, input_ids=ids, attention_mask=enc["attention_mask"].to(model.device)).last_hidden_state
        for b, k in enumerate(ks):
            n = int(enc["attention_mask"][b].sum())
            # project only the real positions (full-vocab logits for a whole batch would not fit on 12 GB)
            with torch.no_grad():
                lg = _logits(model, h[b, :n - 1])
                lp = torch.log_softmax(lg, -1).gather(1, ids[b, 1:n, None])[:, 0].cpu().numpy()
            out[k] = (offs[b][1:n], lp)
    return out


def span_logprob(offsets, lps, span):
    """Mean log-prob of the tokens overlapping a character span (None if the span starts the text)."""
    s, e = span
    v = [lp for (a, b), lp in zip(offsets, lps) if b > s and a < e and b > a]
    return float(np.mean(v)) if v else None
