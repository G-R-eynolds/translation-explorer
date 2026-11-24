"""Logit lens and word -> token-id sets, adapted from the method of Wendler et al. [2402.10588]
(github.com/epfl-dlab/llm-latent-language, Translation.ipynb: prompt format, prefix-token sets, per-layer
decoding).  Re-implemented rather than copied: that repository ships no licence, and its token handling
assumes Llama-2's SentencePiece '▁' vocabulary, whereas Llama-3 / Qwen / Aya use byte-level BPE.
"""
import functools
import unicodedata

import torch


@torch.no_grad()
def unembed(model, h, final=False):
    """Logit lens: final norm (skipped for the last hidden state, which HF already normalises) then the
    unembedding.  Cohere/Aya scales logits by config.logit_scale; that is applied too."""
    from ut.arch import final_norm, logits
    if not final:
        h = final_norm(model)(h)
    return logits(model, h)


@torch.no_grad()
def layer_logprobs(model, hidden_states, pos=-1):
    """hidden_states: tuple from output_hidden_states (batch 1).  Returns [L+1, V] log-probs at `pos`."""
    last = len(hidden_states) - 1
    return torch.stack([torch.log_softmax(unembed(model, h[0, pos], final=(l == last)), -1)
                        for l, h in enumerate(hidden_states)])


@functools.lru_cache(maxsize=4)
def _decoded_vocab(tok):
    return [(i, tok.decode([i])) for i in range(len(tok))]


def _norm(s):
    return unicodedata.normalize("NFC", s)


def prefix_token_ids(tok, word, min_prefix=1):
    """Ids of every vocabulary token whose text (leading space stripped) is a prefix of the word, in its
    given and capitalised forms, with or without a leading space -- Wendler et al.'s 'token_prefixes',
    adapted to byte-level BPE.  min_prefix=1 matches their choice (single letters count)."""
    forms = {_norm(word), _norm(word[:1].upper() + word[1:]), _norm(word.lower())}
    ids = set()
    for i, s in _decoded_vocab(tok):
        t = _norm(s.lstrip())
        if len(t) >= min_prefix and any(f.startswith(t) for f in forms):
            ids.add(i)
    return ids


def disjoint_sets(tok, words_a, words_b, min_prefix=1):
    """Token-id sets for two answer words with shared tokens removed from both, so a prefix such as 'n'
    for nuage/nube cannot count for either language."""
    a = set().union(*(prefix_token_ids(tok, w, min_prefix) for w in words_a))
    b = set().union(*(prefix_token_ids(tok, w, min_prefix) for w in words_b))
    return sorted(a - b), sorted(b - a)
