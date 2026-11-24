"""Word alignment (HANDOFF_internals §3.2): SimAlign's "itermax" with xlm-roberta-base, re-implemented in batches.

Method (Jalili Sabet et al. 2020, github.com/cisnlp/simalign, MIT licence): contextual subword embeddings from layer 8
of xlm-roberta-base; cosine similarity between all source and target subwords; itermax = mutual argmax, then up to
one more round of mutual argmax among still-unaligned rows/columns with the alpha=0.9 mask; a word pair is linked if
any of their subwords are.  The `iter_max` function follows SimAlign's code.  Only 1-to-1 word links are kept, since
the N1 pairs need one source word per target word; what has no 1-to-1 counterpart is dropped, and the share of hot
words that get a link is reported (§13: unaligned hot words are themselves evidence).

    python -m internals align --set what='["es-en","hi-en","flores"]'
Outputs: $UT_CORPUS/align_<name>.jsonl ({id, links: [[i, j], ...]} over sent.data.WORD words) and
align_<name>_spotcheck.txt (30 random sentences, for checking by eye).
"""
import json
import random

import numpy as np

from internals.common import CORPUS, stage
from sent.data import WORD

ALIGN_MODEL, ALIGN_LAYER = "FacebookAI/xlm-roberta-base", 8
FLORES_PAIRS = [("es", "en"), ("hi", "en"), ("es", "hi"), ("bho", "en")]


def iter_max(sim, max_count=2, alpha_ratio=0.9):
    """SimAlign's itermax on one similarity matrix [m, n] -> 0/1 matrix of links."""
    m, n = sim.shape
    forward = np.eye(n)[sim.argmax(axis=1)]
    backward = np.eye(m)[sim.argmax(axis=0)]
    inter = forward * backward.transpose()
    if min(m, n) <= 2:
        return inter
    count = 1
    while count < max_count:
        mask_x = 1.0 - np.tile(inter.sum(1)[:, None], (1, n)).clip(0.0, 1.0)
        mask_y = 1.0 - np.tile(inter.sum(0)[None, :], (m, 1)).clip(0.0, 1.0)
        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
            mask *= 0.0; mask_zeros *= 0.0
        new_sim = sim * mask
        fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
        bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
        new_inter = fwd * bac
        if np.array_equal(inter + new_inter, inter):
            break
        inter = inter + new_inter
        count += 1
    return inter


class Aligner:
    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ALIGN_MODEL)
        self.m = AutoModel.from_pretrained(ALIGN_MODEL).to(self.device).eval()

    def embed(self, word_lists, bs=64):
        """For each sentence (list of words): (subword embeddings [s, 768] unit-norm, subword -> word index)."""
        out = []
        for c in range(0, len(word_lists), bs):
            chunk = word_lists[c:c + bs]
            enc = self.tok(chunk, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True,
                           max_length=256)
            with self.torch.no_grad():
                hs = self.m(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
            H = self.torch.nn.functional.normalize(hs[ALIGN_LAYER].float(), dim=-1).cpu().numpy()
            for b in range(len(chunk)):
                wid = enc.word_ids(b)
                keep = [k for k, w in enumerate(wid) if w is not None]
                out.append((H[b, keep], np.array([wid[k] for k in keep])))
        return out

    def align(self, src_words, tgt_words, bs=64):
        """1-to-1 word links for each sentence pair."""
        E1, E2 = self.embed(src_words, bs), self.embed(tgt_words, bs)
        res = []
        for (a, wa), (b, wb), sw, tw in zip(E1, E2, src_words, tgt_words):
            if len(a) == 0 or len(b) == 0:
                res.append([]); continue
            sim = (a @ b.T + 1) / 2
            L = iter_max(sim)
            links = {(int(wa[i]), int(wb[j])) for i, j in zip(*np.nonzero(L))}
            cs, ct = {}, {}
            for i, j in links:
                cs[i] = cs.get(i, 0) + 1; ct[j] = ct.get(j, 0) + 1
            res.append(sorted([i, j] for i, j in links if cs[i] == 1 and ct[j] == 1))
        return res


def words(s):
    return [m.group() for m in WORD.finditer(s)]


def _write(name, ids, X, Y, links):
    p = CORPUS / f"align_{name}.jsonl"
    with open(p, "w") as f:
        for i, l in zip(ids, links):
            f.write(json.dumps(dict(id=i, links=l)) + "\n")
    rng = random.Random(0)
    q = CORPUS / f"align_{name}_spotcheck.txt"
    with open(q, "w") as f:
        f.write(f"Spot check for {name}: 30 random sentence pairs, every kept 1-to-1 link. Mark wrong links.\n\n")
        for k in rng.sample(range(len(ids)), min(30, len(ids))):
            xw, yw = words(X[k]), words(Y[k])
            f.write(f"[{ids[k]}]\n  {X[k]}\n  {Y[k]}\n  " + "  ".join(f"{xw[i]}={yw[j]}" for i, j in links[k]) + "\n\n")
    n_links = sum(len(l) for l in links)
    n_words = sum(len(words(x)) for x in X)
    print(f"  {name}: {len(ids)} sentences, {n_links} 1-to-1 links ({n_links / max(n_words, 1):.0%} of source words)",
          flush=True)
    return [p, q]


def read_links(name):
    return {r["id"]: r["links"] for r in map(json.loads, open(CORPUS / f"align_{name}.jsonl"))}


@stage("align")
def run(cfg):
    from internals.data import read_corpus, read_flores
    al = Aligner()
    outs = []
    for what in cfg.get("what", ["es-en", "hi-en", "flores"]):
        if what == "flores":
            rows = read_flores()
            for a, b in FLORES_PAIRS:
                X, Y = [r[a] for r in rows], [r[b] for r in rows]
                outs += _write(f"flores_{a}-{b}", [r["id"] for r in rows], X, Y,
                               al.align([words(x) for x in X], [words(y) for y in Y]))
        else:
            rows = read_corpus(what)
            X, Y = [r["x"] for r in rows], [r["y"] for r in rows]
            outs += _write(what, [r["id"] for r in rows], X, Y, al.align([words(x) for x in X], [words(y) for y in Y]))
    return outs
