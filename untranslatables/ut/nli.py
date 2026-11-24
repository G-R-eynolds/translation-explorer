"""Meaning clustering of short translations by bidirectional NLI entailment (Kuhn et al. [2302.09664]);
semantic entropy H and Kernel Language Entropy [2405.20003].

Bare words make poor NLI premises, so each rendering is put in a fixed frame:
    'The <Language> word "<source word>" here means "<rendering>".'
Identical renderings (after lowercasing and stripping punctuation) are merged before any NLI call.
"""
import re
from collections import Counter

import numpy as np

from ut.common import LANG_EN

NLI_MODEL = "microsoft/deberta-large-mnli"


def normalise(s):
    s = s.strip().strip("\"'“”‘’«».,;:!?()").strip()
    return re.sub(r"\s+", " ", s.lower())


class NLI:
    def __init__(self, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        m = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(self.device).eval()
        self.m = m.half() if self.device == "cuda" else m
        lab = {v.lower(): k for k, v in m.config.id2label.items()}
        self.ent = lab["entailment"]
        self._memo = {}

    def p_entail(self, pairs, bs=256):
        todo = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        for c in range(0, len(todo), bs):
            a, b = zip(*todo[c:c + bs])
            enc = self.tok(list(a), list(b), return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.m(**enc).logits.float(), -1)[:, self.ent].cpu().numpy()
            self._memo.update(zip(todo[c:c + bs], p))
        return np.array([self._memo[p] for p in pairs])


def frame(word, lang, rendering):
    return f'The {LANG_EN.get(lang, lang).split(" (")[0]} word "{word}" here means "{rendering}".'


def cluster(nli, renderings, word, lang):
    """Cluster renderings by meaning. Returns dict(labels per rendering, counts per cluster,
    representative text per cluster, P matrix over unique strings, unique strings, unique counts)."""
    norm = [normalise(r) for r in renderings]
    cnt = Counter(n for n in norm if n)
    uniq = [u for u, _ in cnt.most_common()]
    framed = [frame(word, lang, u) for u in uniq]
    n = len(uniq)
    pairs = [(framed[i], framed[j]) for i in range(n) for j in range(n) if i != j]
    P = np.eye(n)
    if pairs:
        P[~np.eye(n, dtype=bool)] = nli.p_entail(pairs)
    # Kuhn et al. greedy: join the first class whose representative it bidirectionally entails
    reps, ulab = [], []
    for i in range(n):
        for c, r in enumerate(reps):
            if P[i, r] > 0.5 and P[r, i] > 0.5:
                ulab.append(c); break
        else:
            reps.append(i); ulab.append(len(reps) - 1)
    lab_of = dict(zip(uniq, ulab))
    labels = [lab_of.get(n, -1) for n in norm]          # -1 = empty output
    counts = Counter(l for l in labels)
    return dict(labels=labels, counts=dict(counts), reps={c: uniq[r] for c, r in enumerate(reps)},
                uniq=uniq, uniq_counts=[cnt[u] for u in uniq], P=P)


def semantic_entropy(counts):
    """H in bits over meaning clusters (empty outputs count as their own cluster)."""
    p = np.array(list(counts.values()), float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum()) + 0.0


def kle(P, weights):
    """Kernel Language Entropy (nats) with the symmetric NLI kernel K = (P + P^T)/2 over unique strings,
    each weighted by its count (rho = W^1/2 K W^1/2 / tr)."""
    K = np.clip((P + P.T) / 2, 0, 1)
    w = np.sqrt(np.asarray(weights, float))
    rho = (w[:, None] * K * w[None, :])
    rho /= np.trace(rho)
    ev = np.clip(np.linalg.eigvalsh(rho), 0, None)
    ev = ev[ev > 1e-12] / ev.sum()
    return float(-(ev * np.log(ev)).sum())
