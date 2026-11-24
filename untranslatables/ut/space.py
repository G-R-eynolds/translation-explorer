"""The concept space at one layer: pools and target words, language identity removed (handoff §5.1-5.2).

Centring (subtract the language centroid, then L2-normalise; no Procrustes alignment [2605.24310]):
  * pool words: their template-averaged state minus the pool's mean, at that layer [2004.05160];
  * a target word's template context k: minus the pool's centroid for template k (same language, same frame);
  * natural and frame contexts: minus the mean of ALL target words' natural+frame contexts in that language
    (templates and running text sit in different regions, so each context type is centred by its own kind).
Romanesco has no pool: its contexts are centred with the Italian centroids (recorded in outputs).
"""
import difflib
import functools
import unicodedata

import numpy as np

from ut.common import load_json, cache_dir
from ut.represent import load_cached

CENTROID_LANG = {"rom": "it"}


def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


@functools.lru_cache(maxsize=32)
def pool(model_key, lang, layer):
    """(words, centred unit matrix [n, d] float32, zipf dict)."""
    cd = cache_dir(model_key)
    X = np.load(cd / f"pool_{lang}.npy", mmap_mode="r")[:, layer].astype(np.float32)
    words = load_json(cd / f"pool_{lang}.json")["words"]
    return words, _unit(X - X.mean(0)), X.mean(0)


@functools.lru_cache(maxsize=32)
def template_centroids(model_key, lang, layer):
    return np.load(cache_dir(model_key) / f"pool_{lang}_tcentroid.npy", mmap_mode="r")[:, layer].astype(np.float32)


class Targets:
    """Target-word context states (cache 'words_t1', written by recipe.py --extract), centred per context kind."""

    def __init__(self, model_key):
        self.H, meta = load_cached(model_key, "words_t1")
        self.items = meta["items"]
        self.index = meta["index"]          # rows: [word_id, kind, template_k or -1, language]
        self.model_key = model_key

    @functools.lru_cache(maxsize=64)
    def at(self, layer):
        """dict word_id -> (kinds list, centred unit vectors [n_ctx, d])."""
        X = self.H[:, layer].astype(np.float32)
        langs = sorted({r[3] for r in self.index})
        nat_cent = {}
        for l in langs:
            rows = [i for i, r in enumerate(self.index) if r[3] == l and r[1] != "template"]
            if rows:
                nat_cent[l] = X[rows].mean(0)
        out = {}
        for i, (wid, kind, k, l) in enumerate(self.index):
            cl = CENTROID_LANG.get(l, l)
            c = template_centroids(self.model_key, cl, layer)[k] if kind == "template" else \
                nat_cent.get(l if l not in CENTROID_LANG else cl, nat_cent.get(l))
            out.setdefault(wid, ([], []))
            out[wid][0].append(kind); out[wid][1].append(X[i] - c)
        return {w: (kinds, _unit(np.stack(v))) for w, (kinds, v) in out.items()}

    def mean_vector(self, layer, wid):
        kinds, V = self.at(layer)[wid]
        return _unit(V.mean(0))


def _fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def cognate_mask(pool_words, forms, extra=(), ratio=0.75, prefix=5):
    """Boolean mask of pool words to EXCLUDE for a target: the word itself, near-identical strings (accent-
    folded SequenceMatcher ratio >= 0.75), strings sharing a 5-letter prefix, and listed extras."""
    forms = [_fold(f) for f in forms if f]
    extra = {_fold(e) for e in extra}
    mask = np.zeros(len(pool_words), bool)
    def similar(a, b):
        sm = difflib.SequenceMatcher(None, a, b)
        return sm.real_quick_ratio() >= ratio and sm.quick_ratio() >= ratio and sm.ratio() >= ratio

    for i, w in enumerate(pool_words):
        fw = _fold(w)
        if fw in extra or any(fw == f or (len(f) >= prefix and len(fw) >= prefix and fw[:prefix] == f[:prefix])
                              or similar(fw, f) for f in forms):
            mask[i] = True
    return mask
