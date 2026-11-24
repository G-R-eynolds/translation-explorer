"""Thread 1: recipes in the concept space (handoff §5.1, §5.4, §6.5).

    python recipe.py --model 3b --extract      # GPU: every target word's contexts, all layers -> cache 'words_t1'
    python recipe.py --model 3b --analyze      # CPU: recipes, usage clouds, nulls, pivots, layer dependence, loanwords

Recipe: non-negative orthogonal matching pursuit (NNLS refit after each pick, k <= 10) of the word's centred
unit vector over another language's centred unit pool vectors [1601.03764].  Card = first 5 ingredients with
their incremental explained share; leftover = 1 - R^2(5).  Per-context recipes give the stable core
(ingredients in >= 70% of contexts) and recipe variability (mean pairwise Jaccard distance of the k=5 sets).
Nulls: random same-language pool words and random unit vectors, decomposed the same way [2602.14486].
"""
import argparse, itertools

import numpy as np
from scipy.optimize import nnls

from ut.common import MODELS, Timer, load_json, load_model, out_dir, save_json
from ut.represent import cached_span_states, unmark
from ut.space import Targets, cognate_mask, pool
from ut.words import PHASE1, load_words

T1_CATS = ["untranslatable", "ordinary", "emotion_plain", "polysemous", "loanword",
           "english_untranslatable", "english_ordinary", "dialect_known_gloss", "dialect_untranslatable"]
POOL_LANGS = ["pt", "de", "it", "fr", "es", "en"]
K, K_CARD, CORE_FRAC = 10, 5, 0.7


def nnomp(D, x, k=K):
    """D [n, d] unit atoms, x [d] unit target.  Returns (indices, weights, R2 after each pick)."""
    r, S, R2, w = x.copy(), [], [], np.zeros(0)
    for _ in range(k):
        c = D @ r
        if S:
            c[S] = -np.inf
        j = int(np.argmax(c))
        if c[j] <= 1e-9:
            break
        S.append(j)
        w, _ = nnls(D[S].T.astype(np.float64), x.astype(np.float64))
        r = x - D[S].T @ w
        R2.append(float(1 - (r @ r) / (x @ x)))
    return S, w, R2


def card(words, S, R2, k=K_CARD):
    gains = np.diff([0.0] + R2[:k])
    return dict(ingredients=[dict(word=words[j], share=float(g)) for j, g in zip(S[:k], gains)],
                r2_curve=R2, leftover=float(1 - (R2[k - 1] if len(R2) >= k else (R2[-1] if R2 else 0.0))))


def jaccard_dist(a, b):
    a, b = set(a), set(b)
    return 1 - len(a & b) / max(1, len(a | b))


# ------------------------------------------------------------------ extraction
def extract(a):
    words = [w for w in load_words() if w.category in T1_CATS]
    items, index = [], []
    for w in words:
        for k, (kind, text) in enumerate(w.t1_contexts()):
            items.append(unmark(text))
            index.append([w.id, kind, k if kind == "template" else -1, w.language])
    tok, model = load_model(a.model)
    with Timer(f"extract {len(items)} contexts"):
        cached_span_states(model, tok, "words_t1", items, batch=64, extra_meta=dict(index=index))


# ------------------------------------------------------------------ analysis
_MASKS, _EXCLUDED = {}, {}       # exclusion masks depend on the pool word lists only, so they are shared across layers


class Decomposer:
    def __init__(self, model_key, layer):
        self.mk, self.layer, self.masks, self.excluded = model_key, layer, _MASKS, _EXCLUDED

    def pool(self, lang):
        return pool(self.mk, lang, self.layer)

    def mask(self, w, lang, keep_loan=False):
        key = (w.id, lang, keep_loan)
        if key not in self.masks:
            pw = self.pool(lang)[0]
            forms = [w.word] + [p for p in w.word.replace("'", " ").split() if len(p) >= 5]
            extra = list(w.exclude_from_pool) + ([w.loan_en] if (w.loan_en and not keep_loan) else [])
            m = cognate_mask(pw, forms, extra)
            if keep_loan and w.loan_en:
                m[[i for i, x in enumerate(pw) if x.lower() == w.loan_en.lower()]] = False
            self.masks[key] = m
            self.excluded[f"{w.id}->{lang}{'+loan' if keep_loan else ''}"] = [pw[i] for i in np.where(m)[0]]
        return self.masks[key]

    def run(self, x, w, lang, keep_loan=False):
        pw, P, _ = self.pool(lang)
        keep = ~self.mask(w, lang, keep_loan)
        idx = np.where(keep)[0]
        S, wts, R2 = nnomp(P[idx], x)
        return card([pw[i] for i in idx], S, R2)


def null_leftovers(dec, src_lang, tgt_lang, n, rng, exclude):
    pw, P, _ = dec.pool(src_lang)
    _, D, _ = dec.pool(tgt_lang)
    cand = [i for i, w in enumerate(pw) if w.lower() not in exclude]
    out = []
    for i in rng.choice(cand, n, replace=False):
        m = cognate_mask(dec.pool(tgt_lang)[0], [pw[i]])
        out.append(nnomp(D[~m], P[i])[2])
    return out                                   # R^2 curves; leftover = 1 - curve[K_CARD - 1]


def leftover_of(R2):
    return 1 - (R2[K_CARD - 1] if len(R2) >= K_CARD else (R2[-1] if R2 else 0.0))


def analyze(a):
    rd = out_dir(a.model)
    layer = a.layer if a.layer is not None else load_json(rd / "layer_choice.json")["concept_layer"]
    words = [w for w in load_words() if w.category in T1_CATS]
    T = Targets(a.model)
    dec = Decomposer(a.model, layer)
    rng = np.random.default_rng(a.seed)
    V = T.at(layer)
    res = dict(layer=layer, k=K, k_card=K_CARD, core_frac=CORE_FRAC, words={}, nulls={}, pivots={},
               layer_dependence={}, loanword={})
    with Timer("recipes"):
        for w in words:
            kinds, C = V[w.id]
            x = T.mean_vector(layer, w.id)
            dirs = ["en"] if w.language != "en" else [l for l in POOL_LANGS if l != "en"]
            entry = dict(word=w.word, language=w.language, category=w.category, gloss_en=w.gloss_en, directions={})
            for tl in dirs:
                main = dec.run(x, w, tl)
                per = [dec.run(c, w, tl) for c in C]
                sets = [[i["word"] for i in p["ingredients"]] for p in per]
                cnt = {}
                for s in sets:
                    for u in s:
                        cnt[u] = cnt.get(u, 0) + 1
                core = sorted([u for u, n in cnt.items() if n >= CORE_FRAC * len(sets)], key=lambda u: -cnt[u])
                var = float(np.mean([jaccard_dist(p, q) for p, q in itertools.combinations(sets, 2)])) if len(sets) > 1 else 0.0
                entry["directions"][tl] = dict(main, contexts=[dict(kind=kd, ingredients=s, leftover=p["leftover"])
                                                               for kd, s, p in zip(kinds, sets, per)],
                                               stable_core=core, core_counts=cnt, variability=var)
            res["words"][w.id] = entry
    # nulls per direction (same-language random words; random unit vectors)
    targets_lower = {w.word.lower() for w in words}
    res["null_curves"] = {}
    for sl in PHASE1:
        res["null_curves"][f"{sl}->en"] = null_leftovers(dec, sl, "en", a.n_null, rng, targets_lower)
    for tl in PHASE1:
        res["null_curves"][f"en->{tl}"] = null_leftovers(dec, "en", tl, a.n_null, rng, targets_lower)
    d = dec.pool("en")[1].shape[1]
    R = rng.standard_normal((a.n_null, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    res["null_curves"]["random_unit->en"] = [nnomp(dec.pool("en")[1], r)[2] for r in R]
    res["nulls"] = {k: [leftover_of(c) for c in v] for k, v in res["null_curves"].items()}
    for wid, e in res["words"].items():
        for tl, dd in e["directions"].items():
            key = f"{e['language']}->{tl}" if e["language"] in PHASE1 or e["language"] == "en" else None
            if key in res["nulls"]:
                dd["null_quantile"] = float(np.mean(np.array(res["nulls"][key]) < dd["leftover"]))
    # within-direction normalisation: leftover minus median ordinary leftover, same direction, same Zipf band
    from wordfreq import zipf_frequency
    zipf = {w.id: zipf_frequency(w.word, w.language) if w.language in POOL_LANGS else None for w in words}
    for wid, e in res["words"].items():
        for tl, dd in e["directions"].items():
            ords = [(res["words"][o]["directions"][tl]["leftover"], zipf[o]) for o in res["words"]
                    if res["words"][o]["category"] in ("ordinary", "english_ordinary") and tl in res["words"][o]["directions"]]
            z = zipf[wid]
            band = [lv for lv, oz in ords if z is not None and oz is not None and abs(oz - z) <= 0.5]
            ref = band if len(band) >= 3 else [lv for lv, _ in ords]
            dd["score_vs_ordinary"] = float(dd["leftover"] - np.median(ref)) if ref else None
            dd["score_reference_n"] = len(ref)
    # pivots (§6.5): untranslatables and their controls into every other pool
    for w in words:
        if w.category in ("untranslatable", "ordinary", "emotion_plain") and w.language in PHASE1:
            x = T.mean_vector(layer, w.id)
            res["pivots"][w.id] = {tl: dec.run(x, w, tl)["leftover"] for tl in POOL_LANGS if tl != w.language}
    # layer dependence (§6.5): leftover into English at every layer (mean vector)
    n_layers = T.H.shape[1]
    sel = [w for w in words if w.category in ("untranslatable", "ordinary", "emotion_plain") and w.language in PHASE1]
    with Timer("layer dependence"):
        for layer_l in range(1, n_layers):
            dl = Decomposer(a.model, layer_l)
            for w in sel:
                x = T.mean_vector(layer_l, w.id)
                res["layer_dependence"].setdefault(w.id, []).append(dl.run(x, w, "en")["leftover"])
            pool.cache_clear()
    # loanword sub-test: with vs without the English loan form in the pool
    for w in words:
        if w.category == "loanword" and w.loan_en:
            x = T.mean_vector(layer, w.id)
            in_pool = any(p.lower() == w.loan_en.lower() for p in dec.pool("en")[0])
            res["loanword"][w.id] = dict(loan=w.loan_en, loan_in_pool=in_pool,
                                         without=dec.run(x, w, "en"), with_loan=dec.run(x, w, "en", keep_loan=True))
    res["exclusions"] = dec.excluded
    res["centring"] = ("templates: per-template pool centroid; natural/frame: mean of all targets' natural+frame "
                       "contexts in that language; rom uses Italian centroids")
    save_json(rd / "recipes.json", res)
    for wid in [w.id for w in words if w.category in ("untranslatable", "emotion_plain")][:6]:
        e = res["words"][wid]["directions"]["en"]
        print(wid, round(e["leftover"], 3), [i["word"] for i in e["ingredients"]], "core", e["stable_core"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.extract:
        extract(a)
    if a.analyze:
        analyze(a)
