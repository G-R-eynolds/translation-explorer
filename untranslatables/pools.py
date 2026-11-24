"""Vocabulary pools (handoff §5.2) and dictionary pairs for layer choice (§5.3).

    python pools.py --build                    # CPU, model-independent: data/pools/<lang>.json, data/pools/muse_pairs.json
    python pools.py --casing --model 3b        # GPU, once: German noun capitalisation + proper-noun removal
    python pools.py --extract --model 3b       # GPU: template-averaged states for every pool word, all layers

Pools: content words from wordfreq (stopwords, non-alphabetic strings, words under 4 letters, and strings at
least as frequent in another of the six languages removed),
Zipf 2.5-6.0, frequency-matched across pt/de/it/fr/es/en by sampling the same number of words from every
0.25-Zipf bin (the minimum over languages), then scaled to --size words each.  wordfreq lower-cases
everything, so the --casing pass picks German noun capitalisation, and drops likely proper nouns, by
comparing model log-probs of the two casings mid-sentence (decided once, with the 3B; recorded).

Cache layout (cache/<model>/):  pool_<lang>.npy  [n_words, L+1, d] float16, the mean over the 4 templates;
pool_<lang>_tcentroid.npy  [4, L+1, d] float32, the per-template language centroid.
"""
import argparse, json, re, urllib.request
from collections import Counter

import numpy as np

from ut.common import DATA, MODELS, Progress, Timer, cache_dir, load_json, load_model, load_yaml, save_json
from ut.represent import span_states, unmark

LANGS = ["pt", "de", "it", "fr", "es", "en"]
MUSE = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{a}-{b}.txt"   # Conneau et al. 2018, CC-BY-NC 4.0
ALPHA = re.compile(r"^[^\W\d_]+$")
VOWEL = re.compile(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüæœ]")
REPEAT = re.compile(r"(.)\1\1")


def plausible(w):
    return bool(ALPHA.match(w)) and len(w) >= 4 and bool(VOWEL.search(w)) and not REPEAT.search(w)


def templates(lang):
    return load_yaml(DATA / "words" / f"{lang}.yaml")["templates"]


def build(a):
    from wordfreq import top_n_list, zipf_frequency
    import stopwordsiso
    rng = np.random.default_rng(a.seed)
    cand = {}
    for l in LANGS:
        sw = stopwordsiso.stopwords(l)
        ws = [w for w in top_n_list(l, 80000) if plausible(w) and w not in sw]
        cand[l] = {w: zipf_frequency(w, l) for w in ws}
        cand[l] = {w: z for w, z in cand[l].items() if a.zmin <= z <= a.zmax}
        # drop strings at least as frequent in another pool language (English in German text, shared names)
        before = len(cand[l])
        cand[l] = {w: z for w, z in cand[l].items() if all(zipf_frequency(w, o) < z for o in LANGS if o != l)}
        print(l, len(cand[l]), f"candidates ({before - len(cand[l])} more frequent in another language dropped)")
    pools = match_bins(cand, a.size, rng)
    for l in LANGS:
        save_json(DATA / "pools" / f"{l}.json", dict(
            language=l, n=len(pools[l]), zipf_range=[a.zmin, a.zmax], seed=a.seed, casing="lowercase (wordfreq)",
            words=[dict(w=w, zipf=round(cand[l][w], 3)) for w in pools[l]]))
        print(l, len(pools[l]), "pool words")
    # MUSE pairs en<->xx for layer choice: both words in their pools, not identical / near-identical strings
    pairs = {}
    for l in LANGS[:-1]:
        url = MUSE.format(a=l, b="en")
        txt = urllib.request.urlopen(url, timeout=60).read().decode()
        ps, seen = [], set()
        en, src = set(pools["en"]), set(pools[l])
        for line in txt.splitlines():
            x, y = line.split()[:2]
            if x in src and y in en and x not in seen and not _near(x, y):
                ps.append([x, y]); seen.add(x)
        sel = [ps[i] for i in sorted(rng.choice(len(ps), min(a.pairs_per_lang, len(ps)), replace=False))]
        pairs[l] = sel
        print(l, "MUSE pairs in pools:", len(ps), "-> using", len(sel))
    save_json(DATA / "pools" / "muse_pairs.json", dict(source=MUSE.format(a="xx", b="en"), licence="CC-BY-NC 4.0",
                                                      rule="both words in the pools; near-identical strings removed",
                                                      pairs=pairs))


def match_bins(cand, size, rng):
    """cand[lang] = {word: zipf}.  Same number of words from every 0.25-Zipf bin in every language (the
    minimum over languages), scaled down to `size` in total.  Returns {lang: sorted word list}."""
    binof = lambda z: int(z / 0.25)
    bins = {l: Counter(binof(z) for z in cand[l].values()) for l in cand}
    take = {b: min(bins[l][b] for l in cand) for b in set().union(*bins.values())}
    scale = min(1.0, size / max(1, sum(take.values())))
    take = {b: int(round(n * scale)) for b, n in take.items()}
    out = {}
    for l in cand:
        by = {}
        for w, z in cand[l].items():
            by.setdefault(binof(z), []).append(w)
        chosen = []
        for b, n in sorted(take.items()):
            ws = sorted(by.get(b, []))
            chosen += [ws[i] for i in rng.choice(len(ws), n, replace=False)] if n else []
        out[l] = sorted(chosen)
    return out


def muse_all(a):
    """Raw MUSE dictionaries for every ordered pair of the six languages (first 5000 entries each), used for
    few-shot examples (patching.py) and triangulated latent-language items (latent_lang.py)."""
    out = DATA / "pools" / "muse"
    out.mkdir(exist_ok=True)
    for x in LANGS:
        for y in LANGS:
            if x == y or (out / f"{x}-{y}.json").exists():
                continue
            txt = urllib.request.urlopen(MUSE.format(a=x, b=y), timeout=60).read().decode()
            pairs = [l.split()[:2] for l in txt.splitlines() if len(l.split()) >= 2][:5000]
            save_json(out / f"{x}-{y}.json", dict(source=MUSE.format(a=x, b=y), licence="CC-BY-NC 4.0", pairs=pairs))
            print(x, y, len(pairs))


def muse(x, y):
    return load_json(DATA / "pools" / "muse" / f"{x}-{y}.json")["pairs"]


def _near(x, y):
    import difflib
    return difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio() >= 0.8


def casing(a):
    """Choose German capitalisation; drop likely proper nouns and acronyms in every language (Capitalised or
    ALL-CAPS form > 3 nats more probable mid-sentence than the lower-case form; for German, ALL-CAPS only,
    and the better of lower/Capitalised is kept).  Writes the decision into data/pools/<lang>.json."""
    from ut.lm import token_logprobs
    tok, model = load_model(a.model)
    frames = {"de": "Gestern habe ich über {} gelesen.", "pt": "Ontem li sobre {} no jornal.",
              "it": "Ieri ho letto qualcosa su {} nel giornale.", "fr": "Hier, j'ai lu un article sur {}.",
              "es": "Ayer leí algo sobre {} en el periódico.", "en": "Yesterday I read about {} in the paper."}
    kept, info = {}, {}
    for l in LANGS:
        p = load_json(DATA / "pools" / f"{l}.json")
        ws = [e["w"] for e in p["words"]]
        texts = [frames[l].format(v) for w in ws for v in (w, w[:1].upper() + w[1:], w.upper())]
        with Timer(f"casing {l}"):
            res = token_logprobs(model, tok, texts, batch=128)
        tot = np.array([lp.sum() for _, lp in res]).reshape(-1, 3)
        keep, dropped = [], []
        for e, (low, cap, caps) in zip(p["words"], tot):
            if l == "de":
                best = max(low, cap)
                if caps - best > 3.0:
                    dropped.append(e["w"]); continue
                e["w"] = e["w"][:1].upper() + e["w"][1:] if cap > low else e["w"]
                keep.append(e)
            elif max(cap, caps) - low > 3.0:
                dropped.append(e["w"])
            else:
                keep.append(e)
        kept[l] = {e["w"]: e["zipf"] for e in keep}
        info[l] = dict(p, dropped_proper_nouns=dropped,
                       casing=f"decided by {MODELS[a.model]['hf_id']} log-prob in '{frames[l]}'")
        print(l, "kept", len(keep), "dropped", len(dropped), dropped[:15])
    # re-equalise pool sizes and frequency profiles after the drops
    pools = match_bins(kept, min(len(v) for v in kept.values()), np.random.default_rng(a.seed))
    for l in LANGS:
        p = info[l]
        p.update(words=[dict(w=w, zipf=kept[l][w]) for w in pools[l]], n=len(pools[l]))
        save_json(DATA / "pools" / f"{l}.json", p)
    # MUSE pairs whose words fell out of a pool (or changed case) are refreshed from the kept pools
    mp = load_json(DATA / "pools" / "muse_pairs.json")
    en = {w.lower(): w for w in pools["en"]}
    for l, ps in mp["pairs"].items():
        src = {w.lower(): w for w in pools[l]}
        mp["pairs"][l] = [[src[x.lower()], en[y.lower()]] for x, y in ps if x.lower() in src and y.lower() in en]
    save_json(DATA / "pools" / "muse_pairs.json", mp)


def pool_words(lang):
    return [e["w"] for e in load_json(DATA / "pools" / f"{lang}.json")["words"]]


def extra_en_words():
    """Loan forms (loanword sub-test) and English cognates named in words.yaml that must exist in the
    English pool even if wordfreq sampling missed them."""
    from ut.words import load_words
    return sorted({w.loan_en for w in load_words() if w.loan_en})


def extract(a):
    tok, model = load_model(a.model)
    cd = cache_dir(a.model)
    from ut.arch import hidden_size, n_layers
    L1, d = n_layers(model) + 1, hidden_size(model)
    for l in LANGS:
        base = pool_words(l)[: a.limit] if a.limit else pool_words(l)      # --limit: smoke tests only
        ws = base + ([w for w in extra_en_words() if w not in set(base)] if l == "en" else [])
        tps = templates(l)
        npy = cd / f"pool_{l}.npy"
        meta = cd / f"pool_{l}.json"
        if meta.exists() and load_json(meta).get("words") == ws:
            print(l, "cached"); continue
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(len(ws), L1, d))
        tsum = np.zeros((len(tps), L1, d), dtype=np.float64)
        pr = Progress(f"pool {l}", len(ws))
        for c in range(0, len(ws), a.chunk):
            chunk = ws[c:c + a.chunk]
            items = [unmark(t.format(w=f"[[{w}]]")) for t in tps for w in chunk]
            H = span_states(model, tok, items, batch=a.batch).astype(np.float32).reshape(len(tps), len(chunk), L1, d)
            tsum += H.sum(1)
            mm[c:c + len(chunk)] = H.mean(0).astype(np.float16)
            pr.tick(c + len(chunk))
        mm.flush()
        np.save(cd / f"pool_{l}_tcentroid.npy", (tsum / len(ws)).astype(np.float32))
        json.dump(dict(words=ws, templates=tps, extra_en=extra_en_words() if l == "en" else []),
                  open(meta, "w"), ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--casing", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--size", type=int, default=8000)
    ap.add_argument("--zmin", type=float, default=2.5)
    ap.add_argument("--zmax", type=float, default=6.0)
    ap.add_argument("--pairs-per-lang", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="first N pool words only (smoke tests)")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    if a.build:
        build(a)
        muse_all(a)
    if a.casing:
        casing(a)
    if a.extract:
        extract(a)
