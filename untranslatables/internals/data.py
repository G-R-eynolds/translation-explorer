"""Corpora for N1 (HANDOFF_internals §3.3).  CPU + network.

  es-en  25k Europarl (Helsinki-NLP/europarl en-es) + 25k OPUS-100 (Helsinki-NLP/opus-100 en-es)
  hi-en  25k OPUS-100 en-hi + 25k Samanantar (ai4bharat/samanantar, hi)
Both sides 8-40 words (sent.data.WORD), deduplicated on the source side, shuffled with a fixed seed; the first 5,000
sentence pairs are the held-out TEST split, the rest FIT.  Two domains on purpose: parliament alone would make
"conventional translation practice" mean parliamentary register.  The domain of every pair is recorded.
FLORES+ dev+devtest (openlanguagedata/flores_plus): spa_Latn, eng_Latn, hin_Deva, bho_Deva, joined on id.
"""
import json
import random

from internals.common import CORPUS, stage
from sent.data import WORD

SOURCES = {
    "es-en": [("europarl", "Helsinki-NLP/europarl", "en-es", "es"), ("opus100", "Helsinki-NLP/opus-100", "en-es", "es")],
    "hi-en": [("opus100", "Helsinki-NLP/opus-100", "en-hi", "hi"), ("samanantar", "ai4bharat/samanantar", "hi", "hi")],
}
N_PER_SOURCE, N_TEST, MIN_W, MAX_W, SEED = 25000, 5000, 8, 40, 0


def nwords(s):
    return len(WORD.findall(s))


def _pairs(repo, config, lang):
    """Yield (x, en) from a streaming dataset, whatever its schema."""
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train", streaming=True).shuffle(seed=SEED, buffer_size=200_000)
    for r in ds:
        if "translation" in r:
            yield r["translation"][lang], r["translation"]["en"]
        elif "src" in r and "tgt" in r:                         # samanantar: src = English, tgt = Indic
            yield r["tgt"], r["src"]
        else:
            raise KeyError(f"unknown schema {list(r)}")


def clean_pair(x, y, lang):
    """Drop software strings and subtitle debris (OPUS-100 has both) and pairs whose source is not in its script."""
    if any(c in x + y for c in "~{}<>|\\_#@") or x.startswith(("-", "\"")) or y.startswith("-"):
        return False
    letters = [c for c in x if c.isalpha()]
    if lang == "hi":
        return len(letters) > 0 and sum("\u0900" <= c <= "\u097f" for c in letters) / len(letters) > 0.9
    return len(letters) > 0 and sum(c.isascii() or c in "áéíóúüñÁÉÍÓÚÜÑ" for c in letters) / len(letters) > 0.95


def read_corpus(pair):
    return [json.loads(l) for l in open(CORPUS / f"{pair}.jsonl")]


@stage("corpus")
def build_corpus(cfg):
    pair = cfg["pair"]
    CORPUS.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for domain, repo, config, lang in SOURCES[pair]:
        k = 0
        for x, y in _pairs(repo, config, lang):
            x, y = x.strip(), y.strip()
            if not (MIN_W <= nwords(x) <= MAX_W and MIN_W <= nwords(y) <= MAX_W) or x in seen or x == y \
                    or not clean_pair(x, y, lang):
                continue
            seen.add(x)
            rows.append(dict(x=x, y=y, domain=domain))
            k += 1
            if k >= cfg.get("n_per_source", N_PER_SOURCE):
                break
        print(f"  {pair} {domain}: {k}", flush=True)
    random.Random(SEED).shuffle(rows)
    n_test = cfg.get("n_test", N_TEST)
    for i, r in enumerate(rows):
        r["id"] = f"{pair}:{i}"
        r["split"] = "test" if i < n_test else "fit"
    p = CORPUS / f"{pair}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [p]


FLORES_LANGS = {"es": "spa_Latn", "en": "eng_Latn", "hi": "hin_Deva", "bho": "bho_Deva"}


def read_flores():
    return [json.loads(l) for l in open(CORPUS / "flores.jsonl")]


@stage("flores")
def build_flores(cfg):
    from huggingface_hub import hf_hub_download
    CORPUS.mkdir(parents=True, exist_ok=True)
    by = {}
    for split in ("dev", "devtest"):
        for k, code in FLORES_LANGS.items():
            f = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", repo_type="dataset")
            for l in open(f):
                r = json.loads(l)
                by.setdefault((split, r["id"]), dict(id=f"flores:{split}:{r['id']}", split=split))[k] = r["text"]
    rows = [r for r in by.values() if all(k in r for k in FLORES_LANGS)]
    p = CORPUS / "flores.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  flores: {len(rows)} multiway sentences", flush=True)
    return [p]
