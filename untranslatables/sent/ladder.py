"""Step 1: the budget ladder (handoff §3).  For every sentence and rung B0-B7: one greedy candidate plus
--n samples at T=0.8.  Every prompt, raw output, cleaned text and length is kept.

    python -m sent.ladder --model 3b            # -> results/sent/3b/candidates.jsonl
    python -m sent.ladder --model 3b --rungs B0 B1 B2 B3 B4 B5 B6     # drop B7 if time is short (§9)
"""
import argparse
import json

from ut.common import MODELS, Timer, chat, describe, generate, load_model, save_json
from sent.data import LENGTH_TOKENIZER, load_sentences, sent_dir
from sent.prompts import RUNG_ORDER, RUNGS, clean, rung_prompt, strip_source

_LT = None


def length_tokenizer():
    global _LT
    if _LT is None:
        from transformers import AutoTokenizer
        _LT = AutoTokenizer.from_pretrained(LENGTH_TOKENIZER)
    return _LT


def n_tokens(text):
    return len(length_tokenizer()(text, add_special_tokens=False)["input_ids"])


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--rungs", nargs="*", default=RUNG_ORDER)
    ap.add_argument("--n", type=int, default=5, help="samples per rung at --temperature, plus one greedy")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dialect", action="store_true", help="include the optional Romanesco items")
    ap.add_argument("--only", nargs="*", default=None, help="sentence ids (smoke tests)")
    ap.add_argument("--greedy-only", nargs="*", default=[], help="rungs that get the greedy candidate only (R2: B7)")
    ap.add_argument("--langs", nargs="*", default=None, help="only sentences in these languages")
    a = ap.parse_args()
    sents = load_sentences(dialect=a.dialect, only=a.only, langs=a.langs)
    n_tokens("warm-up")                                       # fail early if the length tokenizer is unavailable
    tok, model = load_model(a.model)
    rows = []
    print(f"{len(sents)} sentences x {len(a.rungs)} rungs x (1 greedy + {a.n} samples)", flush=True)
    for rung in a.rungs:
        users = [rung_prompt(rung, s) for s in sents]
        prompts = [chat(tok, u) for u in users]
        mx = RUNGS[rung]["max_new_tokens"]
        with Timer(f"{rung} ({RUNGS[rung]['practice']})"):
            greedy = generate(model, tok, prompts, n=1, max_new_tokens=mx, temperature=0, batch=a.batch, seed=a.seed)
            n = 0 if rung in a.greedy_only else a.n
            sampled = generate(model, tok, prompts, n=n, max_new_tokens=mx, temperature=a.temperature,
                               batch=a.batch, seed=a.seed + 1 + RUNG_ORDER.index(rung)) if n else [[]] * len(sents)
        for s, u, g, smp in zip(sents, users, greedy, sampled):
            for k, raw in [(-1, g[0])] + list(enumerate(smp)):
                t = clean(raw, s.text)
                rows.append(dict(cid=f"{s.id}|{rung}|{k}", sid=s.id, lang=s.lang, category=s.category, rung=rung,
                                 practice=RUNGS[rung]["practice"], sample=k,
                                 temperature=0.0 if k < 0 else a.temperature, prompt=u, raw=raw, text=t,
                                 quoted_source=strip_source(raw, s.text)[1] > 0,
                                 len=n_tokens(t),
                                 truncated=len(tok(raw, add_special_tokens=False)["input_ids"]) >= mx - 2))
        g = [r for r in rows if r["rung"] == rung and r["sample"] < 0]
        print(f"  e.g. {g[0]['sid']}: {g[0]['text'][:160]!r}", flush=True)
    d = sent_dir(a.model)
    write_jsonl(d / "candidates.jsonl", rows)
    save_json(d / "ladder_meta.json", dict(model=describe(a.model), rungs={r: RUNGS[r] for r in a.rungs}, n=a.n,
                                           greedy_only=a.greedy_only,
                                           temperature=a.temperature, seed=a.seed, length_tokenizer=LENGTH_TOKENIZER,
                                           sentences=[s.record() for s in sents]))
    trunc = sum(r["truncated"] for r in rows)
    print(f"wrote {len(rows)} candidates ({trunc} hit max_new_tokens) -> {d / 'candidates.jsonl'}")


if __name__ == "__main__":
    main()
