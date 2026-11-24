"""Intended-sense recovery (HANDOFF_internals §2 R2.3), GPU.

For every hot span (and every control's probe span) with an English `senses.en` gloss, and every candidate of the
rungs B1, B3, B4, B5, B6 (and B2_borrow, reported separately):
    sense recovery = mean log p(gloss | t) - mean log p(gloss | s)      (per gloss token, teacher-forced)
where the conditioning text is the user turn  "<context>\\n\\nIn the original text, the word for '<span>' meant:"
and the gloss is the assistant reply.  t is the candidate with source words and close cognates masked (the default)
and, as a secondary, unmasked; s is the source sentence itself.  Near 0: t carries the intended sense as well as the
source does; negative: it does not (the "elf" problem).

    python -m sent.sense --model gemma12b       # after sent.score; -> <sent_dir>/sense.json
"""
import argparse
from collections import defaultdict

import numpy as np

from ut.common import MODELS, Timer, chat, describe, load_model, save_json
from sent.data import load_sentences, sent_dir
from sent.ladder import read_jsonl
from sent.score import rt_logprobs

RUNGS = ("B1", "B2_borrow", "B3", "B4", "B5", "B6")
ASK = "{context}\n\nIn the original text, the word for '{span}' meant:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=MODELS)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    d = sent_dir(a.model)
    cands = [c for c in read_jsonl(d / "candidates.jsonl") if c["rung"] in RUNGS]
    rows = {r["rid"]: r for r in read_jsonl(d / "raw" / "rows.jsonl")}
    masked = {r["parent"]: r["text"] for r in rows.values() if r["role"] == "masked"}
    sids = {c["sid"] for c in cands}
    sents = [s for s in load_sentences() if s.id in sids]
    tok, model = load_model(a.model)
    gl = lambda g: tok(g, add_special_tokens=False)["input_ids"]           # the gloss as the reply's tokens
    pre = lambda u: tok(chat(tok, u), add_special_tokens=False)["input_ids"]
    jobs, keys = [], []
    by_sid = defaultdict(list)
    for c in cands:
        by_sid[c["sid"]].append(c)
    for s in sents:
        for x in s.spans():
            if not x["sense_en"]:
                continue
            g = gl(x["sense_en"])
            key = (s.id, x["start"], x["end"])
            jobs.append((pre(ASK.format(context=s.text, span=x["text"])), g)); keys.append((key, "source", None))
            for c in by_sid[s.id]:
                jobs.append((pre(ASK.format(context=masked.get(c["cid"], c["text"]), span=x["text"])), g))
                keys.append((key, "masked", c["cid"]))
                if c["cid"] in masked:
                    jobs.append((pre(ASK.format(context=c["text"], span=x["text"])), g))
                    keys.append((key, "unmasked", c["cid"]))
    print(f"{len(jobs)} sense-recovery forward passes", flush=True)
    with Timer("sense recovery"):
        lps = rt_logprobs(model, tok, jobs, batch=a.batch, label="sense")
    acc = defaultdict(lambda: dict(cands={}))
    for (key, kind, cid), lp in zip(keys, lps):
        e = acc[key]
        m = float(np.mean(lp))
        if kind == "source":
            e["lp_source"] = m
        else:
            e["cands"].setdefault(cid, {})[kind] = m
    out = []
    by = {s.id: s for s in sents}
    for (sid, st, en), e in acc.items():
        x = next(z for z in by[sid].spans() if (z["start"], z["end"]) == (st, en))
        cc = {}
        for cid, v in e["cands"].items():
            rec = {k: v[k] - e["lp_source"] for k in v}
            rec.setdefault("unmasked", rec["masked"])
            cc[cid] = rec
        per_rung = {}
        for rung in RUNGS:
            vals = [cc[c["cid"]]["masked"] for c in by_sid[sid] if c["rung"] == rung and c["cid"] in cc]
            g = cc.get(f"{sid}|{rung}|-1")
            if vals:
                per_rung[rung] = dict(greedy=g["masked"] if g else None, best=float(max(vals)), mean=float(np.mean(vals)),
                                      greedy_unmasked=g["unmasked"] if g else None)
        out.append(dict(sid=sid, lang=by[sid].lang, category=by[sid].category, span=x, lp_source=e["lp_source"],
                        rungs=per_rung, candidates=cc))
    save_json(d / "sense.json", dict(model=describe(a.model), ask=ASK, rungs=RUNGS,
                                     definition="mean log p(gloss | masked t) - mean log p(gloss | s)", spans=out))
    for r in out:
        b1 = r["rungs"].get("B1", {}).get("greedy")
        b6 = r["rungs"].get("B6", {}).get("best")
        print(f"  {r['sid']:24s} {r['span']['text']:24s} B1 {b1 if b1 is None else round(b1, 3)}  "
              f"B6 best {b6 if b6 is None else round(b6, 3)}")
    print("wrote", d / "sense.json")


if __name__ == "__main__":
    main()
