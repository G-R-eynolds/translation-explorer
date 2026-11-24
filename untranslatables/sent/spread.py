"""Step 3a (GPU): spread H_S per annotated span (handoff §1, "the second axis").

20 standard (B1) translations per sentence at T=1; for each hot span (and each control's probe span) the model
is asked which words of each translation render the span (greedy; checked to be a substring of the
translation); the renderings are clustered by meaning with bidirectional NLI (ut.nli, as in thread 2) and
H_S is the semantic entropy over the clusters, in bits, computed over renderings that are verbatim substrings of
their translation or an explicit "(none)" (unaligned extractor answers are kept in the output but not counted).

    python -m sent.spread --model 3b        # -> results/sent/3b/spread.json
"""
import argparse

import numpy as np

from ut.common import MODELS, Timer, chat, describe, generate, load_model, save_json
from ut.nli import NLI, cluster, normalise, semantic_entropy
from sent.data import load_sentences, sent_dir
from sent.prompts import EXTRACT, clean, extract_prompt, rung_prompt


MIN_ALIGNED = 10        # H is reported only when at least this many renderings are usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dialect", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    sents = load_sentences(dialect=a.dialect, only=a.only)
    tok, model = load_model(a.model)
    with Timer("B1 samples"):
        tr = generate(model, tok, [chat(tok, rung_prompt("B1", s)) for s in sents], n=a.n, max_new_tokens=80,
                      temperature=a.temperature, batch=a.batch, seed=a.seed)
    tr = [[clean(t, s.text) for t in ts] for s, ts in zip(sents, tr)]
    jobs = [(i, x, k) for i, s in enumerate(sents) for x in s.spans() for k in range(a.n)]
    with Timer(f"span extraction ({len(jobs)} prompts)"):
        ext = generate(model, tok, [chat(tok, extract_prompt(sents[i], tr[i][k], x["text"])) for i, x, k in jobs],
                       n=1, max_new_tokens=20, temperature=0, batch=a.batch * 2, stop_at_newline=True)
    del model
    nli = NLI()
    out = []
    by = {}
    for (i, x, k), e in zip(jobs, ext):
        by.setdefault((i, x["start"], x["end"]), (x, []))[1].append((k, clean(e[0])))
    with Timer("NLI clustering"):
        for (i, _, _), (x, lst) in by.items():
            s = sents[i]
            rend, aligned = [], []
            for k, e in lst:
                none = normalise(e) in ("(none)", "none", "")
                ok = not none and normalise(e) in normalise(tr[i][k])
                rend.append("" if none else e); aligned.append(ok)
            # H over the renderings that really occur in their translation (or are explicitly "(none)", i.e. the
            # span was left out); unaligned answers are the extractor's errors, kept but not counted
            use = [r for r, ok in zip(rend, aligned) if ok or r == ""]
            cl = cluster(nli, use, x["text"], s.lang) if use else dict(counts={}, reps={}, labels=[])
            H = semantic_entropy(cl["counts"]) if len(use) >= MIN_ALIGNED else None
            n = max(len(use), 1)
            rend_used = use
            clusters = sorted(([dict(rep=cl["reps"].get(c, "(nothing)"), share=v / n,
                                     members=sorted({r for r, l in zip(rend_used, cl["labels"]) if l == c}))
                                for c, v in cl["counts"].items()]), key=lambda d: -d["share"])
            out.append(dict(sid=s.id, lang=s.lang, category=s.category, text=s.text, span=x, H_bits=H,
                            n_used=len(use), n_clusters=len(clusters), clusters=clusters, renderings=rend, aligned=aligned,
                            aligned_rate=float(np.mean(aligned)), none_rate=float(np.mean([r == "" for r in rend])),
                            translations=[tr[i][k] for k, _ in lst]))
    d = sent_dir(a.model)
    save_json(d / "spread.json", dict(model=describe(a.model), n=a.n, temperature=a.temperature,
                                      extract_prompt=EXTRACT,
                                      spans=out))
    for r in out:
        H = "  n/a" if r["H_bits"] is None else f"{r['H_bits']:.2f}"
        top = f"{r['clusters'][0]['rep']!r} {r['clusters'][0]['share']:.0%}" if r["clusters"] else "-"
        print(f"  {r['sid']:24s} {r['span']['text']:24s} H={H} bits  clusters={r['n_clusters']}"
              f"  aligned={r['aligned_rate']:.0%}  top: {top}")
    print("wrote", d / "spread.json")


if __name__ == "__main__":
    main()
