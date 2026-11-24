"""Step 3b (GPU, after sent.analyze): the targeting test (handoff §3), i.e. whether commentary helps most where
the heat map says.

For every non-control sentence, start from the greedy B1 translation and append short bracketed glosses for k
source words.  The model writes each gloss on its own (<= 8 words, greedy; source-language words masked out of it,
as in sent.score), and the code appends them as "<translation> [gloss 1; gloss 2]", identically in both arms
(asking a small model to rewrite the translation with inline glosses was unreliable: it often added none).
  targeted  the k content words with the most leftover meaning (weight x leftover) in the B1 heat map
  random    k other content words, drawn at random (--draws draws per sentence)
k = min(2, number of content words - 1).  The two arms differ only in WHERE the glosses go; their lengths are
reported so a length difference can be ruled out.  Every variant is scored in both spaces like a candidate.

    python -m sent.targeting --model 3b      # -> results/sent/3b/targeting.json
"""
import argparse

import numpy as np

from ut.common import MODELS, Timer, chat, describe, generate, load_json, load_model, save_json
from ut.stats import boot_ci
from sent.data import CENTROID_LANG, load_sentences, sent_dir
from sent.ladder import n_tokens
from sent.measure import cl_token, rt_token, weighted
from sent.prompts import GLOSS, clean, gloss_prompt
from sent.score import centroids, mask_source_words, run_cl, run_rt


MAX_GLOSS_WORDS = 8      # hard cap, so the two arms stay length-matched whatever the model writes


def pick(heat, content, k, draws, rng):
    """Targeted = top-k content words by weight x leftover; random = `draws` sets of k from the rest."""
    cw = {(a, b) for a, b, _ in content}
    cand = [h for h in heat if (h["start"], h["end"]) in cw]
    mass = lambda h: (h["weight"] or 0) * (h["leftover"] or 0)
    ranked = sorted(cand, key=mass, reverse=True)
    top = ranked[:k]
    rest = [h for h in cand if h not in top]
    rand = [list(rng.choice(len(rest), size=k, replace=False)) for _ in range(draws)] if len(rest) >= k else []
    return [h["word"] for h in top], [[rest[j]["word"] for j in r] for r in rand]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--space", default="rt", choices=["rt", "cl"], help="which heat map chooses the targets")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    d = sent_dir(a.model)
    curves = load_json(d / "curves.json")
    layer, tau = curves["layer"], curves["tau"]
    sents = [s for s in load_sentences(dialect=True) if s.id in curves["sentences"] and not s.is_control]
    rng = np.random.default_rng(a.seed)
    plan = []
    for s in sents:
        C = curves["sentences"][s.id]
        cw = s.content_words()
        k = min(2, len(cw) - 1)
        if k < 1:
            continue
        top, rand = pick(C["spaces"][a.space]["heat"], cw, k, a.draws, rng)
        if not rand:
            continue
        plan.append(dict(sid=s.id, base=C["text_B1"], targeted=top, random=rand))
    tok, model = load_model(a.model)
    by = {s.id: s for s in sents}
    base = {p["sid"]: p["base"] for p in plan}
    need = sorted({(p["sid"], w) for p in plan for w in p["targeted"] + [x for r in p["random"] for x in r]})
    with Timer(f"glosses ({len(need)})"):
        outs = generate(model, tok, [chat(tok, gloss_prompt(by[sid], w)) for sid, w in need], n=1, max_new_tokens=24,
                        temperature=0, batch=a.batch, stop_at_newline=True)
    gloss = {k: " ".join(mask_source_words(clean(o[0], by[k[0]].text), by[k[0]]).split()[:MAX_GLOSS_WORDS]).rstrip(".")
             for k, o in zip(need, outs)}

    def variant(sid, words):
        return base[sid].rstrip() + " [" + "; ".join(gloss[(sid, w)] for w in words) + "]"
    rows = [dict(rid=f"{p['sid']}|targeted|0", sid=p["sid"], role="targeted", parent=None,
                 text=variant(p["sid"], p["targeted"])) for p in plan]
    rows += [dict(rid=f"{p['sid']}|random|{j}", sid=p["sid"], role="random", parent=None, text=variant(p["sid"], r))
             for p in plan for j, r in enumerate(p["random"])]
    rows += [dict(rid=f"{sid}|base", sid=sid, role="base", parent=None, text=base[sid]) for sid in base]
    sl = [by[sid] for sid in base]
    with Timer("scoring variants"):
        rt = run_rt(model, tok, sl, rows, batch=a.batch)
        langs = sorted({CENTROID_LANG.get(s.lang, s.lang) for s in sl} | {"en"})
        mu = centroids(model, tok, langs, d / "raw" / f"centroids_{a.model}.npz")
        lse, _ = run_cl(model, tok, sl, rows, mu)
    res = {}
    for r, ls in zip(rows, lse):
        sr = rt["sentences"][r["sid"]]
        w, l = rt_token(rt["rows"][r["rid"]], sr["lp_null"], sr["lp_oracle"])
        w0 = rt_token(sr["lp_oracle"], sr["lp_null"], sr["lp_oracle"])[0]
        res[r["rid"]] = dict(text=r["text"], len=n_tokens(r["text"]), d_rt=float(weighted(w, l)),
                             d_cl=float(weighted(w0, cl_token(ls[layer], tau))))
    per, summ = [], {}
    for p in plan:
        sid = p["sid"]
        b, t = res[f"{sid}|base"], res[f"{sid}|targeted|0"]
        rr = [res[f"{sid}|random|{j}"] for j in range(len(p["random"]))]
        per.append(dict(**p, base_scores=b, targeted_scores=t, random_scores=rr,
                        added_len=dict(targeted=t["len"] - b["len"], random=float(np.mean([x["len"] for x in rr])) - b["len"]),
                        glosses={w: gloss[(sid, w)] for w in p["targeted"] + [x for r in p["random"] for x in r]}))
    for sp in ("rt", "cl"):
        cut_t = np.array([x["base_scores"][f"d_{sp}"] - x["targeted_scores"][f"d_{sp}"] for x in per])
        cut_r = np.array([x["base_scores"][f"d_{sp}"] - np.mean([y[f"d_{sp}"] for y in x["random_scores"]]) for x in per])
        summ[sp] = dict(cut_targeted=boot_ci(cut_t), cut_random=boot_ci(cut_r), targeted_minus_random=boot_ci(cut_t - cut_r),
                        share_sentences_targeted_better=float(np.mean(cut_t > cut_r)))
    lens = dict(targeted=boot_ci([x["added_len"]["targeted"] for x in per]),
                random=boot_ci([x["added_len"]["random"] for x in per]))
    save_json(d / "targeting.json", dict(model=describe(a.model), chooser=a.space, layer=layer, tau=tau,
                                         prompt=GLOSS, summary=summ, added_length=lens, sentences=per))
    for sp, v in summ.items():
        print(f"[{sp}] cut targeted {v['cut_targeted'][0]:+.3f}  random {v['cut_random'][0]:+.3f}  "
              f"difference {v['targeted_minus_random'][0]:+.3f} ({v['targeted_minus_random'][1]:+.3f}, "
              f"{v['targeted_minus_random'][2]:+.3f})")
    print(f"added tokens: targeted {lens['targeted'][0]:.1f}, random {lens['random'][0]:.1f}")


if __name__ == "__main__":
    main()
