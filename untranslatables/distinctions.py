"""Lost through English?  Distinction collapse (handoff §6.3).

Translate every item of data/distinctions.yaml from its source into its target language (greedy, plus
--n-samples samples at T=1), score each output with the target-class regexes (correct / error / other), and
compare error rates: english_lacks sets vs their english_keeps control.  The excess error rate is the
behavioural English-bias score [2601.12549].  Logit lens: for each greedy output, at the position where the
first class word begins, the per-layer probability of the English collapsed word ('know', 'be') -- does it rise
in the middle layers more on error items than on correct ones?

    python distinctions.py --model 3b      # -> results/<model>/distinctions.json
"""
import argparse, re

import numpy as np
import torch

from ut.arch import forward_hidden, n_layers
from ut.common import DATA, LANG_EN, MODELS, Timer, chat, generate, load_model, load_yaml, out_dir, save_json
from ut.lens import layer_logprobs, prefix_token_ids
from ut.stats import boot_ci

PROMPT = "Translate this {src} sentence into {tgt}. Reply with the translation only.\n\n{s}"


def load_sets():
    return {p["id"]: p for p in load_yaml(DATA / "distinctions.yaml")["pairs"]}


def prompt_text(tok, p, item):
    return chat(tok, PROMPT.format(src=LANG_EN[p["src"]], tgt=LANG_EN[p["tgt"]], s=item["s"]))


def classes_found(text, classes):
    t = text.lower()
    return {c for c, pats in classes.items() if any(re.search(pt, t) for pt in pats)}


def score(text, p, expected):
    m = classes_found(text, p["classes"])
    if expected in m and len(m) == 1:
        return "correct"
    if m and expected not in m:
        return "error"
    return "other"


def error_rate(labels):
    c, e = labels.count("correct"), labels.count("error")
    return e / (c + e) if c + e else np.nan


def first_class_offset(text, classes):
    t, best = text.lower(), None
    for pats in classes.values():
        for pt in pats:
            m = re.search(pt, t)
            if m and (best is None or m.start() < best):
                best = m.start()
    return best


def translate_all(model, tok, sets, n_samples, seed=0):
    out = {}
    for sid, p in sets.items():
        prompts = [prompt_text(tok, p, it) for it in p["items"]]
        greedy = [g[0] for g in generate(model, tok, prompts, n=1, temperature=0, max_new_tokens=60, batch=32)]
        samples = generate(model, tok, prompts, n=n_samples, temperature=1.0, max_new_tokens=60, batch=64,
                           seed=seed) if n_samples else [[] for _ in prompts]
        out[sid] = dict(greedy=greedy, samples=samples)
    return out


def summarise(sets, trans):
    res = {}
    for sid, p in sets.items():
        g = [score(o, p, it["x"]) for o, it in zip(trans[sid]["greedy"], p["items"])]
        s_item = [error_rate([score(o, p, it["x"]) for o in so]) for so, it in zip(trans[sid]["samples"], p["items"])]
        res[sid] = dict(type=p["type"], greedy_labels=g, greedy_error=error_rate(g),
                        greedy_error_ci=boot_ci([1.0 if x == "error" else 0.0 for x in g if x != "other"]),
                        greedy_other_rate=g.count("other") / len(g),
                        sample_error=boot_ci([x for x in s_item if not np.isnan(x)]))
    for sid, p in sets.items():
        if p.get("control"):
            c = res[p["control"]]
            res[sid]["excess_error_greedy"] = res[sid]["greedy_error"] - c["greedy_error"]
            res[sid]["excess_error_samples"] = res[sid]["sample_error"][0] - c["sample_error"][0]
    return res


def lens_check(model, tok, sets, trans, summ):
    out = {}
    for sid, p in sets.items():
        words = p.get("english_collapse") or []
        if not words:
            continue
        ids = sorted(set().union(*(prefix_token_ids(tok, w, min_prefix=2) for w in words)))
        curves = {"error": [], "correct": []}
        for it, o, lab in zip(p["items"], trans[sid]["greedy"], summ[sid]["greedy_labels"]):
            off = first_class_offset(o, p["classes"])
            if off is None or lab not in curves:
                continue
            enc = tok(prompt_text(tok, p, it) + o[:off], return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                hs = forward_hidden(model, **enc, output_hidden_states=True).hidden_states
            curves[lab].append(layer_logprobs(model, hs).exp()[:, ids].sum(1).cpu().numpy())
        out[sid] = {k: dict(n=len(v), mean=np.mean(v, 0).tolist() if v else None) for k, v in curves.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--n-samples", type=int, default=10)
    a = ap.parse_args()
    sets = load_sets()
    tok, model = load_model(a.model)
    with Timer("translations"):
        trans = translate_all(model, tok, sets, a.n_samples)
    summ = summarise(sets, trans)
    with Timer("logit lens"):
        lens = lens_check(model, tok, sets, trans, summ)
    save_json(out_dir(a.model) / "distinctions.json", dict(prompt=PROMPT, n_samples=a.n_samples, translations=trans,
                                                           summary=summ, lens=lens))
    for sid, r in summ.items():
        print(f"{sid:28s} greedy error {r['greedy_error']:.2f}  other {r['greedy_other_rate']:.2f}"
              + (f"  excess {r['excess_error_greedy']:+.2f}" if "excess_error_greedy" in r else ""))


if __name__ == "__main__":
    main()
