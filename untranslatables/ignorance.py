"""Thread 2 diagnostics (handoff §7.2-7.5).  Kept separate, never summed [2209.03302], [2603.24967].

Per (word, context, condition) the samples are clustered by meaning (ut.nli) giving the translation entropy H
(bits [2302.09664]) and KLE [2405.20003].  Per word, averaged over contexts:
  ignorance              H(bare) - H(gloss)                    (bootstrap CI over contexts x samples)
  multiplicity           H(gloss)
  sense information      I(output; sense) = H(pooled gloss+sense_k samples) - mean_k H(gloss+sense_k)
                         (clarification ensembling [2311.08718], adapted)
  context-resolvability  H(gloss) - H(gloss+context)
Other ignorance signals (§7.3): cross-model disagreement = total variation distance between models' reading
distributions (bare) [2604.17112]; small vs large = H_bare(3b) - H_bare(8b) [2402.03563]; iterative-prompting
flip rate [2406.02543].  Agreement between the four signals = Spearman across words.
2x2 validation (§7.5): predicted 'high ignorance' if ignorance >= 0.5 bit with CI above 0; 'high multiplicity'
if H(gloss) >= 1 bit (two equally likely readings).  Fixed thresholds, stated before running.
Human reference (§7.4): data/human_renderings.yaml {word_id: [English renderings]} if filled in.

    python ignorance.py --model 8b --compare qwen7b aya8b --small 3b
"""
import argparse, itertools
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from ut.common import DATA, MODELS, Timer, load_json, load_yaml, out_dir, save_json
from ut.nli import NLI, cluster, kle, semantic_entropy
from ut.stats import tvd

IGN_T, MULT_T = 0.5, 1.0


def load(model_key):
    return load_json(out_dir(model_key) / "spread_samples.json")["rows"]


def cluster_rows(nli, rows):
    for r in rows:
        c = cluster(nli, r["samples"], r["word"], r["language"])
        r["labels"], r["H"] = c["labels"], semantic_entropy(c["counts"])
        r["KLE"] = kle(c["P"], c["uniq_counts"]) if len(c["uniq"]) else 0.0
    return rows


def off_language(samples, lang):
    """Share of answers that look like the source language rather than English (wordfreq: every word of the
    answer at least as frequent in the source language as in English).  Romanesco is checked against Italian."""
    from wordfreq import zipf_frequency
    src = {"rom": "it"}.get(lang, lang)
    def off(s):
        ws = [w for w in s.lower().replace("-", " ").split() if w.isalpha()]
        return bool(ws) and all(zipf_frequency(w, src) >= zipf_frequency(w, "en") for w in ws)
    return float(sum(off(s) for s in samples) / max(1, len(samples)))


def boot_diff(a_lab, b_lab, n=500, seed=0):
    """CI of mean_c [H(a_c) - H(b_c)] resampling contexts and samples within each."""
    rng = np.random.default_rng(seed)
    H = lambda lab: semantic_entropy(Counter(lab))
    vals = []
    for _ in range(n):
        cs = rng.integers(0, len(a_lab), len(a_lab))
        vals.append(np.mean([H(rng.choice(a_lab[c], len(a_lab[c]))) - H(rng.choice(b_lab[c], len(b_lab[c]))) for c in cs]))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def per_word(nli, rows):
    by = {}
    for r in rows:
        by.setdefault(r["word_id"], []).append(r)
    out = {}
    for wid, rs in by.items():
        cond = lambda name: {r["context"]: r for r in rs if r["condition"] == name}
        bare, gloss, ctxp = cond("bare"), cond("gloss"), cond("gloss+context")
        ks = sorted(set(bare) & set(gloss))
        d = dict(category=rs[0]["category"], cell=rs[0]["cell"], word=rs[0]["word"],
                 H_bare=float(np.mean([bare[k]["H"] for k in ks])), H_gloss=float(np.mean([gloss[k]["H"] for k in ks])),
                 KLE_bare=float(np.mean([bare[k]["KLE"] for k in ks])), KLE_gloss=float(np.mean([gloss[k]["KLE"] for k in ks])))
        d["ignorance"] = d["H_bare"] - d["H_gloss"]
        d["off_language"] = {c: off_language([x for r in rs if r["condition"] == c for x in r["samples"]], rs[0]["language"])
                             for c in sorted({r["condition"] for r in rs})}
        d["ignorance_ci"] = boot_diff([np.array(bare[k]["labels"]) for k in ks], [np.array(gloss[k]["labels"]) for k in ks])
        d["multiplicity"] = d["H_gloss"]
        sense_names = sorted({r["condition"] for r in rs if r["condition"].startswith("gloss+sense_")})
        if sense_names:
            mi = []
            for k in ks:
                srs = [r for r in rs if r["condition"] in sense_names and r["context"] == k]
                pooled = cluster(nli, [x for r in srs for x in r["samples"]], rs[0]["word"], rs[0]["language"])
                mi.append(semantic_entropy(pooled["counts"]) - np.mean([r["H"] for r in srs]))
            d["sense_information"] = float(np.mean(mi))
        if ctxp:
            kk = sorted(set(ctxp) & set(gloss))
            d["context_resolvability"] = float(np.mean([gloss[k]["H"] - ctxp[k]["H"] for k in kk]))
        again = [r["again"] for r in rs if r["condition"] == "bare" and r.get("again")]
        flips = []
        for g in again:
            c = cluster(nli, [g["first"]] + g["answers"], rs[0]["word"], rs[0]["language"])
            flips += [lab != c["labels"][0] for lab in c["labels"][1:]]
        d["flip_rate"] = float(np.mean(flips)) if flips else None
        d["predicted_cell"] = ["unknown" if d["ignorance"] >= IGN_T and d["ignorance_ci"][0] > 0 else "known",
                               "many" if d["multiplicity"] >= MULT_T else "one"]
        out[wid] = d
    return out


def cross_model(nli, models):
    """Mean TVD between every pair of models' reading distributions, bare condition, joint clustering."""
    rows = {m: {(r["word_id"], r["context"]): r for r in load(m) if r["condition"] == "bare"} for m in models}
    keys = set.intersection(*(set(v) for v in rows.values()))
    per = {}
    for key in keys:
        rs = [rows[m][key] for m in models]
        c = cluster(nli, [x for r in rs for x in r["samples"]], rs[0]["word"], rs[0]["language"])
        labs, i = {}, 0
        for m, r in zip(models, rs):
            labs[m] = Counter(c["labels"][i:i + len(r["samples"])]); i += len(r["samples"])
        per.setdefault(key[0], []).append(np.mean([tvd(labs[x], labs[y]) for x, y in itertools.combinations(models, 2)]))
    return {w: float(np.mean(v)) for w, v in per.items()}


def human_reference(nli, words):
    p = DATA / "human_renderings.yaml"
    if not p.exists():
        return {}
    ref = load_yaml(p) or {}
    out = {}
    for wid, rend in ref.items():
        if rend and wid in words:
            c = cluster(nli, rend, words[wid]["word"], wid.split(":")[0])
            out[wid] = dict(n=len(rend), H_human=semantic_entropy(c["counts"]), clusters=c["reps"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=MODELS)
    ap.add_argument("--compare", nargs="*", default=[], choices=MODELS)
    ap.add_argument("--small", default=None, choices=MODELS)
    a = ap.parse_args()
    nli = NLI()
    with Timer("clustering"):
        rows = cluster_rows(nli, load(a.model))
    words = per_word(nli, rows)
    if a.compare:
        for w, v in cross_model(nli, [a.model] + a.compare).items():
            words[w]["cross_model_tvd"] = v
    if a.small:
        small = cluster_rows(nli, [r for r in load(a.small) if r["condition"] == "bare"])
        hs = {}
        for r in small:
            hs.setdefault(r["word_id"], []).append(r["H"])
        for w, d in words.items():
            if w in hs:
                d["small_minus_large_H"] = float(np.mean(hs[w]) - d["H_bare"])
    signals = ["ignorance", "cross_model_tvd", "small_minus_large_H", "flip_rate"]
    agree = {}
    for x, y in itertools.combinations(signals, 2):
        pairs = [(d[x], d[y]) for d in words.values() if d.get(x) is not None and d.get(y) is not None]
        if len(pairs) >= 5:
            agree[f"{x}~{y}"] = dict(spearman=float(spearmanr(*zip(*pairs))[0]), n=len(pairs))
    cells = [("known", "one"), ("known", "many"), ("unknown", "one"), ("unknown", "many")]
    conf = {f"{t[0]}/{t[1]}": {f"{p[0]}/{p[1]}": sum(1 for d in words.values() if tuple(d["cell"]) == t
                                                     and tuple(d["predicted_cell"]) == p) for p in cells} for t in cells}
    acc = np.mean([tuple(d["cell"]) == tuple(d["predicted_cell"]) for d in words.values()])
    rd = out_dir(a.model)
    save_json(rd / "ignorance.json", dict(thresholds=dict(ignorance_bits=IGN_T, multiplicity_bits=MULT_T),
                                          words=words, signal_agreement=agree, confusion=conf, cell_accuracy=float(acc),
                                          human=human_reference(nli, words), compare=a.compare, small=a.small,
                                          rows=[{k: v for k, v in r.items() if k != "prompt"} for r in rows]))
    print("2x2 accuracy", round(float(acc), 2)); print(conf)


if __name__ == "__main__":
    main()
