"""Patching readout: what the model 'would say' (handoff §5.6), following Dumas et al. [2411.08745].

Source prompt (Dumas-faithful, user decision 18 Nov): 5-shot word translation from the word's language into a
NON-English language, ending at the word, e.g.  Português: "saudade" - Italiano: "
The latent is the residual state at the source prompt's LAST token at layer j.  Target prompt: 5-shot X -> English
with a different (dummy) word at the end.  A single-layer patch overwrites the target's last position after block j
(ut.represent.patching); later layers still see the target context, which carries the output language.
  * Layer j: swept over every 2nd layer on ordinary MUSE words, where the correct English is known; j* = the
    layer whose patched greedy output most often equals that English word.  Untranslatables then use j*.
  * At j*: 50 samples (T=1) give the distribution of English outputs, compared with the recipe.
  * Variants: (a) 'word_span': mean state over the word's tokens in its natural contexts (thread-1 cache);
    (b) 'mean_target_langs': the latent averaged over source prompts into 4 different non-English languages,
    which keeps the concept and averages out the target language.

    python patching.py --model 3b
"""
import argparse
from collections import Counter

import numpy as np

from pools import muse
from ut.arch import forward_hidden, n_layers
from ut.common import LANG_NAMES, MODELS, Timer, generate, load_model, out_dir, save_json
from ut.represent import load_cached, patching
from ut.words import PHASE1, load_words

SRC2TGT = {"pt": "it", "it": "fr", "fr": "es", "es": "pt", "de": "it"}
ALT_TGTS = {l: [x for x in PHASE1 if x != l] for l in PHASE1}
DUMMY_EN_SRC = "de"          # language of the target prompt's few-shot examples and its dummy word


def fewshot(x, y, n, seed, skip=()):
    rng = np.random.default_rng(seed)
    ps = [p for p in muse(x, y)[:2000] if p[0] not in skip and p[0].isalpha() and p[1].isalpha()]
    sel = [ps[i] for i in rng.choice(len(ps), n, replace=False)]
    return "".join(f'{LANG_NAMES[x]}: "{a}" - {LANG_NAMES[y]}: "{b}"\n' for a, b in sel)


def src_prompt(word, x, y, seed=0):
    return fewshot(x, y, 5, seed, skip={word}) + f'{LANG_NAMES[x]}: "{word}" - {LANG_NAMES[y]}: "'


def tgt_prompt(seed=1):
    dummy = muse(DUMMY_EN_SRC, "en")[1000 + seed][0]
    return fewshot(DUMMY_EN_SRC, "en", 5, seed, skip={dummy}) + f'{LANG_NAMES[DUMMY_EN_SRC]}: "{dummy}" - English: "'


def last_states(model, tok, prompts):
    import torch
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=True).to(model.device)
        with torch.no_grad():
            hs = forward_hidden(model, **ids, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1].float() for h in hs]).cpu())
    return torch.stack(out)                     # [n, L+1, d]


def clean(o):
    return o.split('"')[0].strip().lower()


def readout(model, tok, latent, layer, n, temperature, seed):
    """n samples from the target prompt with one latent patched in."""
    with patching(model, {layer: latent}):
        outs = generate(model, tok, [tgt_prompt()], n=n, max_new_tokens=8, temperature=temperature, seed=seed, batch=n)[0]
    return [clean(o) for o in outs]


def readout_many(model, tok, latents, layer, batch=64):
    """Greedy output for each of many latents (one per batch row)."""
    out = []
    for c in range(0, len(latents), batch):
        chunk = latents[c:c + batch]
        with patching(model, {layer: chunk}):
            outs = generate(model, tok, [tgt_prompt()] * len(chunk), n=1, max_new_tokens=8, temperature=0,
                            batch=len(chunk))
        out += [clean(o[0]) for o in outs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--n-calib", type=int, default=40, help="ordinary MUSE words per language for choosing j")
    ap.add_argument("--n-samples", type=int, default=50)
    a = ap.parse_args()
    tok, model = load_model(a.model)
    L = n_layers(model)
    grid = list(range(1, L + 1, 2))
    # 1) calibrate j on ordinary words with known English
    calib = []
    for x in PHASE1:
        en = dict(muse(x, "en"))
        cand = [w for w, _ in muse(x, SRC2TGT[x])[100:2000] if w in en and w.isalpha()]
        calib += [(x, w, en[w].lower()) for w in cand[:a.n_calib]]
    with Timer(f"calibration latents ({len(calib)})"):
        H = last_states(model, tok, [src_prompt(w, x, SRC2TGT[x]) for x, w, _ in calib])
    acc = {}
    with Timer("calibration sweep"):
        for j in grid:
            outs = readout_many(model, tok, H[:, j], j)
            acc[j] = float(np.mean([o == e for o, (_, _, e) in zip(outs, calib)]))
    jstar = max(acc, key=acc.get)
    print("patch accuracy by layer:", acc, "-> j* =", jstar)
    # 2) untranslatables (+ controls) at j*
    words = [w for w in load_words(langs=PHASE1) if w.category in ("untranslatable", "ordinary", "emotion_plain")]
    res = {}
    Hw, meta = load_cached(a.model, "words_t1")
    index = meta["index"]
    with Timer("readouts"):
        for w in words:
            lat = last_states(model, tok, [src_prompt(w.word, w.language, SRC2TGT[w.language])])[0, jstar]
            lat_mean = last_states(model, tok, [src_prompt(w.word, w.language, t) for t in ALT_TGTS[w.language]])[:, jstar].mean(0)
            rows = [i for i, r in enumerate(index) if r[0] == w.id and r[1] == "natural"]
            import torch
            lat_span = torch.tensor(np.asarray(Hw[rows, jstar], dtype=np.float32).mean(0))
            res[w.id] = {name: dict(Counter(readout(model, tok, v, jstar, a.n_samples, 1.0, 0)).most_common())
                         for name, v in (("dumas", lat), ("mean_target_langs", lat_mean), ("word_span", lat_span))}
            print(w.id, list(res[w.id]["dumas"].items())[:5])
    save_json(out_dir(a.model) / "patching.json", dict(
        layer_sweep=acc, j_star=jstar, calibration_n=len(calib), source_to_target=SRC2TGT,
        target_prompt=tgt_prompt(), example_source_prompt=src_prompt("saudade", "pt", "it"),
        n_samples=a.n_samples, temperature=1.0, words=res))


if __name__ == "__main__":
    main()
