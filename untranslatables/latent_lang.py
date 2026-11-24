"""Does it 'think' in English?  Latent-language share by layer (handoff §6.1), after Wendler et al. [2402.10588].

Word-translation prompts between NON-English languages (fr->es, de->it, pt->fr), 4-shot in their format
    Français: "nuage" - Español: "nube"\n ... Français: "<x>" - Español: "
At the last position, at every layer, the logit lens gives P(English answer) and P(target answer), each the total
probability of the answer's prefix tokens (ut.lens.prefix_token_ids; tokens shared by both answers removed).
Items are triangulated through MUSE: x->y from the x-y dictionary, with x->e and y->e agreeing on the English e;
items where e and y contain each other (cognates, loans) are dropped, as in the original.
Tuned lens: not used. Pretrained lenses exist only for Llama-2 and Llama-3.0-8B (checked 18 Nov 2026), not for
these four checkpoints, and a 3.0 lens is not valid on 3.1 weights.

    python latent_lang.py --model 3b      # -> results/<model>/latent_lang.json
"""
import argparse

import numpy as np
import torch

from pools import muse
from ut.arch import forward_hidden, n_layers
from ut.common import LANG_NAMES, MODELS, Progress, load_model, out_dir, save_json
from ut.lens import disjoint_sets, layer_logprobs
from ut.stats import boot_ci

PAIRS = [("fr", "es"), ("de", "it"), ("pt", "fr")]


def items(x, y, n, seed):
    ex, ey = dict(muse(x, "en")), dict(muse(y, "en"))
    out, seen = [], set()
    for a, b in muse(x, y):
        e = ex.get(a)
        if e and ey.get(b) == e and a not in seen and all(s.isalpha() for s in (a, b, e)) \
                and e.lower() not in b.lower() and b.lower() not in e.lower() and a.lower() != b.lower():
            out.append((a, b, e)); seen.add(a)
    rng = np.random.default_rng(seed)
    return [out[i] for i in sorted(rng.choice(len(out), min(n, len(out)), replace=False))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok, model = load_model(a.model)
    res = {}
    for x, y in PAIRS:
        its = items(x, y, a.n, a.seed)
        rng = np.random.default_rng(a.seed)
        P_en, P_tg, single = [], [], []
        pr = Progress(f"{x}->{y}", len(its))
        for i, (src, tgt, en) in enumerate(its):
            shots = [its[k] for k in rng.choice([k for k in range(len(its)) if k != i], 4, replace=False)]
            prompt = "".join(f'{LANG_NAMES[x]}: "{s}" - {LANG_NAMES[y]}: "{t}"\n' for s, t, _ in shots)
            prompt += f'{LANG_NAMES[x]}: "{src}" - {LANG_NAMES[y]}: "'
            en_ids, tg_ids = disjoint_sets(tok, [en], [tgt])
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                hs = forward_hidden(model, **ids, output_hidden_states=True).hidden_states
            lp = layer_logprobs(model, hs).exp()                      # [L+1, V]
            P_en.append(lp[:, en_ids].sum(1).cpu().numpy() if en_ids else np.zeros(len(hs)))
            P_tg.append(lp[:, tg_ids].sum(1).cpu().numpy() if tg_ids else np.zeros(len(hs)))
            single.append(len(tok(tgt, add_special_tokens=False)["input_ids"]) == 1)
            if i % 20 == 0:
                pr.tick(i + 1)
        P_en, P_tg = np.array(P_en), np.array(P_tg)
        share = P_en / (P_en + P_tg + 1e-12)
        curve = lambda M: [boot_ci(M[:, l]) for l in range(M.shape[1])]
        res[f"{x}->{y}"] = dict(n=len(its), n_single_token_target=int(sum(single)), items=its,
                                p_en=curve(P_en), p_target=curve(P_tg), english_share=curve(share),
                                english_share_single_token=curve(share[np.array(single)]) if any(single) else None,
                                peak_layer=int(np.argmax(share.mean(0))))
        print(x, y, "peak English share", round(float(share.mean(0).max()), 3), "at layer", res[f"{x}->{y}"]["peak_layer"])
    save_json(out_dir(a.model) / "latent_lang.json", dict(pairs=res, lens="logit lens (final norm + unembedding)",
                                                          format="Wendler et al. 4-shot word translation"))


if __name__ == "__main__":
    main()
