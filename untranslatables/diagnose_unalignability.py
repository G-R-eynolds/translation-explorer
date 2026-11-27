"""Untranslatability vs. Model Ignorance & Hsieh-Lian (2610.08303) Diagnostic.

Measures four orthogonal dimensions for curated word groups across Gemma 3 4B / 12B / 27B
(and CPU `tiny` for smoke testing):
  1. Source Precision & Micro-Shifts: contextual cosine shift, TwoNN intrinsic dim, source NLL.
  2. Cross-Lingual Dispersion & Leftover: NNOMP English-basis leftover %, sound-alike rate,
     English translation semantic spread, and round-trip Cycle-Consistency (CC).
  3. Directional Steerability / Inverse-RIG: Output-Fisher top-eigenvalue concentration
     (lambda_1 / tr(F)) + KL-calibrated steering fluency vs. breakdown.

Usage:
    PYTHONPATH=untranslatables python3 -m diagnose_unalignability --model g3_4b_pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch

from explorer.directions import alpha_for, calibrate, grads_at
from explorer.live_engine import CursorHook
from explorer.recipe_live import (
    LiveEnglishRecipeDecomposer,
    default_en_pool_path,
    extract_en_pool_from_model,
)
from ut import arch
from ut.common import ROOT, Timer, load_model, save_json

# Curated 4-group diagnostic set:
#   - ordinary: clear 1-to-1 English equivalents
#   - stylistic: high surface choice, tight semantic cluster ('volverán', 'bank')
#   - untranslatable: sharp source usage + high structured English dispersion
#   - ignorance: Romanesco dialect (from data/words/rom.yaml) + low-resource Welsh/Basque/Yami
DIAGNOSTIC_ITEMS: list[dict[str, Any]] = [
    # 1. Ordinary equivalents
    {
        "id": "es_camisa", "word": "camisa", "lang": "Spanish", "group": "ordinary",
        "contexts": [
            "Se puso una camisa blanca para la entrevista.",
            "La camisa de lino estaba arrugada en la maleta.",
            "Planchó la camisa antes de salir de casa.",
            "El botón de su camisa se soltó en el tren.",
            "Compró una camisa azul en el mercado.",
            "Llevaba la camisa abierta por el calor.",
        ],
    },
    {
        "id": "it_domani", "word": "domani", "lang": "Italian", "group": "ordinary",
        "contexts": [
            "Partiremo domani mattina presto per Roma.",
            "Ci vediamo domani pomeriggio davanti al caffè.",
            "Spero che domani faccia bel tempo.",
            "Il lavoro va consegnato entro domani sera.",
            "Domani inizia la settimana di vacanza.",
            "Non lasciare a domani quello che puoi fare oggi.",
        ],
    },
    {
        "id": "pt_tristeza", "word": "tristeza", "lang": "Portuguese", "group": "ordinary",
        "contexts": [
            "Sentiu uma profunda tristeza ao ouvir a notícia.",
            "A tristeza tomou conta da sala silenciosa.",
            "Havia uma certa tristeza no seu olhar cansado.",
            "Falou sem tristeza dos anos que passaram.",
            "A música expressava a tristeza da despedida.",
            "Tentou esconder a tristeza com um sorriso breve.",
        ],
    },
    # 2. Stylistic / polysemous controls
    {
        "id": "es_volveran", "word": "volverán", "lang": "Spanish", "group": "stylistic",
        "contexts": [
            "Volverán las oscuras golondrinas en tu balcón sus nidos a colgar.",
            "Los viajeros volverán al pueblo antes del invierno.",
            "Esos días felices nunca volverán a repetirse.",
            "Las lluvias volverán con el cambio de estación.",
            "Prometieron que volverán cuando termine la cosecha.",
            "Los pájaros volverán a cantar al amanecer.",
        ],
    },
    {
        "id": "en_bank_mixed", "word": "bank", "lang": "English", "group": "stylistic",
        "contexts": [
            "We sat on the grassy bank watching the river flow past.",
            "The boat ran aground on the muddy bank at dusk.",
            "She walked to the bank to deposit her paycheck.",
            "The bank approved the small business loan yesterday.",
            "Reeds grew thick along the opposite bank of the stream.",
            "He waited outside the bank before the vault opened.",
        ],
    },
    # 3. Genuine untranslatables (high-resource languages)
    {
        "id": "es_duende", "word": "duende", "lang": "Spanish", "group": "untranslatable",
        "contexts": [
            "Cuando canta Camarón, tiene duende; me gusta mucho el duende.",
            "Aquel guitarrista tocaba sin técnica perfecta, pero con un duende inexplicable.",
            "El duende no está en la garganta, sube por dentro desde la planta de los pies.",
            "Esa noche el tablao entero quedó en silencio ante su duende.",
            "Hay artistas correctos y hay artistas que tienen duende.",
            "Lorca escribió que el duende es un poder misterioso que todos sienten.",
        ],
    },
    {
        "id": "pt_saudade", "word": "saudade", "lang": "Portuguese", "group": "untranslatable",
        "contexts": [
            "Tenho muita saudade das tardes de verão em Lisboa.",
            "O fado nasce dessa saudade que mistura dor e ternura.",
            "Sentia saudade até dos dias que nunca chegou a viver.",
            "A carta trazia uma saudade serena da infância distante.",
            "Olhava o mar com uma saudade sem nome nem remédio.",
            "Há uma saudade doce naquela melodia antiga.",
        ],
    },
    {
        "id": "it_sprezzatura", "word": "sprezzatura", "lang": "Italian", "group": "untranslatable",
        "contexts": [
            "Castiglione descrive la sprezzatura come una grazia che nasconde lo sforzo.",
            "Suonava il violino con tale sprezzatura da far sembrare facile ogni passaggio.",
            "Nel suo modo di vestire c'era una sprezzatura studiata e leggera.",
            "L'arte vera richiede sprezzatura, mai affettazione.",
            "Rispose alla critica con elegante sprezzatura.",
            "Dietro quella sprezzatura si celavano anni di disciplina.",
        ],
    },
    {
        "id": "de_sehnsucht", "word": "Sehnsucht", "lang": "German", "group": "untranslatable",
        "contexts": [
            "Eine tiefe Sehnsucht nach der Ferne erfüllte sein Herz.",
            "In ihren Gedichten klingt eine stille Sehnsucht nach Geborgenheit.",
            "Diese Sehnsucht richtet sich auf keinen bestimmten Ort.",
            "Der Klang des Horns weckte alte Sehnsucht in den Zuhörern.",
            "Zwischen Erinnerung und Hoffnung wohnt die Sehnsucht.",
            "Er blickte voller Sehnsucht über die grauen Dächer.",
        ],
    },
    {
        "id": "es_sobremesa", "word": "sobremesa", "lang": "Spanish", "group": "untranslatable",
        "contexts": [
            "La sobremesa del domingo se alargó hasta el anochecer entre risas y café.",
            "En casa de mis abuelos la sobremesa duraba más que la comida.",
            "Nadie tenía prisa por levantarse durante aquella sobremesa.",
            "En la sobremesa surgieron las mejores historias de la familia.",
            "El vino y la conversación animaron la sobremesa.",
            "Extraño las largas sobremesas de verano en el patio.",
        ],
    },
    # 4. Ignorance controls (Romanesco dialect + low-resource Basque / Welsh / Yami)
    {
        "id": "rom_schertri", "word": "schertri", "lang": "Romanesco", "group": "ignorance",
        "contexts": [
            "Usciranno li schertri da la terra a ppecorone.",
            "Li schertri vecchi stavano sepolti sotto la chiesa.",
            "Parevano du' schertri che camminavano ar buio.",
            "Co' queli schertri nun se poteva scherzà.",
            "Sotto er selciato c'erano rimasti solo li schertri.",
            "Li schertri ballavano ne la notte der giudizzio.",
        ],
    },
    {
        "id": "rom_bbiocca", "word": "bbiocca", "lang": "Romanesco", "group": "ignorance",
        "contexts": [
            "Come purcini attorno de la bbiocca.",
            "E sta bbiocca sarà Ddio bbenedetto.",
            "La bbiocca chiama li fiji sotto l'ale.",
            "Quela bbiocca nun se move dar nido suo.",
            "Li pulcini curreno subito da la bbiocca.",
            "La bbiocca cova l'ova dentro er cortile.",
        ],
    },
    {
        "id": "cy_hiraeth", "word": "hiraeth", "lang": "Welsh", "group": "ignorance",
        "contexts": [
            "Mae hiraeth mawr arnaf am fy hen gartref yn y cwm.",
            "Clywais hiraeth yn ei llais wrth iddi ganu'r hen alaw.",
            "Roedd y llythyr yn llawn hiraeth am ddyddiau gynt.",
            "Ni all neb esbonio'r hiraeth sy'n aros yn y galon.",
            "Daeth ton o hiraeth drosof wrth edrych ar y môr.",
            "Mae hiraeth am Gymru yn rhan o'r gân honno.",
        ],
    },
    {
        "id": "eu_bihotz", "word": "bihotz", "lang": "Basque", "group": "ignorance",
        "contexts": [
            "Bere bihotz beroak lagun guztiak bildu zituen.",
            "Etxe hartako bihotz bizia zen amona.",
            "Hitz goxo haiek bihotz barruraino iritsi zitzaizkion.",
            "Bihotz onez eman zion azken ogi puska.",
            "Neguko arratsaldean bihotz hutsez abestu zuten.",
            "Mendiko isiltasunak bihotz lasaia ekarri zion.",
        ],
    },
    {
        "id": "tao_anito", "word": "anito", "lang": "Yami", "group": "ignorance",
        "contexts": [
            "O anito am ya mangay do pongso.",
            "No makavava o tao am, ji na ngina o anito.",
            "Ya mapia o onowned do anito no rarakeh.",
            "Ko ya taziyen o ciriciring do anito.",
            "No mangay do wawa am, manavatava so anito.",
            "O rarakeh am, ya makahehet so anito.",
        ],
    },
]


def _twonn_dim(X: np.ndarray) -> float:
    """Estimate TwoNN intrinsic dimensionality (Facco et al. 2017) on point cloud X (N, d)."""
    n = len(X)
    if n < 4:
        return 1.0
    # Pairwise Euclidean distances
    diffs = X[:, None, :] - X[None, :, :]
    D = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(D, np.inf)
    D_sorted = np.sort(D, axis=1)
    r1 = np.maximum(D_sorted[:, 0], 1e-8)
    r2 = np.maximum(D_sorted[:, 1], r1 + 1e-8)
    mu = np.sort(r2 / r1)
    # Empirical CDF F_emp(i) = i / n for i=1..n-1
    valid = mu[:-1] > 1.0 + 1e-6
    if not np.any(valid):
        return 1.0
    x_val = np.log(mu[:-1][valid])
    y_val = -np.log(1.0 - np.arange(1, n)[valid] / float(n))
    denom = float(np.dot(x_val, x_val))
    if denom <= 1e-12:
        return 1.0
    return float(max(0.5, min(float(n), np.dot(x_val, y_val) / denom)))


def _mean_pairwise_cos_dist(X: np.ndarray) -> float:
    if len(X) < 2:
        return 0.0
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    U = X / np.maximum(norms, 1e-8)
    C = np.clip(U @ U.T, -1.0, 1.0)
    triu = C[np.triu_indices(len(U), k=1)]
    return float(np.mean(1.0 - triu))


def _pairwise_dist_vec(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    U = X / np.maximum(norms, 1e-8)
    C = np.clip(U @ U.T, -1.0, 1.0)
    return 1.0 - C[np.triu_indices(len(U), k=1)]


def _extract_target_states_and_nll(
    model: Any,
    tok: Any,
    contexts: list[str],
    target_word: str,
    layer: int,
) -> tuple[np.ndarray, float]:
    """Return contextualized layer-`layer` vectors (N, d) for `target_word` and mean token NLL."""
    dev = arch.text_model(model).embed_tokens.weight.device
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    vecs: list[np.ndarray] = []
    nlls: list[float] = []
    w_low = target_word.lower()

    for ctx in contexts:
        ids = tok(ctx, add_special_tokens=True)["input_ids"]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = arch.forward_hidden(model, input_ids=x, output_hidden_states=True)
            hs = out.hidden_states[layer + 1][0].float()  # (T, d)
            logits = arch.logits(model, out.last_hidden_state[0]).float()  # (T, V)

        # Find token span matching target_word (greedy substring match on decoded prefix)
        pieces = [tok.decode([int(t)]) for t in ids]
        matched: list[int] = []
        for i in range(len(pieces)):
            acc = ""
            for j in range(i, min(len(pieces), i + 6)):
                acc += pieces[j]
                if w_low in acc.lower():
                    matched = list(range(i, j + 1))
                    break
            if matched:
                break
        if not matched:
            matched = [max(0, len(ids) - 1)]

        vecs.append(hs[matched].mean(dim=0).cpu().numpy())
        # Compute NLL on matched tokens (where index > 0)
        tok_nlls = []
        for idx in matched:
            if idx >= 1:
                lp = logits[idx - 1].log_softmax(-1)
                tok_nlls.append(float(-lp[int(ids[idx])].item()))
        if tok_nlls:
            nlls.append(float(np.mean(tok_nlls)))

    return np.stack(vecs, axis=0), (float(np.mean(nlls)) if nlls else 8.0)


def _greedy_translate_batch(
    model: Any,
    tok: Any,
    prompts: list[str],
    max_new: int = 24,
) -> tuple[list[str], np.ndarray]:
    """Greedy-decode short translations & return decoded strings + mean final-layer states."""
    dev = arch.text_model(model).embed_tokens.weight.device
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    eos = tok.eos_token_id
    out_texts: list[str] = []
    out_vecs: list[np.ndarray] = []
    try:
        for p in prompts:
            ids = tok(p, add_special_tokens=True)["input_ids"]
            cur = torch.tensor([ids], dtype=torch.long, device=dev)
            gen: list[int] = []
            last_vecs: list[torch.Tensor] = []
            with torch.no_grad():
                res = arch.forward_hidden(model, input_ids=cur, use_cache=True)
                pkv = res.past_key_values
                h = res.last_hidden_state[0, -1]
                nxt = int(arch.logits(model, h[None])[0].argmax().item())
                for _ in range(max_new):
                    if eos is not None and nxt == eos:
                        break
                    gen.append(nxt)
                    txt_so_far = tok.decode(gen)
                    if "\n" in txt_so_far:
                        break
                    res = arch.forward_hidden(
                        model,
                        input_ids=torch.tensor([[nxt]], dtype=torch.long, device=dev),
                        past_key_values=pkv,
                        use_cache=True,
                    )
                    pkv = res.past_key_values
                    h = res.last_hidden_state[0, -1]
                    last_vecs.append(h.detach().float().cpu())
                    nxt = int(arch.logits(model, h[None])[0].argmax().item())
            clean = tok.decode(gen).split("\n")[0].strip()
            out_texts.append(clean)
            if last_vecs:
                out_vecs.append(torch.stack(last_vecs).mean(0).numpy())
            else:
                out_vecs.append(h.detach().float().cpu().numpy())
    finally:
        tok.padding_side = old_side
    return out_texts, np.stack(out_vecs, axis=0)


def _measure_steerability_rig(
    model: Any,
    tok: Any,
    hook: CursorHook,
    prompt_text: str,
    layer: int,
    n_samples: int = 6,
) -> dict[str, float]:
    """Measure Output-Fisher top eigenvalue ratio and KL-calibrated steering contrast."""
    dev = arch.text_model(model).embed_tokens.weight.device
    prefix = tok(prompt_text, add_special_tokens=True)["input_ids"]
    b = len(prefix) - 1
    eos = tok.eos_token_id
    conts: list[list[int]] = []
    # Sample n_samples diverse short continuations
    for s_idx in range(n_samples):
        torch.manual_seed(100 + s_idx)
        cur = torch.tensor([prefix], dtype=torch.long, device=dev)
        gen: list[int] = []
        with torch.no_grad():
            res = arch.forward_hidden(model, input_ids=cur, use_cache=True)
            pkv = res.past_key_values
            h = res.last_hidden_state[0, -1]
            for _ in range(10):
                probs = arch.logits(model, h[None])[0].float().softmax(-1)
                nxt = int(torch.multinomial(probs, num_samples=1).item())
                if eos is not None and nxt == eos:
                    break
                gen.append(nxt)
                if "\n" in tok.decode(gen):
                    break
                res = arch.forward_hidden(
                    model,
                    input_ids=torch.tensor([[nxt]], dtype=torch.long, device=dev),
                    past_key_values=pkv,
                    use_cache=True,
                )
                pkv = res.past_key_values
                h = res.last_hidden_state[0, -1]
        conts.append(gen if gen else [int(prefix[-1])])

    G = grads_at(model, prefix, conts, layer, b, batch=min(4, n_samples))
    Gc = G - G.mean(0, keepdims=True)
    # Gram SVD for top eigenvalue concentration
    K = Gc @ Gc.T
    eigvals = np.maximum(np.linalg.eigvalsh(K), 0.0)[::-1]
    tr = float(np.sum(eigvals))
    lam1_ratio = float(eigvals[0] / tr) if tr > 1e-12 else 0.0

    # Top singular vector steering test
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    v1 = Vt[0]
    v1 = v1 / max(float(np.linalg.norm(v1)), 1e-12)
    a0 = alpha_for(G, v1, 0.25)
    if not np.isfinite(a0) or a0 <= 0 or a0 > 1e4:
        a0 = 1.0
    a_cal = calibrate(model, hook, prefix + conts[0], b, v1, target=0.25, mode="cursor", a0=a0, iters=3)

    # Compare steered (+a_cal v1 vs -a_cal v1) greedy outputs
    outs = []
    for sign in (1.0, -1.0):
        vec = torch.tensor(sign * a_cal * v1, dtype=torch.float32)
        hook.set(vec, b, only_at_b=False)
        try:
            txt, _ = _greedy_translate_batch(model, tok, [prompt_text], max_new=14)
            outs.append(txt[0])
        finally:
            hook.set(None)

    distinct_poles = 1.0 if (outs[0].strip().lower() != outs[1].strip().lower() and bool(outs[0].strip()) and bool(outs[1].strip())) else 0.0
    # Directional RIG proxy: 1 - (lam1_ratio * distinct_poles) -> low when clean, steerable fork exists; high when flat/inoperable
    rig_proxy = float(max(0.0, min(1.0, 1.0 - (0.6 * lam1_ratio + 0.4 * distinct_poles))))
    return {
        "fisher_lam1_ratio": round(lam1_ratio, 4),
        "distinct_poles": distinct_poles,
        "rig_proxy": round(rig_proxy, 4),
        "pole_pos": outs[0],
        "pole_neg": outs[1],
    }


def run_diagnostic(model_key: str, limit: int | None = None) -> dict[str, Any]:
    tok, model = load_model(model_key)
    layer = arch.n_layers(model) // 2

    en_pool_path = default_en_pool_path(model_key)
    if en_pool_path.exists():
        decomposer = LiveEnglishRecipeDecomposer.from_npz(en_pool_path)
    else:
        decomposer = extract_en_pool_from_model(model, tok, layer=layer, max_words=1500)
        decomposer.save(en_pool_path)

    hook = CursorHook(model, layer)
    items = DIAGNOSTIC_ITEMS[:limit] if limit else DIAGNOSTIC_ITEMS
    rows: list[dict[str, Any]] = []

    try:
        for item in items:
            lang = item["lang"]
            word = item["word"]
            ctxs = item["contexts"]

            # 1. Source precision & micro-shifts
            src_vecs, src_nll = _extract_target_states_and_nll(model, tok, ctxs, word, layer)
            ctx_shift = _mean_pairwise_cos_dist(src_vecs)
            twonn = _twonn_dim(src_vecs)

            # 2. English-basis leftover recipe
            rec = decomposer.decompose(word, src_vecs.mean(0), k=5)
            sa_rate = float(np.mean([1.0 if ing.get("sound_alike") else 0.0 for ing in rec["ingredients"]])) if rec["ingredients"] else 0.0

            # 3. English translation spread + Cycle-Consistency (CC)
            fwd_prompts = [f"Translate from {lang} to English:\n{lang}: {c}\nEnglish:" for c in ctxs]
            en_texts, en_vecs = _greedy_translate_batch(model, tok, fwd_prompts, max_new=24)
            en_spread = _mean_pairwise_cos_dist(en_vecs)

            cyc_prompts = [f"Translate from English to {lang}:\nEnglish: {e}\n{lang}:" for e in en_texts]
            cyc_texts, cyc_vecs = _greedy_translate_batch(model, tok, cyc_prompts, max_new=24)

            d_src = _pairwise_dist_vec(src_vecs)
            d_cyc = _pairwise_dist_vec(cyc_vecs)
            if len(d_src) >= 3 and np.std(d_src) > 1e-8 and np.std(d_cyc) > 1e-8:
                rho = float(spearmanr(d_src, d_cyc).statistic)
                cc = float(max(0.0, min(2.0, 1.0 - (0.0 if np.isnan(rho) else rho))))
            else:
                cc = 1.0

            # 4. Directional Steerability / RIG
            rig_info = _measure_steerability_rig(model, tok, hook, fwd_prompts[0], layer, n_samples=5)

            # Composite sharpness: high micro-shift + low NLL + low sound-alike pull
            sharpness = float(ctx_shift * (1.0 / (1.0 + 0.15 * src_nll)) * (1.0 - 0.5 * sa_rate))

            row = {
                "id": item["id"],
                "word": word,
                "lang": lang,
                "group": item["group"],
                "contextual_shift": round(ctx_shift, 4),
                "twonn_dim": round(twonn, 4),
                "source_nll": round(src_nll, 4),
                "source_sharpness": round(sharpness, 4),
                "en_leftover": rec["leftover"],
                "en_ingredients": rec["ingredients"],
                "sound_alike_rate": round(sa_rate, 4),
                "en_semantic_spread": round(en_spread, 4),
                "cycle_consistency_cc": round(cc, 4),
                **rig_info,
            }
            rows.append(row)
            print(
                f"  [{item['group']:12s}] {word:14s} sharp={row['source_sharpness']:.3f} "
                f"nll={row['source_nll']:.2f} left={row['en_leftover']:.2f} "
                f"CC={row['cycle_consistency_cc']:.2f} RIG={row['rig_proxy']:.2f}",
                flush=True,
            )
    finally:
        hook.remove()

    out = {"model": model_key, "layer": layer, "rows": rows}
    out_path = ROOT / "results" / "explorer" / model_key / "unalignability_diag.json"
    save_json(out_path, out)
    print(f"Saved {len(rows)} diagnostic rows -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Untranslatability vs. Model Ignorance diagnostic.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    with Timer(f"diagnose_unalignability({args.model})"):
        run_diagnostic(args.model, limit=args.limit)


if __name__ == "__main__":
    main()
