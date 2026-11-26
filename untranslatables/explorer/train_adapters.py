"""Training routines for diversity-preserving multi-reference LoRA and continuous Spectral ReFT.

CLI:
    python -m explorer.train_adapters --mode lora|reft|axes --model tiny --steps 10 --out out.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from explorer.adapters import (
    LoRALinear,
    SpectralReFTController,
    inject_lora,
    lora_disabled,
    lora_parameters,
    save_lora,
)
from explorer.frame import build_prompt, load as load_frame_model
from explorer.global_axes import default_layers, extract_global_axes, load_tone_axes_dataset
from ut import arch
from ut.common import seed_all


def _model_device(model) -> torch.device:
    return arch.text_model(model).embed_tokens.weight.device


def _prepare_examples(
    tok,
    dataset: dict,
    axis_names: list[str] | None = None,
    prompt_style: str = "legacy",
) -> list[list[dict]]:
    """Pre-tokenize all items grouped by source sentence so sampling is uniform over items and readings."""
    from explorer.frame import build_doc_prompt

    grouped: list[list[dict]] = []
    for item in dataset["items"]:
        if prompt_style == "doc":
            prompt_ids, start = build_doc_prompt(tok, item["source"])
            lead = ""
        else:
            prompt_ids, start = build_prompt(tok, item["source"])
            lead = " "
        entries: list[dict] = []
        for tr in item["translations"]:
            cont_ids = tok(lead + tr["text"], add_special_tokens=False)["input_ids"]
            if not cont_ids:
                continue
            entry: dict = {
                "prompt_ids": prompt_ids,
                "start": start,
                "cont_ids": cont_ids,
            }
            if axis_names is not None:
                coords = tr.get("coords", {})
                entry["z"] = torch.tensor(
                    [float(coords.get(a, 0.0)) for a in axis_names],
                    dtype=torch.float32,
                )
            entries.append(entry)
        if entries:
            grouped.append(entries)
    if not grouped:
        raise ValueError("No valid training examples found in dataset")
    return grouped


def _translation_logprobs(model, prompt_ids: list[int], start: int, cont_ids: list[int]) -> torch.Tensor:
    """Return log_softmax logits (1, L_cont, V) over translation positions only."""
    dev = _model_device(model)
    seq = prompt_ids + cont_ids
    x = torch.tensor([seq], device=dev, dtype=torch.long)
    h = arch.forward_hidden(model, input_ids=x).last_hidden_state
    h_pred = h[:, start - 1 : start - 1 + len(cont_ids), :]
    return arch.logits(model, h_pred).log_softmax(-1)


def _translation_nll(logprobs: torch.Tensor, cont_ids: list[int]) -> torch.Tensor:
    target = torch.tensor([cont_ids], device=logprobs.device, dtype=torch.long)
    tok_lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -tok_lp.mean()


def train_multiref_lora(
    model,
    tok,
    dataset: dict | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    lr: float = 2e-4,
    steps: int = 100,
    kl_beta: float = 0.1,
    seed: int = 0,
    save_path: str | Path | None = None,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "o_proj"),
    layers: list[int] | None = None,
    prompt_style: str = "legacy",
) -> dict:
    """Diversity-preserving multi-reference SFT with optional KL anchor to the base model.

    Samples uniformly across source items and uniformly across all contrasting translations
    of each item, minimizing translation-token NLL + kl_beta * KL(p_base || p_lora) where
    p_base is computed cleanly via `with lora_disabled(model):`.
    """
    seed_all(seed)
    rng = random.Random(seed)
    if dataset is None:
        dataset = load_tone_axes_dataset()

    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        inject_lora(model, rank=rank, alpha=alpha, target_modules=target_modules, layers=layers)
    else:
        model.requires_grad_(False)
        for p in lora_parameters(model):
            p.requires_grad_(True)

    params = lora_parameters(model)
    if not params:
        raise ValueError("No trainable LoRA parameters found in model")

    grouped = _prepare_examples(tok, dataset, prompt_style=prompt_style)
    opt = torch.optim.AdamW(params, lr=lr)

    losses: list[float] = []
    kl_losses: list[float] = []

    with torch.enable_grad():
        for _ in range(int(steps)):
            item_trs = rng.choice(grouped)
            ex = rng.choice(item_trs)
            prompt_ids, start, cont_ids = ex["prompt_ids"], ex["start"], ex["cont_ids"]

            opt.zero_grad(set_to_none=True)
            logp_lora = _translation_logprobs(model, prompt_ids, start, cont_ids)
            nll = _translation_nll(logp_lora, cont_ids)

            if kl_beta > 0.0:
                with torch.no_grad(), lora_disabled(model):
                    logp_base = _translation_logprobs(model, prompt_ids, start, cont_ids)
                    p_base = logp_base.exp()
                kl = (p_base * (logp_base - logp_lora)).sum(-1).mean()
                loss = nll + float(kl_beta) * kl
                kl_val = float(kl.detach().item())
            else:
                loss = nll
                kl_val = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses.append(float(loss.detach().item()))
            kl_losses.append(kl_val)

    saved: str | None = None
    if save_path is not None:
        saved = str(save_lora(model, save_path))

    return {
        "losses": losses,
        "kl_losses": kl_losses,
        "final_loss": losses[-1] if losses else 0.0,
        "saved": saved,
    }


def _calibrate_reft_delta_norm(model, prompt_ids: list[int], cont_ids: list[int], layers: list[int]) -> float:
    """Estimate a safe, scale-invariant SpectralReFT norm bound (10% of median hooked hidden norm, floor 5.0)."""
    dev = _model_device(model)
    x = torch.tensor([prompt_ids + cont_ids[:16]], device=dev, dtype=torch.long)
    norms: list[float] = []
    with torch.no_grad():
        out = arch.forward_hidden(model, input_ids=x, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            for l_idx in layers:
                if 0 <= int(l_idx) + 1 < len(hs):
                    h_l = hs[int(l_idx) + 1][0].float()
                    norms.append(float(h_l.norm(dim=-1).median().item()))
    if not norms:
        return 5.0
    med_norm = float(sorted(norms)[len(norms) // 2])
    if not (med_norm > 0.0) or med_norm != med_norm:
        return 5.0
    return float(max(5.0, round(0.10 * med_norm, 2)))


def train_spectral_reft(
    model,
    tok,
    dataset: dict | None = None,
    layers: list[int] | None = None,
    rank: int = 8,
    lr: float = 1e-3,
    steps: int = 100,
    seed: int = 0,
    save_path: str | Path | None = None,
    max_delta_norm: float | None = None,
    contrast_weight: float = 0.2,
    margin: float = 1.0,
    prompt_style: str = "legacy",
) -> dict:
    """Train continuous Spectral ReFT blocks conditioned on tone/practice slider coordinates.

    Freezes the base model, attaches SpectralReFTController across `layers`, and trains on
    (source, translation, z_target) triples using translation NLL at z = z_target plus a
    gentle contrastive margin loss ensuring z = -z_target does not raise the likelihood
    of the positive-coordinate translation.
    """
    seed_all(seed)
    rng = random.Random(seed)
    if dataset is None:
        dataset = load_tone_axes_dataset()

    model.requires_grad_(False)
    axis_names = list(dataset["axes"].keys())
    if layers is None:
        layers = default_layers(model)

    grouped = _prepare_examples(tok, dataset, axis_names=axis_names, prompt_style=prompt_style)
    if max_delta_norm is None:
        sample_ex = grouped[0][0]
        resolved_max_delta = _calibrate_reft_delta_norm(
            model, sample_ex["prompt_ids"], sample_ex["cont_ids"], layers
        )
    else:
        resolved_max_delta = float(max_delta_norm)

    controller = SpectralReFTController(
        model=model,
        layers=layers,
        axis_names=axis_names,
        rank=rank,
        max_delta_norm=resolved_max_delta,
    )
    params = list(controller.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    losses: list[float] = []
    try:
        with torch.enable_grad():
            for _ in range(int(steps)):
                item_trs = rng.choice(grouped)
                ex = rng.choice(item_trs)
                prompt_ids, start, cont_ids, z_target = (
                    ex["prompt_ids"],
                    ex["start"],
                    ex["cont_ids"],
                    ex["z"],
                )
                edit_pos = max(start - 1, 0)

                opt.zero_grad(set_to_none=True)
                use_contrast = contrast_weight > 0.0 and float(z_target.abs().sum().item()) > 0.0
                nll_neg_val: float | None = None
                if use_contrast:
                    with torch.no_grad():
                        controller.set_coords(-z_target, start_pos=edit_pos)
                        nll_neg_val = float(
                            _translation_nll(
                                _translation_logprobs(model, prompt_ids, start, cont_ids), cont_ids
                            ).item()
                        )

                controller.set_coords(z_target, start_pos=edit_pos)
                logp_pos = _translation_logprobs(model, prompt_ids, start, cont_ids)
                nll_pos = _translation_nll(logp_pos, cont_ids)
                nll_pos_val = float(nll_pos.detach().item())

                margin_active = use_contrast and (float(margin) + nll_pos_val - float(nll_neg_val)) > 0.0
                if margin_active:
                    ((1.0 + float(contrast_weight)) * nll_pos).backward()
                    del logp_pos, nll_pos
                    controller.set_coords(-z_target, start_pos=edit_pos)
                    logp_neg = _translation_logprobs(model, prompt_ids, start, cont_ids)
                    nll_neg = _translation_nll(logp_neg, cont_ids)
                    (-float(contrast_weight) * nll_neg).backward()
                    loss_val = nll_pos_val + float(contrast_weight) * (
                        float(margin) + nll_pos_val - float(nll_neg.detach().item())
                    )
                    del logp_neg, nll_neg
                else:
                    nll_pos.backward()
                    loss_val = nll_pos_val
                    del logp_pos, nll_pos

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                losses.append(loss_val)
    finally:
        controller.set_coords(None)

    saved: str | None = None
    if save_path is not None:
        saved = str(controller.save(save_path))

    return {
        "losses": losses,
        "controller": controller,
        "saved": saved,
    }


def _parse_layers(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train Phase 3 adapters or extract global tone axes.")
    ap.add_argument("--mode", choices=("lora", "reft", "axes"), required=True)
    ap.add_argument("--model", default="tiny", help="Model key from ut.common.MODELS (e.g. tiny, qwen05, g3_4b_pt)")
    ap.add_argument("--dataset", default=None, help="Optional path to tone_axes.yaml")
    ap.add_argument("--layers", default=None, help="Comma-separated layer indices (default: auto)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--kl-beta", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--prompt-style", choices=("legacy", "doc"), default="legacy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output path (.pt for lora/reft, .npz for axes)")
    args = ap.parse_args(argv)

    tok, model = load_frame_model(args.model)
    dataset = load_tone_axes_dataset(args.dataset)
    layers = _parse_layers(args.layers)

    if args.mode == "lora":
        lr = args.lr if args.lr is not None else 2e-4
        res = train_multiref_lora(
            model,
            tok,
            dataset=dataset,
            rank=args.rank,
            alpha=args.alpha,
            lr=lr,
            steps=args.steps,
            kl_beta=args.kl_beta,
            seed=args.seed,
            save_path=args.out,
            layers=layers,
            prompt_style=args.prompt_style,
        )
        print(json.dumps({"mode": "lora", "steps": args.steps, "final_loss": res["final_loss"], "saved": res["saved"]}))
    elif args.mode == "reft":
        lr = args.lr if args.lr is not None else 1e-3
        res = train_spectral_reft(
            model,
            tok,
            dataset=dataset,
            layers=layers,
            rank=args.rank,
            lr=lr,
            steps=args.steps,
            seed=args.seed,
            save_path=args.out,
            prompt_style=args.prompt_style,
        )
        res["controller"].remove()
        print(json.dumps({
            "mode": "reft",
            "steps": args.steps,
            "max_delta_norm": res["controller"].max_delta_norm,
            "final_loss": res["losses"][-1] if res["losses"] else 0.0,
            "saved": res["saved"],
        }))
    else:
        bank = extract_global_axes(
            model,
            tok,
            dataset=dataset,
            layers=layers,
            eps=args.eps,
            max_items=args.max_items,
            prompt_style=args.prompt_style,
        )
        saved = str(bank.save(args.out)) if args.out else None
        print(json.dumps({
            "mode": "axes",
            "axes": bank.axis_names,
            "layers": bank.layers,
            "scales": bank.scales,
            "saved": saved,
        }))


if __name__ == "__main__":
    main()
