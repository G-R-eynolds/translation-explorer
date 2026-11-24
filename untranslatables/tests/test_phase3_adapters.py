"""CPU tests for Phase 3 (global tone axes, LoRA, Spectral ReFT, and training).

Run: .venv-cpu/bin/python -m pytest -q tests/test_phase3_adapters.py
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from blind.spectrum import Comparator
from explorer.adapters import (
    LoRALinear,
    SpectralReFTBlock,
    SpectralReFTController,
    inject_lora,
    load_lora,
    lora_disabled,
    lora_parameters,
    save_lora,
)
from explorer.directions import grads_at
from explorer.frame import build_prompt, load
from explorer.global_axes import (
    GlobalAxisBank,
    extract_global_axes,
    from_blind_comparator,
    grads_at_layers,
    load_tone_axes_dataset,
)
from explorer.train_adapters import main as train_cli_main
from explorer.train_adapters import train_multiref_lora, train_spectral_reft
from ut import arch


@pytest.fixture(scope="module")
def tiny():
    return load("tiny")


# ------------------------------------------------------------------ Dataset
def test_tone_axes_dataset_schema():
    ds = load_tone_axes_dataset()
    axes = ds["axes"]
    items = ds["items"]

    expected_axes = {
        "literal_vs_idiomatic",
        "plain_vs_poetic",
        "foreignising_vs_domesticating",
        "concise_vs_glossed",
    }
    assert set(axes.keys()) == expected_axes
    for name, desc in axes.items():
        assert isinstance(desc.get("neg"), str) and desc["neg"].strip()
        assert isinstance(desc.get("pos"), str) and desc["pos"].strip()

    assert len(items) >= 12
    assert len({it["id"] for it in items}) == len(items)

    pos_counts = {a: 0 for a in expected_axes}
    neg_counts = {a: 0 for a in expected_axes}
    for it in items:
        assert isinstance(it["source"], str) and it["source"].strip()
        trs = it["translations"]
        assert 4 <= len(trs) <= 6
        for tr in trs:
            assert isinstance(tr["text"], str) and tr["text"].strip()
            coords = tr["coords"]
            assert set(coords.keys()) == expected_axes
            for a, val in coords.items():
                fval = float(val)
                assert -1.0 <= fval <= 1.0, (it["id"], a, fval)
                if fval >= 0.5:
                    pos_counts[a] += 1
                if fval <= -0.5:
                    neg_counts[a] += 1

    for a in expected_axes:
        assert pos_counts[a] >= len(items)
        assert neg_counts[a] >= len(items)


# ------------------------------------------------------------------ Global axes & blind bridge
def test_grads_at_layers_matches_single_layer(tiny):
    tok, model = tiny
    ids, start = build_prompt(tok, "Hola.")
    conts = [[5, 6, 7], [8, 9]]
    b = start - 1
    multi = grads_at_layers(model, ids, conts, [0, 1], b)
    for l in (0, 1):
        single = grads_at(model, ids, conts, l, b)
        assert np.allclose(multi[l], single, atol=1e-5, rtol=1e-4)


def test_extract_global_axes_compose_and_roundtrip(tiny, tmp_path):
    tok, model = tiny
    d = arch.hidden_size(model)
    bank = extract_global_axes(model, tok, layers=[0, 1], max_items=3)

    assert len(bank.axis_names) == 4
    assert bank.layers == [0, 1]
    assert bank.model == "tiny"

    for axis in bank.axis_names:
        assert math.isfinite(bank.scales[axis]) and bank.scales[axis] > 0
        for l in (0, 1):
            v = bank.vectors[axis][l]
            assert v.shape == (d,)
            assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)

    # Compose at zero yields exact zero vectors
    zeros = bank.compose({})
    assert set(zeros.keys()) == {0, 1}
    assert all(np.all(zeros[l] == 0.0) for l in (0, 1))

    # Compose at non-zero slider values equals manual linear combination
    coords = {"literal_vs_idiomatic": 0.75, "plain_vs_poetic": -0.4}
    composed = bank.compose(coords)
    for l in (0, 1):
        expected = (
            0.75 * bank.scales["literal_vs_idiomatic"] * bank.vectors["literal_vs_idiomatic"][l]
            - 0.4 * bank.scales["plain_vs_poetic"] * bank.vectors["plain_vs_poetic"][l]
        )
        assert np.allclose(composed[l], expected, atol=1e-12)

    # Save / load round-trip
    out_path = tmp_path / "global_axes.npz"
    bank.save(out_path)
    assert out_path.exists()
    loaded = GlobalAxisBank.load(out_path)
    assert loaded.axis_names == bank.axis_names
    assert loaded.layers == bank.layers
    assert loaded.scales == bank.scales
    assert loaded.descriptions == bank.descriptions
    for axis in bank.axis_names:
        for l in (0, 1):
            assert np.allclose(loaded.vectors[axis][l], bank.vectors[axis][l], atol=1e-12)
    assert np.allclose(loaded.compose(coords)[0], composed[0], atol=1e-12)


def test_from_blind_comparator():
    rng = np.random.default_rng(42)
    d = 16
    n_pos = 3

    # Plant a strong source-sensitive direction that target is blind to
    u_pos = rng.normal(size=(n_pos, d))
    u_pos /= np.linalg.norm(u_pos)
    u_flat = u_pos.reshape(-1)

    src_blocks = rng.normal(scale=0.1, size=(24, n_pos * d)) + np.outer(rng.normal(scale=3.0, size=24), u_flat)
    tgt_blocks = rng.normal(scale=0.1, size=(24, n_pos * d))
    cmp = Comparator({"src": src_blocks, "plain": tgt_blocks})

    # Multi-position reduction to d_model
    bank_multi = from_blind_comparator(cmp, "src", "plain", layer=1, axis_name="blind_spot_1", top_k=2, d_model=d)
    assert bank_multi.axis_names == ["blind_spot_1", "blind_spot_2"]
    assert bank_multi.layers == [1]
    for name in bank_multi.axis_names:
        v = bank_multi.vectors[name][1]
        assert v.shape == (d,)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
        assert bank_multi.scales[name] > 0

    # Single-position Comparator (D == d)
    cmp_single = Comparator({"src": src_blocks[:, :d], "plain": tgt_blocks[:, :d]})
    bank_single = from_blind_comparator(cmp_single, "src", "plain", layer=0, axis_name="blind_duende", top_k=1)
    assert bank_single.axis_names == ["blind_duende"]
    assert bank_single.vectors["blind_duende"][0].shape == (d,)
    assert np.isclose(np.linalg.norm(bank_single.vectors["blind_duende"][0]), 1.0, atol=1e-6)

    # Merge two banks
    merged = bank_multi.merge(bank_single)
    assert set(merged.axis_names) == {"blind_spot_1", "blind_spot_2", "blind_duende"}
    assert merged.layers == [0, 1]


# ------------------------------------------------------------------ LoRA
def test_lora_identity_grads_merge_and_roundtrip(tiny, tmp_path):
    tok, base_model = tiny
    model = copy.deepcopy(base_model)
    x = torch.tensor([[1, 5, 6, 7, 8]])

    with torch.no_grad():
        logits_before = model(input_ids=x).logits.clone()

    injected = inject_lora(model, rank=4, alpha=8.0, target_modules=("q_proj", "v_proj", "o_proj"))
    assert len(injected) == arch.n_layers(model) * 3

    # Exact bit-identity at init (B == 0)
    with torch.no_grad():
        logits_after_init = model(input_ids=x).logits.clone()
    assert torch.equal(logits_before, logits_after_init)

    # Gradients flow ONLY to LoRA parameters
    lora_param_ids = {id(p) for p in lora_parameters(model)}
    assert len(lora_param_ids) == len(injected) * 2
    for p in model.parameters():
        if id(p) in lora_param_ids:
            assert p.requires_grad
        else:
            assert not p.requires_grad

    with torch.enable_grad():
        loss = model(input_ids=x).logits.pow(2).mean()
        loss.backward()

    for p in model.parameters():
        if id(p) not in lora_param_ids:
            assert p.grad is None

    # At init B=0 so lora_B must receive nonzero gradients
    b_grads = [m.lora_B.grad for m in model.modules() if isinstance(m, LoRALinear)]
    assert all(g is not None and g.abs().sum().item() > 0 for g in b_grads)

    # Perturb LoRA weights and verify lora_disabled + merge + save/load round-trip
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.2)

    with torch.no_grad():
        logits_adapted = model(input_ids=x).logits.clone()
    assert not torch.allclose(logits_before, logits_adapted, atol=1e-5)

    with torch.no_grad(), lora_disabled(model):
        logits_disabled = model(input_ids=x).logits.clone()
    assert torch.equal(logits_before, logits_disabled)

    # LoRALinear.merge() matches unmerged forward
    first_lora = next(m for m in model.modules() if isinstance(m, LoRALinear))
    inp_vec = torch.randn(2, first_lora.in_features)
    fused = first_lora.merge()
    assert torch.allclose(first_lora(inp_vec), fused(inp_vec), atol=1e-5)

    # Save / load round-trip on a fresh model copy
    ckpt = tmp_path / "lora.pt"
    save_lora(model, ckpt)
    assert ckpt.exists()

    model2 = copy.deepcopy(base_model)
    loaded_names = load_lora(model2, ckpt)
    assert len(loaded_names) == len(injected)
    with torch.no_grad():
        logits_loaded = model2(input_ids=x).logits.clone()
    assert torch.allclose(logits_adapted, logits_loaded, atol=1e-6)


# ------------------------------------------------------------------ Spectral ReFT
def test_spectral_reft_zero_identity_bounded_and_roundtrip(tiny, tmp_path):
    tok, base_model = tiny
    model = copy.deepcopy(base_model)
    d = arch.hidden_size(model)
    axis_names = ["a0", "a1", "a2", "a3"]

    # Unit test block norm bound and exact zero at z=0
    block = SpectralReFTBlock(d_model=d, n_axes=4, rank=4, max_delta_norm=2.5)
    h = torch.randn(2, 5, d)
    delta_zero = block(h, torch.zeros(4))
    assert torch.equal(delta_zero, torch.zeros_like(h))

    delta_huge = block(h, torch.full((4,), 1e4))
    norms = delta_huge.norm(dim=-1)
    assert float(norms.max().item()) <= 2.5 + 1e-5
    assert float(norms.min().item()) > 0.0

    # Gradients at z=0 must be finite (no NaN from norm clamp!)
    z_grad = torch.zeros(4, requires_grad=True)
    out_z = block(h, z_grad).sum()
    out_z.backward()
    assert torch.isfinite(z_grad.grad).all()

    # Controller on tiny model
    x = torch.tensor([[1, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        logits_base = model(input_ids=x).logits.clone()

    ctrl = SpectralReFTController(model, layers=[0, 1], axis_names=axis_names, rank=4, max_delta_norm=3.0)
    try:
        # Exact zero diff at z=0 (dict and tensor)
        ctrl.set_coords({a: 0.0 for a in axis_names}, start_pos=2)
        with torch.no_grad():
            logits_zero_dict = model(input_ids=x).logits.clone()
        assert torch.equal(logits_base, logits_zero_dict)

        ctrl.set_coords(torch.zeros(4), start_pos=2)
        with torch.no_grad():
            logits_zero_tensor = model(input_ids=x).logits.clone()
        assert torch.equal(logits_base, logits_zero_tensor)

        # Non-zero bounded diff at z != 0, respecting start_pos
        z_active = {"a0": 0.8, "a1": -0.5, "a2": 0.2, "a3": 1.0}
        ctrl.set_coords(z_active, start_pos=3)
        with torch.no_grad():
            logits_steered = model(input_ids=x).logits.clone()
        assert torch.equal(logits_base[:, :3], logits_steered[:, :3])
        assert not torch.allclose(logits_base[:, 3:], logits_steered[:, 3:], atol=1e-6)

        # Save / load round-trip on a fresh controller
        ckpt = tmp_path / "reft.pt"
        ctrl.save(ckpt)
        assert ckpt.exists()
    finally:
        ctrl.remove()

    # Removing hooks restores exact base output
    with torch.no_grad():
        assert torch.equal(logits_base, model(input_ids=x).logits)

    # Load into fresh controller via .load() and via .from_checkpoint()
    ctrl2 = SpectralReFTController(model, layers=[0, 1], axis_names=axis_names, rank=4, max_delta_norm=3.0)
    try:
        ctrl2.load(ckpt)
        ctrl2.set_coords(z_active, start_pos=3)
        with torch.no_grad():
            logits_loaded = model(input_ids=x).logits.clone()
        assert torch.allclose(logits_steered, logits_loaded, atol=1e-6)
    finally:
        ctrl2.remove()

    ctrl3 = SpectralReFTController.from_checkpoint(model, ckpt)
    try:
        ctrl3.set_coords(z_active, start_pos=3)
        with torch.no_grad():
            logits_from_ckpt = model(input_ids=x).logits.clone()
        assert torch.allclose(logits_steered, logits_from_ckpt, atol=1e-6)
    finally:
        ctrl3.remove()


# ------------------------------------------------------------------ Training smoke tests
def test_train_multiref_lora_smoke(tiny, tmp_path):
    tok, base_model = tiny
    model = copy.deepcopy(base_model)
    save_path = tmp_path / "multiref_lora.pt"

    res = train_multiref_lora(
        model,
        tok,
        rank=4,
        alpha=8.0,
        lr=1e-3,
        steps=3,
        kl_beta=0.1,
        seed=0,
        save_path=save_path,
    )
    assert len(res["losses"]) == 3
    assert len(res["kl_losses"]) == 3
    assert all(math.isfinite(x) for x in res["losses"])
    assert all(math.isfinite(x) and x >= -1e-6 for x in res["kl_losses"])
    # Step 0 has exact identity LoRA -> KL(p_base || p_lora) == 0
    assert res["kl_losses"][0] < 1e-6
    assert math.isfinite(res["final_loss"])
    assert save_path.exists()
    assert res["saved"] == str(save_path)


def test_train_spectral_reft_smoke(tiny, tmp_path):
    tok, base_model = tiny
    model = copy.deepcopy(base_model)
    save_path = tmp_path / "spectral_reft.pt"

    res = train_spectral_reft(
        model,
        tok,
        layers=[0, 1],
        rank=4,
        lr=1e-3,
        steps=3,
        seed=0,
        save_path=save_path,
    )
    ctrl = res["controller"]
    try:
        assert len(res["losses"]) == 3
        assert all(math.isfinite(x) for x in res["losses"])
        assert save_path.exists()
        assert res["saved"] == str(save_path)
    finally:
        ctrl.remove()


def test_train_adapters_cli_axes_smoke(tmp_path):
    out_path = tmp_path / "cli_axes.npz"
    train_cli_main(["--mode", "axes", "--model", "tiny", "--layers", "0,1", "--max-items", "2", "--out", str(out_path)])
    assert out_path.exists()
    bank = GlobalAxisBank.load(out_path)
    assert len(bank.axis_names) == 4
    assert bank.layers == [0, 1]
