"""CPU tests for Phase 4 ForkHead side-head: .venv-cpu/bin/python -m pytest -q tests/test_phase4_fork_head.py"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from explorer.fork_head import (
    ForkHead,
    LiveForkScorer,
    extract_fork_dataset,
    load_dataset,
    save_dataset,
    train_fork_head,
)
from explorer.frame import build_prompt
from ut import arch


@pytest.fixture(scope="module")
def tiny():
    from explorer.frame import load
    return load("tiny")


def test_fork_head_shapes_orthonormality_and_params():
    torch.manual_seed(0)
    d_model = 64
    head = ForkHead(d_model=d_model, hidden_dim=32, n_dirs=2)
    head.eval()

    # 2D input: (B, d_model)
    h2 = torch.randn(5, d_model)
    out2 = head(h2)
    assert out2["fork_logit"].shape == (5,)
    assert out2["entropy"].shape == (5,)
    assert out2["fisher_top"].shape == (5,)
    assert out2["dirs"].shape == (5, 2, d_model)
    assert torch.all(out2["entropy"] >= 0)
    assert torch.all(out2["fisher_top"] >= 0)

    # Exact orthonormality: D @ D^T ≈ I_2
    D = out2["dirs"]
    gram = torch.matmul(D, D.transpose(-1, -2))
    eye = torch.eye(2).expand_as(gram)
    assert torch.allclose(gram, eye, atol=1e-5)

    # 3D input: (B, T, d_model) + differentiability through Gram-Schmidt
    head.train()
    h3 = torch.randn(2, 4, d_model, requires_grad=True)
    out3 = head(h3)
    assert out3["fork_logit"].shape == (2, 4)
    assert out3["dirs"].shape == (2, 4, 2, d_model)
    gram3 = torch.matmul(out3["dirs"], out3["dirs"].transpose(-1, -2))
    assert torch.allclose(gram3, torch.eye(2).expand_as(gram3), atol=1e-5)

    loss = (
        out3["fork_logit"].sum()
        + out3["entropy"].sum()
        + out3["fisher_top"].sum()
        + (out3["dirs"] ** 2).sum()
    )
    loss.backward()
    assert h3.grad is not None and torch.isfinite(h3.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())

    # Ultra-tiny <1M param head when n_dirs=0 even at Gemma-3-4B width (d_model=2560)
    tiny_head = ForkHead(d_model=2560, hidden_dim=256, n_dirs=0)
    assert tiny_head.predict_dirs is False
    assert tiny_head.param_count() == sum(p.numel() for p in tiny_head.parameters())
    assert tiny_head.param_count() < 1_000_000
    out_nodir = tiny_head(torch.randn(3, 2560))
    assert out_nodir["dirs"] is None


def test_fork_head_save_load_roundtrip(tmp_path):
    torch.manual_seed(1)
    for n_dirs in (0, 2):
        head = ForkHead(d_model=32, hidden_dim=16, n_dirs=n_dirs)
        head.eval()
        p = tmp_path / f"head_{n_dirs}.pt"
        head.save(p, meta={"layer": 17, "note": "roundtrip"})

        loaded = ForkHead.load(p)
        assert loaded.param_count() == head.param_count()
        assert loaded.config_dict() == head.config_dict()
        assert loaded.meta == {"layer": 17, "note": "roundtrip"}

        h = torch.randn(4, 32)
        with torch.no_grad():
            o1 = head(h)
            o2 = loaded(h)
        for k in ("fork_logit", "entropy", "fisher_top"):
            assert torch.allclose(o1[k], o2[k], atol=1e-6)
        if n_dirs == 0:
            assert o1["dirs"] is None and o2["dirs"] is None
        else:
            assert torch.allclose(o1["dirs"], o2["dirs"], atol=1e-6)


def test_extract_fork_dataset_on_tiny(tiny, tmp_path):
    tok, model = tiny
    d_model = arch.hidden_size(model)
    (tmp_path / "branches").mkdir()
    (tmp_path / "samples").mkdir()

    # Text 1: exact g3_4b_pt schema with samples/<id>.json + branches/<id>.json
    src1 = "Cuando canta tiene duende."
    y_ref1 = [10, 11, 12, 13, 14]
    y_other1 = [10, 20, 21, 22]
    sample1 = {
        "id": "synth_one",
        "text": src1,
        "kind": "hard",
        "ys": [y_other1, y_ref1],
        "texts": [tok.decode(y_other1).strip(), tok.decode(y_ref1).strip()],
        "logp": [-5.0, -2.0],
        "lens": [len(y_other1), len(y_ref1)],
        "labels": [1, 0],
        "readings": [0, 1],
        "shares": {"0": 0.5, "1": 0.5},
    }
    branch1 = {
        "id": "synth_one",
        "ref": tok.decode(y_ref1).strip(),
        "ref_tokens": [tok.decode([x]) for x in y_ref1],
        "t": 2,
        "specificity": 0.7,
        "why": "reference commits to its reading here",
        "entropy": [0.2, 0.4, 0.8, 2.1, 0.3],     # fork at t=2 (B["t"]) and t=3 (entropy >= 1.5)
        "fisher_top": [0.05, 0.08, 0.30, 0.45, 0.02],
    }
    (tmp_path / "samples" / "synth_one.json").write_text(json.dumps(sample1))
    (tmp_path / "branches" / "synth_one.json").write_text(json.dumps(branch1))

    # Text 2: second text with different length
    src2 = "El museo abre a las diez."
    y_ref2 = [15, 16, 17]
    sample2 = {
        "id": "synth_two",
        "text": src2,
        "kind": "control",
        "ys": [y_ref2],
        "texts": [tok.decode(y_ref2).strip()],
        "logp": [-1.5],
        "lens": [len(y_ref2)],
        "labels": [0],
        "readings": [0],
        "shares": {"0": 1.0},
    }
    branch2 = {
        "id": "synth_two",
        "ref": tok.decode(y_ref2).strip(),
        "ref_tokens": [tok.decode([x]) for x in y_ref2],
        "t": 1,
        "specificity": None,
        "why": "no reading-specific token: max output Fisher",
        "entropy": [0.1, 1.6, 0.2],               # fork at t=1 (both B["t"] and entropy >= 1.5)
        "fisher_top": [0.01, 0.25, 0.03],
    }
    (tmp_path / "samples" / "synth_two.json").write_text(json.dumps(sample2))
    (tmp_path / "branches" / "synth_two.json").write_text(json.dumps(branch2))

    ds = extract_fork_dataset(model, tok, tmp_path, layer=0, entropy_fork_thresh=1.5)
    assert ds["H"].shape == (8, d_model)
    assert ds["y_fork"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert np.allclose(ds["y_entropy"].tolist(), branch1["entropy"] + branch2["entropy"], atol=1e-6)
    assert np.allclose(ds["y_fisher"].tolist(), branch1["fisher_top"] + branch2["fisher_top"], atol=1e-6)
    assert ds["meta"]["layer"] == 0
    assert ds["meta"]["n_texts"] == 2
    assert ds["meta"]["n_tokens"] == 8

    # Verify extracted hidden states match manual forward_hidden at h[start + t - 1]
    ids1, start1 = build_prompt(tok, src1)
    x1 = torch.tensor([ids1 + y_ref1], dtype=torch.long)
    with torch.no_grad():
        out1 = arch.forward_hidden(model, input_ids=x1, attention_mask=torch.ones_like(x1), output_hidden_states=True)
        h_expected = out1.hidden_states[1][0, start1 - 1 : start1 - 1 + len(y_ref1)].float()
    assert torch.allclose(ds["H"][:5], h_expected, atol=1e-5)

    # save_dataset / load_dataset round-trip
    ds_path = tmp_path / "ds.pt"
    save_dataset(ds, ds_path)
    ds2 = load_dataset(ds_path)
    assert torch.allclose(ds2["H"], ds["H"])
    assert ds2["meta"] == ds["meta"]


def _make_synthetic_dataset(n: int = 240, d_model: int = 32, with_dirs: bool = False, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    # Create an orthonormal basis in R^{d_model} for clean, linearly separable clusters
    Q, _ = torch.linalg.qr(torch.randn(d_model, 4, generator=g))
    w_fork, w_aux, d_a, d_b = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]

    # Imbalanced (~25% fork tokens) with a clean linear margin along w_fork
    n_pos = n // 4
    n_neg = n - n_pos
    y_fork = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
    shift = torch.where(y_fork > 0.5, 2.2, -1.0)[:, None] * w_fork[None, :]
    H = shift + 0.5 * torch.randn(n, d_model, generator=g)

    proj = H @ w_fork
    aux = H @ w_aux
    y_entropy = torch.nn.functional.softplus(1.1 * proj + 0.3 * aux + 0.6)
    y_fisher = torch.nn.functional.softplus(0.5 * proj + 0.2)

    ds = {
        "H": H,
        "y_fork": y_fork,
        "y_entropy": y_entropy,
        "y_fisher": y_fisher,
        "meta": {"layer": 0, "d_model": d_model},
    }
    if with_dirs:
        # Smooth input-conditioned 2D orthonormal subspace (rotates between fork and non-fork regimes)
        gate = torch.sigmoid(2.0 * proj)[:, None]
        u1 = gate * d_a[None, :] + (1.0 - gate) * w_aux[None, :]
        u1 = u1 / u1.norm(dim=-1, keepdim=True)
        u2 = gate * d_b[None, :] + (1.0 - gate) * w_fork[None, :]
        u2 = u2 - (u2 * u1).sum(dim=-1, keepdim=True) * u1
        u2 = u2 / u2.norm(dim=-1, keepdim=True)
        ds["y_dirs"] = torch.stack([u1, u2], dim=1)  # (N, 2, d_model)
    return ds


def test_train_fork_head_converges_and_saves(tmp_path):
    ds = _make_synthetic_dataset(n=240, d_model=32, with_dirs=False, seed=7)

    # Naive untrained baseline entropy MAE
    torch.manual_seed(0)
    untrained = ForkHead(d_model=32, hidden_dim=64, n_dirs=0).eval()
    with torch.no_grad():
        init_ent_mae = float((untrained(ds["H"])["entropy"] - ds["y_entropy"]).abs().mean().item())

    ckpt = tmp_path / "fork_head.pt"
    res = train_fork_head(
        ds,
        hidden_dim=64,
        n_dirs=0,
        epochs=60,
        lr=3e-3,
        val_frac=0.2,
        seed=0,
        save_path=ckpt,
    )
    m = res["val_metrics"]
    assert m["accuracy"] > 0.90, f"Expected >90% accuracy, got {m}"
    assert m["auroc_or_acc"] > 0.95, f"Expected >0.95 AUROC, got {m}"
    assert m["entropy_mae"] < 0.25 * init_ent_mae, f"Entropy MAE did not drop sharply: {m['entropy_mae']} vs {init_ent_mae}"
    assert m["fisher_mae"] < 0.20
    assert res["saved"] == str(ckpt) and ckpt.exists()

    reloaded = ForkHead.load(ckpt)
    assert reloaded.param_count() == res["head"].param_count()


def test_train_fork_head_with_dir_supervision():
    ds = _make_synthetic_dataset(n=200, d_model=24, with_dirs=True, seed=11)
    res = train_fork_head(
        ds,
        hidden_dim=64,
        n_dirs=2,
        epochs=80,
        lr=3e-3,
        val_frac=0.2,
        seed=0,
    )
    head = res["head"]
    m = res["val_metrics"]
    assert head.predict_dirs is True
    assert m["accuracy"] > 0.90
    assert m["dir_subspace_sim"] > 0.90, f"Expected strong subspace alignment, got {m}"

    with torch.no_grad():
        pred = head(ds["H"][:10])
    D = pred["dirs"]
    assert D.shape == (10, 2, 24)
    gram = torch.matmul(D, D.transpose(-1, -2))
    assert torch.allclose(gram, torch.eye(2).expand_as(gram), atol=1e-5)


def test_live_fork_scorer_prefill_and_kv_decode(tiny):
    tok, model = tiny
    d_model = arch.hidden_size(model)
    x_prefill = torch.tensor([[1, 5, 6, 7, 8, 9]], dtype=torch.long)
    decode_tokens = [10, 11, 12]

    # Reference logits without any scorer attached
    with torch.no_grad():
        ref_out = model(input_ids=x_prefill, use_cache=True)
        ref_logits = ref_out.logits.detach().clone()

    for n_dirs in (0, 2):
        head = ForkHead(d_model=d_model, hidden_dim=32, n_dirs=n_dirs)
        scorer = LiveForkScorer(head, model, layer=0)
        assert scorer.latest() is None and scorer.history() == []

        with torch.no_grad():
            out = model(input_ids=x_prefill, use_cache=True)
            assert torch.allclose(out.logits, ref_logits, atol=1e-6)
            pkv = out.past_key_values
            for tok_id in decode_tokens:
                step_in = torch.tensor([[tok_id]], dtype=torch.long)
                out = model(input_ids=step_in, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values

        hist = scorer.history()
        assert len(hist) == 4
        assert scorer.latest() == hist[-1]
        assert [entry["seq_len"] for entry in hist] == [6, 1, 1, 1]
        assert [entry["step"] for entry in hist] == [0, 1, 2, 3]
        for entry in hist:
            assert 0.0 <= entry["fork_prob"] <= 1.0
            assert entry["pred_entropy"] >= 0.0
            assert entry["pred_fisher_top"] >= 0.0
            if n_dirs == 0:
                assert entry["dirs"] is None
            else:
                D = entry["dirs"]
                assert D.shape == (2, d_model)
                assert torch.allclose(D @ D.T, torch.eye(2), atol=1e-5)

        # reset clears recorded history
        scorer.reset()
        assert scorer.latest() is None and scorer.history() == []

        # remove detaches hook cleanly
        scorer.remove()
        scorer.remove()  # idempotent
        with torch.no_grad():
            model(input_ids=x_prefill, use_cache=False)
        assert scorer.history() == []


def test_cli_and_default_layer(tiny, tmp_path):
    from explorer.fork_head import main

    tok, model = tiny
    (tmp_path / "branches").mkdir()
    (tmp_path / "samples").mkdir()

    # Branch with matching sample
    y_ref = [10, 11, 12, 13]
    (tmp_path / "samples" / "case1.json").write_text(json.dumps({
        "id": "case1",
        "text": "Cuando canta tiene duende.",
        "ys": [y_ref],
        "logp": [-1.0],
        "labels": [0],
        "readings": [0],
    }))
    (tmp_path / "branches" / "case1.json").write_text(json.dumps({
        "id": "case1",
        "ref": tok.decode(y_ref).strip(),
        "ref_tokens": [tok.decode([x]) for x in y_ref],
        "t": 2,
        "entropy": [0.2, 0.3, 2.0, 0.1],
        "fisher_top": [0.05, 0.06, 0.4, 0.02],
    }))

    # Branch without samples file (tests text lookup / re-tokenization fallback + default layer = n_layers // 2)
    ref_text = " The house is near."
    ref_ids = tok(ref_text, add_special_tokens=False)["input_ids"]
    (tmp_path / "branches" / "case2.json").write_text(json.dumps({
        "id": "case2",
        "text": "La casa está cerca.",
        "ref": ref_text.strip(),
        "ref_tokens": [tok.decode([x]) for x in ref_ids],
        "t": 0,
        "entropy": [1.8] + [0.2] * (len(ref_ids) - 1),
        "fisher_top": [0.3] + [0.05] * (len(ref_ids) - 1),
    }))

    out_ckpt = tmp_path / "trained.pt"
    ds_cache = tmp_path / "dataset.pt"
    summary = main([
        "--model", "tiny",
        "--explorer-dir", str(tmp_path),
        "--epochs", "5",
        "--hidden-dim", "16",
        "--out", str(out_ckpt),
        "--dataset", str(ds_cache),
    ])
    assert summary["layer"] == arch.n_layers(model) // 2
    assert summary["n_tokens"] == len(y_ref) + len(ref_ids)
    assert out_ckpt.exists() and ds_cache.exists()
    loaded = ForkHead.load(out_ckpt)
    assert loaded.param_count() == summary["params"]

