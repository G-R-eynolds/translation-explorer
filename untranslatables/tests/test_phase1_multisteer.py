"""CPU tests for Phase 1 multi-layer steering & gradient-space readings.

Run from untranslatables/:
    .venv-cpu/bin/python -m pytest -q tests/test_phase1_multisteer.py
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from explorer.frame import build_prompt
from explorer.grad_readings import (
    cluster_gradients,
    cluster_hybrid,
    spectral_basis,
    whiten_gradients,
)
from explorer.multisteer import (
    MultiLayerCursorHook,
    default_layer_band,
    fast_calibrate,
    layer_weights,
    measured_kl_multi,
    multi_layer_grads_at,
)
from ut import arch


@pytest.fixture(scope="module")
def tiny():
    from explorer.frame import load
    return load("tiny")


def _capture_layers(model, layers, run):
    rec = {l: [] for l in layers}
    blocks = arch.layers(model)
    handles = [
        blocks[l].register_forward_hook(
            lambda m, i, o, l=l: rec[l].append((o[0] if isinstance(o, tuple) else o).detach().clone())
        )
        for l in layers
    ]
    try:
        run()
    finally:
        for h in handles:
            h.remove()
    return rec


def test_default_layer_band_and_weights(tiny):
    _, model = tiny
    band = default_layer_band(model)
    n = arch.n_layers(model)
    assert len(band) >= 1 and band == sorted(band)
    assert all(0 <= l < n for l in band)
    assert band == [0, 1]  # 2-layer tiny covers both layers

    # Works for 1-layer and deep (34-layer Gemma-like) configs
    for n_l in (1, 2, 12, 34):
        dummy = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=n_l))
        b = default_layer_band(dummy)
        assert len(b) >= 1 and b[0] >= 0 and b[-1] < n_l

    w_u = layer_weights([10, 11, 12, 13, 14], profile="uniform")
    assert np.isclose(sum(w_u.values()), 1.0)
    assert all(np.isclose(v, 0.2) for v in w_u.values())

    w_g = layer_weights([10, 11, 12, 13, 14], profile="gaussian")
    assert np.isclose(sum(w_g.values()), 1.0)
    assert w_g[12] > w_g[11] > w_g[10]
    assert np.isclose(w_g[10], w_g[14]) and np.isclose(w_g[11], w_g[13])
    assert layer_weights([3], profile="gaussian") == {3: 1.0}


def test_multilayer_cursor_hook_positions_and_additive(tiny):
    _, model = tiny
    layers = [0, 1]
    d = arch.hidden_size(model)
    x = torch.tensor([[1, 5, 6, 7, 8, 9]])

    def run():
        with torch.no_grad():
            out = model(input_ids=x, use_cache=True)
            pkv, nxt = out.past_key_values, torch.tensor([[10]])
            for _ in range(2):
                out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values

    base = _capture_layers(model, layers, run)
    hk = MultiLayerCursorHook(model, layers)

    try:
        # 1. local_vecs only (cursor vs only_at_b) across prefill + 2 KV-cache decode steps
        v0 = torch.full((d,), 0.4)
        v1 = torch.full((d,), -0.3)
        for only, want in ((False, [3, 4, 5, 6, 7]), (True, [3])):
            hk.set({0: v0, 1: v1}, b=3, only_at_b=only)
            steered = _capture_layers(model, layers, run)
            assert hk.pos == {0: 8, 1: 8}

            # Layer 0 receives unperturbed input, so its output delta is exact at every position
            d0 = [(steered[0][0] - base[0][0])[0, j] for j in range(6)]
            d0 += [(steered[0][i] - base[0][i])[0, 0] for i in (1, 2)]
            for j, dv in enumerate(d0):
                if j in want:
                    assert torch.allclose(dv, v0, atol=1e-5), (only, j)
                else:
                    assert dv.abs().max().item() < 1e-6, (only, j)

            # Layer 1 must NOT leak before position 3
            d1_pre = [(steered[1][0] - base[1][0])[0, j].abs().max().item() for j in range(3)]
            assert all(val < 1e-6 for val in d1_pre)
            assert (steered[1][0] - base[1][0])[0, 3].abs().max().item() > 1e-4

        # 2. Additive local_vecs (b=4) + global_vecs (global_start=2) without leaking before 2 or 4
        gl0 = torch.linspace(0.1, 0.2, d)
        lo0 = torch.linspace(-0.25, -0.05, d)
        hk.set(local_vecs={0: lo0, 1: v1}, b=4, only_at_b=False,
               global_vecs={0: gl0, 1: v0}, global_start=2)
        steered = _capture_layers(model, layers, run)
        assert hk.pos == {0: 8, 1: 8}

        d0 = [(steered[0][0] - base[0][0])[0, j] for j in range(6)]
        d0 += [(steered[0][i] - base[0][i])[0, 0] for i in (1, 2)]
        for j in (0, 1):
            assert d0[j].abs().max().item() < 1e-6, f"leaked before global_start at pos {j}"
            assert (steered[1][0] - base[1][0])[0, j].abs().max().item() < 1e-6
        for j in (2, 3):
            assert torch.allclose(d0[j], gl0, atol=1e-5), f"expected global only at pos {j}"
        for j in (4, 5, 6, 7):
            assert torch.allclose(d0[j], gl0 + lo0, atol=1e-5), f"expected global+local at pos {j}"

        # 3. Single 1D vector broadcasts uniformly (1/len(layers)) across layers
        hk.set(local_vecs=v0, b=5, only_at_b=True)
        steered = _capture_layers(model, layers, run)
        assert torch.allclose((steered[0][0] - base[0][0])[0, 5], 0.5 * v0, atol=1e-5)
        assert (steered[0][0] - base[0][0])[0, 4].abs().max().item() < 1e-6

        # 4. clear() restores baseline identically
        hk.clear()
        cleared = _capture_layers(model, layers, run)
        for l in layers:
            for t_idx in range(3):
                assert torch.allclose(cleared[l][t_idx], base[l][t_idx], atol=1e-6)
    finally:
        hk.remove()
    assert hk.handles == []


def test_multilayer_cursor_hook_sync_pos_with_kv_cache(tiny):
    _, model = tiny
    layers = [0, 1]
    d = arch.hidden_size(model)
    v0_loc = torch.full((d,), 0.35)
    v1_loc = torch.full((d,), -0.25)
    v0_glb = torch.full((d,), 0.15)
    v1_glb = torch.full((d,), 0.10)

    hk = MultiLayerCursorHook(model, layers)
    try:
        # Uncached 6-token reference with global_start=2 and local b=4
        x_full = torch.tensor([[1, 2, 3, 4, 5, 6]])
        hk.set(
            local_vecs={0: v0_loc, 1: v1_loc},
            b=4,
            only_at_b=False,
            global_vecs={0: v0_glb, 1: v1_glb},
            global_start=2,
        )
        with torch.no_grad():
            ref_logits = model(input_ids=x_full, use_cache=False).logits[:, -1, :]

        # Cached prefill (4 tokens) + multi-token cached append (T=2 at pos=4) via sync_pos
        hk.set(
            local_vecs={0: v0_loc, 1: v1_loc},
            b=4,
            only_at_b=False,
            global_vecs={0: v0_glb, 1: v1_glb},
            global_start=2,
        )
        hk.sync_pos(0)
        with torch.no_grad():
            out0 = model(input_ids=x_full[:, :4], use_cache=True)
            assert hk.pos == {0: 4, 1: 4}
            hk.sync_pos(4)
            out1 = model(input_ids=x_full[:, 4:6], past_key_values=out0.past_key_values, use_cache=True)
            assert hk.pos == {0: 6, 1: 6}
            cached_logits = out1.logits[:, -1, :]
        assert torch.allclose(cached_logits, ref_logits, atol=1e-5)

        # Cropped-cache single-token replay at pos=3 (global applies, local b=4 must NOT leak)
        with torch.no_grad():
            hk.clear()
            base_out = model(input_ids=x_full[:, :3], use_cache=True)
            base_l0 = _capture_layers(
                model,
                [0],
                lambda: model(input_ids=x_full[:, 3:4], past_key_values=base_out.past_key_values, use_cache=False),
            )[0][0]

        hk.set(
            local_vecs={0: v0_loc, 1: v1_loc},
            b=4,
            only_at_b=False,
            global_vecs={0: v0_glb, 1: v1_glb},
            global_start=2,
        )
        with torch.no_grad():
            hk.sync_pos(0)
            pre3 = model(input_ids=x_full[:, :3], use_cache=True).past_key_values
            hk.sync_pos(3)
            steered_l0 = _capture_layers(
                model,
                [0],
                lambda: model(input_ids=x_full[:, 3:4], past_key_values=pre3, use_cache=False),
            )[0][0]
            assert hk.pos == {0: 4, 1: 4}
        # Position 3 received global_vecs only (not local_vecs which starts at b=4)
        assert torch.allclose((steered_l0 - base_l0)[0, 0], v0_glb, atol=1e-5)
    finally:
        hk.remove()



def test_multi_layer_grads_at_matches_fd(tiny):
    from explorer.directions import grads_at
    from explorer.sample import score

    tok, model = tiny
    ids, _ = build_prompt(tok, "Hola.")
    conts = [[5, 6, 7], [8, 9], [10, 11, 12]]
    layers = [0, 1]
    b = len(ids) - 1

    G_multi = multi_layer_grads_at(model, ids, conts, layers, b, batch=2)
    assert set(G_multi.keys()) == set(layers)
    d = arch.hidden_size(model)
    for l in layers:
        assert G_multi[l].shape == (len(conts), d)
        assert G_multi[l].dtype == np.float64
        # Exact agreement with single-layer grads_at
        G_single = grads_at(model, ids, conts, l, b, batch=2)
        assert np.allclose(G_multi[l], G_single, rtol=1e-5, atol=1e-6)

    # Central finite differences simultaneously across ALL layers
    rng = np.random.default_rng(7)
    us = {l: rng.normal(size=d) for l in layers}
    us = {l: u / np.linalg.norm(u) for l, u in us.items()}
    eps = 1e-3
    hk = MultiLayerCursorHook(model, layers)
    try:
        hk.set({l: torch.tensor(eps * us[l], dtype=torch.float32) for l in layers}, b=b, only_at_b=True)
        lp_p = score(model, ids, conts)
        hk.set({l: torch.tensor(-eps * us[l], dtype=torch.float32) for l in layers}, b=b, only_at_b=True)
        lp_m = score(model, ids, conts)
    finally:
        hk.remove()
    fd = (lp_p - lp_m) / (2 * eps)
    analytic = sum(G_multi[l] @ us[l] for l in layers)
    assert np.allclose(fd, analytic, rtol=0.05, atol=2e-3)


def test_fast_calibrate_hits_target(tiny):
    tok, model = tiny
    ids, _ = build_prompt(tok, "Hola.")
    seq = ids + [5, 6, 7, 8]
    b = len(ids) - 1
    layers = default_layer_band(model)
    d = arch.hidden_size(model)
    rng = np.random.default_rng(3)
    unit_vecs = {}
    for l in layers:
        v = rng.normal(size=d)
        unit_vecs[l] = v / np.linalg.norm(v)

    hk = MultiLayerCursorHook(model, layers)
    try:
        target = 0.003  # tiny random model saturates near ~0.007
        for mode in ("cursor", "at_b"):
            for a0 in (1.0, 0.05, 10.0):
                alpha, achieved, n_evals = fast_calibrate(
                    model, hk, seq, b, unit_vecs, target=target, mode=mode, a0=a0, max_evals=7, tol=0.05
                )
                assert n_evals <= 7
                assert alpha > 0.0
                assert abs(achieved - target) <= 0.1 * target, (mode, a0, achieved, n_evals)
                check = measured_kl_multi(
                    model, hk, seq, b, {l: alpha * v for l, v in unit_vecs.items()}, mode=mode
                )
                assert np.isclose(check, achieved, atol=1e-6)
    finally:
        hk.remove()


def test_whiten_gradients_matches_dense():
    rng = np.random.default_rng(42)
    n, d = 30, 50
    G = rng.normal(size=(n, d))
    # Inject a huge shared nuisance direction to test Fisher equalisation
    nuisance = rng.normal(size=d)
    nuisance /= np.linalg.norm(nuisance)
    G += 25.0 * rng.normal(size=(n, 1)) * nuisance[None, :]

    eps = 1e-2
    W = whiten_gradients(G, eps=eps)
    assert W.shape[0] == n and W.shape[1] <= min(n, d)
    assert np.allclose(np.linalg.norm(W, axis=1), 1.0, atol=1e-10)

    # Compare Gram matrix against dense d x d Fisher whitening
    G_hat = (G.T @ G) / n
    evals, evecs = np.linalg.eigh(G_hat)
    pos = evals[evals > 1e-12 * evals.max()]
    tr = float(pos.sum())
    eff_r = (tr * tr) / float((pos * pos).sum())
    lam = eps * tr / max(eff_r, 1.0)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 0.0) + lam)) @ evecs.T
    W_dense = G @ inv_sqrt
    W_dense /= np.linalg.norm(W_dense, axis=1, keepdims=True)
    assert np.allclose(W @ W.T, W_dense @ W_dense.T, atol=1e-10)


def test_grad_and_hybrid_clustering_poetry_blind_spot():
    from explorer.readings import readings_from

    rng = np.random.default_rng(123)
    d_grad, d_sem = 64, 32
    counts = (22, 18, 14)
    n_good = sum(counts)
    n_bad = 6
    n_total = n_good + n_bad  # 60 -> quantile(0.1) drops exactly the 6 lowest-logp tail samples

    # Plant 3 distinct gradient clusters + a strong shared nuisance direction
    centers = rng.normal(size=(3, d_grad))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    shared = rng.normal(size=d_grad)
    shared /= np.linalg.norm(shared)

    G_parts, true_labels = [], []
    for cid, cnt in enumerate(counts):
        pts = 10.0 * shared[None, :] + 1.5 * centers[cid][None, :] + 0.25 * rng.normal(size=(cnt, d_grad))
        G_parts.append(pts)
        true_labels.extend([cid] * cnt)
    # Tail garbage samples with very low logp
    G_parts.append(rng.normal(scale=5.0, size=(n_bad, d_grad)))
    true_labels.extend([-1] * n_bad)

    G = np.vstack(G_parts)
    true_labels = np.array(true_labels)
    lp = np.concatenate([rng.uniform(-1.0, -0.2, size=n_good), rng.uniform(-9.0, -5.0, size=n_bad)])

    # Case 1: Identical semantic embeddings (simulating all-mpnet-base-v2 collapse on poetry paraphrases)
    e0 = rng.normal(size=d_sem)
    e0 /= np.linalg.norm(e0)
    E_identical = np.tile(e0, (n_total, 1))

    assert readings_from(E_identical, lp, drop_q=0.1)["k"] == 1  # semantic blind spot confirmed

    res_w = cluster_gradients(G, lp, drop_q=0.1, whiten=True)
    assert res_w["method"] == "grad_whitened"
    assert res_w["n_dropped"] == n_bad
    assert res_w["k"] == 3 and len(res_w["readings"]) == 3
    lab_w = np.array(res_w["labels"])
    assert np.all(lab_w[n_good:] == -1)
    assert adjusted_rand_score(true_labels[:n_good], lab_w[:n_good]) > 0.95

    res_p = cluster_gradients(G, lp, drop_q=0.1, whiten=False)
    assert res_p["method"] == "grad_plain"
    assert res_p["k"] == 3 and adjusted_rand_score(true_labels[:n_good], np.array(res_p["labels"])[:n_good]) > 0.95

    res_hyb_id = cluster_hybrid(E_identical, G, lp, alpha_grad=0.5, drop_q=0.1, whiten=True)
    assert res_hyb_id["k"] == 3 and len(res_hyb_id["readings"]) == 3
    assert adjusted_rand_score(true_labels[:n_good], np.array(res_hyb_id["labels"])[:n_good]) > 0.95

    # Case 2: Pure random noise semantic embeddings
    E_noise = rng.normal(size=(n_total, d_sem))
    E_noise /= np.linalg.norm(E_noise, axis=1, keepdims=True)
    assert readings_from(E_noise, lp, drop_q=0.1)["k"] == 1

    res_hyb_noise = cluster_hybrid(E_noise, G, lp, alpha_grad=0.5, drop_q=0.1, whiten=True)
    assert res_hyb_noise["k"] == 3 and len(res_hyb_noise["readings"]) == 3
    assert adjusted_rand_score(true_labels[:n_good], np.array(res_hyb_noise["labels"])[:n_good]) > 0.95

    # Case 3: High-variance varying syntax nuisance (whiten=True succeeds where plain gradients fail)
    nuis = rng.normal(size=(1, d_grad))
    nuis /= np.linalg.norm(nuis, axis=1, keepdims=True)
    G_nuis_parts = [
        8.0 * rng.normal(size=(cnt, 1)) @ nuis + 2.0 * centers[cid][None, :] + 0.08 * rng.normal(size=(cnt, d_grad))
        for cid, cnt in enumerate(counts)
    ]
    G_nuis_parts.append(rng.normal(scale=5.0, size=(n_bad, d_grad)))
    G_nuis = np.vstack(G_nuis_parts)
    res_plain_nuis = cluster_gradients(G_nuis, lp, drop_q=0.1, whiten=False)
    res_white_nuis = cluster_gradients(G_nuis, lp, drop_q=0.1, whiten=True)
    ari_plain = adjusted_rand_score(true_labels[:n_good], np.array(res_plain_nuis["labels"])[:n_good])
    ari_white = adjusted_rand_score(true_labels[:n_good], np.array(res_white_nuis["labels"])[:n_good])
    assert ari_white > 0.90 and ari_white > ari_plain + 0.5


def test_spectral_basis():
    from explorer.directions import natural

    rng = np.random.default_rng(9)
    d = 48
    mus = rng.normal(size=(3, d))
    G0 = mus[0] + 0.2 * rng.normal(size=(15, d))
    G1 = mus[1] + 0.2 * rng.normal(size=(12, d))
    G2 = mus[2] + 0.2 * rng.normal(size=(10, d))
    G = np.vstack([G0, G1, G2])
    labels = np.array([0] * 15 + [1] * 12 + [2] * 10)
    readings = [0, 1, 2]

    for whiten in (True, False):
        sb = spectral_basis(G, labels, readings, n_dims=2, whiten=whiten)
        axes = sb["axes"]
        assert axes.shape == (2, d)
        assert np.allclose(axes @ axes.T, np.eye(2), atol=1e-10)
        assert 0.0 <= sb["variance_explained"] <= 1.0 + 1e-12
        assert set(sb["anchors"].keys()) == {0, 1, 2}
        assert set(sb["dots"].keys()) == {0, 1, 2}

        anc = np.array([sb["anchors"][r] for r in readings])
        assert anc.shape == (3, 2)
        assert np.isclose(np.linalg.norm(anc, axis=1).max(), 1.0, atol=1e-10)
        assert np.allclose(anc.mean(0), 0.0, atol=1e-10)

        for r, cnt in zip(readings, (15, 12, 10)):
            pts = np.array(sb["dots"][r])
            assert pts.shape == (cnt, 2)
            assert np.allclose(pts.mean(0), sb["anchors"][r], atol=1e-10)

    # With 2 readings and whiten=True, axis 0 aligns with explorer.directions.natural(G0, G1)
    G2r = np.vstack([G0, G1])
    lab2 = np.array([0] * len(G0) + [1] * len(G1))
    sb2 = spectral_basis(G2r, lab2, [0, 1], n_dims=2, whiten=True, eps=1e-2)
    assert sb2["axes"].shape == (2, d)
    assert np.allclose(sb2["axes"] @ sb2["axes"].T, np.eye(2), atol=1e-10)
    assert np.isclose(sb2["variance_explained"], 1.0, atol=1e-10)
    v_nat = natural(G0, G1, eps=1e-2)
    assert abs(float(np.dot(sb2["axes"][0], v_nat))) > 1.0 - 1e-6
    assert np.isclose(abs(sb2["anchors"][0][0]), 1.0, atol=1e-10)
    assert np.isclose(sb2["anchors"][0][1], 0.0, atol=1e-10)
