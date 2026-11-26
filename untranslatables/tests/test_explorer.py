"""CPU tests for explorer/ (tiny random Llama):  .venv-cpu/bin/python -m pytest -q tests/test_explorer.py"""
import numpy as np
import pytest
import torch

from explorer.frame import build_prompt, cut, load_texts


@pytest.fixture(scope="module")
def tiny():
    from explorer.frame import load
    return load("tiny")


def test_frame_and_cut(tiny):
    tok, _ = tiny
    ids, start = build_prompt(tok, "Hola.")
    assert start == len(ids) and tok.decode(ids).rstrip().endswith("English:")
    y = tok(" I am here.\n\nSpanish: x", add_special_tokens=False)["input_ids"]
    assert tok.decode(cut(tok, y)).strip() == "I am here."
    y2 = tok(" No stop here", add_special_tokens=False)["input_ids"]
    assert cut(tok, y2) == y2


def test_texts():
    t = load_texts()
    assert len(t) == 9 and sum(x["kind"] == "control" for x in t) == 1
    assert len({x["id"] for x in t}) == 9


def _block_outputs(model, layer, run):
    from ut import arch
    rec = []
    h = arch.layers(model)[layer].register_forward_hook(lambda m, i, o: rec.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
    run()
    h.remove()
    return rec


def test_cursor_hook_positions(tiny):
    from explorer.steer import CursorHook
    tok, model = tiny
    x = torch.tensor([[1, 5, 6, 7, 8, 9]])

    def run():
        with torch.no_grad():
            out = model(input_ids=x, use_cache=True)
            pkv, nxt = out.past_key_values, torch.tensor([[10]])
            for _ in range(2):
                out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values

    d = model.config.hidden_size
    v = torch.ones(d) * 0.5
    base = _block_outputs(model, 0, run)
    hk = CursorHook(model, 0)
    for only, want in ((False, [3, 4, 5, 6, 7]), (True, [3])):
        hk.set(v, 3, only_at_b=only)
        steered = _block_outputs(model, 0, run)   # recorded AFTER our hook (registered later) -> includes the add
        diffs = [(steered[0] - base[0])[0, j].abs().max().item() for j in range(6)]
        diffs += [(steered[i] - base[i]).abs().max().item() for i in (1, 2)]
        got = [j for j, dv in enumerate(diffs) if dv > 1e-6]
        assert got[0] == 3 and (got == want if only else set(want) <= set(got))
        assert all(diffs[j] < 1e-6 for j in range(3))
    hk.remove()


def test_sample_score_steered(tiny):
    from explorer.sample import sample, score, steered
    from explorer.steer import CursorHook
    tok, model = tiny
    ids, start = build_prompt(tok, "Hola.")
    ys = sample(model, tok, ids, 3, 6, seed=0)
    assert len(ys) == 3 and all(len(y) <= 6 for y in ys)
    y = next(y for y in ys if y) if any(ys) else [5, 6]
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids + y])).logits.float().log_softmax(-1)[0]
    manual = sum(lg[len(ids) - 1 + j, t].item() for j, t in enumerate(y))
    assert np.isclose(score(model, ids, [y])[0], manual, atol=1e-3)
    hk = CursorHook(model, 0)
    g0 = sample(model, tok, ids, 1, 6, temperature=0.0)
    assert steered(model, tok, hk, ids, None, start, 6) == g0
    hk.remove()


def test_cluster_planted():
    from explorer.readings import assign, centroids, cluster
    r = np.random.default_rng(0)
    mus = r.normal(size=(3, 40))
    X = np.vstack([mu + 0.15 * r.normal(size=(n, 40)) for mu, n in zip(mus, (30, 20, 14))])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    c = cluster(X)
    assert c["k"] == 3 and len(c["readings"]) == 3
    C = centroids(X, c["labels"], c["readings"])
    assert assign(X[0], C) == 0                     # largest planted cluster is reading 0
    noise = r.normal(size=(64, 40))
    assert cluster(noise / np.linalg.norm(noise, axis=1, keepdims=True))["k"] == 1


def test_natural_matches_dense():
    from explorer.directions import natural, alpha_for
    r = np.random.default_rng(0)
    GA, GB = r.normal(size=(20, 50)) + 0.3, r.normal(size=(20, 50))
    Z = np.vstack([GA, GB]); n = len(Z)
    G = Z.T @ Z / n
    lam = 1e-2 * np.trace(G) / n
    x = np.linalg.solve(G + lam * np.eye(50), GA.mean(0) - GB.mean(0))
    assert np.allclose(natural(GA, GB), x / np.linalg.norm(x), atol=1e-8)
    v = natural(GA, GB)
    a = alpha_for(Z, v, 0.5)
    assert np.isclose(0.5 * a * a * np.mean((Z @ v) ** 2), 0.5)


def test_grads_at_fd(tiny):
    from explorer.directions import grads_at
    from explorer.sample import score
    from explorer.steer import CursorHook
    tok, model = tiny
    ids, _ = build_prompt(tok, "Hola.")
    conts = [[5, 6, 7], [8, 9]]
    G = grads_at(model, ids, conts, 0, len(ids) - 1)
    u = np.random.default_rng(1).normal(size=G.shape[1]); u /= np.linalg.norm(u)
    hk = CursorHook(model, 0)
    t = 1e-3
    hk.set(torch.tensor(t * u, dtype=torch.float32), len(ids) - 1, only_at_b=True)
    lp_p = score(model, ids, conts)
    hk.set(torch.tensor(-t * u, dtype=torch.float32), len(ids) - 1, only_at_b=True)
    lp_m = score(model, ids, conts)
    hk.remove()
    assert np.allclose((lp_p - lp_m) / (2 * t), G @ u, rtol=0.05, atol=2e-3)


def test_choose_t_and_timeline(tiny):
    from explorer.branches import choose_t, timeline
    from explorer.branches import align_index
    # reference reading 0 commits at token 3 ("magic"=7 vs "spirit"=9); punctuation variants (5) do not matter
    mem = {0: [[1, 2, 3, 7, 4]] * 3 + [[1, 5, 2, 3, 7, 4]], 1: [[1, 2, 3, 9, 4]] * 2 + [[1, 5, 2, 3, 9, 4]]}
    assert choose_t([1, 2, 3, 7, 4], mem, 0, [0] * 5)[0] == 3
    assert choose_t([1, 2, 3], {0: [[1, 2, 3]], 1: [[1, 2, 3]]}, 0, [0.1, 0.5, 0.2])[0] == 1   # no specific token
    assert align_index([1, 2, 3, 7, 4], [1, 5, 2, 3, 9, 4], 3) == 4
    tok, model = tiny
    ids, _ = build_prompt(tok, "Hola.")
    tl = timeline(model, ids, [5, 6, 7], iters=5)
    assert len(tl["entropy"]) == 3 and all(v >= 0 for v in tl["fisher_top"])
    # power iteration agrees with the dense Fisher on the tiny model
    with torch.no_grad():
        from ut import arch
        H = arch.forward_hidden(model, input_ids=torch.tensor([ids + [5]])).last_hidden_state[0, len(ids) - 1]
        p = arch.logits(model, H[None])[0].softmax(-1).double()
        W = model.lm_head.weight.double()
        Wp = W.T @ p
        F = (W * p[:, None]).T @ W - torch.outer(Wp, Wp)
        assert abs(float(torch.linalg.eigvalsh(F)[-1]) - timeline(model, ids, [5], iters=200)["fisher_top"][0]) < 1e-3 * float(torch.linalg.eigvalsh(F)[-1]) + 1e-6


def test_end_to_end(tmp_path, monkeypatch):
    import explorer.run as R
    import explorer.report as REP
    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr(REP, "OUT", tmp_path)
    monkeypatch.setattr(R, "KLS", {"at_b": [0.5], "cursor": [0.1]})
    monkeypatch.setattr(R, "N_STEER", 2)
    # no embedder download in tests: hash-based fake embeddings; force two readings so the steering branch runs
    import explorer.readings as RD
    def fake_embed(texts):
        E = np.array([np.random.default_rng(abs(hash(t)) % 2 ** 32).normal(size=16) for t in texts])
        return E / np.linalg.norm(E, axis=1, keepdims=True)
    monkeypatch.setattr(RD, "embed_texts", fake_embed)
    real = RD.readings_from
    monkeypatch.setattr(RD, "readings_from", lambda E, lp, **k: {**real(E, lp, **k), "k": 2, "silhouette": 0.5,
                                                                 "labels": [i % 2 for i in range(len(E))],
                                                                 "readings": [0, 1], "shares": {0: 0.5, 1: 0.5}})
    R.run(dict(model="tiny", n=8, texts=["duende_bare", "control_museo"], layer=0))
    out = tmp_path / "tiny"
    for f in ["samples/duende_bare.json", "branches/duende_bare.json", "steering/duende_bare.jsonl", "summary.json"]:
        assert (out / f).exists(), f
    s = __import__("json").loads((out / "summary.json").read_text())
    assert {"n_hard_with_2plus", "steer_success_rate", "pass"} <= set(s)
    assert REP.run(dict(model="tiny")).exists()


def test_calibrate_hits_target(tiny):
    from explorer.directions import calibrate, measured_kl
    from explorer.steer import CursorHook
    tok, model = tiny
    ids, _ = build_prompt(tok, "Hola.")
    seq = ids + [5, 6, 7, 8]
    v = np.random.default_rng(0).normal(size=model.config.hidden_size); v /= np.linalg.norm(v)
    hk = CursorHook(model, 0)
    for mode in ("cursor", "at_b"):
        a = calibrate(model, hk, seq, len(ids) - 1, v, 0.003, mode, 1.0)   # tiny model saturates near 0.007
        got = measured_kl(model, hk, seq, len(ids) - 1, torch.tensor(a * v, dtype=torch.float32), mode)
        assert abs(got - 0.003) < 0.0003, (mode, got)
    hk.remove()


def test_normalize_text_and_raw_dispersion_gate():
    from explorer.readings import cluster, normalize_text, raw_dispersion, readings_from

    s = "Ten\u00a0o\u2019clock\u2026 \u201chello\u201d\r\n  line\u2003two  "
    assert normalize_text(s, collapse_newlines=False) == 'Ten o\'clock... "hello"\nline two'
    assert normalize_text(s, collapse_newlines=True) == 'Ten o\'clock... "hello" line two'

    # raw_dispersion edge cases
    assert raw_dispersion(np.zeros((0, 8))) == 0.0
    assert raw_dispersion(np.ones((1, 8))) == 0.0
    assert raw_dispersion(np.zeros((5, 8))) == 0.0
    rng = np.random.default_rng(0)
    u = rng.normal(size=32); u /= np.linalg.norm(u)
    assert raw_dispersion(np.tile(u, (10, 1))) < 1e-12

    # Tight surface-variant cluster collapses cleanly under default disp_min=0.035
    v = rng.normal(size=32); v -= np.dot(v, u) * u; v /= np.linalg.norm(v)
    tight = np.vstack([
        np.tile(u + 0.05 * v, (20, 1)),
        np.tile(u - 0.05 * v, (20, 1)),
    ])
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)
    lp = np.zeros(len(tight))
    r_tight = readings_from(tight, lp, drop_q=0.1)
    assert r_tight["raw_disp"] < 0.035
    assert r_tight["k"] == 1 and r_tight["silhouette"] == 0.0
    assert r_tight["readings"] == [0] and r_tight["shares"] == {0: 1.0}
    # Disabling gate splits the tight synthetic clusters
    assert readings_from(tight, lp, drop_q=0.1, disp_min=None)["k"] == 2

    # Empty & tiny inputs in readings_from / cluster
    r_empty = readings_from(np.zeros((0, 8)), np.zeros(0))
    assert r_empty["k"] == 0 and r_empty["readings"] == [] and r_empty["raw_disp"] == 0.0
    assert cluster(np.zeros((0, 8)))["k"] == 0
    c2 = cluster(tight[:2])
    assert c2["k"] == 1 and c2["silhouette"] == 0.0


def test_real_g3_4b_pt_samples_raw_dispersion_gate():
    import json
    from pathlib import Path
    from explorer.readings import readings_from

    samples_dir = Path(__file__).resolve().parents[1] / "results" / "explorer" / "g3_4b_pt" / "samples"
    if not samples_dir.is_dir():
        pytest.skip("g3_4b_pt saved samples not present")

    checked = 0
    for sp in sorted(samples_dir.glob("*.json")):
        S = json.loads(sp.read_text())
        tid = S["id"]
        E = np.load(samples_dir / f"{tid}_emb.npy")
        lpt = np.array(S["logp"]) / np.maximum(np.array(S["lens"]), 1)
        res = readings_from(E, lpt, drop_q=0.1, disp_min=0.035)
        checked += 1
        if tid == "control_museo":
            assert res["raw_disp"] < 0.035, res["raw_disp"]
            assert res["k"] == 1 and res["readings"] == [0] and res["silhouette"] == 0.0
        else:
            assert res["raw_disp"] >= 0.035, (tid, res["raw_disp"])
            assert len(res["readings"]) >= 2, (tid, res["k"], res["readings"])
    assert checked == 9


def test_cluster_hybrid_dispersion_gate_and_blind_spot():
    from explorer.grad_readings import cluster_gradients, cluster_hybrid

    rng = np.random.default_rng(7)
    n, d_sem, d_grad = 40, 24, 32
    lp = rng.uniform(-1.0, -0.2, size=n)

    # Identical / near-identical semantics AND gradients -> collapses cleanly to k=1
    e0 = rng.normal(size=d_sem); e0 /= np.linalg.norm(e0)
    g0 = rng.normal(size=d_grad); g0 /= np.linalg.norm(g0)
    E_same = np.tile(e0, (n, 1)) + 1e-4 * rng.normal(size=(n, d_sem))
    G_same = np.tile(g0, (n, 1)) + 1e-4 * rng.normal(size=(n, d_grad))
    res_same = cluster_hybrid(E_same, G_same, lp, alpha_grad=0.5, disp_min=0.035)
    assert res_same["k"] == 1 and res_same["silhouette"] == 0.0
    assert res_same["readings"] == [0] and res_same["raw_disp"] < 0.035

    res_grad_const = cluster_gradients(np.tile(g0, (n, 1)), lp, whiten=False)
    assert res_grad_const["k"] == 1 and res_grad_const["silhouette"] == 0.0

    # Poetry blind spot: identical semantics + 3 strong gradient clusters still splits cleanly!
    centers = rng.normal(size=(3, d_grad))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    G_3_shared = np.vstack([
        8.0 * g0[None, :] + 1.5 * centers[i][None, :] + 0.15 * rng.normal(size=(14, d_grad))
        for i in range(3)
    ])
    E_id = np.tile(e0, (len(G_3_shared), 1))
    lp3 = rng.uniform(-1.0, -0.2, size=len(G_3_shared))
    res_bs = cluster_hybrid(E_id, G_3_shared, lp3, alpha_grad=0.5, whiten=True, disp_min=0.035)
    assert res_bs["k"] == 3 and len(res_bs["readings"]) == 3
    assert res_bs["raw_disp"] >= 0.035 and res_bs["raw_disp_sem"] < 1e-9

    G_3_plain = np.vstack([centers[i][None, :] + 0.08 * rng.normal(size=(14, d_grad)) for i in range(3)])
    res_bs_plain = cluster_hybrid(E_id, G_3_plain, lp3, alpha_grad=0.5, whiten=False, disp_min=0.035)
    assert res_bs_plain["k"] == 3 and len(res_bs_plain["readings"]) == 3
    assert res_bs_plain["raw_disp"] >= 0.035


def test_run_text_hybrid_fallback_and_doc_style(tiny, tmp_path, monkeypatch):
    import explorer.directions as DIR
    import explorer.readings as RD
    from explorer.run import run_text
    from explorer.steer import CursorHook

    tok, model = tiny
    hk = CursorHook(model, 0)

    seen_embed_inputs = []
    def fake_embed(texts):
        seen_embed_inputs.append(list(texts))
        E = np.array([np.random.default_rng(abs(hash(t)) % 2 ** 32).normal(size=16) for t in texts])
        return E / np.linalg.norm(E, axis=1, keepdims=True)
    monkeypatch.setattr(RD, "embed_texts", fake_embed)

    txt = {"id": "poem_doc", "kind": "hard", "text": "En la plaza\n  cae la lluvia.\n\nSolo el viento."}
    try:
        # 1. Hybrid clustering + doc prompt style
        out1 = tmp_path / "run_hybrid"
        S1, _, _ = run_text(model, tok, hk, 0, txt, out1, n=6, prompt_style="doc", clustering="hybrid", disp_min=1.0)
        assert S1["prompt_style"] == "doc"
        assert S1["clustering"] == "hybrid"
        assert S1["method"].startswith("hybrid")
        assert "raw_disp" in S1
        assert all("\n" not in s for s in seen_embed_inputs[-1])

        # 2. Hybrid fallback when grads_at fails
        def boom(*a, **k):
            raise RuntimeError("synthetic OOM")
        monkeypatch.setattr(DIR, "grads_at", boom)
        out2 = tmp_path / "run_fallback"
        S2, _, _ = run_text(model, tok, hk, 0, txt, out2, n=6, prompt_style="legacy", clustering="hybrid", disp_min=1.0)
        assert "clustering_fallback" in S2 and "RuntimeError" in S2["clustering_fallback"]
        assert S2["k"] == 1
    finally:
        hk.remove()


def test_mapgrid_build_map_basis_ranks():
    from explorer.mapgrid import build_map_basis

    rng = np.random.default_rng(21)
    d = 32

    # R == 2: preserves rank-2 (axis 1 from orthogonal within-reading variance)
    m0, m1 = rng.normal(size=d), rng.normal(size=d)
    G2 = {0: m0 + 0.3 * rng.normal(size=(12, d)), 1: m1 + 0.3 * rng.normal(size=(10, d))}
    b2 = build_map_basis(G2, [0, 1], whiten=True)
    assert b2["axes"].shape == (2, d)
    assert np.allclose(b2["axes"] @ b2["axes"].T, np.eye(2), atol=1e-10)
    assert all(len(b2["anchors"][r]) == 2 for r in (0, 1))
    assert all(len(pt) == 2 for r in (0, 1) for pt in b2["dots"][r])

    # R == 3
    m2 = rng.normal(size=d)
    G3 = {0: G2[0], 1: G2[1], 2: m2 + 0.3 * rng.normal(size=(8, d))}
    b3 = build_map_basis(G3, [0, 1, 2], whiten=True)
    assert b3["axes"].shape == (2, d)
    assert np.allclose(b3["axes"] @ b3["axes"].T, np.eye(2), atol=1e-10)

    # R == 1
    b1 = build_map_basis({0: G2[0]}, [0], whiten=True)
    assert b1["axes"].shape == (2, d)
    assert np.allclose(b1["axes"] @ b1["axes"].T, np.eye(2), atol=1e-10)
    assert len(b1["anchors"][0]) == 2

    # Degenerate rank-1 / identical gradients still produce orthonormal (2, d) via Gram-Schmidt padding
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    G_deg = {0: np.tile(v, (5, 1)), 1: np.tile(-v, (5, 1))}
    b_deg = build_map_basis(G_deg, [0, 1], whiten=False)
    assert b_deg["axes"].shape == (2, d)
    assert np.allclose(b_deg["axes"] @ b_deg["axes"].T, np.eye(2), atol=1e-10)
    assert all(len(b_deg["anchors"][r]) == 2 for r in (0, 1))


def test_mapgrid_phrases_by_reading_generalized():
    import json
    from pathlib import Path
    from explorer.mapgrid import phrases_by_reading

    # 1. Real saved non-duende texts
    samples_dir = Path(__file__).resolve().parents[1] / "results" / "explorer" / "g3_4b_pt" / "samples"
    if samples_dir.is_dir():
        for tid in ("machado_caminante", "becquer_xxi", "monte_catalejo", "duende_bare", "rosalia_dicen"):
            S = json.loads((samples_dir / f"{tid}.json").read_text())
            rd = S["readings"]
            cands = phrases_by_reading(S, rd, top=3)
            for r in rd:
                assert len(cands[r]) >= 1, (tid, r, cands)
        S_mach = json.loads((samples_dir / "machado_caminante.json").read_text())
        c_mach = phrases_by_reading(S_mach, S_mach["readings"], top=3)
        joined = {r: " ".join(c_mach[r]).lower() for r in S_mach["readings"]}
        assert "traveler" in joined[0]
        assert "walker" in joined[1]
        assert "wayfarer" in joined[3]

    # 2. Multi-stanza poem where divergence occurs ONLY in stanza 2 (lines after line 1 preserved!)
    S_poem = {
        "texts": [
            "The dark swallows return\n  to your balcony.\n\nOnly the ember endures.",
            "The dark swallows return\n  to your balcony.\n\nOnly the ember remains.",
            "The dark swallows return\n  to your balcony.\n\nOnly the shadow lingers.",
            "The dark swallows return\n  to your balcony.\n\nOnly the shadow waits.",
        ],
        "labels": [0, 0, 1, 1],
    }
    c_poem = phrases_by_reading(S_poem, [0, 1], top=3)
    assert any("ember" in p.lower() for p in c_poem[0]), c_poem
    assert any("shadow" in p.lower() for p in c_poem[1]), c_poem

    # 3. Mode 1: fork continuations supplied directly
    c_cont = phrases_by_reading(
        S_poem,
        [0, 1],
        top=2,
        cont_texts_by_reading={
            0: ["raw soul; it moves me.", "raw soul, and nothing more."],
            1: ["dark magic; I love it.", "dark magic, fierce and strange."],
        },
    )
    assert c_cont[0][0].lower() == "raw soul"
    assert c_cont[1][0].lower() == "dark magic"


def test_readings_labels_yaml_and_report(tmp_path):
    import json
    import re
    from pathlib import Path
    from explorer import report as REP

    labels_data = REP.load_reading_labels()
    assert "texts" in labels_data and "models" in labels_data
    bench_ids = {t["id"] for t in load_texts()}
    assert bench_ids <= set(labels_data["texts"].keys())
    for tid in bench_ids:
        meta = REP.text_metadata(labels_data, tid)
        assert meta["title"] and meta["description"]

    samples_dir = Path(__file__).resolve().parents[1] / "results" / "explorer" / "g3_4b_pt" / "samples"
    if samples_dir.is_dir():
        g3_map = labels_data["models"]["g3_4b_pt"]
        for sp in sorted(samples_dir.glob("*.json")):
            S = json.loads(sp.read_text())
            tid = S["id"]
            assert tid in g3_map
            curated_keys = {int(k) for k in g3_map[tid].keys()}
            assert set(int(r) for r in S["readings"]) <= curated_keys, (tid, S["readings"], curated_keys)

    # Single-reading collapse guard
    S_ctrl_collapsed = {
        "id": "control_museo",
        "text": "El museo abre a las diez de la mañana.",
        "texts": ["The museum opens at ten in the morning."] * 4,
        "labels": [0, 0, 0, 0],
        "readings": [0],
        "shares": {"0": 1.0},
        "logp": [-1.0] * 4,
        "ys": [[1, 2]] * 4,
        "k": 1,
        "silhouette": 0.0,
        "entropy_bits": 0.0,
    }
    res_ctrl = REP.resolve_reading_labels(labels_data, "g3_4b_pt", "control_museo", S_ctrl_collapsed)
    assert res_ctrl == {0: "Single unanimous reading"}

    # Legacy flat schema support
    legacy = {"custom_tid": {0: "Literal", 1: "Poetic"}}
    S_custom = {
        "id": "custom_tid",
        "text": "Hola",
        "texts": ["Hello world", "Greetings earth"],
        "labels": [0, 1],
        "readings": [0, 1],
        "shares": {"0": 0.5, "1": 0.5},
        "logp": [-1.0, -1.2],
        "ys": [[1], [2]],
        "k": 2,
        "silhouette": 0.4,
        "entropy_bits": 1.0,
    }
    assert REP.resolve_reading_labels(legacy, "any_model", "custom_tid", S_custom) == {0: "Literal", 1: "Poetic"}

    # Unknown model auto-labeler never outputs bare "reading N"
    auto = REP.resolve_reading_labels(labels_data, "unseen_model_72b", "custom_tid", S_custom)
    for lbl in auto.values():
        assert not re.fullmatch(r"reading\s+\d+", lbl.strip(), flags=re.I), lbl

    # Empty cluster fallback
    auto_empty = REP.auto_label_readings([], [], [0, 1])
    assert auto_empty == {0: "Cluster #1", 1: "Cluster #2"}

    # End-to-end report.run with results_dir override
    res_dir = tmp_path / "rep_out"
    (res_dir / "samples").mkdir(parents=True)
    (res_dir / "samples" / "control_museo.json").write_text(json.dumps(S_ctrl_collapsed))
    (res_dir / "samples" / "custom_tid.json").write_text(json.dumps(S_custom))
    html_path = REP.run({"model": "g3_4b_pt", "results_dir": str(res_dir)})
    html_txt = html_path.read_text(encoding="utf-8")
    assert "white-space:pre-wrap" in html_txt
    assert "Museum Opening Time (Control)" in html_txt
    assert "Single unanimous reading" in html_txt
    assert not re.search(r">reading \d+<", html_txt, flags=re.I)

