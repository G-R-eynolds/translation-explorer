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
