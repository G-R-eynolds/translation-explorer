"""CPU tests for blind/ (tiny random Llama):  .venv-cpu/bin/python -m pytest -q tests/test_blind.py"""
import numpy as np
import pytest
import scipy.linalg
import torch

from blind import spectrum as sp
from blind.common import instruction, load_builtin, load_e2, load_e3


def test_comparator_matches_dense():
    r = np.random.default_rng(0)
    S, L = r.normal(size=(20, 300)), r.normal(size=(20, 300)) * np.r_[np.ones(150), 0.1 * np.ones(150)]
    c = sp.Comparator({"s": S, "l": L})
    lam, C, I = c.eig(c.idx("s"), c.idx("l"), vectors=True)
    Q, _ = np.linalg.qr(np.vstack([S, L]).T)
    A, B = Q.T @ S.T @ S @ Q / 20, Q.T @ L.T @ L @ Q / 20
    B += 1e-2 * np.trace(B) / 20 * np.eye(40)
    assert np.allclose(lam, scipy.linalg.eigh(A, B, eigvals_only=True)[::-1], rtol=1e-6)
    v = c.vec(C, I, 0)
    assert abs(np.linalg.norm(v) - 1) < 1e-8
    assert np.isclose(c.quad(c.idx("s"), C, I, 0), sp.quad_v(S, v))
    # well-sampled case: the top blind direction lives where L is weak
    S2, L2 = r.normal(size=(400, 20)), r.normal(size=(400, 20)) * np.r_[np.ones(10), 0.1 * np.ones(10)]
    c2 = sp.Comparator({"s": S2, "l": L2})
    lam2, C2, I2 = c2.eig(c2.idx("s"), c2.idx("l"), vectors=True)
    assert (c2.vec(C2, I2, 0)[10:] ** 2).sum() > 0.9


def test_null_is_null():
    r = np.random.default_rng(1)
    X = r.normal(size=(64, 200))
    Y = r.normal(size=(64, 200))
    c = sp.Comparator({"src": X, "t": Y})
    s = sp.summarise(sp.split_spectra(c, "src", "t", n_splits=10))
    assert s["U"]["med"] <= s["U_null"]["hi"] * 1.5


def test_U_of_padding():
    assert sp.U_of(np.array([4.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])) == pytest.approx(1.0)


def test_material_loads():
    b, types = load_builtin()
    assert len(b) == 42 and all(i.group in types for i in b)
    e2 = {i.id: i for i in load_e2()}
    assert "es_duende_flamenco" in e2 and e2["es_passage_dense"].parts
    assert len(load_e3()) == 27


@pytest.fixture(scope="module")
def tiny():
    from ut.common import load_model
    from blind.fisher import Hooks, Prompter
    tok, model = load_model("tiny")
    model.requires_grad_(False)
    return tok, model, Prompter(tok), Hooks(model, [0, 1])


def test_source_ids_shared(tiny):
    tok, model, pr, _ = tiny
    a, P = pr.build("Estoy cansada.", instruction("src", "es"))
    b, P2 = pr.build("Estoy cansada.", instruction("en", "es"))
    assert P == P2 and a[:P[1]] == b[:P[1]] and a != b


def test_per_sample_grads_independent(tiny):
    from blind.fisher import per_sample_grads
    tok, model, pr, hooks = tiny
    ids, P = pr.build("Estoy cansada.", instruction("src", "es"))
    ys = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    G, lp = per_sample_grads(model, hooks, ids, P, ys, [0], batch=3)
    G1, lp1 = per_sample_grads(model, hooks, ids, P, ys[1:2], [0], batch=1)
    assert np.allclose(G[0][1], G1[0][0], atol=1e-5) and np.allclose(lp[1], lp1[0], atol=1e-4)
