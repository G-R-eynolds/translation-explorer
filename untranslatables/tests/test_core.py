"""CPU unit tests on a tiny random Llama: hooks, span extraction, recipe maths, scoring helpers.
    .venv-cpu/bin/python -m pytest -q
"""
import numpy as np
import pytest
import torch

from recipe import nnomp
from ut.represent import patching, span_states, steering, unmark


@pytest.fixture(scope="module")
def tiny():
    from ut.common import load_model
    return load_model("tiny")


def states(model, tok, text):
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model.model(**ids, output_hidden_states=True).hidden_states


def test_unmark():
    assert unmark("Sinto [[saudade]] hoje") == ("Sinto saudade hoje", (6, 13))
    with pytest.raises(ValueError):
        unmark("no span")


def test_span_states_shape(tiny):
    tok, model = tiny
    H = span_states(model, tok, [unmark("a [[word]] here"), ("last token", None)])
    assert H.shape == (2, model.config.num_hidden_layers + 1, model.config.hidden_size)


def test_steering_changes_downstream_and_gamma0_is_noop(tiny):
    tok, model = tiny
    base = states(model, tok, "hello world")
    v = {1: torch.ones(model.config.hidden_size, device=model.device)}
    with steering(model, v, 0.0):
        same = states(model, tok, "hello world")
    with steering(model, v, 2.0):
        moved = states(model, tok, "hello world")
    assert all(torch.allclose(a, b) for a, b in zip(base, same))
    assert not torch.allclose(base[2], moved[2]) and not torch.allclose(base[-1], moved[-1])


def test_patching_only_last_position_downstream(tiny):
    tok, model = tiny
    base = states(model, tok, "one two three")
    lat = torch.full((model.config.hidden_size,), 3.0, device=model.device)
    with patching(model, {1: lat}):
        p = states(model, tok, "one two three")
    assert not torch.allclose(p[2][0, -1], base[2][0, -1])          # last position changed downstream
    assert torch.allclose(p[2][0, :-1], base[2][0, :-1], atol=1e-5)  # causal: earlier positions untouched


def test_nnomp_recovers_sparse_mix():
    rng = np.random.default_rng(0)
    D = rng.standard_normal((200, 64)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    x = 0.7 * D[3] + 0.3 * D[17]; x /= np.linalg.norm(x)
    S, w, R2 = nnomp(D, x, k=5)
    assert set(S[:2]) == {3, 17} and R2[1] > 0.999


def test_distinction_scoring():
    from distinctions import load_sets, score
    p = load_sets()["fr-es_savoir-connaitre"]
    assert score("Conozco bien este barrio.", p, "conocer") == "correct"
    assert score("Sé bien este barrio.", p, "conocer") == "error"
    assert score("Se fue a casa.", p, "saber") == "other"          # reflexive 'se' is not 'sé'


def test_word_data_valid():
    from ut.words import load_words, validate
    hard = [i for i in validate(load_words()) if "need exactly one" in i or "unknown category" in i or "duplicate" in i]
    assert not hard, hard
