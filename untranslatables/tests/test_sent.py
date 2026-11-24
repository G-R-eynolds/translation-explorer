"""CPU tests for the sentence experiments (sent/).    .venv-cpu/bin/python -m pytest -q tests/test_sent.py"""
import numpy as np
import pytest
import torch

from sent.measure import (area, cl_token, envelope, rt_token, span_mask, to_words, token_to_word,
                          untranslatability, weighted)


def test_cl_closed_form_matches_pot():
    """Semi-relaxed UOT, free target (reg_m = (tau, 0)), POT's default KL reference: our closed form."""
    ot = pytest.importorskip("ot")
    rng = np.random.default_rng(0)
    for n, m, tau in [(4, 7, 0.3), (6, 20, 1.0), (3, 2, 5.0)]:
        C = rng.uniform(0.1, 1.2, (n, m))
        a = rng.uniform(0.2, 1, n); a /= a.sum()
        eps = 0.05
        P = ot.unbalanced.sinkhorn_unbalanced(a, np.ones(m), C, reg=eps, reg_m=(tau, 0.0), numItermax=20000,
                                              stopThr=1e-13)
        lse = np.log(np.exp(-C / eps).sum(1))
        l_pot = 1 - np.minimum(1, P.sum(1) / a)
        assert np.allclose(cl_token(lse, tau, eps), l_pot, atol=1e-6)


def test_cl_leftover_monotone():
    lse_close, lse_far = np.log(np.exp(-0.1 / 0.05)), np.log(np.exp(-0.9 / 0.05))
    assert cl_token(lse_close, 1.0) < cl_token(lse_far, 1.0)
    assert cl_token(lse_far, 10.0) < cl_token(lse_far, 0.1)          # bigger tau = more mass moved
    assert cl_token(-1e4, 1.0) == pytest.approx(1.0)                   # nothing to match: all left over


def test_rt_token():
    w, l = rt_token(lp_t=[-1, -5, -2], lp_null=[-6, -5.2, -3], lp_oracle=[-0.5, -5.1, -0.1])
    assert w[1] == 0 and l[1] == 0                                     # oracle gain 0.1 < delta: dropped
    assert w[0] == pytest.approx(5.5) and l[0] == pytest.approx(1 - 5 / 5.5)
    assert l[2] == pytest.approx(1 - 1 / 2.9)
    w, l = rt_token([0, -9], [-6, -6], [-1, -1])                       # better than oracle / worse than nothing
    assert l[0] == 0 and l[1] == 1


def test_envelope_and_U():
    Ls, D = envelope([10, 40, 20, 80, 5], [0.5, 0.2, 0.6, 0.1, 0.9])
    assert list(Ls) == [5, 10, 20, 40, 80] and list(D) == [0.9, 0.5, 0.5, 0.2, 0.1]
    U, cens, raw = untranslatability(Ls, D, L0=10, eps=0.2)
    assert U == pytest.approx(2.0) and not cens
    U, cens, _ = untranslatability(Ls, D, L0=10, eps=0.05)
    assert cens and U == pytest.approx(3.0)
    U, cens, raw = untranslatability(Ls, D, L0=10, eps=0.95)           # reached below L0: floored at 0
    assert U == 0 and raw < 0
    assert 0.1 <= area(Ls, D, 10, K=3) <= 0.5


def test_words_and_spans():
    offs = [(0, 2), (2, 5), (5, 11), (11, 17), (17, 24)]               # 'Me| gu|sta mu|...' style pieces
    words = [(0, 2, "Me"), (3, 8, "gusta"), (9, 14, "mucho")]
    t2w = token_to_word([(0, 2), (2, 5), (5, 8), (8, 14)], words)
    assert list(t2w) == [0, 1, 1, 2]
    W, L = to_words(np.array([1., 2., 2., 0.]), np.array([.5, .2, .4, .9]), t2w, 3)
    assert W[1] == 4 and L[1] == pytest.approx(.3) and np.isnan(L[2])
    assert list(span_mask(offs, 3, 8)) == [0, 1, 1, 0, 0]
    assert weighted([1, 1, 2], [0, 1, .5], [0, 1, 1]) == pytest.approx(2 / 3)


def test_sentences_valid_and_fathers_example():
    from sent.data import load_sentences, validate
    s = load_sentences(dialect=True)
    assert not validate(s)
    d = {x.id: x for x in s}["es_duende"]
    assert {x["text"] for x in d.spans()} == {"duende", "gusta"}


def test_spelling_and_masking():
    from sent.data import load_sentences
    from sent.score import mask_source_words, spelling_control
    s = {x.id: x for x in load_sentences(dialect=True)}
    sp = spelling_control("El tren sale a las ocho.")
    assert sp.lower().startswith("elton trenly sal") and "tren" in sp.lower()
    assert mask_source_words("I really like *duende* a lot", s["es_duende"]) == "I really like *…* a lot"
    assert mask_source_words("The doctor reached the hospital", s["pt_medico"]) == "The doctor reached the hospital"


def test_clean():
    from sent.prompts import clean
    assert clean('Translation: "I really like the *duende*."') == "I really like the duende."
    assert clean("I like it.\n\n\n\nNote: x") == "I like it.\n\nNote: x"


@pytest.fixture(scope="module")
def tiny():
    from ut.common import load_model
    return load_model("tiny")


def test_rt_logprobs_match_full_forward_and_ignore_padding(tiny):
    from sent.score import rt_logprobs
    tok, model = tiny
    jobs = [([1, 5, 9, 12], [40, 41, 42]), ([1, 7], [40, 41, 42]), ([1, 3, 3, 3, 3, 3, 3, 8], [77, 2])]
    got = rt_logprobs(model, tok, jobs, batch=3)
    for (p, t), lp in zip(jobs, got):
        ids = torch.tensor([p + t])
        with torch.no_grad():
            ref = torch.log_softmax(model(ids).logits[0].float(), -1)
        exp = [ref[len(p) - 1 + i, t[i]].item() for i in range(len(t))]
        assert np.allclose(lp, exp, atol=1e-4)


def test_strip_source():
    from sent.prompts import clean
    t = clean('The translation of "Me gusta mucho el duende." is "I love the charm."', "Me gusta mucho el duende.")
    assert "gusta" not in t and "I love the charm" in t
