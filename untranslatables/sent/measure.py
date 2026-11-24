"""The two meaning spaces' leftovers, and the distortion-length curve (handoff §1-2).  Pure numpy, no model.

Round trip (§2.2), per source token i:
    R_i(t) = log p(s_i | s_<i, t) - log p(s_i | s_<i, empty);   R*_i = R_i(t = s)  (the copy task)
    w_i = R*_i if R*_i > delta else 0;   l_i = 1 - clip(R_i / R*_i, 0, 1)   (0 where w_i = 0)
Concept layer (§2.1): semi-relaxed unbalanced OT between source tokens (mass a_i = w_i / sum w) and target tokens
(mass 1 each), cost C_ij = 1 - cos, entropic regulariser eps_ot, KL penalty tau on the source marginal and NO
penalty on the target marginal ("free": commentary is not penalised).  With POT's default KL reference
(reg_type="kl") this has a closed form, checked against ot.unbalanced.sinkhorn_unbalanced in tests/test_sent.py:
    transported_i / a_i = exp( eps/(tau+eps) * lse_i ),   lse_i = logsumexp_j(-C_ij / eps)
so the leftover depends on each token's soft-min cost to the target only, and tau can be recalibrated on the
CPU from lse alone.  (The handoff's "reg_m = (tau, inf)" would make the target marginal binding; its text asks
for a free target, which is reg_m = (tau, 0).)
"""
import numpy as np

EPS_OT = 0.05
DELTA = 0.5
TAU_GRID = np.round(np.logspace(-2, 1.5, 36), 5)


# ------------------------------------------------------------------ token level
def rt_token(lp_t, lp_null, lp_oracle, delta=DELTA):
    lp_t, lp_null, lp_oracle = (np.asarray(x, float) for x in (lp_t, lp_null, lp_oracle))
    R, Rs = lp_t - lp_null, lp_oracle - lp_null
    keep = Rs > delta
    w = np.where(keep, Rs, 0.0)
    l = np.where(keep, 1 - np.clip(R / np.where(keep, Rs, 1.0), 0, 1), 0.0)
    return w, l


def cl_token(lse, tau, eps=EPS_OT):
    """Per-token leftover in the concept layer, from lse_i (any leading shape, e.g. [layers, n])."""
    return 1 - np.minimum(1.0, np.exp(eps / (tau + eps) * np.asarray(lse, float)))


def cl_cost(lse, softcost, w, tau, eps=EPS_OT):
    """<P, C>: transported mass times the softmax-weighted cost of each source token (secondary quantity)."""
    a = np.asarray(w, float) / max(np.sum(w), 1e-12)
    return float((a * np.exp(eps / (tau + eps) * np.asarray(lse, float)) * np.asarray(softcost, float)).sum(-1))


def weighted(w, l, mask=None):
    w, l = np.asarray(w, float), np.asarray(l, float)
    if mask is not None:
        w = w * np.asarray(mask, float)
    s = w.sum(-1)
    return np.where(s > 0, (w * l).sum(-1) / np.where(s > 0, s, 1), np.nan)


def token_to_word(offsets, words):
    """Index of the word each token belongs to (by the token's first non-space character), or -1."""
    out = []
    for a, b in offsets:
        j = -1
        for k, (ws, we, _) in enumerate(words):
            if ws <= a < we or (a < ws < b):
                j = k; break
        out.append(j)
    return np.array(out, int)


def span_mask(offsets, start, end):
    return np.array([b > start and a < end and b > a for a, b in offsets], float)


def to_words(w, l, t2w, n_words):
    """Word-level weights (sum) and leftovers (w-weighted mean; nan when the word carries no weight)."""
    W, Lw = np.zeros(n_words), np.full(n_words, np.nan)
    for k in range(n_words):
        m = t2w == k
        W[k] = w[m].sum()
        if W[k] > 0:
            Lw[k] = (w[m] * l[m]).sum() / W[k]
    return W, Lw


# ------------------------------------------------------------------ distortion-length curve
def envelope(lengths, ds):
    """Lower envelope D_s(L): (sorted lengths, running minimum of d).  Candidates with nan d are dropped."""
    L, d = np.asarray(lengths, float), np.asarray(ds, float)
    ok = ~np.isnan(d)
    o = np.argsort(L[ok], kind="stable")
    return L[ok][o], np.minimum.accumulate(d[ok][o])


def d_at(Ls, D, L):
    """D_s at budget L (nan below the shortest candidate)."""
    k = np.searchsorted(Ls, L, side="right") - 1
    return np.where(k >= 0, D[np.clip(k, 0, len(D) - 1)], np.nan) if len(D) else np.full(np.shape(L), np.nan)


def untranslatability(Ls, D, L0, eps):
    """U_eps = log2(L*/L0), floored at 0 (reaching eps at or below the standard length scores 0).
    Returns (U, censored, raw log2 ratio).  Censored: the ladder never reached eps; U is then the lower bound
    log2(L_max / L0)."""
    hit = np.nonzero(D <= eps)[0]
    if len(hit) == 0:
        return float(np.log2(Ls[-1] / L0)), True, None
    raw = float(np.log2(Ls[hit[0]] / L0))
    return max(raw, 0.0), False, raw


def grid(L0, K, step=0.25):
    return L0 * 2.0 ** np.arange(0, K + 1e-9, step)


def area(Ls, D, L0, K, step=0.25):
    """A(s): mean of D_s over log-spaced budgets from L0 to L0 * 2^K (D held flat beyond the longest
    candidate, i.e. no extra candidates were tried there)."""
    v = d_at(Ls, D, grid(L0, K, step))
    return float(np.nanmean(v)) if np.any(~np.isnan(v)) else np.nan


def calibrate_tau(lse_matched, w_matched, target=0.15, grid_=TAU_GRID, eps=EPS_OT):
    """Smallest tau on the grid whose mean d_CL over matched (control, own translation) pairs is <= target.
    Smaller tau = harsher, so this is the most discriminating tau that still lets easy sentences through."""
    for tau in grid_:
        d = [weighted(w, cl_token(l, tau, eps)) for l, w in zip(lse_matched, w_matched)]
        if np.nanmean(d) <= target:
            return float(tau)
    return float(grid_[-1])
