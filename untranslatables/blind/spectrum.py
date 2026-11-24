"""Blind-spot spectra from stored per-sample gradients (CPU; §1, §2.2, §2.3).

Rows are per-sample scores g = d log p(y|h)/dh, flattened over the source positions used (all of P, a span, or one
token).  Ĝ_c = (1/n) Σ g gᵀ (uncentred: E[g] = 0 for a score).  For a comparison of a set S (src) with a set L
(target), everything happens in Q = an orthonormal basis of the rows involved (≤ 2n dims):
    A = Qᵀ Ĝ_S Q,   B = Qᵀ Ĝ_L Q + ε (tr Ĝ_L / rank Ĝ_L) I,   A w = λ B w   (scipy.linalg.eigh)
Implementation: one Gram matrix K = Z Zᵀ over all samples of all channels of an item at a layer; for a subset I of
rows, Q = Z_Iᵀ U Λ^{-1/2} from eigh(K_II), so the coordinates of any row j are K_jI U Λ^{-1/2}.  The D-dimensional
vectors (D = |P|·d, up to ~10⁵) are only formed when a direction is needed for steering.

Null (§2.3): src half a vs src half b, the same sizes as every real comparison (src half a vs target half a); 20 random
splits.  U_L = ½ Σ_i log2 max(1, λ_i / λ0_i) with λ0 = the index-wise median null spectrum; the same formula applied to
each split's own null spectrum gives the null distribution of U.  n_blind = #{i: λ_i > q95(λ0_i)}.
Two scalar variants (exact, because ε is relative to tr Ĝ_L):
  trace-normalised   λ · tr Ĝ_L / tr Ĝ_S  (only the shape of each channel's sensitivity counts, not its overall size)
  length-normalised  each Ĝ divided by its channel's mean output length in tokens (E3): λ · len_L / len_S
"""
import numpy as np
import scipy.linalg

from blind.common import EPS_HEAD, N_SPLITS


class Comparator:
    """All samples of an item at one layer (one position set).  blocks: {channel: (n, D) array}."""

    def __init__(self, blocks):
        self.names = list(blocks)
        self.off, o = {}, 0
        for c in self.names:
            self.off[c] = (o, o + len(blocks[c])); o += len(blocks[c])
        self.Z = np.concatenate([np.asarray(blocks[c], np.float64) for c in self.names])
        self.K = self.Z @ self.Z.T

    def idx(self, c):
        a, b = self.off[c]
        return np.arange(a, b)

    def n(self, c):
        a, b = self.off[c]
        return b - a

    def _coords(self, I):
        KI = self.K[np.ix_(I, I)]
        lam, U = np.linalg.eigh(KI)
        keep = lam > 1e-10 * max(lam.max(), 1e-300)
        return U[:, keep] / np.sqrt(lam[keep])          # M: coords(row j) = K[j, I] @ M

    def eig(self, iS, iL, eps=EPS_HEAD, vectors=False):
        """λ (descending) of Ĝ_S w = λ (Ĝ_L + ε) w in span(rows iS ∪ iL).  vectors=True also returns
        C (|I| x r) and I such that the unit direction i in D-space is Z[I].T @ C[:, i]."""
        I = np.concatenate([iS, iL])
        M = self._coords(I)
        YS, YL = self.K[np.ix_(iS, I)] @ M, self.K[np.ix_(iL, I)] @ M
        r = M.shape[1]
        A, Bl = YS.T @ YS / len(iS), YL.T @ YL / len(iL)
        B = Bl + eps * (np.trace(Bl) / max(min(len(iL), r), 1)) * np.eye(r)
        lam, W = scipy.linalg.eigh(A, B)
        lam, W = np.clip(lam[::-1], 0, None), W[:, ::-1]
        if not vectors:
            return lam
        C = M @ W                                        # direction = Z_I^T C  (Q = Z_I^T M)
        nrm = np.sqrt(np.einsum("ij,ik,kj->j", C, self.K[np.ix_(I, I)], C))
        return lam, C / nrm, I

    def vec(self, C, I, i):
        return self.Z[I].T @ C[:, i]

    def quad(self, rows, C, I, i):
        """vᵀĜv = mean over the given rows of (g·v)², without forming v."""
        return float(np.mean((self.K[np.ix_(rows, I)] @ C[:, i]) ** 2))


def _pad(x, r):
    return np.pad(x, (0, r - len(x))) if len(x) < r else x[:r]


def U_of(lam, lam0):
    """½ Σ log2 max(1, λ_i/λ0_i); indices where the null is 0 (rank-deficient padding) count as no evidence."""
    ok = lam0 > 0
    ratio = np.ones_like(lam)
    ratio[ok] = lam[ok] / lam0[ok]
    return 0.5 * float(np.log2(np.maximum(1.0, ratio)).sum())


def split_spectra(cmp, src, tgt, eps=EPS_HEAD, n_splits=N_SPLITS, seed=0, reverse=True):
    """Matched split-half spectra of src vs tgt.  Returns arrays (n_splits, r)."""
    rng = np.random.default_rng(seed)
    iS, iT = cmp.idx(src), cmp.idx(tgt)
    h = min(len(iS), len(iT)) // 2
    r = 2 * h
    real, null, rev, trr = [], [], [], []
    for _ in range(n_splits):
        ps, pt = rng.permutation(iS), rng.permutation(iT)
        Sa, Sb, Ta = ps[:h], ps[h:2 * h], pt[:h]
        real.append(_pad(cmp.eig(Sa, Ta, eps), r))
        null.append(_pad(cmp.eig(Sa, Sb, eps), r))
        if reverse:
            rev.append(_pad(cmp.eig(Ta, Sa, eps), r))
        trr.append(float(np.trace(cmp.K[np.ix_(Ta, Ta)]) / max(np.trace(cmp.K[np.ix_(Sa, Sa)]), 1e-300)))
    return dict(real=np.array(real), null=np.array(null), rev=np.array(rev) if reverse else None,
                tr_ratio=np.array(trr))


def _q(x):
    x = np.asarray(x, float)
    return dict(med=float(np.median(x)), lo=float(np.quantile(x, 0.05)), hi=float(np.quantile(x, 0.95)))


def summarise(sp, len_ratio=None, keep_splits=True):
    """Statistics of one comparison (median and 90% interval over splits).
    len_ratio = mean output length of the target channel / that of src (length-normalised U)."""
    real, null = sp["real"], sp["null"]
    lam0 = np.median(null, 0)
    q95 = np.quantile(null, 0.95, 0)
    U = np.array([U_of(x, lam0) for x in real])
    U0 = np.array([U_of(x, lam0) for x in null])
    out = dict(U=_q(U), U_null=_q(U0), n_blind=_q([(x > q95).sum() for x in real]),
               lam1_ratio=_q(real[:, 0] / max(lam0[0], 1e-300)),
               U_tracenorm=_q([U_of(x * t, lam0) for x, t in zip(real, sp["tr_ratio"])]),
               tr_ratio=_q(sp["tr_ratio"]), lam_med=np.median(real, 0)[:16].tolist(), lam0_med=lam0[:16].tolist(),
               lam0_q95=q95[:16].tolist())
    if keep_splits:
        out.update(U_splits=U.round(4).tolist(), U_null_splits=U0.round(4).tolist())
    if sp.get("rev") is not None:
        out.update(U_rev=_q([U_of(x, lam0) for x in sp["rev"]]),
                   n_seen=_q([(x > q95).sum() for x in sp["rev"]]))
    if len_ratio is not None:
        out["U_lennorm"] = _q([U_of(x * len_ratio, lam0) for x in real])
    return out


def flat(G, positions=None):
    """(n, |P|, d) -> (n, |S|·d) restricted to the given source positions (span version)."""
    g = G if positions is None else G[:, positions]
    return g.reshape(len(g), -1)


# ------------------------------------------------------------------ directions
def directions(cmp, src, tgt, n_dirs, eps=EPS_HEAD, seen=False):
    """Top directions from the full samples (src vs tgt; tgt vs src when seen=True).  n_dirs comes from the split
    analysis (how many eigenvalues are above the null).  Returns [dict(lam, v, q_src, q_tgt)] with v unit-norm."""
    iS, iT = cmp.idx(src), cmp.idx(tgt)
    lam, C, I = cmp.eig(iT, iS, eps, vectors=True) if seen else cmp.eig(iS, iT, eps, vectors=True)
    out = []
    for i in range(int(min(n_dirs, len(lam)))):
        out.append(dict(lam=float(lam[i]), v=cmp.vec(C, I, i), q_src=cmp.quad(iS, C, I, i), q_tgt=cmp.quad(iT, C, I, i)))
    return out


def alpha_for(q_src, q_tgt, target_kl=0.5):
    """Step α so that the larger predicted KL over the two channels, ½ α² vᵀĜv, equals target_kl (§2.4: 0.5 nats on
    src for a blind direction; for a 'seen' direction src is insensitive, so the target channel sets the scale)."""
    return float(np.sqrt(2 * target_kl / max(q_src, q_tgt, 1e-300)))


def position_profile(v, n_pos):
    w = (np.asarray(v).reshape(n_pos, -1) ** 2).sum(1)
    return (w / max(w.sum(), 1e-300)).tolist()


def held_out_rho(cmp, src, plain, others, n_dirs, eps=EPS_HEAD, seed=0):
    """E3: blind directions fitted on src half a vs plain half a; for each, on held-out samples,
    ρ_c = vᵀĜ_c v / vᵀĜ_src(b) v for plain half b and every other channel (all its samples)."""
    rng = np.random.default_rng(seed)
    iS, iP = rng.permutation(cmp.idx(src)), rng.permutation(cmp.idx(plain))
    h = min(len(iS), len(iP)) // 2
    lam, C, I = cmp.eig(iS[:h], iP[:h], eps, vectors=True)
    ev = {"plain": iP[h:2 * h], **{c: cmp.idx(c) for c in others}}
    out = []
    for i in range(int(min(n_dirs, len(lam)))):
        qs = cmp.quad(iS[h:2 * h], C, I, i)
        out.append(dict(lam_fit=float(lam[i]),
                        rho={c: cmp.quad(rows, C, I, i) / max(qs, 1e-300) for c, rows in ev.items()}))
    return out


def quad_v(Gflat, v):
    """vᵀĜv for an explicit direction v (n, D) @ (D,)."""
    return float(np.mean((np.asarray(Gflat, np.float64) @ v) ** 2))
