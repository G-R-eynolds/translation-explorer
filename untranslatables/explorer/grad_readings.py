"""Gradient-space and hybrid reading discovery + Fisher-natural spectral basis.

Solves the semantic-embedder blind spot on poetry and tight paraphrases (where all-mpnet-base-v2
collapses distinct interpretive readings into one cluster) by clustering in Fisher-whitened score
(gradient) space or a weighted hybrid of semantic embeddings and gradient features:

  whiten_gradients   Fisher-whitened coordinates in the low-rank span of G via K = G G^T (no d x d solve)
  cluster_gradients  tail-filtered clustering on (whitened or plain) centred gradients
  cluster_hybrid     tail-filtered clustering combining centred semantic embeddings + gradient features
  spectral_basis     orthonormal principal axes, normalized anchors, and per-sample dots in ambient d-space
"""
import numpy as np

from explorer.readings import centre, cluster, raw_dispersion


def _row_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return np.where(norms > 1e-12, X / norms.clip(1e-12), 0.0)


def whiten_gradients(G, eps=1e-2):
    """Fisher-whitened coordinates in the low-rank span of G: (n, d) -> (n, r), L2-normalised per row.

    Using K = G G^T = U diag(s) U^T and the orthonormal row-span basis Q = G^T U_r diag(s_r^{-1/2}),
    the empirical Fisher is G_hat = G^T G / n = Q diag(eig) Q^T with eig = s_r / n.  Regularisation
    uses the effective rank eff_r = tr(G_hat)^2 / tr(G_hat^2), i.e. lambda = eps * tr(G_hat) / eff_r,
    which prevents tiny high-dimensional noise directions from being blown up when n <= d.
    In the Q basis, (G_hat + lambda I)^{-1/2} g_i has coordinates U_r * sqrt(s_r / (eig + lambda)),
    preserving all inner products of the dense d-dimensional whitened vectors exactly.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"Expected 2D gradient matrix, got shape {G.shape}")
    n = len(G)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    K = G @ G.T
    K = 0.5 * (K + K.T)
    s, U = np.linalg.eigh(K)
    s = s[::-1]
    U = U[:, ::-1]
    s_max = float(s[0]) if len(s) else 0.0
    if s_max <= 1e-30:
        return np.zeros((n, 1), dtype=np.float64)
    keep = s > 1e-12 * s_max
    s_r = s[keep]
    U_r = U[:, keep]
    eig = s_r / n
    tr = float(eig.sum())
    eff_r = (tr * tr) / max(float((eig * eig).sum()), 1e-30)
    lam = float(eps) * tr / max(eff_r, 1.0)
    scale = np.sqrt(s_r / (eig + max(lam, 1e-30)))
    W = U_r * scale[None, :]
    return _row_normalize(W)


def _grad_features(G_keep, whiten=True, eps=1e-2):
    if whiten:
        return whiten_gradients(G_keep, eps=eps)
    return _row_normalize(np.asarray(G_keep, dtype=np.float64))


def cluster_gradients(G, logp_per_token, drop_q=0.1, whiten=True, eps=1e-2, disp_min=0.0, **kw):
    """Filter the lowest `drop_q` tail by logp_per_token, optionally Fisher-whiten G[keep], centre and
    L2-normalise, then cluster via explorer.readings.cluster.

    Returns the same schema as explorer.readings.readings_from plus method='grad_whitened'|'grad_plain'
    and raw_disp.
    """
    G = np.asarray(G, dtype=np.float64)
    lp = np.asarray(logp_per_token, dtype=np.float64)
    if len(G) != len(lp):
        raise ValueError(f"Length mismatch: len(G)={len(G)} vs len(logp_per_token)={len(lp)}")
    method = "grad_whitened" if whiten else "grad_plain"
    if len(G) == 0:
        return dict(labels=[], k=0, silhouette=0.0, readings=[], shares={}, mu=[], n_dropped=0,
                    method=method, raw_disp=0.0)
    keep = lp >= np.quantile(lp, drop_q)
    if not np.any(keep):
        keep = np.ones(len(G), dtype=bool)
    X = _grad_features(G[keep], whiten=whiten, eps=eps)
    mu = X.mean(0)
    raw_disp = raw_dispersion(X)
    labels = np.full(len(G), -1, dtype=int)
    if len(X) < 2 or raw_disp <= 1e-12 or (disp_min is not None and float(disp_min) > 0.0 and raw_disp < float(disp_min)):
        labels[keep] = 0
        return {
            "labels": labels.tolist(),
            "k": 1,
            "silhouette": 0.0,
            "readings": [0],
            "shares": {0: 1.0},
            "mu": mu.tolist(),
            "n_dropped": int((~keep).sum()),
            "method": method,
            "raw_disp": float(raw_disp),
        }
    c = cluster(centre(X, mu), **kw)
    labels[keep] = c["labels"]
    return {
        **c,
        "labels": labels.tolist(),
        "mu": mu.tolist(),
        "n_dropped": int((~keep).sum()),
        "method": method,
        "raw_disp": float(raw_disp),
    }


def cluster_hybrid(E_sem, G, logp_per_token, alpha_grad=0.5, drop_q=0.1, whiten=True, eps=1e-2, disp_min=0.035, **kw):
    """Combine L2-normalised centred semantic embeddings (weight sqrt(1 - alpha_grad)) with L2-normalised
    centred (optionally Fisher-whitened) gradient features (weight sqrt(alpha_grad)), then cluster.

    Applies an uncentred hybrid raw-dispersion gate (`disp_min=0.035`) so surface-identical controls
    collapse to k=1 while genuine gradient-only (poetry blind-spot) or semantic forks still split.
    """
    E_sem = np.asarray(E_sem, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    lp = np.asarray(logp_per_token, dtype=np.float64)
    if not (len(E_sem) == len(G) == len(lp)):
        raise ValueError(
            f"Length mismatch: len(E_sem)={len(E_sem)}, len(G)={len(G)}, len(logp_per_token)={len(lp)}"
        )
    alpha_grad = float(alpha_grad)
    if not (0.0 <= alpha_grad <= 1.0):
        raise ValueError(f"alpha_grad must be in [0, 1], got {alpha_grad}")
    method = "hybrid_whitened" if whiten else "hybrid_plain"
    if len(E_sem) == 0:
        return dict(labels=[], k=0, silhouette=0.0, readings=[], shares={}, mu=[], mu_grad=[],
                    n_dropped=0, method=method, alpha_grad=alpha_grad,
                    raw_disp=0.0, raw_disp_sem=0.0, raw_disp_grad=0.0)

    keep = lp >= np.quantile(lp, drop_q)
    if not np.any(keep):
        keep = np.ones(len(G), dtype=bool)
    E_k = E_sem[keep]
    mu_sem = E_k.mean(0)
    E_unit = _row_normalize(E_k)

    Xg = _grad_features(G[keep], whiten=whiten, eps=eps)
    mu_grad = Xg.mean(0)

    w_sem = np.sqrt(1.0 - alpha_grad)
    w_grad = np.sqrt(alpha_grad)
    raw_disp_sem = raw_dispersion(E_k)
    raw_disp_grad = raw_dispersion(Xg)
    raw_disp = raw_dispersion(np.hstack([w_sem * E_unit, w_grad * Xg]))

    labels = np.full(len(G), -1, dtype=int)
    if len(E_k) < 2 or (disp_min is not None and raw_disp < float(disp_min)):
        labels[keep] = 0
        return {
            "labels": labels.tolist(),
            "k": 1,
            "silhouette": 0.0,
            "readings": [0],
            "shares": {0: 1.0},
            "mu": mu_sem.tolist(),
            "mu_grad": mu_grad.tolist(),
            "n_dropped": int((~keep).sum()),
            "method": method,
            "alpha_grad": alpha_grad,
            "raw_disp": float(raw_disp),
            "raw_disp_sem": float(raw_disp_sem),
            "raw_disp_grad": float(raw_disp_grad),
        }

    Z_sem = centre(E_k, mu_sem)
    Z_grad = centre(Xg, mu_grad)
    H = _row_normalize(np.hstack([w_sem * Z_sem, w_grad * Z_grad]))

    c = cluster(H, **kw)
    labels[keep] = c["labels"]
    return {
        **c,
        "labels": labels.tolist(),
        "mu": mu_sem.tolist(),
        "mu_grad": mu_grad.tolist(),
        "n_dropped": int((~keep).sum()),
        "method": method,
        "alpha_grad": alpha_grad,
        "raw_disp": float(raw_disp),
        "raw_disp_sem": float(raw_disp_sem),
        "raw_disp_grad": float(raw_disp_grad),
    }


def _orthonormal_append(axes_list, candidates, singular_vals, target_k):
    if len(axes_list) >= target_k or len(singular_vals) == 0:
        return
    s0 = max(float(singular_vals[0]), 1e-30)
    for j, sv in enumerate(singular_vals):
        if len(axes_list) >= target_k:
            break
        if float(sv) <= 1e-10 * s0:
            break
        v = candidates[j].astype(np.float64).copy()
        for u in axes_list:
            v -= float(np.dot(v, u)) * u
        nv = float(np.linalg.norm(v))
        if nv > 1e-10:
            axes_list.append(v / nv)


def spectral_basis(G, labels, readings, n_dims=2, whiten=True, eps=1e-2):
    """Compute reading mean vectors M (optionally Fisher-whitened natural directions relative to global mean),
    top-`n_dims` orthonormal principal axes (k_dims, d) in ambient d-space, normalized anchors {r: [...]},
    projected per-sample dots {r: [[...], ...]}, and variance_explained.
    """
    G = np.asarray(G, dtype=np.float64)
    labels = np.asarray(labels)
    rd = [int(r) for r in readings]
    if not rd:
        raise ValueError("readings must be non-empty")
    d = G.shape[1]

    member_idx = {}
    slices = {}
    all_parts = []
    offset = 0
    for r in rd:
        idx = np.where(labels == r)[0]
        if len(idx) == 0:
            raise ValueError(f"Reading {r} has no member samples in labels")
        member_idx[r] = idx
        all_parts.append(G[idx])
        slices[r] = slice(offset, offset + len(idx))
        offset += len(idx)

    Z = np.vstack(all_parts)
    m = len(Z)
    R = len(rd)
    M_raw = np.array([Z[slices[r]].mean(0) for r in rd], dtype=np.float64)
    c_raw = M_raw.mean(0)

    if whiten:
        K = Z @ Z.T
        K = 0.5 * (K + K.T)
        eigvals = np.linalg.eigvalsh(K)
        emax = float(eigvals.max()) if len(eigvals) else 0.0
        rank = max(int(np.sum(eigvals > 1e-12 * max(emax, 1e-30))), 1)
        lam = float(eps) * (float(np.trace(K)) / m) / rank
        A = K / m + max(lam, 1e-12) * np.eye(m)

        # c_bar gives the global mean of reading means: c_raw = Z^T c_bar
        c_bar = np.zeros(m, dtype=np.float64)
        C = np.zeros((m, R), dtype=np.float64)
        for j, r in enumerate(rd):
            sl = slices[r]
            nr = sl.stop - sl.start
            c_bar[sl] = 1.0 / (R * nr)
            C[sl, j] = 1.0 / nr
        C -= c_bar[:, None]

        # Natural reading directions relative to global mean: (G_hat + lam I)^{-1} (m_r - c_raw)
        W_means = np.linalg.solve(A, C)               # (m, R)
        M = W_means.T @ Z                             # (R, d)

        # Sample natural displacements relative to global mean: (G_hat + lam I)^{-1} (g_i - c_raw)
        E_c = np.eye(m, dtype=np.float64) - c_bar[:, None]
        W_samp = np.linalg.solve(A, E_c)              # (m, m)
    else:
        M = M_raw - c_raw
        W_samp = None

    _, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
    total_var_m = float((s_m ** 2).sum())
    axes_list = []
    _orthonormal_append(axes_list, Vt_m, s_m, int(n_dims))
    n_from_m = len(axes_list)
    var_expl = float((s_m[:n_from_m] ** 2).sum() / total_var_m) if total_var_m > 1e-30 else 1.0

    # When fewer than n_dims independent reading directions exist (e.g. R == 2, n_dims == 2),
    # fill remaining axes from orthogonal within-reading sample variance so the plane is well-defined.
    if len(axes_list) < int(n_dims):
        V_samp = (W_samp.T @ Z) if whiten else (Z - c_raw)
        if axes_list:
            A_cur = np.vstack(axes_list)
            V_res = V_samp - (V_samp @ A_cur.T) @ A_cur
        else:
            V_res = V_samp
        _, s_r, Vt_r = np.linalg.svd(V_res, full_matrices=False)
        _orthonormal_append(axes_list, Vt_r, s_r, int(n_dims))

    axes = np.vstack(axes_list) if axes_list else np.zeros((0, d), dtype=np.float64)
    anchors_raw = M @ axes.T                          # (R, k_dims)
    dots_all = (W_samp.T @ (Z @ axes.T)) if whiten else ((Z - c_raw) @ axes.T)

    scale = float(np.linalg.norm(anchors_raw, axis=1).max()) if len(anchors_raw) else 0.0
    if scale <= 1e-12:
        scale = float(np.linalg.norm(dots_all, axis=1).max()) if len(dots_all) else 1.0
    if scale <= 1e-12:
        scale = 1.0

    anchors = {int(r): (anchors_raw[i] / scale).tolist() for i, r in enumerate(rd)}
    dots = {int(r): (dots_all[slices[r]] / scale).tolist() for r in rd}

    return dict(
        axes=axes,
        anchors=anchors,
        dots=dots,
        variance_explained=float(var_expl),
        M=M,
        scale=float(scale),
        readings=rd,
    )
