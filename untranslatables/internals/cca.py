"""N1: the round-trip operator spectrum (HANDOFF_internals §3).  Pure torch; runs on CPU or GPU.

Given paired centred states (x_i, y_i) at one layer, with covariances Sxx, Syy, Sxy and ridge
Sxx(lam) = Sxx + lam * tr(Sxx)/d * I:
  Wx = Sxx(lam)^(-1/2),  K = Wx Sxy Wy = U P V^T,  canonical directions A = Wx U (columns a_k), B = Wy V.
  The round trip M = B_map A_map = Sxy Syy(lam)^-1 Syx Sxx(lam)^-1 has eigenvalues rho_k^2 (the identity in §3.1).
  Held-out rho_k = corr(a_k^T x, b_k^T y) on TEST pairs, directions fitted on FIT pairs; only held-out values are
  reported, and the retention/U scores use clip(held-out rho, 0, 1).
  Per-span: c = A^T (x - mu_x) (whitened canonical coordinates); R_n = sum rho_k^(2n) c_k^2 / sum c_k^2;
  U = 1 - R_1; half-life h = n with R_n = 1/2 (interpolated, capped at 50); U_raw = |x - Mx|^2 / |x|^2 (centred raw).
  English bias: E_X = tr(Sxy Syy(lam)^-1 Syx) / tr Sxx (share of X's variance linearly predictable from Y), E_Y
  likewise; also as held-out R^2, and in per-dimension standardised coordinates.
"""
import json

import numpy as np
import torch

from internals.common import cache_dir, res_dir, stage

LAM_GRID = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
TOP = 50
H_CAP = 50.0


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ covariances from streamed sums
def covs(acc, subtract=None):
    """acc: dict of float64 tensors (shifted sums) -> (mu_x, mu_y, Sxx, Syy, Sxy, Sxy_null, n).
    Shifted-data algorithm: sums are of (x - c); Sigma = (S - n d d^T)/n with d = mean(x - c)."""
    g = lambda k: acc[k] - (subtract[k] if subtract is not None else 0)
    n = float(g("n"))
    dx, dy = g("sx") / n, g("sy") / n
    Sxx = g("xx") / n - torch.outer(dx, dx)
    Syy = g("yy") / n - torch.outer(dy, dy)
    Sxy = g("xy") / n - torch.outer(dx, dy)
    Sxn = acc["xyn"] / n - torch.outer(dx, acc["syn"] / n) if "xyn" in acc and subtract is None else None
    return acc["cx"] + dx, acc["cy"] + dy, Sxx, Syy, Sxy, Sxn, n


def eig(S):
    w, V = torch.linalg.eigh(S.double())
    return w.clamp_min(0), V


def ridge(w, lam):
    return lam * w.sum() / len(w)


def inv_sqrt(w, V, lam):
    r = ridge(w, lam)
    return (V * (w + r).rsqrt()) @ V.T


def fit(ex, ey, Sxy, lam, device=None):
    """Canonical directions at one ridge value.  Returns dict(A, B, rho_in) as float32 on `device`."""
    device = device or dev()
    Wx = inv_sqrt(*ex, lam).float().to(device)
    Wy = inv_sqrt(*ey, lam).float().to(device)
    K = Wx @ Sxy.float().to(device) @ Wy
    U, P, Vh = torch.linalg.svd(K)
    return dict(A=Wx @ U, B=Wy @ Vh.T, rho_in=P)


def heldout_rho(A, B, Xc, Yc, chunk=4096):
    """Per-direction correlation of the canonical variates on held-out pairs (Xc, Yc centred with FIT means)."""
    d = A.shape[1]
    sxy, sxx, syy = torch.zeros(d, device=A.device), torch.zeros(d, device=A.device), torch.zeros(d, device=A.device)
    mx, my = torch.zeros(d, device=A.device), torch.zeros(d, device=A.device)
    n = 0
    for c in range(0, len(Xc), chunk):
        u = torch.as_tensor(np.asarray(Xc[c:c + chunk]), device=A.device).float() @ A
        v = torch.as_tensor(np.asarray(Yc[c:c + chunk]), device=A.device).float() @ B
        sxy += (u * v).sum(0); sxx += (u * u).sum(0); syy += (v * v).sum(0); mx += u.sum(0); my += v.sum(0)
        n += len(u)
    mx, my = mx / n, my / n
    cov = sxy / n - mx * my
    return (cov / ((sxx / n - mx ** 2).clamp_min(1e-12) * (syy / n - my ** 2).clamp_min(1e-12)).sqrt()).cpu()


def retention(c2, rho2, n):
    """R_n for coefficient energies c2 [..., d] and spectrum rho2 [d]."""
    return (c2 * rho2 ** n).sum(-1) / c2.sum(-1).clamp_min(1e-12)


def half_life(c2, rho2):
    """Real n where R_n = 1/2, by interpolation on a grid; capped at H_CAP."""
    grid = torch.cat([torch.linspace(0, 5, 51), torch.linspace(5.5, H_CAP, 90)])
    R = torch.stack([retention(c2, rho2, float(n)) for n in grid], -1)       # [..., G]
    below = (R <= 0.5)
    out = torch.full(R.shape[:-1], H_CAP)
    has = below.any(-1)
    k = below.float().argmax(-1)
    k0 = (k - 1).clamp_min(0)
    r0, r1 = R.gather(-1, k0[..., None])[..., 0], R.gather(-1, k[..., None])[..., 0]
    g0, g1 = grid[k0], grid[k]
    frac = ((r0 - 0.5) / (r0 - r1).clamp_min(1e-9)).clamp(0, 1)
    out[has] = (g0 + frac * (g1 - g0))[has]
    return out


class Basis:
    """Everything needed to score states at one layer (saved as .npz for the key layers)."""

    def __init__(self, A, rho_ho, mu_x, M=None, SxxA=None, r=None, lam=None):
        self.A, self.rho_ho, self.mu_x, self.M, self.SxxA, self.r, self.lam = A, rho_ho, mu_x, M, SxxA, r, lam

    def coeffs(self, X):
        Xc = torch.as_tensor(np.asarray(X), device=self.A.device).float() - self.mu_x.to(self.A.device).float()
        return Xc @ self.A, Xc

    def scores(self, X, ns=(1, 2, 5, 10)):
        """U, U_raw, half-life, retention curve for each row of X [n, d]."""
        c, Xc = self.coeffs(X)
        c2 = (c ** 2).cpu()
        rho2 = self.rho_ho.clamp(0, 1) ** 2
        out = dict(U=(1 - retention(c2, rho2, 1)).numpy(), h=half_life(c2, rho2).numpy(),
                   R={n: retention(c2, rho2, n).numpy() for n in range(0, 11)})
        if self.M is not None:
            Mx = Xc @ self.M.T
            out["U_raw"] = (((Xc - Mx) ** 2).sum(-1) / (Xc ** 2).sum(-1).clamp_min(1e-12)).cpu().numpy()
        return out

    def save(self, path):
        np.savez(path, A=self.A.cpu().numpy(), rho_ho=self.rho_ho.numpy(), mu_x=self.mu_x.numpy(),
                 M=None if self.M is None else self.M.cpu().numpy(),
                 SxxA=None if self.SxxA is None else self.SxxA.cpu().numpy(), r=self.r, lam=self.lam)

    @classmethod
    def load(cls, path, device=None):
        z = np.load(path, allow_pickle=True)
        device = device or dev()
        t = lambda k: None if z[k].dtype == object else torch.tensor(z[k], device=device)
        return cls(t("A"), torch.tensor(z["rho_ho"]), torch.tensor(z["mu_x"]), t("M"), t("SxxA"), int(z["r"]),
                   float(z["lam"]))


# ------------------------------------------------------------------ English bias
def bias(Sxx, Syy, Sxy, ex, ey, lam):
    """E_X, E_Y in raw and per-dimension standardised coordinates (in-sample formula with ridge)."""
    def share(Sab, Sbb_inv, Sba, Saa):
        return float(torch.trace(Sab @ Sbb_inv @ Sba) / torch.trace(Saa))
    inv = lambda w, V: (V * (w + ridge(w, lam)).reciprocal()) @ V.T
    raw = dict(E_X=share(Sxy, inv(*ey), Sxy.T, Sxx), E_Y=share(Sxy.T, inv(*ex), Sxy, Syy))
    dx, dy = Sxx.diagonal().clamp_min(1e-12).rsqrt(), Syy.diagonal().clamp_min(1e-12).rsqrt()
    Zxx, Zyy, Zxy = dx[:, None] * Sxx * dx, dy[:, None] * Syy * dy, dx[:, None] * Sxy * dy
    std = dict(E_X=share(Zxy, inv(*eig(Zyy)), Zxy.T, Zxx), E_Y=share(Zxy.T, inv(*eig(Zxx)), Zxy, Zyy))
    for d_ in (raw, std):
        d_["bias"] = d_["E_Y"] - d_["E_X"]
    return dict(raw=raw, standardised=std)


def heldout_r2(Sxx, Syy, Sxy, ex, ey, lam, Xc, Yc, device=None):
    """Held-out R^2 of predicting X from Y (E_X) and Y from X (E_Y) with the ridge least-squares maps."""
    device = device or dev()
    inv = lambda w, V: ((V * (w + ridge(w, lam)).reciprocal()) @ V.T).float().to(device)
    Bm = (Sxy.float().to(device) @ inv(*ey))          # x_hat = Bm y
    Am = (Sxy.T.float().to(device) @ inv(*ex))        # y_hat = Am x
    X = torch.as_tensor(np.asarray(Xc), device=device).float(); Y = torch.as_tensor(np.asarray(Yc), device=device).float()
    r2 = lambda T, P: float(1 - ((T - P) ** 2).sum() / (T ** 2).sum())
    return dict(E_X=r2(X, Y @ Bm.T), E_Y=r2(Y, X @ Am.T))


# ------------------------------------------------------------------ one layer, end to end
def analyse_layer(acc, Xte, Yte, first_te=None, acc_first=None, keep_basis=False, robustness=False, device=None):
    """Fit, choose lambda on held-out pairs, null spectrum, bias, optional robustness.  Returns (summary, basis)."""
    device = device or dev()
    mx, my, Sxx, Syy, Sxy, Sxn, n = covs(acc)
    ex, ey = eig(Sxx), eig(Syy)
    Xc = torch.as_tensor(np.asarray(Xte, dtype=np.float32)) - mx.float()
    Yc = torch.as_tensor(np.asarray(Yte, dtype=np.float32)) - my.float()
    per_lam = {}
    best = None
    for lam in LAM_GRID:
        f = fit(ex, ey, Sxy, lam, device)
        rho = heldout_rho(f["A"], f["B"], Xc, Yc)
        s = float(rho[:TOP].sum())
        per_lam[lam] = s
        if best is None or s > best[0]:
            best = (s, lam, f, rho)
    _, lam, f, rho = best
    # null: mismatched pairs (fit on the mismatched cross-covariance, evaluate on mismatched test pairs)
    perm = np.roll(np.arange(len(Yc)), len(Yc) // 2)
    fn = fit(ex, ey, Sxn, lam, device)
    rho_null = heldout_rho(fn["A"], fn["B"], Xc, Yc[perm])
    thr = float(np.quantile(rho_null.numpy(), 0.99))
    shared = rho > thr
    r = int(shared.sum())
    summ = dict(n_fit=n, n_test=len(Xc), lam=lam, ridge_per_lam=per_lam, rho_heldout=rho.numpy().round(5).tolist(),
                rho_insample=f["rho_in"].cpu().numpy().round(5).tolist(), rho_null_heldout_q99=thr,
                rho_null_heldout_top=rho_null.numpy().round(5)[:200].tolist(), n_shared=r,
                n_shared_top_contiguous=int(np.argmin(np.r_[shared.numpy(), False])),
                mean_rho2_shared=float((rho[shared] ** 2).mean()) if r else 0.0,
                sum_top50_heldout=float(rho[:TOP].sum()),
                bias=bias(Sxx, Syy, Sxy, ex, ey, lam), bias_heldout=heldout_r2(Sxx, Syy, Sxy, ex, ey, lam, Xc, Yc, device))
    if robustness:
        summ["robustness"] = robust(acc, acc_first, Sxx, Syy, Sxy, mx, my, ex, ey, lam, Xc, Yc, first_te, thr, device)
    basis = None
    if keep_basis:
        inv = lambda w, V: ((V * (w + ridge(w, lam)).reciprocal()) @ V.T).float().to(device)
        M = Sxy.float().to(device) @ inv(*ey) @ Sxy.T.float().to(device) @ inv(*ex)
        Sl = (Sxx + ridge(ex[0], lam) * torch.eye(len(Sxx), dtype=Sxx.dtype)).float().to(device)
        order = torch.argsort(-rho)                  # shared directions = the held-out-ranked top r
        A_sh = f["A"][:, order[:r].to(f["A"].device)]
        basis = Basis(f["A"], rho, mx, M=M, SxxA=Sl @ A_sh, r=r, lam=lam)
        basis.A_shared = A_sh
    return summ, basis


def robust(acc, acc_first, Sxx, Syy, Sxy, mx, my, ex, ey, lam, Xc, Yc, first_te, thr, device):
    out = {}
    def summary(rho):
        return dict(n_shared=int((rho > thr).sum()), sum_top50=float(rho[:TOP].sum()))
    if acc_first is not None:                                   # (a) drop pairs where either word starts its sentence
        _, _, a, b, c, _, n2 = covs(acc, subtract=acc_first)
        keep = ~np.asarray(first_te, bool)
        f = fit(eig(a), eig(b), c, lam, device)
        out["no_first_word"] = dict(summary(heldout_rho(f["A"], f["B"], Xc[keep], Yc[keep])), n_fit=n2)
    # (b) remove the 5 dimensions with the largest RMS ("massive activations") in each language
    def top5(S, mu):
        return torch.argsort(-(S.diagonal() + mu ** 2))[:5]
    kx, ky = top5(Sxx, mx), top5(Syy, my)
    mxk = torch.ones(len(Sxx), dtype=bool); mxk[kx] = False
    myk = torch.ones(len(Syy), dtype=bool); myk[ky] = False
    f = fit(eig(Sxx[mxk][:, mxk]), eig(Syy[myk][:, myk]), Sxy[mxk][:, myk], lam, device)
    out["drop_massive_dims"] = dict(summary(heldout_rho(f["A"], f["B"], Xc[:, mxk], Yc[:, myk])),
                                    dims_x=kx.tolist(), dims_y=ky.tolist())
    # (c) PCA to 512 dimensions per language before CCA
    k = min(512, len(Sxx) // 2)
    Px, Py = ex[1][:, -k:], ey[1][:, -k:]
    f = fit(eig(Px.T @ Sxx @ Px), eig(Py.T @ Syy @ Py), Px.T @ Sxy @ Py, lam, device)
    out["pca512"] = summary(heldout_rho(f["A"], f["B"], Xc @ Px.float(), Yc @ Py.float()))
    return out


def load_acc(path):
    z = torch.load(path, weights_only=True)
    return z


@stage("cca")
def run(cfg):
    """All stored layers for one model and pair: spectrum json per layer; bases (.npz) for the key layers."""
    m, pair = cfg["model"], cfg["pair"]
    cd = cache_dir(m, "n1", pair)
    meta = json.loads((cd / "meta.json").read_text())
    rd = res_dir(m)
    outs = []
    key = set(meta["key_layers"])
    first_te = np.array([p["first"] for p in meta["test_pairs"]], bool)
    rows = []
    for l in meta["layers"]:
        acc = load_acc(cd / f"acc_L{l}.pt")
        acc_f = load_acc(cd / f"accfirst_L{l}.pt") if (cd / f"accfirst_L{l}.pt").exists() else None
        Xte = np.load(cd / f"test_x_L{l}.npy", mmap_mode="r")
        Yte = np.load(cd / f"test_y_L{l}.npy", mmap_mode="r")
        summ, basis = analyse_layer(acc, Xte, Yte, first_te, acc_f, keep_basis=True,
                                    robustness=l == meta.get("peak_layer") and acc_f is not None)
        summ.update(model=m, pair=pair, layer=l, key_layer=l in key)
        p = rd / f"n1_spectrum_{pair}_L{l}.json"
        p.write_text(json.dumps(summ))
        outs.append(p)
        if basis is not None:
            q = cache_dir(m, "n1", pair) / f"basis_L{l}.npz"
            basis.save(q)
            outs.append(q)
        rows.append((l, summ["n_shared"], summ["bias"]["raw"]["bias"], summ["sum_top50_heldout"]))
        print(f"  L{l:>2}: shared {summ['n_shared']:>5}  top50 {summ['sum_top50_heldout']:.2f}  lam {summ['lam']}  "
              f"bias(E_Y-E_X) {summ['bias']['raw']['bias']:+.3f}", flush=True)
    return outs


@stage("hub")
def hub(cfg):
    """Is English the hub? (§3.3, Q4) es-hi vs es-en vs hi-en on the same FLORES sentences, PCA-128 per language,
    held-out by sentence (80/20); plus Bhojpuri scored with the hi-en maps (Q in §3.3)."""
    from internals.spans import flores_hub
    return flores_hub(cfg)
