"""Small statistics helpers: bootstrap CIs, AUROC with CI, frequency-matched comparisons, permutation nulls."""
import numpy as np


def boot_ci(x, stat=np.mean, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(stat(x)), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2)))


def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def auroc_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    b = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return (auroc(pos, neg), float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975)))


def matched_pairs(pos_cov, neg_cov):
    """Greedy 1:1 nearest-neighbour matching of positives to negatives on standardised covariates
    (e.g. Zipf frequency, token count). Returns list of (i_pos, j_neg)."""
    P, N = np.asarray(pos_cov, float), np.asarray(neg_cov, float)
    mu, sd = np.r_[P, N].mean(0), np.r_[P, N].std(0) + 1e-9
    P, N = (P - mu) / sd, (N - mu) / sd
    D = ((P[:, None] - N[None]) ** 2).sum(-1)
    pairs, used = [], set()
    for i in np.argsort(D.min(1)):
        j = next((j for j in np.argsort(D[i]) if j not in used), None)
        if j is not None:
            used.add(j); pairs.append((int(i), int(j)))
    return pairs


def matched_auroc(pos, neg, pos_cov, neg_cov):
    """AUROC restricted to a covariate-matched subset, plus the residualised AUROC (score regressed on the
    covariates over all items, then compared)."""
    pairs = matched_pairs(pos_cov, neg_cov)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    m = auroc(pos[[i for i, _ in pairs]], neg[[j for _, j in pairs]])
    X = np.c_[np.ones(len(pos) + len(neg)), np.r_[np.asarray(pos_cov, float), np.asarray(neg_cov, float)]]
    y = np.r_[pos, neg]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return dict(matched=m, n_pairs=len(pairs), residualised=auroc(r[:len(pos)], r[len(pos):]))


def null_quantile(value, null):
    """Where a value falls in a permutation/null sample (fraction of null below it)."""
    null = np.asarray(null)
    return float((null < value).mean())


def entropy_bits(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum()) + 0.0


def tvd(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)
