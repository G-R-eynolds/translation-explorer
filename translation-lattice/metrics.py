"""Distances, kernels, clustering, validation and the clusters_*.json outputs.

    python metrics.py --model 3b --stage spine      # M1 clusters before controls exist -> central route
    python metrics.py --model 3b --stage all        # everything, after controls.py and layers.py

Metrics (all give a distance matrix per line, and a distance per control pair):
  m1   back-translation Jensen-Shannon divergence (from score.py's L matrices)
  m2   cosine distance of LEACE-erased hidden states at the selected layer (from layers.py)
  b0   char 2-4-gram TF-IDF cosine distance
  b1   cosine distance of raw hidden states (from layers.py)
  b2   NLI: 1 - mean bidirectional entailment probability (microsoft/deberta-large-mnli)
"""
import argparse, itertools, json
from collections import Counter, defaultdict

import numpy as np

from common import DATA, MODELS, load_json, run_dir, save_json

LN2 = float(np.log(2))


# ---------------------------------------------------------------- M1: back-translation JSD
def jsd_from_L(L, owner, n_boot=200, seed=0):
    """MC estimate of JSD(q_i || q_j) for all pairs, with bootstrap SE over back-translations.

    L[j, b] = log q_j(b); owner[b] = index of the unit whose q sampled b.
    T[i, j] = mean_{b in B_i} [L[i,b] - log M_ij(b)],  log M = logaddexp(L_i, L_j) - log 2
    JSD_ij  = (T[i,j] + T[j,i]) / 2, clamped to [0, ln 2].
    """
    N = L.shape[0]
    idx = [np.where(owner == i)[0] for i in range(N)]

    def T_of(cols):
        T = np.zeros((N, N))
        for i in range(N):
            Li = L[i, cols[i]][None, :]                  # (1, m)
            Lall = L[:, cols[i]]                         # (N, m)
            T[i] = (Li - (np.logaddexp(Li, Lall) - LN2)).mean(1)
        return T

    T = T_of(idx)
    D = np.clip((T + T.T) / 2, 0, LN2)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        Tb = T_of([rng.choice(c, len(c), replace=True) for c in idx])
        boots.append(np.clip((Tb + Tb.T) / 2, 0, LN2))
    se = np.std(boots, 0)
    np.fill_diagonal(D, 0); np.fill_diagonal(se, 0)
    return D, se


def jeffreys_rate_from_L(L, owner, lens, n_boot=200, seed=0):
    """MC estimate of the symmetric KL divergence RATE (nats/token) between back-translation processes:
    J_ij = 1/2 E_{b~q_i}[(log q_i(b) - log q_j(b)) / |b|] + 1/2 E_{b~q_j}[(log q_j(b) - log q_i(b)) / |b|].
    Unlike string-level JSD it does not saturate when small per-token wording preferences add up."""
    N = L.shape[0]
    idx = [np.where(owner == i)[0] for i in range(N)]
    R = L / lens[None, :]

    def T_of(cols):
        return np.stack([(R[i, cols[i]][None, :] - R[:, cols[i]]).mean(1) for i in range(N)])

    T = T_of(idx)
    J = np.clip((T + T.T) / 2, 0, None)
    rng = np.random.default_rng(seed)
    boots = [np.clip((lambda Tb: (Tb + Tb.T) / 2)(T_of([rng.choice(c, len(c)) for c in idx])), 0, None)
             for _ in range(n_boot)]
    se = np.std(boots, 0)
    np.fill_diagonal(J, 0); np.fill_diagonal(se, 0)
    return J, se


def load_m1(model_key, name, kind="m1"):
    """{group: (D, SE, uids)} for one L_<name>.npz.  kind: 'm1' string-level JSD, 'm1r' KL rate."""
    z = np.load(run_dir(model_key) / "results" / f"L_{name}.npz")
    out = {}
    for k in z.files:
        if k.endswith("__L"):
            g = k[:-3]
            L = z[f"{g}__L"].astype(np.float64)
            if kind == "m1":
                D, se = jsd_from_L(L, z[f"{g}__owner"])
            else:
                D, se = jeffreys_rate_from_L(L, z[f"{g}__owner"], z[f"{g}__len"].astype(np.float64))
            out[g] = (D, se, [str(u) for u in z[f"{g}__uids"]])
    return out


# ---------------------------------------------------------------- kernels, entropy, clustering
def kernel(D, metric, lam=None):
    """Similarity kernel: 1 - JSD/ln2 for M1, exp(-J/lam) for the KL rate M1r (lam = tau, the 95th
    percentile of J over style-only pairs, so style-level differences get similarity >= 1/e; the spine
    stage, before controls exist, uses the lattice median), 1 - distance otherwise."""
    if metric == "m1":
        return 1 - D / LN2
    if metric == "m1r":
        return np.exp(-D / lam)
    return 1 - D


def kle(K):
    """Kernel Language Entropy (von Neumann entropy of K / tr K), in nats. Negative eigenvalues
    (K need not be PSD for M1/NLI kernels) are clipped and the spectrum renormalised."""
    rho = (K + K.T) / 2 / np.trace(K)
    ev = np.clip(np.linalg.eigvalsh(rho), 0, None)
    ev = ev / ev.sum()
    ev = ev[ev > 1e-12]
    return float(-(ev * np.log(ev)).sum())


def eigengap_k(K, kmax=5):
    A = np.clip((K + K.T) / 2, 0, None)
    np.fill_diagonal(A, 1)
    d = A.sum(1)
    Dm = np.diag(1 / np.sqrt(d))
    Lap = np.eye(len(A)) - Dm @ A @ Dm
    ev = np.sort(np.linalg.eigvalsh(Lap))[: kmax + 1]
    gaps = np.diff(ev)              # gaps[k-1] = lambda_{k+1} - lambda_k  (1-indexed)
    return int(np.argmax(gaps[:kmax]) + 1), ev.tolist()


def spectral(K, k, seed=0):
    from sklearn.cluster import SpectralClustering
    if k == 1:
        return np.zeros(len(K), dtype=int)
    A = np.clip((K + K.T) / 2, 0, None)
    np.fill_diagonal(A, 1)
    return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed,
                              assign_labels="cluster_qr").fit_predict(A)


def cluster_line(D, metric, tau=None, kmax=5, k_fixed=None, lam=None):
    """Returns labels, k, info.  k=1 if every pairwise distance is below tau (the style-noise level)."""
    K = kernel(D, metric, lam)
    k_eg, ev = eigengap_k(K, kmax)
    off = D[np.triu_indices(len(D), 1)]
    if k_fixed is not None:
        k = k_fixed
    elif tau is not None and off.max() < tau:
        k = 1
    else:
        k = k_eg
    lab = spectral(K, k)
    return lab, k, dict(eigengap_k=k_eg, laplacian_eigs=ev, frac_pairs_above_tau=None if tau is None else
                        float((off > tau).mean()))


def tau_cut(D, tau):
    """Average-linkage agglomerative clustering cut at tau: two groups stay apart only if their mean
    pairwise distance exceeds the style-noise level.  k=1 iff the whole line merges below tau."""
    from scipy.cluster.hierarchy import average, fcluster
    from scipy.spatial.distance import squareform
    Z = average(squareform(np.clip((D + D.T) / 2, 0, None), checks=False))  # float32 cosines can dip below 0
    return fcluster(Z, t=tau, criterion="distance") - 1


def medoid(D, members):
    sub = D[np.ix_(members, members)]
    return members[int(np.argmin(sub.sum(1)))]


def entropy_bits(labels):
    p = np.array(list(Counter(labels).values()), float); p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def nli_greedy(P_ent):
    """Kuhn et al. (2023) semantic-entropy clustering: i joins the first class whose representative
    it bidirectionally entails (argmax label = entailment both ways)."""
    reps, lab = [], []
    for i in range(len(P_ent)):
        for c, r in enumerate(reps):
            if P_ent[i, r] > 0.5 and P_ent[r, i] > 0.5:
                lab.append(c); break
        else:
            reps.append(i); lab.append(len(reps) - 1)
    return np.array(lab)


# ---------------------------------------------------------------- baselines
def lexical_D(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    return np.clip(cosine_distances(X), 0, 1)


class NLI:
    def __init__(self, name="microsoft/deberta-large-mnli"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval().half()
        lab = {v.lower(): k for k, v in self.m.config.id2label.items()}
        self.ent, self.con = lab["entailment"], lab["contradiction"]
        self.name = name

    def probs(self, pairs, bs=128):
        out = []
        with self.torch.no_grad():
            for c in range(0, len(pairs), bs):
                a, b = zip(*pairs[c:c + bs])
                enc = self.tok(list(a), list(b), return_tensors="pt", padding=True, truncation=True).to("cuda")
                out.append(self.torch.softmax(self.m(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def matrix(self, texts):
        """P_ent[i, j] = P(texts[i] entails texts[j])."""
        n = len(texts)
        pairs = [(texts[i], texts[j]) for i in range(n) for j in range(n) if i != j]
        p = self.probs(pairs)[:, self.ent]
        P = np.eye(n)
        P[~np.eye(n, dtype=bool)] = p
        return P


def nli_D(P):
    return np.clip(1 - (P + P.T) / 2, 0, 1)


# ---------------------------------------------------------------- outputs
def clusters_json(per_line, samples, clusterer):
    """per_line[i] = dict(D, labels, uids) over this run's samples (uids 'SID:i')."""
    lines = []
    for i, r in sorted(per_line.items()):
        sids = [u.split(":")[0] for u in r["uids"]]
        cl = []
        for c in sorted(set(r["labels"]), key=lambda c: -(r["labels"] == c).sum()):
            mem = list(np.where(r["labels"] == c)[0])
            med = medoid(r["D"], mem)
            text = next(s for s in samples if s["id"] == sids[med])["lines"][i]
            cl.append(dict(id=f"c{len(cl) + 1}", gloss=f"“{text}” (medoid {sids[med]}; {len(mem)} samples)",
                           members=[sids[x] for x in mem], medoid=sids[med]))
        lines.append(dict(line=i, note="", clusters=cl))
    return dict(clusterer=clusterer, lines=lines)


# ================================================================ driver
METRICS = ["m1", "m1r", "m2", "b0", "b1", "b2"]
NAMES = {"m1": "M1 back-translation JSD (string level)", "m1r": "M1r back-translation KL rate (nats/token)",
         "m2": "M2 hidden states, LEACE-erased", "b0": "B0 lexical char-n-gram", "b1": "B1 hidden states, raw",
         "b2": "B2 NLI bidirectional entailment"}


def lattice_lam(per_line):
    return float(np.median(np.concatenate([D[np.triu_indices(len(D), 1)] for D in per_line.values()])))


def stage_spine(a):
    rd = run_dir(a.model)
    fam = MODELS[a.model]["family"]
    samples = load_json(rd / f"samples_{fam}.json")["samples"]
    m1r = load_m1(a.model, "lattice", "m1r")
    per = {int(g[4:]): D for g, (D, se, u) in m1r.items()}
    uids = {int(g[4:]): u for g, (D, se, u) in m1r.items()}
    lam = lattice_lam(per)
    spine_ids, spine_lines, info = [], [], []
    for i in sorted(per):
        lab, k, inf = cluster_line(per[i], "m1r", lam=lam)
        big = Counter(lab).most_common(1)[0][0]
        med = medoid(per[i], list(np.where(lab == big)[0]))
        sid = uids[i][med].split(":")[0]
        spine_ids.append(sid)
        spine_lines.append(next(s for s in samples if s["id"] == sid)["lines"][i])
        info.append(dict(line=i, k=k, largest=int((lab == big).sum())))
    save_json(rd / "spine.json", dict(description="Central route: medoid of the largest M1r cluster per line "
                                      "(eigengap spectral clustering, no tau), used as the fixed common prefix for M2.",
                                      lam=lam, sample_ids=spine_ids, lines=spine_lines, info=info))
    for i, (sid, t) in enumerate(zip(spine_ids, spine_lines)):
        print(f"{i + 1:2d} {sid}  {t}")


def auroc(d_style, d_mean):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(d_style)), np.ones(len(d_mean))]
    return float(roc_auc_score(y, np.r_[d_style, d_mean])) if len(d_style) and len(d_mean) else None


def stage_all(a):
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score
    rd = run_dir(a.model)
    fam = MODELS[a.model]["family"]
    samples = load_json(rd / f"samples_{fam}.json")["samples"]
    demo_samples = load_json(DATA / "samples.json")["samples"]
    hand = load_json(DATA / "clusters.json")
    ctrl = load_json(rd / "controls.json")
    items = {it["id"]: it for it in ctrl["items"]}
    pairs = ctrl["pairs"]
    lat_texts = {i: [s["lines"][i] for s in samples] for i in range(14)}
    lat_ids = [s["id"] for s in samples]
    demo_texts = {i: [s["lines"][i] for s in demo_samples] for i in range(14)}
    demo_ids = [s["id"] for s in demo_samples]
    text_of = {**{f"{s['id']}:{i}": s["lines"][i] for s in demo_samples for i in range(14)},
               **{k: v["text"] for k, v in items.items()}}

    LAT, DEMO, PAIR, SE = {}, {}, {}, {}   # metric -> {line: D}, {line: D}, {pair_id: d}
    # ---- M1 / M1r
    for kind in ("m1", "m1r"):
        lat = load_m1(a.model, "lattice", kind)
        dem = load_m1(a.model, "demo", kind)
        con = load_m1(a.model, "controls", kind)
        LAT[kind] = {int(g[4:]): D for g, (D, se, u) in lat.items()}
        SE[kind] = {int(g[4:]): se for g, (D, se, u) in lat.items()}
        assert all(lat[f"line{i}"][2] == [f"{x}:{i}" for x in lat_ids] for i in range(14))
        DEMO[kind] = {int(g[4:]): D for g, (D, se, u) in dem.items()}
        PAIR[kind] = {}
        for p in pairs:
            if p["set"] == "demo":
                D = DEMO[kind][p["line"]]
                PAIR[kind][p["id"]] = float(D[demo_ids.index(p["a"].split(":")[0]), demo_ids.index(p["b"].split(":")[0])])
            else:
                D, se, u = con[items[p["a"]]["family"]]
                PAIR[kind][p["id"]] = float(D[u.index(p["a"]), u.index(p["b"])])
    # ---- B0 lexical (TF-IDF fitted per line on every text of that line)
    LAT["b0"], DEMO["b0"], PAIR["b0"] = {}, {}, {}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
    for i in range(14):
        ctl = [k for k, v in items.items() if v["line"] == i]
        corpus = lat_texts[i] + demo_texts[i] + [items[k]["text"] for k in ctl]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(corpus)
        f = lambda T: vec.transform(T)
        LAT["b0"][i] = np.clip(cosine_distances(f(lat_texts[i])), 0, 1)
        DEMO["b0"][i] = np.clip(cosine_distances(f(demo_texts[i])), 0, 1)
        for p in pairs:
            if p["line"] == i:
                PAIR["b0"][p["id"]] = float(np.clip(cosine_distances(f([text_of[p["a"]]]), f([text_of[p["b"]]]))[0, 0], 0, 1))
    # ---- B2 NLI
    nli = NLI()
    LAT["b2"], DEMO["b2"], PAIR["b2"], PENT, DPENT = {}, {}, {}, {}, {}
    for i in range(14):
        PENT[i] = nli.matrix(lat_texts[i])
        LAT["b2"][i] = nli_D(PENT[i])
        DPENT[i] = nli.matrix(demo_texts[i])
        DEMO["b2"][i] = nli_D(DPENT[i])
    pp = [(text_of[p["a"]], text_of[p["b"]]) for p in pairs]
    fw = nli.probs(pp)[:, nli.ent]
    bw = nli.probs([(y, x) for x, y in pp])[:, nli.ent]
    for p, x, y in zip(pairs, fw, bw):
        PAIR["b2"][p["id"]] = float(1 - (x + y) / 2)
    del nli
    # ---- M2 / B1 from layers.py; layer chosen by AUROC on the generated controls, cross-fitted by line parity
    have_m2 = (rd / "results" / "m2_distances.npz").exists()
    active = METRICS if have_m2 else [m for m in METRICS if m not in ("m2", "b1")]
    layer_auc = {}
    z = np.load(rd / "results" / "m2_distances.npz") if have_m2 else None
    nL = 0 if not have_m2 else 1 + max(int(k.split("|")[2]) for k in z.files if "|controls|" in k)
    ctl_pairs = [p for p in pairs if p["set"] == "controls"]
    for rname, metric in ((("erased", "m2"), ("raw", "b1")) if have_m2 else ()):
        per_layer = []
        for l in range(nL):
            d = dict(zip([str(x) for x in z[f"{rname}|controls|{l}|ids"]], z[f"{rname}|controls|{l}|d"]))
            per_layer.append(d)
        aucs = {}
        for l, d in enumerate(per_layer):
            for par in (0, 1, None):
                sel = [p for p in ctl_pairs if par is None or p["line"] % 2 == par]
                aucs[(l, par)] = auroc([d[p["id"]] for p in sel if p["type"] == "style"],
                                       [d[p["id"]] for p in sel if p["type"] == "meaning"])
        best = {par: max(range(nL), key=lambda l: aucs[(l, par)]) for par in (0, 1, None)}
        # cross-fitted: choose on one parity, evaluate on the other
        xfit = float(np.mean([aucs[(best[0], 1)], aucs[(best[1], 0)]]))
        lsel = best[None]
        layer_auc[metric] = dict(per_layer=[aucs[(l, None)] for l in range(nL)], selected_layer=lsel,
                                 best_even=best[0], best_odd=best[1], crossfit_auroc=xfit)
        LAT[metric] = {i: z[f"{rname}|lattice|{i}|{lsel}"].astype(np.float64) for i in range(14)}
        DEMO[metric] = {i: z[f"{rname}|demo|{i}|{lsel}"].astype(np.float64) for i in range(14)}
        PAIR[metric] = dict(per_layer[lsel])
        for p in pairs:
            if p["set"] == "demo":
                D = DEMO[metric][p["line"]]
                PAIR[metric][p["id"]] = float(D[demo_ids.index(p["a"].split(":")[0]), demo_ids.index(p["b"].split(":")[0])])

    # ---- validation
    val = {}
    lam = {m: None for m in active}
    for m in active:
        r = {}
        for block in ("controls", "controls_ownctx", "demo", "all"):
            if block == "controls_ownctx":   # own-context restyles vs the same meaning edits
                sel = [p for p in pairs if p["set"] == block or (p["set"] == "controls" and p["type"] == "meaning")]
            else:
                sel = [p for p in pairs if (block == "all" and p["set"] != "controls_ownctx") or p["set"] == block]
            r[f"auroc_{block}"] = auroc([PAIR[m][p["id"]] for p in sel if p["type"] == "style"],
                                        [PAIR[m][p["id"]] for p in sel if p["type"] == "meaning"])
        st = np.array([PAIR[m][p["id"]] for p in ctl_pairs if p["type"] == "style"])
        me = np.array([PAIR[m][p["id"]] for p in ctl_pairs if p["type"] == "meaning"])
        tau = float(np.percentile(st, 95))
        r.update(tau=tau, meaning_above_tau=float((me > tau).mean()), style_median=float(np.median(st)),
                 meaning_median=float(np.median(me)), n_style=len(st), n_meaning=len(me))
        if m in layer_auc:
            r["auroc_controls_crossfit_layer"] = layer_auc[m]["crossfit_auroc"]
            r["layer"] = layer_auc[m]["selected_layer"]
        # ARI vs hand clusters (weak reference), own k (eigengap / NLI greedy) and hand k
        ari_own, ari_k, ari_alt = [], [], []
        if m == "m1r":
            lam[m] = tau  # kernel exp(-J / tau): a pair at the style-noise level has similarity 1/e
        lam_demo = lam[m]
        for L in hand["lines"]:
            i = L["line"]
            hl = {mm: c["id"] for c in L["clusters"] for mm in c["members"]}
            href = [hl[x] for x in demo_ids]
            own = tau_cut(DEMO[m][i], tau)
            if m == "b2":
                alt = nli_greedy(DPENT[i])
            else:
                alt, _, _ = cluster_line(DEMO[m][i], m, tau=tau, lam=lam_demo)
            ari_alt.append(adjusted_rand_score(href, alt))
            fixed, _, _ = cluster_line(DEMO[m][i], m, k_fixed=len(L["clusters"]), lam=lam_demo)
            ari_own.append(adjusted_rand_score(href, own)); ari_k.append(adjusted_rand_score(href, fixed))
        r.update(ari_hand_own_k=float(np.mean(ari_own)), ari_hand_hand_k=float(np.mean(ari_k)),
                 ari_hand_alt=float(np.mean(ari_alt)))
        val[m] = r
    # ---- lattice clusters + per-line stats
    per_line, labels = defaultdict(dict), {}
    for m in active:
        tau = val[m]["tau"]
        pl = {}
        for i in range(14):
            D = LAT[m][i]
            if m == "b2":
                lab_sp = nli_greedy(PENT[i]); inf = {}
            else:
                lab_sp, _, inf = cluster_line(D, m, tau=tau, lam=lam[m])
            lab = tau_cut(D, tau)
            k = len(set(lab))
            labels[(m, i)] = lab
            off = D[np.triu_indices(len(D), 1)]
            per_line[i][m] = dict(k=int(k), kle=kle(kernel(D, m, lam[m])), sem_entropy_bits=entropy_bits(lab),
                                  frac_pairs_above_tau=float((off > tau).mean()), mean_d=float(off.mean()),
                                  eigengap_k=inf.get("eigengap_k"), k_alt=int(len(set(lab_sp))),
                                  n_shared_clusters=int(sum(c >= 2 for c in Counter(lab).values())),
                                  largest_cluster=int(max(Counter(lab).values())),
                                  mean_boot_se=float(SE[m][i][np.triu_indices(len(D), 1)].mean()) if m in SE else None)
            pl[i] = dict(D=D, labels=lab, uids=[f"{x}:{i}" for x in lat_ids])
        cj = clusters_json(pl, samples, f"{NAMES[m]}; average-linkage clustering cut at tau={tau:.4g} "
                           "(95th percentile of this metric over style-only control pairs)")
        save_json(rd / f"clusters_{m}.json", cj)
    # ---- agreement among metrics on the lattice
    agree = {}
    for x, y in itertools.combinations(active, 2):
        dx = np.concatenate([LAT[x][i][np.triu_indices(48, 1)] for i in range(14)])
        dy = np.concatenate([LAT[y][i][np.triu_indices(48, 1)] for i in range(14)])
        agree[f"{x}~{y}"] = dict(spearman=float(spearmanr(dx, dy)[0]),
                                 mean_ari=float(np.mean([adjusted_rand_score(labels[(x, i)], labels[(y, i)]) for i in range(14)])))
    # stance vs cluster (does the branch just track the stance?)
    st_lab = [s["stance"] for s in samples]
    for m in active:
        val[m]["ari_clusters_vs_stance"] = float(np.mean([adjusted_rand_score(st_lab, labels[(m, i)]) for i in range(14)]))
    save_json(rd / "results" / "validation.json", dict(metrics=NAMES, validation=val, layer_auroc=layer_auc,
                                                       agreement=agree, per_line=per_line, lam_m1r=lam["m1r"]))
    np.savez_compressed(rd / "results" / "lattice_distances.npz",
                        **{f"{m}|{i}": LAT[m][i] for m in active for i in range(14)})
    print(json.dumps(val, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--stage", choices=["spine", "all"], required=True)
    a = ap.parse_args()
    {"spine": stage_spine, "all": stage_all}[a.stage](a)
