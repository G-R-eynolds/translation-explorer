"""N1, §3.4 Q5-Q6: what lives in the private directions, and is linear enough?  CPU (spaCy / stanza) + torch.

Probes (Q5).  Held-out es-en (hi-en) pairs at the peak-sharing layer.  Each source word gets morphological labels from
spaCy es_core_news_md (stanza for Hindi) by character offsets.  States are expressed in whitened canonical coordinates
c = A^T (x - mu) (one coordinate per canonical direction).  For each binary feature, a logistic probe (balanced classes,
5-fold CV, L2 strength chosen by CV) reports
  shared share  = the probe weight energy on the shared directions (held-out rho above the null) / total, against the
                  random-direction expectation r/d;
  accuracy      with all coordinates, shared-only and private-only.
Features (es): gender (Masc/Fem), number, subjunctive (Mood=Sub vs Ind), imperfect vs preterite (Tense=Imp vs Past,
indicative), formal address (usted/ustedes vs tu/ti/vosotros), dative experiencer (an experiencer verb with a dative
clitic, vs other finite verbs), ser vs estar.  (hi): gender, number, the ne-ergative (word followed by 'ne'),
honorific aap vs tum, dative subject (dative pronoun in a pasand/lag/chahiye construction vs elsewhere).

MLP (Q6).  On the held-out pairs (split by sentence 75/25), PCA-512 per language: ridge linear map vs a 2-layer MLP
(GELU, early stopping), both directions; held-out R^2.  "Recovered" = (R2_mlp - R2_lin) / (1 - R2_lin): the share of
the variance the linear map loses that a nonlinear map gets back.
"""
import json
from collections import defaultdict

import numpy as np
import torch

from internals.cca import Basis, dev
from internals.common import cache_dir, res_dir, stage

ES_EXPERIENCER = {"gustar", "encantar", "faltar", "doler", "importar", "olvidar", "quedar", "parecer", "molestar",
                  "interesar", "apetecer", "preocupar", "sorprender", "costar", "tocar", "pasar"}
ES_DATIVE = {"me", "te", "le", "nos", "os", "les"}
HI_DATIVE = {"मुझे", "तुझे", "तुम्हें", "उसे", "हमें", "उन्हें", "आपको", "इसे", "मुझको", "उसको"}
HI_EXP = {"पसंद", "लगता", "लगती", "लगा", "लगी", "लगते", "चाहिए", "मालूम", "पता"}
MIN_PER_CLASS = 30


def es_labels(rows, pairs):
    import spacy
    nlp = spacy.load("es_core_news_md", disable=["ner"])
    texts = {r["id"]: r["x"] for r in rows}
    docs = {sid: nlp(texts[sid]) for sid in sorted({p["sid"] for p in pairs})}
    lab = []
    for p in pairs:
        doc = docs[p["sid"]]
        t = next((t for t in doc if t.idx <= p["xs"] < t.idx + len(t.text) + 1), None)
        if t is None:
            lab.append({}); continue
        mo = t.morph.to_dict()
        L = {}
        if mo.get("Gender") in ("Masc", "Fem"):
            L["gender"] = int(mo["Gender"] == "Fem")
        if mo.get("Number") in ("Sing", "Plur"):
            L["number"] = int(mo["Number"] == "Plur")
        if t.pos_ in ("VERB", "AUX") and mo.get("Mood") in ("Sub", "Ind"):
            L["subjunctive"] = int(mo["Mood"] == "Sub")
        if t.pos_ in ("VERB", "AUX") and mo.get("Mood") == "Ind" and mo.get("Tense") in ("Imp", "Past"):
            L["imperfect_vs_preterite"] = int(mo["Tense"] == "Imp")
        w = t.text.lower()
        if w in ("usted", "ustedes"):
            L["formal_address"] = 1
        elif w in ("tú", "ti", "vosotros", "vosotras"):
            L["formal_address"] = 0
        if t.pos_ in ("VERB", "AUX") and "Fin" in mo.get("VerbForm", "Fin"):
            prev = {x.text.lower() for x in doc[max(0, t.i - 2):t.i]}
            L["dative_experiencer"] = int(t.lemma_ in ES_EXPERIENCER and bool(prev & ES_DATIVE))
        if t.lemma_ in ("ser", "estar"):
            L["ser_vs_estar"] = int(t.lemma_ == "estar")
        lab.append(L)
    return lab


def hi_labels(rows, pairs):
    import stanza
    try:
        nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma", verbose=False, use_gpu=torch.cuda.is_available())
    except Exception:
        stanza.download("hi", verbose=False)
        nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma", verbose=False, use_gpu=torch.cuda.is_available())
    texts = {r["id"]: r["x"] for r in rows}
    sids = sorted({p["sid"] for p in pairs})
    docs = dict(zip(sids, nlp.bulk_process([texts[s] for s in sids])))
    lab = []
    for p in pairs:
        doc = docs[p["sid"]]
        words = [w for s in doc.sentences for w in s.words]
        k = next((i for i, w in enumerate(words) if w.start_char is not None and w.start_char <= p["xs"] < w.end_char + 1), None)
        if k is None:
            lab.append({}); continue
        w = words[k]
        f = dict(x.split("=") for x in (w.feats or "").split("|") if "=" in x)
        L = {}
        if f.get("Gender") in ("Masc", "Fem"):
            L["gender"] = int(f["Gender"] == "Fem")
        if f.get("Number") in ("Sing", "Plur"):
            L["number"] = int(f["Number"] == "Plur")
        nxt = words[k + 1].text if k + 1 < len(words) else ""
        if w.upos in ("NOUN", "PROPN", "PRON"):
            L["ne_ergative"] = int(nxt == "ने")
        if w.text in ("आप", "आपका", "आपके", "आपकी", "आपको", "आपने"):
            L["honorific_aap_vs_tum"] = 1
        elif w.text in ("तुम", "तुम्हारा", "तुम्हारे", "तुम्हारी", "तुम्हें", "तुमने", "तू"):
            L["honorific_aap_vs_tum"] = 0
        if w.text in HI_DATIVE:
            L["dative_subject"] = int(any(x.text in HI_EXP for x in words[k + 1:k + 5]))
        lab.append(L)
    return lab


def probe(C, y, shared, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    n = min(len(pos), len(neg), 2000)
    idx = np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
    X, Y = C[idx], y[idx]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = max((0.001, 0.01, 0.1), key=lambda c: cross_val_score(LogisticRegression(C=c, max_iter=2000), X, Y, cv=cv).mean())
    acc = lambda Z: float(cross_val_score(LogisticRegression(C=best, max_iter=2000), Z, Y, cv=cv).mean())
    w = LogisticRegression(C=best, max_iter=2000).fit(X, Y).coef_[0]
    return dict(n_per_class=int(n), C=best, shared_share=float((w[shared] ** 2).sum() / (w ** 2).sum()),
                random_expectation=float(shared.mean()), acc_all=acc(X), acc_shared=acc(X[:, shared]) if shared.any() else None,
                acc_private=acc(X[:, ~shared]))


@stage("probes")
def run(cfg):
    from internals.data import read_corpus
    m = cfg["model"]
    out = {}
    for pair, labeller in (("es-en", es_labels), ("hi-en", hi_labels)):
        cd = cache_dir(m, "n1", pair)
        if not (cd / "meta.json").exists():
            continue
        meta = json.loads((cd / "meta.json").read_text())
        l = meta["peak_layer"]
        B = Basis.load(cd / f"basis_L{l}.npz")
        spec = json.loads((res_dir(m) / f"n1_spectrum_{pair}_L{l}.json").read_text())
        shared = np.asarray(spec["rho_heldout"]) > spec["rho_null_heldout_q99"]
        X = np.load(cd / f"test_x_L{l}.npy", mmap_mode="r")
        C = np.concatenate([B.coeffs(X[c:c + 4096])[0].cpu().numpy() for c in range(0, len(X), 4096)])
        rows = [r for r in read_corpus(pair) if r["split"] == "test"]
        labs = labeller(rows, meta["test_pairs"])
        res = {}
        for feat in sorted({k for L in labs for k in L}):
            ii = np.array([k for k, L in enumerate(labs) if feat in L])
            y = np.array([labs[k][feat] for k in ii])
            counts = (int((y == 1).sum()), int((y == 0).sum()))
            if min(counts) < MIN_PER_CLASS:
                res[feat] = dict(skipped=f"too few examples (1: {counts[0]}, 0: {counts[1]})"); continue
            res[feat] = dict(probe(C[ii], y, shared), counts=counts)
            print(f"  {pair} {feat:24s} shared share {res[feat]['shared_share']:.3f} (random {res[feat]['random_expectation']:.3f})"
                  f"  acc all {res[feat]['acc_all']:.2f} shared {res[feat]['acc_shared']} private {res[feat]['acc_private']:.2f}",
                  flush=True)
        out[pair] = dict(layer=l, n_shared=int(shared.sum()), d=len(shared), features=res)
    p = res_dir(m) / "n1_probes.json"
    p.write_text(json.dumps(out))
    return [p]


@stage("mlp")
def mlp(cfg):
    m = cfg["model"]
    device = dev()
    out = {}
    for pair in ("es-en", "hi-en"):
        cd = cache_dir(m, "n1", pair)
        if not (cd / "meta.json").exists():
            continue
        meta = json.loads((cd / "meta.json").read_text())
        l = meta["peak_layer"]
        X = torch.tensor(np.load(cd / f"test_x_L{l}.npy").astype(np.float32))
        Y = torch.tensor(np.load(cd / f"test_y_L{l}.npy").astype(np.float32))
        sids = np.array([p["sid"] for p in meta["test_pairs"]])
        us = np.unique(sids)
        rng = np.random.default_rng(0)
        te_s = set(rng.choice(us, len(us) // 4, replace=False))
        te = torch.tensor([s in te_s for s in sids])
        res = {}
        for name, (A, Bm) in (("X_to_Y", (X, Y)), ("Y_to_X", (Y, X))):
            mu_a, mu_b = A[~te].mean(0), Bm[~te].mean(0)
            Pa = torch.linalg.eigh(torch.cov((A[~te] - mu_a).T))[1][:, -512:]
            Pb = torch.linalg.eigh(torch.cov((Bm[~te] - mu_b).T))[1][:, -512:]
            a, b = ((A - mu_a) @ Pa), ((Bm - mu_b) @ Pb)
            sa = a[~te].std(0, keepdim=True); a = a / sa
            r2 = lambda pred, tgt: float(1 - ((pred - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum())
            # ridge linear map (ridge chosen on a validation part of train)
            tr = (~te).nonzero()[:, 0]
            va, fi = tr[: len(tr) // 6], tr[len(tr) // 6:]
            best = None
            for lam in (1e-2, 1e-1, 1, 10, 100):
                W = torch.linalg.solve(a[fi].T @ a[fi] + lam * len(fi) / 1000 * torch.eye(512), a[fi].T @ b[fi])
                s = r2(a[va] @ W, b[va])
                if best is None or s > best[0]:
                    best = (s, lam, W)
            W = torch.linalg.solve(a[tr].T @ a[tr] + best[1] * len(tr) / 1000 * torch.eye(512), a[tr].T @ b[tr])
            lin = r2(a[te] @ W, b[te])
            # MLP with early stopping on the same validation part
            torch.manual_seed(0)
            net = torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512)).to(device)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
            ad, bd = a.to(device), b.to(device)
            best_v, best_state, bad = -1e9, None, 0
            for ep in range(300):
                perm = fi[torch.randperm(len(fi))]
                for c in range(0, len(perm), 512):
                    k = perm[c:c + 512].to(device)
                    loss = ((net(ad[k]) - bd[k]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    v = r2(net(ad[va.to(device)]).cpu(), b[va])
                if v > best_v + 1e-4:
                    best_v, best_state, bad = v, {k_: t.clone() for k_, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 15:
                        break
            net.load_state_dict(best_state)
            with torch.no_grad():
                mlp_r2 = r2(net(ad[te.nonzero()[:, 0].to(device)]).cpu(), b[te])
            res[name] = dict(linear_r2=lin, mlp_r2=mlp_r2, recovered=(mlp_r2 - lin) / max(1 - lin, 1e-9),
                             ridge=best[1], epochs=ep + 1, n_train=int((~te).sum()), n_test=int(te.sum()))
            print(f"  {pair} {name}: linear R2 {lin:.3f}  MLP R2 {mlp_r2:.3f}  recovered {res[name]['recovered']:.3f}", flush=True)
        out[pair] = dict(layer=l, pca=512, **res)
    p = res_dir(m) / "n1_mlp.json"
    p.write_text(json.dumps(out))
    return [p]
