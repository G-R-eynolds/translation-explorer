"""N1 span scores and their validation (HANDOFF_internals §3.1 per-span definitions, §3.4 Q2, Q4, §7.1).  CPU (+GPU
if present).

For every stored layer and the evaluation items of `collect.eval_states`:
  token   U, U_raw, h, R_0..R_10 of each token state of data/sentences.yaml (es with the es-en maps, hi with hi-en)
  word    the same for each word's mean state
  span    each annotated span: (a) its mean state, (b) the token-weighted U_S = sum w_i U(x_i) / sum w_i with the
          round-trip information weights w_i (copy oracle minus null, as in sent/; tokens below delta get 0)
  wordctx the target word of each data/words/es.yaml natural context (untranslatable vs ordinary words)
Validation (es, at the peak-sharing layer): AUROC of word U for hot-span words vs control content words (must exceed
0.6, else N3 is not run); the father's order duende > gusta > mucho; U by category; Spearman of sentence U with the R2
round-trip U; the share of hot-span words with a 1-to-1 alignment to the model's own standard translation.
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from internals.cca import Basis
from internals.common import RESULTS, cache_dir, res_dir, stage
from sent.data import load_sentences
from sent.measure import rt_token, span_mask
from ut.stats import auroc_ci, boot_ci

PAIR_OF = {"es": "es-en", "hi": "hi-en"}
FIG1_WORDS = [("es_duende", "duende"), ("es_duende", "gusta"), ("es_sobremesa", "sobremesa"), ("hi_jugaad", "जुगाड़"),
              ("es_tren", "tren"), ("es_casa", "hermana")]


def _words(s, offsets):
    """[(start, end, word, token indices)] for each word of the sentence."""
    out = []
    for ws, we, w in s.words():
        idx = [k for k, (a, b) in enumerate(offsets) if b > ws and a < we and b > a]
        if idx:
            out.append((ws, we, w, idx))
    return out


def _kind(s, ws, we):
    for x in s.spans():
        if ws < x["end"] and we > x["start"]:
            return x["kind"]
    return "other"


def score_layer(m, layer, sents, index, T, li, words_items=None, W=None):
    """All rows for one layer.  T: sent_tokens [n_layers, n_tok, d]; li: index of `layer` in the stored list."""
    rows = []
    for lang, pair in PAIR_OF.items():
        bp = cache_dir(m, "n1", pair) / f"basis_L{layer}.npz"
        ss = [s for s in sents if s.lang == lang and s.id in index]
        if not bp.exists() or not ss:
            continue
        B = Basis.load(bp)
        for s in ss:
            ix = index[s.id]
            X = T[li, ix["start"]:ix["end"]].astype(np.float32)
            w = rt_token(ix["lp_oracle"], ix["lp_null"], ix["lp_oracle"])[0]
            tok = B.scores(X)
            cw = {(a, b) for a, b, _ in s.content_words()}
            for ws, we, wd, idx in _words(s, ix["offsets"]):
                sc = B.scores(X[idx].mean(0, keepdims=True))
                rows.append(dict(pair=pair, layer=layer, sid=s.id, lang=lang, category=s.category, unit="word", text=wd,
                                 start=ws, end=we, kind=_kind(s, ws, we), content=(ws, we) in cw,
                                 U=float(sc["U"][0]), U_raw=float(sc["U_raw"][0]), h=float(sc["h"][0]),
                                 R=[float(sc["R"][n][0]) for n in range(11)], w=float(w[idx].sum())))
            for x in s.spans():
                mask = span_mask(ix["offsets"], x["start"], x["end"]).astype(bool)
                if not mask.any():
                    continue
                sc = B.scores(X[mask].mean(0, keepdims=True))
                wt = w[mask]
                Uw = float((wt * tok["U"][mask]).sum() / wt.sum()) if wt.sum() > 0 else None
                rows.append(dict(pair=pair, layer=layer, sid=s.id, lang=lang, category=s.category, unit="span",
                                 text=x["text"], start=x["start"], end=x["end"], kind=x["kind"], U=float(sc["U"][0]),
                                 U_raw=float(sc["U_raw"][0]), h=float(sc["h"][0]), U_weighted=Uw,
                                 R=[float(sc["R"][n][0]) for n in range(11)]))
            wsum = w.sum()
            rows.append(dict(pair=pair, layer=layer, sid=s.id, lang=lang, category=s.category, unit="sentence",
                             text=s.text, U_weighted=float((w * tok["U"]).sum() / wsum) if wsum > 0 else None,
                             U_raw_weighted=float((w * tok["U_raw"]).sum() / wsum) if wsum > 0 else None,
                             tokens=dict(U=np.round(tok["U"], 4).tolist(), U_raw=np.round(tok["U_raw"], 4).tolist(),
                                         h=np.round(tok["h"], 3).tolist(), w=np.round(w, 3).tolist())))
        if lang == "es" and words_items is not None:
            sc = B.scores(W[:, li].astype(np.float32))
            for k, it in enumerate(words_items):
                rows.append(dict(pair=pair, layer=layer, unit="wordctx", text=it["word"], category=it["category"],
                                 k=it["k"], U=float(sc["U"][k]), U_raw=float(sc["U_raw"][k]), h=float(sc["h"][k])))
    return rows


def validate(rows, layer, r2_curves=None, lang="es"):
    R = [r for r in rows if r["layer"] == layer and r.get("lang", "es") == lang]
    out = {}
    hot = [r for r in R if r["unit"] == "word" and r["kind"] == "hot"]
    ctrl = [r for r in R if r["unit"] == "word" and r["category"] == "control" and r["content"]]
    for key in ("U", "U_raw", "h"):
        if hot and ctrl:
            out[f"auroc_hot_vs_control_{key}"] = auroc_ci([r[key] for r in hot], [r[key] for r in ctrl])
    out["n_hot_words"], out["n_control_words"] = len(hot), len(ctrl)
    if lang == "es":
        f = {r["text"].lower(): r for r in R if r["unit"] == "word" and r["sid"] == "es_duende"}
        if all(k in f for k in ("duende", "gusta", "mucho")):
            out["fathers_sentence"] = dict(U={k: f[k]["U"] for k in ("duende", "gusta", "mucho")},
                                           U_raw={k: f[k]["U_raw"] for k in ("duende", "gusta", "mucho")},
                                           order_holds_U=f["duende"]["U"] > f["gusta"]["U"] > f["mucho"]["U"],
                                           order_holds_U_raw=f["duende"]["U_raw"] > f["gusta"]["U_raw"] > f["mucho"]["U_raw"])
        wc = [r for r in R if r["unit"] == "wordctx"]
        u = [r["U"] for r in wc if r["category"] == "untranslatable"]
        o = [r["U"] for r in wc if r["category"] == "ordinary"]
        if u and o:
            out["wordctx_auroc_untranslatable_vs_ordinary"] = auroc_ci(u, o)
            out["wordctx_auroc_untranslatable_vs_ordinary_raw"] = auroc_ci(
                [r["U_raw"] for r in wc if r["category"] == "untranslatable"], [r["U_raw"] for r in wc if r["category"] == "ordinary"])
    cat = defaultdict(list)
    for r in R:
        if r["unit"] == "span":
            cat[r["category"]].append(r["U"])
    out["span_U_by_category"] = {c: dict(n=len(v), mean=float(np.mean(v)), ci=boot_ci(v)) for c, v in cat.items()}
    if r2_curves:
        a, b = [], []
        for r in R:
            if r["unit"] == "sentence" and r["sid"] in r2_curves and r["U_weighted"] is not None:
                w = r2_curves[r["sid"]]["spaces"]["rt"]["whole"]
                if w:
                    a.append(r["U_weighted"]); b.append(w["U"])
        if len(a) > 3:
            out["spearman_with_R2_U"] = dict(rho=float(spearmanr(a, b)[0]), n=len(a))
    return out


def hot_alignment_share(m, sents):
    """Share of hot-span words with a 1-to-1 link to the standard (greedy B1) translation of this model (or, if this
    model has no R2 run, of gemma12b)."""
    from internals.align import Aligner, words
    for src in (m, "gemma12b"):
        p = RESULTS / src / "replication_R2" / "candidates.jsonl"
        if p.exists():
            b1 = {c["sid"]: c["text"] for c in map(json.loads, open(p)) if c["rung"] == "B1" and c["sample"] < 0}
            break
    else:
        return None
    al = Aligner()
    ss = [s for s in sents if s.id in b1]
    links = al.align([words(s.text) for s in ss], [words(b1[s.id]) for s in ss])
    res = defaultdict(lambda: [0, 0])
    detail = []
    for s, l in zip(ss, links):
        src_linked = {i for i, _ in l}
        for k, (ws, we, w) in enumerate(s.words()):
            kind = _kind(s, ws, we)
            if kind == "hot":
                res[s.category][0] += k in src_linked; res[s.category][1] += 1
                detail.append(dict(sid=s.id, word=w, aligned=k in src_linked))
    return dict(translations_from=src, by_category={c: dict(aligned=a, total=t, share=a / t) for c, (a, t) in res.items()},
                words=detail)


@stage("spans")
def run(cfg):
    m = cfg["model"]
    cd = cache_dir(m, "n1", "eval")
    idx = json.loads((cd / "sent_index.json").read_text())
    layers, index = idx["layers"], idx["sentences"]
    T = np.load(cd / "sent_tokens.npy", mmap_mode="r")
    W = np.load(cd / "words_es.npy", mmap_mode="r") if (cd / "words_es.npy").exists() else None
    witems = json.loads((cd / "words_es.json").read_text())["items"] if W is not None else None
    sents = load_sentences(langs=["es", "hi"])
    rows = []
    for li, l in enumerate(layers):
        rows += score_layer(m, l, sents, index, T, li, witems, W)
    rd = res_dir(m)
    p = rd / "n1_span_scores.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    r2 = RESULTS / m / "replication_R2" / "curves.json"
    r2c = json.loads(r2.read_text())["sentences"] if r2.exists() else None
    summ = dict(model=m, layers={})
    for pair, lang in (("es-en", "es"), ("hi-en", "hi")):
        mp = cache_dir(m, "n1", pair) / "meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        peak = meta["peak_layer"] or layers[len(layers) // 2]
        summ[pair] = dict(peak_layer=peak, detour_layer=meta.get("detour_layer"), at_peak=validate(rows, peak, r2c, lang),
                          by_layer={l: validate(rows, l, r2c, lang) for l in layers})
        fig1 = {}
        for sid, wd in FIG1_WORDS:
            for r in rows:
                if r["layer"] == peak and r["unit"] == "word" and r.get("sid") == sid and r["text"] == wd:
                    fig1[f"{sid}:{wd}"] = dict(R=r["R"], h=r["h"], U=r["U"])
        summ[pair]["fig1_retention"] = fig1
    if "es-en" in summ:
        v = summ["es-en"]["at_peak"].get("auroc_hot_vs_control_U", (np.nan,) * 3)
        summ["gate_N1"] = dict(auroc=v, passed=bool(v[0] > 0.6),
                               rule="U separates es hot-span words from control content words with AUROC > 0.6 (§7.1)")
    try:
        summ["hot_alignment"] = hot_alignment_share(m, sents)
    except Exception as e:                                              # never lose the scores over the aligner
        summ["hot_alignment"] = dict(error=str(e))
    q = rd / "n1_spans_summary.json"
    q.write_text(json.dumps(summ, ensure_ascii=False, default=float))
    return [p, q]


# ------------------------------------------------------------------ FLORES: English as hub, and Bhojpuri
def flores_hub(cfg):
    import torch
    from internals.align import read_links
    from internals.cca import eig, fit, heldout_rho
    m = cfg["model"]
    cd = cache_dir(m, "n1", "eval")
    S, I = {}, {}
    for lg in ("es", "en", "hi", "bho"):
        if (cd / f"flores_{lg}.npy").exists():
            S[lg] = np.load(cd / f"flores_{lg}.npy", mmap_mode="r")
            I[lg] = {tuple(x): k for k, x in enumerate(json.loads((cd / f"flores_{lg}.json").read_text())["index"])}
    hub_layers = json.loads((cd / "flores_es.json").read_text())["layers"]
    rng = np.random.default_rng(0)
    sids = sorted({k[0] for k in I["es"]})
    test_s = set(rng.choice(sids, len(sids) // 5, replace=False))
    out = dict(model=m, layers={}, pca=128, split="80/20 by sentence", note="same FLORES sentences for every pair")
    for li, l in enumerate(hub_layers):
        res = {}
        for a, b in (("es", "en"), ("hi", "en"), ("es", "hi")):
            links = read_links(f"flores_{a}-{b}")
            fx, fy, tx, ty = [], [], [], []
            for sid, ll in links.items():
                for i, j in ll:
                    ka, kb = I[a].get((sid, i)), I[b].get((sid, j))
                    if ka is None or kb is None:
                        continue
                    (tx if sid in test_s else fx).append(ka); (ty if sid in test_s else fy).append(kb)
            X = lambda lg, ks: torch.tensor(S[lg][li, ks].astype(np.float32))
            Xf, Yf, Xt, Yt = X(a, fx), X(b, fy), X(a, tx), X(b, ty)
            mx, my = Xf.mean(0), Yf.mean(0)
            Xf, Yf, Xt, Yt = Xf - mx, Yf - my, Xt - mx, Yt - my
            Px = eig(Xf.T.double() @ Xf.double() / len(Xf))[1][:, -min(128, Xf.shape[1]):].float()
            Py = eig(Yf.T.double() @ Yf.double() / len(Yf))[1][:, -min(128, Yf.shape[1]):].float()
            u, v = Xf @ Px, Yf @ Py
            ex, ey = eig(u.T @ u / len(u)), eig(v.T @ v / len(v))
            best = None
            for lam in (1e-3, 1e-2, 1e-1):
                f = fit(ex, ey, u.T @ v / len(u), lam)
                rho = heldout_rho(f["A"], f["B"], Xt @ Px, Yt @ Py)
                if best is None or rho[:20].sum() > best[1][:20].sum():
                    best = (lam, rho)
            lam, rho = best
            vn = torch.roll(v, len(v) // 2, 0)
            fn = fit(ex, ey, u.T @ vn / len(u), lam)
            rn = heldout_rho(fn["A"], fn["B"], Xt @ Px, torch.roll(Yt @ Py, len(Yt) // 2, 0))
            thr = float(np.quantile(rn.numpy(), 0.99))
            sh = rho > thr
            from internals.cca import bias
            Suu, Svv, Suv = u.T @ u / len(u), v.T @ v / len(v), u.T @ v / len(u)
            E = bias(Suu.double(), Svv.double(), Suv.double(), eig(Suu), eig(Svv), lam)["raw"]
            res[f"{a}-{b}"] = dict(n_fit=len(fx), n_test=len(tx), lam=lam, n_shared=int(sh.sum()), E_X=E["E_X"], E_Y=E["E_Y"],
                                   mean_rho2_shared=float((rho[sh] ** 2).mean()) if sh.any() else 0.0,
                                   mean_rho2_top20=float((rho[:20] ** 2).mean()), null_q99=thr,
                                   rho_heldout=rho.numpy().round(4).tolist())
        # Bhojpuri vs Hindi on the same sentences, scored with the full hi-en basis at this layer
        bp = cache_dir(m, "n1", "hi-en") / f"basis_L{l}.npz"
        if bp.exists() and "bho" in S:
            B = Basis.load(bp)
            Uh, Ub = B.scores(np.asarray(S["hi"][li], np.float32))["U"], B.scores(np.asarray(S["bho"][li], np.float32))["U"]
            sh_bho = np.asarray(S["bho"][li], np.float32)
            Ub_own = B.scores(sh_bho - sh_bho.mean(0) + B.mu_x.numpy())["U"]
            per_s = defaultdict(lambda: [[], [], []])
            for (sid, _), k in I["hi"].items():
                per_s[sid][0].append(Uh[k])
            for (sid, _), k in I["bho"].items():
                per_s[sid][1].append(Ub[k]); per_s[sid][2].append(Ub_own[k])
            d1 = [np.mean(b_) - np.mean(h_) for h_, b_, _ in per_s.values() if h_ and b_]
            d2 = [np.mean(o_) - np.mean(h_) for h_, _, o_ in per_s.values() if h_ and o_]
            res["bhojpuri_minus_hindi_U"] = dict(hindi_centring=boot_ci(d1), own_centring=boot_ci(d2), n_sentences=len(d1),
                                                 mean_U_hindi=float(np.mean(Uh)), mean_U_bho=float(np.mean(Ub)))
        out["layers"][l] = res
        print(f"  L{l}: " + "  ".join(f"{k} shared {v['n_shared']} rho2 {v['mean_rho2_shared']:.3f}"
                                     for k, v in res.items() if "n_shared" in v), flush=True)
    p = res_dir(m) / "n1_hub.json"
    p.write_text(json.dumps(out, default=float))
    return [p]
