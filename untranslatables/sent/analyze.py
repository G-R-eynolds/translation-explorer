"""Step 2b (CPU, seconds): leftovers, layer choice, tau and epsilon calibration, distortion-length curves,
untranslatability, span heat maps, and the validation of handoff §6.

Order of decisions (everything is fixed on the CONTROL sentences before any test sentence is looked at):
  1. per layer: tau_l = smallest tau whose mean d_CL(control, own B1) <= 0.15        (§2.1 step 5)
  2. layer = argmax over layers of AUROC(mismatched vs matched control pairs), excluding layers where the
     spelling-only control looks at least as close as the real translation                  (§6.1)
  3. epsilon (per space) = 90th percentile of d(control, B1) over all B1 candidates          (§1)
Then the test sentences: curves, U, A, spans, construction gaps, and validation §6.2-6.5.

    python -m sent.analyze --model 3b      # -> results/sent/3b/{scores.jsonl, curves.json, validation.json}
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from ut.common import MODELS, ROOT, describe, load_json, save_json
from ut.stats import auroc, auroc_ci, boot_ci
from sent.data import load_sentences, sent_dir
from sent.ladder import read_jsonl, write_jsonl
from sent.measure import (DELTA, EPS_OT, area, calibrate_tau, cl_cost, cl_token, envelope, rt_token, span_mask,
                          to_words, token_to_word, untranslatability, weighted)
from sent.prompts import PRACTICE, RUNG_ORDER
from sent.score import load_cl

SPACES = ("rt", "cl")
EPS_QUANTILE = 0.9
TAU_TARGET = 0.15
STEP = 0.25


def detour_band(model_key, L):
    """Layers where the logit lens reads English most (word-level run's latent_lang.json), +-2; otherwise the
    proportional band of the 3B's 18-22 of 28."""
    p = ROOT / "results" / model_key / "latent_lang.json"
    if p.exists():
        peaks = [v["peak_layer"] for v in load_json(p)["pairs"].values() if "peak_layer" in v]
        if peaks:
            c = int(np.median(peaks))
            return list(range(max(1, c - 2), min(L, c + 2) + 1)), f"latent_lang.json median peak {c} +- 2"
    lo, hi = round(18 / 28 * L), round(22 / 28 * L)
    return list(range(lo, hi + 1)), "proportional to the 3B's 18-22 (no latent_lang.json for this model)"


class Data:
    """Everything loaded once: sentences, candidates, rows, RT log-probs, CL statistics."""

    def __init__(self, model_key, delta=DELTA):
        d = sent_dir(model_key)
        self.model_key = model_key
        self.cands = {c["cid"]: c for c in read_jsonl(d / "candidates.jsonl")}
        self.rows = read_jsonl(d / "raw" / "rows.jsonl")
        sids = {c["sid"] for c in self.cands.values()}
        self.sents = {s.id: s for s in load_sentences(dialect=True) if s.id in sids}
        self.rt = load_json(d / "raw" / f"rt_{model_key}.json")
        self.cl, cl_meta = load_cl(d / "raw", model_key)
        self.L = cl_meta["n_layers"]
        self.delta = delta
        self.tok = {}                     # sid -> dict(offsets, tokens, t2w, words, w_null...)
        for sid, s in self.sents.items():
            r = self.rt["sentences"][sid]
            words = s.words()
            assert [list(o) for o in cl_meta["tokens"][sid][1]] == [list(o) for o in r["offsets"]], sid
            self.tok[sid] = dict(offsets=r["offsets"], tokens=r["tokens"], words=words,
                                 t2w=token_to_word(r["offsets"], words))
        self.by_role = defaultdict(list)
        for r in self.rows:
            self.by_role[r["role"]].append(r)

    def rt_tokens(self, rid, sid, rt=None):
        rt = rt or self.rt
        s = rt["sentences"][sid]
        return rt_token(rt["rows"][rid], s["lp_null"], s["lp_oracle"], self.delta)

    def weights(self, sid):
        """Information weight w_i (the oracle's gain) of each source token: shared by both spaces."""
        s = self.rt["sentences"][sid]
        return rt_token(s["lp_oracle"], s["lp_null"], s["lp_oracle"], self.delta)[0]

    def cl_tokens(self, rid, layer, tau):
        return cl_token(self.cl[rid][0][layer], tau)


def choose_layer(D, model_key):
    """§2.1 step 5 + §6.1, controls only."""
    ctrl_b1 = [r for r in D.by_role["cand"] if D.sents[r["sid"]].is_control and D.cands[r["rid"]]["rung"] == "B1"]
    mism, spell = D.by_role["mismatch"], D.by_role["spell"]
    per = []
    for l in range(1, D.L + 1):
        lse_m = [D.cl[r["rid"]][0][l] for r in ctrl_b1]
        w_m = [D.weights(r["sid"]) for r in ctrl_b1]
        tau = calibrate_tau(lse_m, w_m, TAU_TARGET)
        dm = [weighted(D.weights(r["sid"]), D.cl_tokens(r["rid"], l, tau)) for r in ctrl_b1]
        dx = [weighted(D.weights(r["sid"]), D.cl_tokens(r["rid"], l, tau)) for r in mism]
        ds = [weighted(D.weights(r["sid"]), D.cl_tokens(r["rid"], l, tau)) for r in spell]
        a = auroc_ci(dx, dm) if dx and dm else (np.nan, np.nan, np.nan)
        sep = (np.nanmean(dx) - np.nanmean(dm)) / (np.sqrt((np.nanvar(dx) + np.nanvar(dm)) / 2) + 1e-9) if dx else np.nan
        per.append(dict(layer=l, tau=tau, matched_mean=float(np.nanmean(dm)), mismatched_mean=float(np.nanmean(dx)),
                        separation=float(sep),
                        spelling_median=float(np.nanmedian(ds)) if ds else np.nan,
                        matched_median=float(np.nanmedian(dm)), auroc=a[0], auroc_ci=a[1:],
                        auroc_spell_vs_matched=auroc(ds, dm) if ds else np.nan,
                        spelling_wins=bool(ds and np.nanmedian(ds) <= np.nanmedian(dm))))
    ok = [p for p in per if not p["spelling_wins"] and not np.isnan(p["auroc"])] or per
    best = max(ok, key=lambda p: (round(p["auroc"], 2), p["separation"]))     # ties (AUROC 1.0 is common): effect size
    return best["layer"], {p["layer"]: p["tau"] for p in per}, per


def sentence_curve(lengths, ds, L0, eps, K):
    Ls, Dv = envelope(lengths, ds)
    if len(Ls) == 0:
        return None
    U, cens, raw = untranslatability(Ls, Dv, L0, eps)
    return dict(L=Ls.tolist(), D=Dv.tolist(), U=U, censored=cens, U_raw=raw, A=area(Ls, Dv, L0, K, STEP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--layer", type=int, default=None, help="override the validated layer (not recommended)")
    a = ap.parse_args()
    D = Data(a.model, a.delta)
    out = sent_dir(a.model)
    V = dict(model=describe(a.model), delta=a.delta, eps_ot=EPS_OT, stop=[])

    # ---- 1-2. layer and tau (controls only)
    layer, taus, per_layer = choose_layer(D, a.model)
    if a.layer is not None:
        V["layer_override"] = dict(chosen=layer, used=a.layer); layer = a.layer
    tau = taus[layer]
    band, band_src = detour_band(a.model, D.L)
    neighbours = [l for l in (layer - 1, layer + 1) if 1 <= l <= D.L]
    V["layer_choice"] = dict(layer=layer, tau=tau, neighbours=neighbours, detour_band=band, detour_source=band_src,
                             per_layer=per_layer, rule="argmax AUROC(mismatched vs matched controls) to 2 d.p., ties "
                             "broken by the standardised mean difference; layers where the spelling control's median d "
                             "<= the matched median are excluded")
    sel = per_layer[layer - 1]
    V["warnings"] = []
    if layer in band:
        V["warnings"].append(f"the validated layer {layer} lies inside the detour band {band}: the §6.5 comparison "
                             "(concept vs detour) is not informative; consider --layer with the best layer outside it")
    if sel["auroc"] < 0.75:
        V["stop"].append(f"layer choice: best AUROC only {sel['auroc']:.2f} (< 0.75): d_CL barely separates "
                         "matched from mismatched controls")

    # ---- per-row scores
    def row_scores(r):
        sid, rid = r["sid"], r["rid"]
        w_rt, l_rt = D.rt_tokens(rid, sid)
        w = D.weights(sid)
        l_cl = D.cl_tokens(rid, layer, tau)
        res = dict(d_rt=float(weighted(w_rt, l_rt)), d_cl=float(weighted(w, l_cl)),
                   cost_cl=cl_cost(D.cl[rid][0][layer], D.cl[rid][1][layer], w, tau),
                   d_cl_neighbours={l: float(weighted(w, D.cl_tokens(rid, l, taus[l]))) for l in neighbours},
                   d_cl_detour=float(np.nanmean([weighted(w, D.cl_tokens(rid, l, taus[l])) for l in band])),
                   l_rt=np.round(l_rt, 4).tolist(), l_cl=np.round(l_cl, 4).tolist())
        return res

    S = {r["rid"]: row_scores(r) for r in D.rows}
    for r in D.by_role["masked"]:
        S[r["parent"]]["d_rt_masked"] = S[r["rid"]]["d_rt"]; S[r["parent"]]["d_cl_masked"] = S[r["rid"]]["d_cl"]
    for cid, c in D.cands.items():
        if cid in S:
            S[cid].setdefault("d_rt_masked", S[cid]["d_rt"]); S[cid].setdefault("d_cl_masked", S[cid]["d_cl"])
    weights = {sid: np.round(D.weights(sid), 4).tolist() for sid in D.sents}
    rows_out = []
    for r in D.rows:
        c = D.cands.get(r["rid"]) or D.cands.get(r["parent"] or "", {})
        rows_out.append(dict(rid=r["rid"], sid=r["sid"], role=r["role"], parent=r["parent"], rung=c.get("rung"),
                             sample=c.get("sample"), len=c.get("len"), text=r["text"], **S[r["rid"]]))
    write_jsonl(out / "scores.jsonl", rows_out)

    # ---- 3. epsilon (controls' B1)
    cand_rows = [r for r in D.by_role["cand"]]
    ctrl_b1 = [r for r in cand_rows if D.sents[r["sid"]].is_control and D.cands[r["rid"]]["rung"] == "B1"]
    eps = {sp: float(np.nanquantile([S[r["rid"]][f"d_{sp}"] for r in ctrl_b1], EPS_QUANTILE)) for sp in SPACES}
    V["epsilon"] = dict(eps, rule=f"{EPS_QUANTILE:.0%} quantile of d(control, B1) over {len(ctrl_b1)} B1 candidates")

    # ---- curves, U, A, spans
    by_sid = defaultdict(list)
    for r in cand_rows:
        by_sid[r["sid"]].append(r)
    L0 = {sid: D.cands[f"{sid}|B1|-1"]["len"] for sid in D.sents if f"{sid}|B1|-1" in D.cands}
    ratios = [max(D.cands[r["rid"]]["len"] for r in by_sid[sid]) / L0[sid] for sid in L0 if L0[sid]]
    K = float(min(6.0, np.ceil(np.log2(np.median(ratios)) / STEP) * STEP)) if ratios else 4.0
    curves = dict(model=describe(a.model), layer=layer, tau=tau, epsilon=eps, K=K, step=STEP, sentences={})
    for sid, s in D.sents.items():
        if sid not in L0:
            continue
        rs = by_sid[sid]
        lens = [D.cands[r["rid"]]["len"] for r in rs]
        tk = D.tok[sid]
        w = D.weights(sid)
        entry = dict(record=s.record(), L0=L0[sid], text_B1=D.cands[f"{sid}|B1|-1"]["text"], spaces={})
        for sp in SPACES:
            key = f"d_{sp}"
            sc = dict(whole=sentence_curve(lens, [S[r["rid"]][key] for r in rs], L0[sid], eps[sp], K),
                      masked=sentence_curve(lens, [S[r["rid"]][f"{key}_masked"] for r in rs], L0[sid], eps[sp], K))
            # per-rung profile
            prof = {}
            for rung in RUNG_ORDER:
                rr = [r for r in rs if D.cands[r["rid"]]["rung"] == rung]
                if rr:
                    v = [S[r["rid"]][key] for r in rr]
                    prof[rung] = dict(min=float(np.nanmin(v)), mean=float(np.nanmean(v)),
                                      greedy=S[f"{sid}|{rung}|-1"][key],
                                      median_len=float(np.median([D.cands[r["rid"]]["len"] for r in rr])))
            sc["rungs"] = prof
            # spans: every word, and every annotated span
            def span_curve(mask):
                ds = []
                for r in rs:
                    wt, lt = (D.rt_tokens(r["rid"], sid) if sp == "rt" else (w, np.array(S[r["rid"]]["l_cl"])))
                    ds.append(float(weighted(wt, lt, mask)))
                return sentence_curve(lens, ds, L0[sid], eps[sp], K)
            # heat map from the greedy B1: per word leftover and weight
            g = f"{sid}|B1|-1"
            wt, lt = (D.rt_tokens(g, sid) if sp == "rt" else (w, np.array(S[g]["l_cl"])))
            Ww, Lw = to_words(np.asarray(wt), np.asarray(lt), tk["t2w"], len(tk["words"]))
            sc["heat"] = [dict(word=x[2], start=x[0], end=x[1], weight=float(Ww[k]),
                               leftover=None if np.isnan(Lw[k]) else float(Lw[k])) for k, x in enumerate(tk["words"])]
            sc["words"] = {}
            for (ws, we, wd) in s.content_words():
                sc["words"][f"{ws}:{we}"] = dict(word=wd, curve=span_curve(span_mask(tk["offsets"], ws, we)))
            sc["spans"] = []
            for x in s.spans():
                cv = span_curve(span_mask(tk["offsets"], x["start"], x["end"]))
                inner = [v for k, v in sc["words"].items()
                         if int(k.split(":")[0]) >= x["start"] and int(k.split(":")[1]) <= x["end"]]
                gap = None
                if cv and len(inner) >= 2 and all(v["curve"] for v in inner):
                    gap = dict(U=cv["U"] - max(v["curve"]["U"] for v in inner),
                               A=cv["A"] - max(v["curve"]["A"] for v in inner),
                               censored=cv["censored"] or any(v["curve"]["censored"] for v in inner))
                sc["spans"].append(dict(span=x, curve=cv, construction_gap=gap))
            entry["spaces"][sp] = sc
        curves["sentences"][sid] = entry
    save_json(out / "curves.json", curves)

    # ---- validation §6.2-6.5 (test sentences = everything but controls)
    test = [sid for sid, s in D.sents.items() if not s.is_control and sid in curves["sentences"]]
    ctrl = [sid for sid, s in D.sents.items() if s.is_control and sid in curves["sentences"]]
    C = curves["sentences"]
    V["ladder"] = {}
    for sp in SPACES:
        prof = {}
        for rung in RUNG_ORDER:
            v = [C[sid]["spaces"][sp]["rungs"][rung] for sid in test if rung in C[sid]["spaces"][sp]["rungs"]]
            if v:
                prof[rung] = dict(practice=PRACTICE[rung], best_mean=float(np.mean([x["min"] for x in v])),
                                  greedy_mean=float(np.nanmean([x["greedy"] for x in v])),
                                  median_len=float(np.median([x["median_len"] for x in v])))
        diff = [C[sid]["spaces"][sp]["rungs"]["B6"]["min"] - C[sid]["spaces"][sp]["rungs"]["B1"]["min"]
                for sid in test if {"B1", "B6"} <= set(C[sid]["spaces"][sp]["rungs"])]
        ci = boot_ci(diff) if diff else (np.nan,) * 3
        order = sorted(prof, key=lambda r: prof[r]["median_len"])
        rho = spearmanr([prof[r]["median_len"] for r in order], [prof[r]["best_mean"] for r in order])[0] \
            if len(order) > 2 else np.nan
        V["ladder"][sp] = dict(profile=prof, B6_minus_B1=ci, length_vs_d_spearman=float(rho),
                               passed=bool(diff and ci[2] < 0))
        if not V["ladder"][sp]["passed"]:
            V["stop"].append(f"ladder ({sp}): best B6 is not reliably below best B1 (diff {ci[0]:+.3f}, "
                             f"95% CI {ci[1]:+.3f} to {ci[2]:+.3f}); this space may not be measuring meaning")

    V["heatmap"] = {}
    for sp in SPACES:
        hot, other = [], []
        for sid in test:
            s = D.sents[sid]
            spans = [x for x in s.spans() if x["kind"] == "hot"]
            for h in C[sid]["spaces"][sp]["heat"]:
                if h["leftover"] is None or not any(cw[0] == h["start"] for cw in s.content_words()):
                    continue
                (hot if any(h["start"] < x["end"] and h["end"] > x["start"] for x in spans) else other).append(h["leftover"])
        a_ = auroc_ci(hot, other) if hot and other else (np.nan,) * 3
        V["heatmap"][sp] = dict(auroc_hot_vs_other_content=a_, n_hot=len(hot), n_other=len(other),
                                mean_hot=float(np.mean(hot)) if hot else None,
                                mean_other=float(np.mean(other)) if other else None)
        if not (a_[0] > 0.5 and a_[1] > 0.5):
            V["stop"].append(f"heat map ({sp}): hot spans are not reliably hotter than other content words "
                             f"(AUROC {a_[0]:.2f}, CI {a_[1]:.2f}-{a_[2]:.2f})")
    if "es_duende" in C:
        fa = {}
        for sp in SPACES:
            h = {x["word"].lower(): x for x in C["es_duende"]["spaces"][sp]["heat"]}
            vals = {k: (h[k]["leftover"] if k in h else None) for k in ("duende", "gusta", "mucho")}
            fa[sp] = dict(leftover=vals, weight={k: h[k]["weight"] for k in vals if k in h},
                          order_holds=None if None in vals.values() else
                          bool(vals["duende"] > vals["gusta"] > vals["mucho"]))
        V["fathers_sentence"] = fa

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        return dict(spearman=float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 else None, n=int(ok.sum()))
    allsid = test + ctrl
    get = lambda sid, sp, k: (C[sid]["spaces"][sp]["whole"] or {}).get(k, np.nan)
    tok_rt, tok_cl = [], []
    for sid in allsid:
        g = f"{sid}|B1|-1"
        w_rt, l_rt = D.rt_tokens(g, sid)
        m = w_rt > 0
        tok_rt += list(l_rt[m]); tok_cl += list(np.array(S[g]["l_cl"])[m])
    V["agreement"] = dict(A=corr([get(s, "rt", "A") for s in allsid], [get(s, "cl", "A") for s in allsid]),
                          U=corr([get(s, "rt", "U") for s in allsid], [get(s, "cl", "U") for s in allsid]),
                          token_leftover_B1=corr(tok_rt, tok_cl),
                          note="U uses the censoring bound for censored sentences; A has no threshold")

    det = {}
    for cat in sorted({D.sents[s].category for s in allsid}):
        ids = [s for s in allsid if D.sents[s].category == cat]
        ch = [S[f"{s}|B1|-1"]["d_cl"] for s in ids]
        dt = [S[f"{s}|B1|-1"]["d_cl_detour"] for s in ids]
        det[cat] = dict(n=len(ids), concept_layer=float(np.nanmean(ch)), detour=float(np.nanmean(dt)),
                        detour_minus_concept=boot_ci(np.array(dt) - np.array(ch)))
    V["detour"] = dict(layers=band, by_category=det, note="d_CL of the greedy B1, each layer with its own "
                       "control-calibrated tau; negative = the English translation looks closer in the detour layers")

    sp_check = {}
    for sp in SPACES:
        g1 = np.array([S[f"{s}|B1|-1"][f"d_{sp}"] for s in test])
        g2 = np.array([S[f"{s}|B2|-1"][f"d_{sp}"] for s in test if f"{s}|B2|-1" in S])
        g2m = np.array([S[f"{s}|B2|-1"][f"d_{sp}_masked"] for s in test if f"{s}|B2|-1" in S])
        g3 = np.array([S[f"{s}|B3|-1"][f"d_{sp}"] for s in test if f"{s}|B3|-1" in S])
        if len(g2) == len(g1) and len(g3) == len(g1):
            gain, gain_m = np.mean(g1 - g2), np.mean(g1 - g2m)
            sp_check[sp] = dict(B2_gain=float(gain), B2_gain_masked=float(gain_m), B3_gain=float(np.mean(g1 - g3)),
                                share_surviving_masking=float(gain_m / gain) if gain > 0.01 else None)
    V["spelling_check"] = sp_check

    V["second_backtranslators"] = {}
    for p in sorted((out / "raw").glob("rt_*.json")):
        bt = p.stem[3:]
        if bt == a.model:
            continue
        rt2 = load_json(p)
        A2, A1 = [], []
        for sid in allsid:
            rs = by_sid[sid]
            ds = [float(weighted(*D.rt_tokens(r["rid"], sid, rt2))) for r in rs]
            cv = sentence_curve([D.cands[r["rid"]]["len"] for r in rs], ds, L0[sid],
                                float(np.nanquantile([float(weighted(*D.rt_tokens(r["rid"], r["sid"], rt2)))
                                                      for r in ctrl_b1], EPS_QUANTILE)), K)
            A2.append(cv["A"] if cv else np.nan); A1.append(get(sid, "rt", "A"))
            C[sid]["spaces"]["rt"].setdefault("other_bt", {})[bt] = cv
        V["second_backtranslators"][bt] = dict(model=rt2["model"], A_vs_main=corr(A1, A2))
    if V["second_backtranslators"]:
        save_json(out / "curves.json", curves)
    save_json(out / "validation.json", V)

    print(f"layer {layer} (tau {tau}), neighbours {neighbours}, detour {band}; AUROC {sel['auroc']:.3f}")
    print(f"epsilon: rt {eps['rt']:.3f}  cl {eps['cl']:.3f}   K = {K}")
    for sp in SPACES:
        lad = V["ladder"][sp]
        print(f"[{sp}] ladder B6-B1 {lad['B6_minus_B1'][0]:+.3f} ({lad['B6_minus_B1'][1]:+.3f}, "
              f"{lad['B6_minus_B1'][2]:+.3f}); heat-map AUROC {V['heatmap'][sp]['auroc_hot_vs_other_content'][0]:.3f}")
        if "fathers_sentence" in V:
            print(f"      duende/gusta/mucho: {V['fathers_sentence'][sp]['leftover']}")
    print("agreement:", {k: v for k, v in V["agreement"].items() if k != "note"})
    for w_ in V["warnings"]:
        print("WARNING:", w_)
    print("STOP:\n  " + "\n  ".join(V["stop"]) if V["stop"] else "validation: no stop flags")


if __name__ == "__main__":
    main()
