"""R2 analysis (HANDOFF_internals §2 R2), CPU: the sentence round trip with the copying loophole fixed.

Differences from sent.analyze (the 19 Nov design):
  * masking is the default: d_rt is scored on the candidate with source words and close cognates replaced by '[·]'
    (sent.score --mask-rule cognate); the unmasked score is kept as d_rt_unmasked;
  * B2_borrow is its own labelled rung, kept OFF the distortion-length curve: `whole` = non-borrowing rungs,
    `with_borrow` = all rungs; "what borrowing buys" = d(B1) - d(B2_borrow), unmasked and masked;
  * no concept-layer (OT) score, no targeting, no dialect items;
  * intended-sense recovery (sent.sense) is summarised per rung, category and span.
Validation kept: §6.2 (ladder monotone) and §6.3 (hot-span AUROC, father's sentence).

    python -m sent.analyze_rt --model gemma12b   # -> <sent_dir>/{scores.jsonl, curves.json, validation.json}
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from ut.common import MODELS, describe, load_json, save_json
from ut.stats import auroc_ci, boot_ci
from sent.analyze import EPS_QUANTILE, STEP, sentence_curve
from sent.data import load_sentences, sent_dir
from sent.ladder import read_jsonl, write_jsonl
from sent.measure import DELTA, rt_token, span_mask, to_words, token_to_word, weighted
from sent.prompts import BORROW, PRACTICE, RUNG_ORDER

SPACES = ("rt", "rt_unmasked")
CURVE_RUNGS = [r for r in RUNG_ORDER if r not in BORROW]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=MODELS)
    ap.add_argument("--delta", type=float, default=DELTA)
    a = ap.parse_args()
    out = sent_dir(a.model)
    cands = {c["cid"]: c for c in read_jsonl(out / "candidates.jsonl")}
    for c in cands.values():
        if c["rung"] == "B2":
            c["rung"] = "B2_borrow"
    rows = read_jsonl(out / "raw" / "rows.jsonl")
    rt = load_json(out / "raw" / f"rt_{a.model}.json")
    sents = {s.id: s for s in load_sentences() if s.id in {c["sid"] for c in cands.values()}}
    masked_of = {r["parent"]: r["rid"] for r in rows if r["role"] == "masked"}

    def tokens(rid, sid, rt_=None):
        r_ = rt_ or rt
        s = r_["sentences"][sid]
        return rt_token(r_["rows"][rid], s["lp_null"], s["lp_oracle"], a.delta)

    S = {}
    for cid, c in cands.items():
        wu, lu = tokens(cid, c["sid"])
        wm, lm = tokens(masked_of.get(cid, cid), c["sid"])
        S[cid] = dict(d_rt=float(weighted(wm, lm)), d_rt_unmasked=float(weighted(wu, lu)),
                      l_rt=np.round(lm, 4).tolist(), l_rt_unmasked=np.round(lu, 4).tolist(), masked=cid in masked_of)
    write_jsonl(out / "scores.jsonl", [dict(rid=cid, sid=c["sid"], rung=c["rung"], sample=c["sample"], len=c["len"],
                                            text=c["text"], **S[cid]) for cid, c in cands.items()])
    V = dict(model=describe(a.model), delta=a.delta, masking="source words + cognates (norm. edit distance <= 0.3) "
             "-> '[·]'", stop=[], warnings=[])
    ctrl_b1 = [cid for cid, c in cands.items() if sents[c["sid"]].is_control and c["rung"] == "B1"]
    eps = {sp: float(np.nanquantile([S[c][f"d_{sp}"] for c in ctrl_b1], EPS_QUANTILE)) for sp in SPACES}
    V["epsilon"] = dict(eps, rule=f"{EPS_QUANTILE:.0%} quantile of d(control, B1) over {len(ctrl_b1)} B1 candidates")

    by_sid = defaultdict(list)
    for cid, c in cands.items():
        by_sid[c["sid"]].append(cid)
    L0 = {sid: cands[f"{sid}|B1|-1"]["len"] for sid in sents if f"{sid}|B1|-1" in cands}
    ratios = [max(cands[c]["len"] for c in by_sid[sid]) / L0[sid] for sid in L0 if L0[sid]]
    K = float(min(6.0, np.ceil(np.log2(np.median(ratios)) / STEP) * STEP)) if ratios else 4.0
    curves = dict(model=describe(a.model), epsilon=eps, K=K, step=STEP, curve_rungs=CURVE_RUNGS, sentences={})
    for sid, s in sents.items():
        if sid not in L0:
            continue
        r_ = rt["sentences"][sid]
        words = s.words()
        t2w = token_to_word(r_["offsets"], words)
        entry = dict(record=s.record(), L0=L0[sid], text_B1=cands[f"{sid}|B1|-1"]["text"], spaces={})
        for sp in SPACES:
            key = f"d_{sp}"
            main_c = [c for c in by_sid[sid] if cands[c]["rung"] not in BORROW]
            curve = lambda cs, f: sentence_curve([cands[c]["len"] for c in cs], [f(c) for c in cs], L0[sid], eps[sp], K)
            sc = dict(whole=curve(main_c, lambda c: S[c][key]), with_borrow=curve(by_sid[sid], lambda c: S[c][key]))
            sc["rungs"] = {}
            for rung in RUNG_ORDER:
                rr = [c for c in by_sid[sid] if cands[c]["rung"] == rung]
                if rr:
                    v = [S[c][key] for c in rr]
                    sc["rungs"][rung] = dict(min=float(np.nanmin(v)), mean=float(np.nanmean(v)),
                                             greedy=S.get(f"{sid}|{rung}|-1", {}).get(key),
                                             median_len=float(np.median([cands[c]["len"] for c in rr])))
            g = f"{sid}|B1|-1"
            wt, lt = tokens(masked_of.get(g, g) if sp == "rt" else g, sid)
            Ww, Lw = to_words(np.asarray(wt), np.asarray(lt), t2w, len(words))
            sc["heat"] = [dict(word=x[2], start=x[0], end=x[1], weight=float(Ww[k]),
                               leftover=None if np.isnan(Lw[k]) else float(Lw[k])) for k, x in enumerate(words)]

            def span_curve(m):
                f = lambda c: float(weighted(*tokens(masked_of.get(c, c) if sp == "rt" else c, sid), m))
                return curve(main_c, f)
            sc["words"] = {f"{ws}:{we}": dict(word=wd, curve=span_curve(span_mask(r_["offsets"], ws, we)))
                           for ws, we, wd in s.content_words()}
            sc["spans"] = []
            for x in s.spans():
                cv = span_curve(span_mask(r_["offsets"], x["start"], x["end"]))
                inner = [v for k, v in sc["words"].items()
                         if int(k.split(":")[0]) >= x["start"] and int(k.split(":")[1]) <= x["end"]]
                gap = None
                if cv and len(inner) >= 2 and all(v["curve"] for v in inner):
                    gap = dict(U=cv["U"] - max(v["curve"]["U"] for v in inner),
                               censored=cv["censored"] or any(v["curve"]["censored"] for v in inner))
                sc["spans"].append(dict(span=x, curve=cv, construction_gap=gap))
            entry["spaces"][sp] = sc
        curves["sentences"][sid] = entry
    C = curves["sentences"]
    test = [sid for sid in C if not sents[sid].is_control]

    # ---- §6.2 ladder (non-borrowing rungs), §6.3 heat map
    V["ladder"], V["heatmap"] = {}, {}
    for sp in SPACES:
        diff = [C[s]["spaces"][sp]["rungs"]["B6"]["min"] - C[s]["spaces"][sp]["rungs"]["B1"]["min"]
                for s in test if {"B1", "B6"} <= set(C[s]["spaces"][sp]["rungs"])]
        ci = boot_ci(diff)
        prof = {r: dict(practice=PRACTICE[r], best_mean=float(np.mean([C[s]["spaces"][sp]["rungs"][r]["min"] for s in test
                                                                        if r in C[s]["spaces"][sp]["rungs"]])))
                for r in RUNG_ORDER if any(r in C[s]["spaces"][sp]["rungs"] for s in test)}
        V["ladder"][sp] = dict(B6_minus_B1=ci, passed=bool(ci[2] < 0), profile=prof)
        hot, other = [], []
        for sid in test:
            s = sents[sid]
            hs = [x for x in s.spans() if x["kind"] == "hot"]
            cw = {w[0] for w in s.content_words()}
            for h in C[sid]["spaces"][sp]["heat"]:
                if h["leftover"] is None or h["start"] not in cw:
                    continue
                (hot if any(h["start"] < x["end"] and h["end"] > x["start"] for x in hs) else other).append(h["leftover"])
        au = auroc_ci(hot, other)
        V["heatmap"][sp] = dict(auroc_hot_vs_other_content=au, n_hot=len(hot), n_other=len(other))
    for sp in ("rt",):
        if not V["ladder"][sp]["passed"]:
            V["stop"].append(f"ladder ({sp}): best B6 not reliably below best B1 ({V['ladder'][sp]['B6_minus_B1']})")
        au = V["heatmap"][sp]["auroc_hot_vs_other_content"]
        if not au[1] > 0.5:
            V["stop"].append(f"heat map ({sp}): hot spans not reliably hotter (AUROC {au[0]:.2f}, CI {au[1]:.2f}-{au[2]:.2f})")
    if "es_duende" in C:
        V["fathers_sentence"] = {}
        for sp in SPACES:
            h = {x["word"].lower(): x for x in C["es_duende"]["spaces"][sp]["heat"]}
            v = {k: h[k]["leftover"] if k in h else None for k in ("duende", "gusta", "mucho")}
            V["fathers_sentence"][sp] = dict(leftover=v, order_holds=None if None in v.values() else
                                             bool(v["duende"] > v["gusta"] > v["mucho"]))

    # ---- what borrowing buys (B2_borrow vs B1), by category
    bb = defaultdict(lambda: defaultdict(list))
    for sid in C:
        b1, b2 = f"{sid}|B1|-1", f"{sid}|B2_borrow|-1"
        if b1 in S and b2 in S:
            cat = sents[sid].category
            bb[cat]["unmasked"].append(S[b1]["d_rt_unmasked"] - S[b2]["d_rt_unmasked"])
            bb[cat]["masked"].append(S[b1]["d_rt"] - S[b2]["d_rt"])
            bb[cat]["kept_source_word"].append(S[b2]["masked"])
    V["borrowing"] = {cat: dict(n=len(v["masked"]), gain_unmasked=boot_ci(v["unmasked"]), gain_masked=boot_ci(v["masked"]),
                                share_keeping_a_source_word=float(np.mean(v["kept_source_word"])))
                      for cat, v in bb.items()}
    V["borrowing_note"] = ("gain = d(B1) - d(B2_borrow), greedy; unmasked = what the loan itself buys (copying "
                           "included); masked = what is left once the loan word is hidden")

    # ---- categories
    cats = defaultdict(lambda: defaultdict(list))
    for sid, e in C.items():
        for sp in SPACES:
            w = e["spaces"][sp]["whole"]
            if w:
                cats[sents[sid].category][sp].append((w["U"], w["censored"], w["A"]))
    V["categories"] = {cat: {sp: dict(n=len(v), U_mean=float(np.mean([x[0] for x in v])), censored=int(sum(x[1] for x in v)),
                                      A_mean=float(np.mean([x[2] for x in v]))) for sp, v in d_.items()} for cat, d_ in cats.items()}
    V["categories_by_lang"] = {}
    for lang in sorted({s.lang for s in sents.values()}):
        d_ = defaultdict(list)
        for sid, e in C.items():
            if sents[sid].lang == lang and e["spaces"]["rt"]["whole"]:
                d_[sents[sid].category].append(e["spaces"]["rt"]["whole"]["U"])
        V["categories_by_lang"][lang] = {k: dict(n=len(v), U_mean=float(np.mean(v))) for k, v in d_.items()}

    # ---- intended-sense recovery
    sp_ = out / "sense.json"
    if sp_.exists():
        sn = load_json(sp_)["spans"]
        summ = {}
        for kind in ("hot", "probe"):
            for rung in ("B1", "B2_borrow", "B3", "B4", "B5", "B6"):
                v = [r["rungs"][rung]["greedy"] for r in sn if r["span"]["kind"] == kind and rung in r["rungs"]
                     and r["rungs"][rung]["greedy"] is not None]
                if v:
                    summ.setdefault(kind, {})[rung] = boot_ci(v)
        b1h = [r["rungs"]["B1"]["greedy"] for r in sn if r["span"]["kind"] == "hot" and "B1" in r["rungs"]]
        b1p = [r["rungs"]["B1"]["greedy"] for r in sn if r["span"]["kind"] == "probe" and "B1" in r["rungs"]]
        b6h = [(r["rungs"]["B6"]["greedy"], r["rungs"]["B1"]["greedy"]) for r in sn
               if r["span"]["kind"] == "hot" and {"B1", "B6"} <= set(r["rungs"])]
        duende = next((r for r in sn if r["sid"] == "es_duende" and r["span"]["text"] == "duende"), None)
        V["sense_recovery"] = dict(
            by_kind_and_rung=summ,
            auroc_probe_vs_hot_B1=auroc_ci(b1p, b1h) if b1h and b1p else None,
            B6_minus_B1_hot=boot_ci([a_ - b_ for a_, b_ in b6h]) if b6h else None,
            duende_B1=duende["rungs"]["B1"]["greedy"] if duende else None,
            duende_rank_among_hot=(int(np.sum(np.array(b1h) < duende["rungs"]["B1"]["greedy"])), len(b1h)) if duende else None,
            note="recovery = mean log p(gloss | masked t) - mean log p(gloss | s); 0 = as good as the source")

    # ---- second back-translators
    V["second_backtranslators"] = {}
    for p in sorted((out / "raw").glob("rt_*.json")):
        bt = p.stem[3:]
        if bt == a.model:
            continue
        rt2 = load_json(p)
        A1, A2 = [], []
        e2 = float(np.nanquantile([float(weighted(*tokens(masked_of.get(c, c), cands[c]["sid"], rt2))) for c in ctrl_b1],
                                  EPS_QUANTILE))
        for sid in C:
            cs = [c for c in by_sid[sid] if cands[c]["rung"] not in BORROW]
            cv = sentence_curve([cands[c]["len"] for c in cs],
                                [float(weighted(*tokens(masked_of.get(c, c), sid, rt2))) for c in cs], L0[sid], e2, K)
            A1.append(C[sid]["spaces"]["rt"]["whole"]["A"]); A2.append(cv["A"] if cv else np.nan)
            C[sid]["spaces"]["rt"].setdefault("other_bt", {})[bt] = cv
        ok = ~np.isnan(A2)
        V["second_backtranslators"][bt] = dict(model=rt2["model"], A_spearman=float(spearmanr(np.array(A1)[ok], np.array(A2)[ok])[0]))
    save_json(out / "curves.json", curves)
    save_json(out / "validation.json", V)

    print(f"epsilon rt {eps['rt']:.3f} (unmasked {eps['rt_unmasked']:.3f}), K {K}")
    for sp in SPACES:
        print(f"[{sp}] ladder B6-B1 {np.round(V['ladder'][sp]['B6_minus_B1'], 3)}  heat AUROC "
              f"{np.round(V['heatmap'][sp]['auroc_hot_vs_other_content'], 2)}")
    if "fathers_sentence" in V:
        print("father:", V["fathers_sentence"]["rt"])
    for cat, v in V["categories"].items():
        print(f"  {cat:28s} U {v['rt']['U_mean']:.2f} (cens {v['rt']['censored']}/{v['rt']['n']})  A {v['rt']['A_mean']:.3f}")
    for cat, v in V["borrowing"].items():
        print(f"  borrowing {cat:22s} unmasked {v['gain_unmasked'][0]:+.3f}  masked {v['gain_masked'][0]:+.3f}")
    if "sense_recovery" in V:
        print("sense:", {k: {r: round(x[0], 3) for r, x in v.items()} for k, v in V["sense_recovery"]["by_kind_and_rung"].items()})
    print("STOP:\n  " + "\n  ".join(V["stop"]) if V["stop"] else "validation: no stop flags")


if __name__ == "__main__":
    main()
