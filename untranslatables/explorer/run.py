"""Step-1 pipeline (spec §Method, §Outputs).  Per text, resumable: samples -> readings -> branch -> directions+steer.

    python -m explorer run --model g3_4b_pt [--n 64] [--texts a,b] [--layer 17]

Steering grid per ordered reading pair (A, B), for the top 3 readings: variant {natural, plain} x mode {cursor, at_b}
x target KL {0.25, 0.5, 1, 2}; each condition = 1 greedy + 8 samples at T 0.7 from the cursor.  A continuation
succeeds if it is assigned to reading A AND its per-token log p under the unsteered model is within 1.5 nats of the
median of A's own members.  The unsteered continuations from the same cursor give the baseline share of A.
Pre-registered decision (spec §6) uses variant=natural, mode=cursor; the other cells are reported, not chosen from.
"""
import json
import time
import zlib
from pathlib import Path

import numpy as np

from ut.common import ROOT

OUT = ROOT / "results" / "explorer"
KLS = {"at_b": [0.25, 0.5, 1.0, 2.0],          # KL of the next-token distribution at the cursor
       "cursor": [0.05, 0.1, 0.25, 0.5]}       # MEAN per-token KL along the reference continuation (added everywhere)
VARIANTS = ["natural", "plain"]
MODES = ["cursor", "at_b"]
MAX_MEMBERS = 24
N_STEER = 8
# Reduced grid (24 Nov, after Bécquer XXI took 108 min under the full grid): cursor mode only, two KL levels,
# 4 samples, continuations capped at the reference's remaining length + 16.  Texts already done keep the full grid;
# the pre-registered cell (natural, cursor) is present in both.
import os
if os.environ.get("EXPLORER_FAST"):
    MODES = ["cursor"]
    KLS = {"cursor": [0.1, 0.25]}
    N_STEER = 4


def _seed(*p):
    return zlib.crc32("|".join(map(str, p)).encode()) % 2 ** 31


def _dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def _generate_for_style(model, tok, ids, n, max_new, prompt_style="legacy", temperature=1.0, seed=0, batch=64):
    from explorer.sample import _dev, generate
    if prompt_style == "legacy":
        return generate(model, tok, ids, n, max_new, temperature=temperature, seed=seed, batch=batch)
    from transformers import GenerationConfig
    import torch
    from explorer.frame import DOC_STOPS, cut_doc
    from ut.common import seed_all

    gc = GenerationConfig(
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        stop_strings=list(DOC_STOPS),
        temperature=temperature if temperature > 0 else None,
        top_k=0 if temperature > 0 else None,
        top_p=1.0 if temperature > 0 else None,
    )
    out = []
    for c in range(0, n, batch):
        m = min(batch, n - c)
        seed_all((seed * 100003 + c) % 2 ** 32)
        x = torch.tensor([ids] * m, device=_dev(model))
        with torch.no_grad():
            g = model.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                generation_config=gc,
                tokenizer=tok,
            )[:, len(ids):].tolist()
        for row in g:
            row = [t for t in row if t not in (tok.pad_token_id, tok.eos_token_id)]
            out.append(cut_doc(tok, row))
    return out


def _steered_for_style(model, tok, hook, ids, vec, b, max_new, prompt_style="legacy",
                       n=1, temperature=0.0, only_at_b=False, seed=0):
    from explorer.sample import steered
    if prompt_style == "legacy":
        return steered(model, tok, hook, ids, vec, b, max_new, n=n,
                       temperature=temperature, only_at_b=only_at_b, seed=seed)
    hook.set(vec, b, only_at_b)
    try:
        return _generate_for_style(model, tok, ids, n, max_new, prompt_style="doc",
                                   temperature=temperature, seed=seed)
    finally:
        hook.set(None)


def run_text(model, tok, hook, layer, txt, out, n, prompt_style="legacy",
             clustering="semantic", alpha_grad=0.5, disp_min=0.035):
    from explorer import readings as R
    from explorer.branches import align_index, choose_t, timeline
    from explorer.directions import alpha_for, calibrate, grads_at, natural, plain
    from explorer.frame import build_doc_prompt, build_prompt
    from explorer.grad_readings import cluster_gradients, cluster_hybrid
    from explorer.sample import score
    import torch

    if prompt_style not in ("legacy", "doc"):
        raise ValueError(f"Unknown prompt_style: {prompt_style!r}")
    if clustering not in ("semantic", "hybrid", "grad"):
        raise ValueError(f"Unknown clustering mode: {clustering!r}")

    tid = txt["id"]
    ids, start = build_doc_prompt(tok, txt["text"]) if prompt_style == "doc" else build_prompt(tok, txt["text"])
    n_src = len(tok(txt["text"], add_special_tokens=False)["input_ids"])
    max_new = 3 * n_src + 16
    sp = out / "samples" / f"{tid}.json"
    if sp.exists():
        S = json.loads(sp.read_text())
        X = np.load(out / "samples" / f"{tid}_emb.npy")
    else:
        t0 = time.time()
        ys = _generate_for_style(model, tok, ids, n, max_new, prompt_style=prompt_style,
                                 temperature=1.0, seed=_seed(tid))
        lp = score(model, ids, ys)
        texts = [tok.decode(y).strip() for y in ys]
        emb_in = [R.normalize_text(t, collapse_newlines=True) for t in texts] if prompt_style == "doc" else texts
        X = R.embed_texts(emb_in)
        lpt_s = lp / np.maximum([len(y) for y in ys], 1)
        if clustering in ("hybrid", "grad"):
            try:
                G_init = grads_at(model, ids, ys, layer, start - 1)
                if clustering == "hybrid":
                    cl = cluster_hybrid(X, G_init, lpt_s, alpha_grad=alpha_grad, disp_min=disp_min)
                else:
                    cl = cluster_gradients(G_init, lpt_s, disp_min=disp_min)
            except Exception as exc:
                cl = R.readings_from(X, lpt_s, disp_min=disp_min)
                cl["clustering_fallback"] = f"semantic ({type(exc).__name__}: {exc})"
        else:
            cl = R.readings_from(X, lpt_s, disp_min=disp_min)
        S = dict(id=tid, text=txt["text"], kind=txt["kind"], prompt_style=prompt_style,
                 clustering=clustering, ys=ys, texts=texts,
                 logp=lp.tolist(), lens=[len(y) for y in ys], **cl, entropy_bits=R.entropy(cl["shares"]),
                 seconds=round(time.time() - t0, 1))
        (out / "samples").mkdir(parents=True, exist_ok=True)
        np.save(out / "samples" / f"{tid}_emb.npy", X)
        _dump(sp, S)
    shares = {int(k): v for k, v in S["shares"].items()}
    print(f"[explorer] {tid}: k={S['k']} silhouette={S['silhouette']:.2f} readings={len(S['readings'])} "
          f"shares={[round(shares[r], 2) for r in S['readings']]}", flush=True)
    labels = np.array(S["labels"])
    ys = S["ys"]
    rd = S["readings"][:3]
    if not rd:
        return S, {}, []
    members = {r: [ys[i] for i in np.where(labels == r)[0]] for r in rd}
    lpt = np.array(S["logp"]) / np.maximum(np.array(S["lens"]), 1)
    ref_i = max(np.where(labels == rd[0])[0], key=lambda i: S["logp"][i])
    y_ref = ys[ref_i]

    bp = out / "branches" / f"{tid}.json"
    if bp.exists():
        Bx = json.loads(bp.read_text())
    else:
        tl = timeline(model, ids, y_ref)
        t, spec, why = choose_t(y_ref, members, rd[0], tl["fisher_top"])
        Bx = dict(id=tid, ref=tok.decode(y_ref).strip(), ref_tokens=[tok.decode([x]) for x in y_ref], t=t,
                  specificity=spec, why=why, **tl)
        _dump(bp, Bx)
    if len(rd) < 2:
        return S, Bx, []

    stp = out / "steering" / f"{tid}.jsonl"
    if stp.exists():
        return S, Bx, [json.loads(l) for l in stp.read_text().splitlines()]
    t = Bx["t"]
    prefix = ids + y_ref[:t]
    b_abs = len(prefix) - 1
    mu_raw = np.asarray(S.get("mu", []), dtype=np.float64)
    if mu_raw.ndim == 1 and len(mu_raw) == X.shape[1]:
        mu = mu_raw
    else:
        kept_mask = labels >= 0
        mu = X[kept_mask].mean(0) if np.any(kept_mask) else X.mean(0)
    C = R.centroids(R.centre(X, mu), labels, rd)
    med = {r: float(np.median(lpt[labels == r])) for r in rd}
    conts = {r: [y[align_index(y_ref, y, t):] for y in members[r][:MAX_MEMBERS]] for r in rd}
    G = {r: grads_at(model, prefix, conts[r], layer, b_abs) for r in rd}
    G_all = np.vstack(list(G.values()))
    max_rest = max(8, max_new - t) if not os.environ.get("EXPLORER_FAST") else len(y_ref) - t + 16

    def judge(conts):
        full = [y_ref[:t] + c for c in conts]
        txts = [tok.decode(f).strip() for f in full]
        j_in = [R.normalize_text(s, collapse_newlines=True) for s in txts] if prompt_style == "doc" else txts
        E = R.centre(R.embed_texts(j_in), mu)
        fl = score(model, ids, full) / np.maximum([len(f) for f in full], 1)
        return [rd[R.assign(e, C)] for e in E], fl.tolist(), txts

    base_conts = _steered_for_style(model, tok, hook, prefix, None, b_abs, max_rest,
                                    prompt_style=prompt_style, n=N_STEER, temperature=0.7,
                                    seed=_seed(tid, "base"))
    base_r, _, base_txt = judge(base_conts)
    rows = []
    for A in rd:
        for B in rd:
            if A == B:
                continue
            for var in VARIANTS:
                v = natural(G[A], G[B]) if var == "natural" else plain(G[A], G[B])
                for mode in MODES:
                    for kl in KLS[mode]:
                        a = calibrate(model, hook, ids + y_ref, b_abs, v, kl, mode, alpha_for(G_all, v, kl))
                        vec = torch.tensor(a * v, dtype=torch.float32)
                        g = _steered_for_style(model, tok, hook, prefix, vec, b_abs, max_rest,
                                               prompt_style=prompt_style, only_at_b=mode == "at_b")
                        s = _steered_for_style(model, tok, hook, prefix, vec, b_abs, max_rest,
                                               prompt_style=prompt_style, n=N_STEER, temperature=0.7,
                                               only_at_b=mode == "at_b", seed=_seed(tid, A, B, var, kl, mode))
                        r_, fl, txts = judge(g + s)
                        ok = [ri == A and f >= med[A] - 1.5 for ri, f in zip(r_, fl)]
                        rows.append(dict(id=tid, A=A, B=B, variant=var, kl=kl,
                                         kl_level=[0.05, 0.1, 0.25, 0.5].index(kl) if mode == "cursor" else KLS[mode].index(kl),
                                         alpha=a, mode=mode, t=t, texts=txts, assigned=r_, fluency=fl, success=ok,
                                         greedy_success=ok[0], success_rate=float(np.mean(ok)),
                                         baseline_share_A=float(np.mean([x == A for x in base_r]))))
        print(f"[explorer] {tid}: steering towards reading {A} done", flush=True)
    stp.parent.mkdir(parents=True, exist_ok=True)
    stp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    _dump(out / "steering" / f"{tid}_baseline.json", dict(assigned=base_r, texts=base_txt))
    return S, Bx, rows


def summarise(out, results):
    hard = [S for S, _, _ in results if S["kind"] == "hard"]
    ctrl = [S for S, _, _ in results if S["kind"] == "control"]
    n2 = sum(len(S["readings"]) >= 2 for S in hard)
    cells = {}
    for _, _, rows in results:
        for r in rows:
            cells.setdefault((r["variant"], r["mode"]), {}).setdefault((r["id"], r["A"], r["B"]), []).append(r)
    by = {}
    for (var, mode), cases in cells.items():
        succ = [any(x["success_rate"] >= 0.5 for x in rs if x["kl_level"] <= 2) for rs in cases.values()]
        lift = [max(x["success_rate"] for x in rs if x["kl_level"] <= 2) - rs[0]["baseline_share_A"] for rs in cases.values()]
        by[f"{var}/{mode}"] = dict(n_cases=len(succ), success_rate=float(np.mean(succ)) if succ else None,
                                   median_lift_over_baseline=float(np.median(lift)) if lift else None)
    head = by.get("natural/cursor", {})
    summ = dict(n_hard=len(hard), n_hard_with_2plus=n2, control_readings=[len(S["readings"]) for S in ctrl],
                steer_success_rate=head.get("success_rate"), by_variant_mode=by,
                rule="pass = n_hard_with_2plus >= n_hard/2 AND natural/cursor success_rate >= 0.6 (success = >=50% of "
                     "the 9 continuations land in A and are fluent, at one of the three lower KL levels: at_b <= 1 nat at the cursor; cursor <= 0.25 nats mean per token)")
    summ["pass"] = bool(n2 >= len(hard) / 2 and (head.get("success_rate") or 0) >= 0.6)
    summ["control_warning"] = any(len(S["readings"]) > 1 for S in ctrl)
    _dump(out / "summary.json", summ)
    return summ


def run(cfg):
    from explorer.frame import load, load_texts
    from explorer.steer import CursorHook
    from ut import arch
    m = cfg.get("model", "g3_4b_pt")
    out = OUT / m
    tok, model = load(m)
    layer = cfg.get("layer") if cfg.get("layer") is not None else arch.n_layers(model) // 2
    hook = CursorHook(model, layer)
    texts = load_texts()
    if cfg.get("texts"):
        texts = [t for t in texts if t["id"] in cfg["texts"]]
    prompt_style = cfg.get("prompt_style", "legacy")
    clustering = cfg.get("clustering", "semantic")
    alpha_grad = cfg.get("alpha_grad", 0.5)
    disp_min = cfg.get("disp_min", 0.035)
    res = []
    t0 = time.time()
    for i, txt in enumerate(texts):
        res.append(run_text(
            model, tok, hook, layer, txt, out, cfg.get("n", 64),
            prompt_style=prompt_style,
            clustering=clustering,
            alpha_grad=alpha_grad,
            disp_min=disp_min,
        ))
        el = time.time() - t0
        print(f"[explorer] {i + 1}/{len(texts)} texts, {el / 60:.1f} min, ~{el / (i + 1) * (len(texts) - i - 1) / 60:.1f} min left",
              flush=True)
    s = summarise(out, res)
    print("[explorer] summary:", json.dumps(s), flush=True)
    return out
