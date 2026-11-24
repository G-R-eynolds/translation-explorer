"""Stages (HANDOFF_blind_spots.md §5).  GPU: e0, collect, fdcheck, steer, sae, atlas_dirs.  CPU: spectra, heatmaps,
practices, validate, atlas_cluster.  Each is run(cfg) -> [outputs]; see blind/__main__.py and blind/queues/.

cfg keys: model, set (e1|e2|e3|atlas), K (64), batch (8), langs (list, overrides the targets), n (atlas size),
layer (overrides the headline layer), items (list of ids), force.
"""
import json
import re
import zlib

import numpy as np

from blind import spectrum as sp
from blind.common import (EPS_HEAD, EPS_REL, K_DEFAULT, N_SPLITS, SAE, channels_for, grad_path, headline_layer,
                          instruction, layers_for, load_builtin, load_set, max_new_tokens, read_jsonl, res_dir,
                          stage, write_jsonl, CACHE)
from ut.common import save_json, load_json
from blind.fisher import dev


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def _load_model(cfg):
    from ut.common import load_model
    tok, model = load_model(cfg["model"])
    model.requires_grad_(False)
    import torch
    # Gemma 3 loads as a multimodal model: its image encoder (~0.8 GB) is never used on text, and on the 12 GB card the
    # extra memory pushed the driver into system-RAM fallback (GPU at 100% but ~70 W, items 10x slower).
    inner = getattr(model, "model", None)
    for name in ("vision_tower", "multi_modal_projector"):
        if getattr(inner, name, None) is not None:
            setattr(inner, name, torch.nn.Identity())      # parameter-free stub: model.device stays the GPU
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tok, model


def _ctx(cfg):
    from blind.fisher import Hooks, Prompter
    tok, model = _load_model(cfg)
    return tok, model, Prompter(tok), Hooks(model, layers_for(cfg["model"]))


def _items(cfg):
    items = load_set(cfg["set"], n=cfg.get("n"))
    if cfg.get("items"):
        items = [i for i in items if i.id in cfg["items"]]
    return items


def _layer(cfg):
    return cfg["layer"] if cfg.get("layer") is not None else headline_layer(cfg["model"])


# ------------------------------------------------------------------ E0
@stage("e0")
def e0(cfg):
    import torch
    from blind.fisher import generate_ids, logp_shifted, per_sample_grads, states_at
    m = cfg["model"]
    tok, model, pr, hooks = _ctx(cfg)
    layers = layers_for(m)
    items = load_builtin()[0]
    out = dict(identical_h=[], grad_fd=[])
    chans = ["src", "src2", "copy", "en", "it", "fr", "pt", "p_keep", "p_gloss", "p_foot", "p_thick"]
    for it in [items[0], items[12], items[30]]:
        ref_ids, P = pr.build(it.text, instruction("src", "es"))
        ref = states_at(model, hooks, ref_ids, P, layers)
        for ch in chans[1:]:
            ids, P2 = pr.build(it.text, instruction(ch, "es"))
            assert P2 == P and ids[:P[1]] == ref_ids[:P[1]], f"source ids differ for {ch}"
            st = states_at(model, hooks, ids, P, layers)
            for k in layers:
                dlt = st[k] - ref[k]
                rel = float((dlt.norm(dim=-1) / ref[k].norm(dim=-1)).max())     # worst source position
                out["identical_h"].append(dict(item=it.id, channel=ch, layer=k, max_abs=float(dlt.abs().max()),
                                               rel=rel, exact=bool((dlt == 0).all())))
    # Exact zero whenever the prompts have the same length; otherwise bf16 kernel-shape noise (~0.4% on Gemma 3 4B;
    # max |dh|/rms is meaningless there because a few residual dims reach ~8e4, where one bf16 step is ~500).
    worst = max(r["rel"] for r in out["identical_h"])
    out["identical_h_max_rel"] = worst
    out["identical_h_n_exact"] = sum(r["exact"] for r in out["identical_h"])
    out["identical_h_n"] = len(out["identical_h"])
    out["identical_h_pass"] = worst <= 3e-2
    print(f"[e0] identical-h: max per-position |dh|/|h| = {worst:.2e}; exactly equal in "
          f"{out['identical_h_n_exact']}/{out['identical_h_n']}", flush=True)
    save_json(res_dir(m) / "e0.json", out)
    assert out["identical_h_pass"], "source states differ across channels beyond bf16 noise"
    # gradient vs finite differences
    it = items[0]
    ids, P = pr.build(it.text, instruction("src", "es"))
    ys = generate_ids(model, tok, ids, 4, max_new_tokens("src", P[1] - P[0]), seed=1)
    G, lp0 = per_sample_grads(model, hooks, ids, P, ys, layers, batch=4)
    base = states_at(model, hooks, ids, P, layers)
    rng = np.random.default_rng(0)
    for k in layers:
        hn = float(base[k].norm())
        for trel in (1e-3, 3e-3, 1e-2, 3e-2):
            an, fd = [], []
            for _ in range(3):
                u = rng.normal(size=G[k].shape[1:]); u /= np.linalg.norm(u)
                t = trel * hn
                ut_ = torch.tensor(u * t, dtype=torch.float32)
                lpp = logp_shifted(model, hooks, ids, P, ys, {k: ut_})
                lpm = logp_shifted(model, hooks, ids, P, ys, {k: -ut_})
                fd += list((lpp - lpm) / (2 * t)); an += list((G[k] * u).sum((1, 2)))
            an, fd = np.array(an), np.array(fd)
            c = float(np.corrcoef(an, fd)[0, 1]) if an.std() > 0 and fd.std() > 0 else None
            out["grad_fd"].append(dict(layer=k, t_rel=trel, corr=c,
                                       med_rel_err=float(np.median(np.abs(an - fd) / (np.abs(fd) + 1e-12))),
                                       analytic=an.tolist(), fd=fd.tolist()))
    best = {}
    for r in out["grad_fd"]:
        if r["corr"] is not None:
            best[r["layer"]] = max(best.get(r["layer"], -1), r["corr"])
    out["grad_fd_best_corr"] = best          # a layer with all-zero gradients (the last block) is left out
    out["grad_fd_pass"] = bool(best) and all(v > 0.9 for v in best.values())
    print(f"[e0] grad vs FD best corr per layer: {best}", flush=True)
    p = res_dir(m) / "e0.json"
    save_json(p, out)
    return [p]


# ------------------------------------------------------------------ collection
@stage("collect")
def collect(cfg):
    import torch
    from blind.fisher import Clock, decode, generate_ids, per_sample_grads, save_grads
    m, S = cfg["model"], cfg["set"]
    K, batch = cfg.get("K", K_DEFAULT), cfg.get("batch", 8)
    tok, model, pr, hooks = _ctx(cfg)
    layers = layers_for(m)
    items = _items(cfg)
    clock = Clock(f"collect {S} {m}", len(items))
    samples, n_new = [], 0
    for it in items:
        tspans = pr.token_spans(it.text, it.spans())
        parts = [[j for j in pr.token_spans(it.text, [("p", a, b)])["p"]] for a, b in it.parts] if it.parts else None
        for ch in channels_for(it, S, cfg.get("langs")):
            path = grad_path(m, S, it.id, ch)
            if path.exists():
                continue
            ids, P = pr.build(it.text, instruction(ch, it.lang))
            mx = max_new_tokens(ch, P[1] - P[0])
            ys = generate_ids(model, tok, ids, K, mx, seed=_seed(it.id, ch))
            b = batch
            while True:
                try:
                    G, lp = per_sample_grads(model, hooks, ids, P, ys, layers, batch=b)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); hooks.z = {}
                    if b == 1:
                        raise
                    b //= 2
                    print(f"  OOM: batch -> {b}", flush=True)
            texts = [decode(tok, y) for y in ys]
            meta = dict(item=it.id, set=S, channel=ch, lang=it.lang, group=it.group, text=it.text, P=list(P),
                        tokens=pr.tokens(it.text), spans=tspans, parts=parts, lens=[len(y) for y in ys],
                        logp=lp.tolist(), n_capped=int(sum(1 for y in ys if len(y) >= mx)), max_new=mx, texts=texts)
            save_grads(path, G, meta)
            del G
            torch.cuda.empty_cache()
            n_new += 1
            samples.append(dict(item=it.id, channel=ch, texts=texts[:8], mean_len=float(np.mean(meta["lens"])),
                                n_capped=meta["n_capped"]))
        clock.tick(it.id)
    p = res_dir(m) / f"samples_{S}.jsonl"
    old = {(r["item"], r["channel"]): r for r in read_jsonl(p)}
    old.update({(r["item"], r["channel"]): r for r in samples})
    write_jsonl(p, old.values())
    print(f"[collect] {n_new} new (item, channel) files", flush=True)
    return [p]


def load_item_grads(m, S, item_id, layers=None):
    from blind.fisher import load_grads
    d = CACHE / m / "grads" / S / item_id
    out = {}
    for f in sorted(d.glob("*.npz")) if d.exists() else []:
        if f.name.endswith(".tmp.npz"):
            continue
        out[f.stem] = load_grads(f, layers)
    return out


# ------------------------------------------------------------------ spectra
def _rows_for(cmp, meta, item, targets, k, eps_list, scope, n_splits, extra=None):
    rows = []
    len_src = float(np.mean(meta["src"]["lens"]))
    for t in targets:
        lr = float(np.mean(meta[t]["lens"])) / len_src
        for eps in eps_list:
            s = sp.summarise(sp.split_spectra(cmp, "src", t, eps, n_splits), len_ratio=lr, keep_splits=scope == "whole")
            rows.append(dict(item=item, group=meta["src"]["group"], lang=meta["src"]["lang"], target=t, layer=k,
                             eps=eps, scope=scope, len_src=len_src, len_tgt=lr * len_src, **(extra or {}), **s))
    return rows


def surface_share(cmp, eps, n_splits=N_SPLITS, top=8, seed=0):
    """Share of U (into en) carried by blind directions that the copy channel registers more strongly than the
    held-out paraphrase does (surface, not meaning; §1 copy diagnostic)."""
    rng = np.random.default_rng(seed)
    iS, iT, iC = cmp.idx("src"), cmp.idx("en"), cmp.idx("copy")
    h = min(len(iS), len(iT)) // 2
    shares = []
    for _ in range(n_splits):
        ps, pt = rng.permutation(iS), rng.permutation(iT)
        lam0 = cmp.eig(ps[:h], ps[h:2 * h], eps)
        lam, C, I = cmp.eig(ps[:h], pt[:h], eps, vectors=True)
        tot = sur = 0.0
        for i in range(min(top, len(lam), len(lam0))):
            if lam0[i] <= 0:
                continue
            c = 0.5 * np.log2(max(1.0, lam[i] / lam0[i]))
            tot += c
            if cmp.quad(iC, C, I, i) > cmp.quad(ps[h:2 * h], C, I, i):
                sur += c
        shares.append(sur / tot if tot > 0 else np.nan)
    return float(np.nanmedian(shares)) if not all(np.isnan(shares)) else None


@stage("spectra")
def spectra(cfg):
    m, S = cfg["model"], cfg["set"]
    layers = layers_for(m)
    n_splits = cfg.get("n_splits", N_SPLITS)
    head = _layer(cfg)
    rows = []
    items = _items(cfg)
    for n_it, it in enumerate(items):
        g = load_item_grads(m, S, it.id)
        if "src" not in g:
            continue
        meta = {c: v[1] for c, v in g.items()}
        targets = [c for c in g if c != "src"]
        for k in layers:
            cmp = sp.Comparator({c: sp.flat(v[0][k]) for c, v in g.items()})
            rows += _rows_for(cmp, meta, it.id, targets, k, EPS_REL, "whole", n_splits)
            if "copy" in g and "en" in g and k == head:
                ss = surface_share(cmp, EPS_HEAD, n_splits)
                for r in rows:
                    if r["item"] == it.id and r["layer"] == k and r["target"] == "en" and r["scope"] == "whole":
                        r["surface_share"] = ss
            del cmp
            if k != head:
                continue
            spans = dict(meta["src"].get("spans") or {})
            for j, pos in enumerate(meta["src"].get("parts") or []):
                spans[f"part{j + 1}"] = pos
            for name, pos in spans.items():
                if not pos:
                    continue
                cmp = sp.Comparator({c: sp.flat(v[0][k], pos) for c, v in g.items()})
                rows += _rows_for(cmp, meta, it.id, targets, k, [EPS_HEAD], f"span:{name}", n_splits,
                                  extra=dict(positions=pos))
        print(f"[spectra {S}] {n_it + 1}/{len(items)} {it.id}", flush=True)
    p = res_dir(m) / f"spectra_{S}.jsonl"
    write_jsonl(p, rows)
    allrows = []
    for q in sorted(res_dir(m).glob("spectra_*.jsonl")):
        allrows += [dict(set=q.stem.split("_", 1)[1], **r) for r in read_jsonl(q)]
    write_jsonl(res_dir(m) / "spectra.jsonl", allrows)
    return [p]


def spectra_index(m, S):
    """{(item, target, layer, eps, scope): row}"""
    return {(r["item"], r["target"], r["layer"], r["eps"], r["scope"]): r
            for r in read_jsonl(res_dir(m) / f"spectra_{S}.jsonl")}


# ------------------------------------------------------------------ heat maps
@stage("heatmaps")
def heatmaps(cfg):
    m, S = cfg["model"], cfg["set"]
    layers = cfg.get("layers") or layers_for(m)
    head = _layer(cfg)
    n_splits = cfg.get("n_splits", N_SPLITS)
    p = res_dir(m) / "heatmaps.json"
    out = load_json(p) if p.exists() else {}
    for it in _items(cfg):
        g = load_item_grads(m, S, it.id)
        if "src" not in g:
            continue
        meta = g["src"][1]
        targets = [c for c in g if c not in ("src", "copy")]
        rec = dict(set=S, group=meta["group"], text=it.text, tokens=meta["tokens"], layers={})
        for k in layers:
            if k != head and cfg.get("head_only", False):
                continue
            per = {t: dict(U=[], U_null_hi=[]) for t in targets}
            for j in range(len(meta["tokens"])):
                cmp = sp.Comparator({c: sp.flat(v[0][k], [j]) for c, v in g.items() if c != "copy"})
                for t in targets:
                    s = sp.summarise(sp.split_spectra(cmp, "src", t, EPS_HEAD, n_splits, reverse=False), keep_splits=False)
                    per[t]["U"].append(s["U"]["med"]); per[t]["U_null_hi"].append(s["U_null"]["hi"])
            rec["layers"][str(k)] = per
        out[it.id] = rec
        print(f"[heatmaps] {it.id}", flush=True)
        save_json(p, out)
    return [p]


# ------------------------------------------------------------------ directions on the GPU: FD check, steering, SAE
def _direction(m, S, it_id, target, k, seen=False, n=1, g=None):
    g = g or load_item_grads(m, S, it_id, [k])
    cmp = sp.Comparator({c: sp.flat(g[c][0][k]) for c in ("src", target)})
    row = spectra_index(m, S).get((it_id, target, k, EPS_HEAD, "whole"))
    nb = (row or {}).get("n_seen" if seen else "n_blind", {}).get("med", 0)
    ds = sp.directions(cmp, "src", target, n, seen=seen)
    for d in ds:
        d["above_null"] = bool(nb >= 1)
        d["n_above_null"] = nb
        d["alpha"] = sp.alpha_for(d["q_src"], d["q_tgt"])
    return ds, g


@stage("fdcheck")
def fdcheck(cfg):
    import torch
    from scipy.stats import spearmanr
    from blind.fisher import generate_ids, logp_shifted
    m, S = cfg["model"], cfg.get("set", "e1")
    k = _layer(cfg)
    tok, model, pr, hooks = _ctx(cfg)
    items = _items({**cfg, "set": S, "items": None})
    default = [i.id for t in ("gender_adj", "plural_you", "formal_you", "lexical") for i in
               [x for x in items if x.group == t][:2]]
    items = [i for i in items if i.id in (cfg.get("items") or default)]
    rows = []
    for it in items:
        if not (CACHE / m / "grads" / S / it.id / "src.npz").exists():
            continue
        blind, g = _direction(m, S, it.id, "en", k, n=2)
        seen, _ = _direction(m, S, it.id, "en", k, seen=True, n=2, g=g)
        P = g["src"][1]["P"]
        npos = P[1] - P[0]
        fresh = {}
        for ch in ("src", "en"):
            ids, _ = pr.build(it.text, instruction(ch, it.lang))
            ys = generate_ids(model, tok, ids, 64, max_new_tokens(ch, npos), seed=_seed(it.id, ch, "fresh"))
            fresh[ch] = (ids, ys, logp_shifted(model, hooks, ids, tuple(P), ys))
        for kind, ds in (("blind", blind), ("seen", seen)):
            for rank, d in enumerate(ds):
                a = d["alpha"]
                delta = torch.tensor((a * d["v"]).reshape(npos, -1), dtype=torch.float32)
                meas = {}
                for ch, (ids, ys, lp0) in fresh.items():
                    lp1 = logp_shifted(model, hooks, ids, tuple(P), ys, {k: delta})
                    meas[ch] = float(np.mean(lp0 - lp1))
                rows.append(dict(item=it.id, group=it.group, kind=kind, rank=rank, layer=k, lam=d["lam"], alpha=a,
                                 above_null=d["above_null"], pred_kl_src=0.5 * a * a * d["q_src"],
                                 pred_kl_en=0.5 * a * a * d["q_tgt"], meas_kl_src=meas["src"], meas_kl_en=meas["en"]))
                print(f"[fd] {it.id} {kind}{rank}: pred src/en {rows[-1]['pred_kl_src']:.2f}/{rows[-1]['pred_kl_en']:.2f}"
                      f"  measured {meas['src']:.2f}/{meas['en']:.2f}", flush=True)
    pred = [r[f"pred_kl_{c}"] for r in rows for c in ("src", "en")]
    meas = [r[f"meas_kl_{c}"] for r in rows for c in ("src", "en")]
    lr = [np.log(max(r["meas_kl_src"], 1e-6) / max(r["meas_kl_en"], 1e-6)) for r in rows]
    pl = [np.log(max(r["lam"], 1e-12)) * (1 if r["kind"] == "blind" else -1) for r in rows]
    out = dict(rows=rows, spearman_pred_meas=float(spearmanr(pred, meas)[0]) if len(rows) > 2 else None,
               spearman_lambda_ratio=float(spearmanr(pl, lr)[0]) if len(rows) > 2 else None)
    p = res_dir(m) / "fd_check.json"
    save_json(p, out)
    return [p]


def _flip_check(it, src_texts, en_texts, base_idx=2):
    """Auto pre-count (§5 E1 test 4): the paraphrase loses the 'from' word and gains the 'to' word at some setting,
    while the English at that setting is the same as at 0.  Final count by reading the texts."""
    if not it.flip:
        return None
    f, t = it.flip
    has = lambda w, s: re.search(rf"(?<!\w){re.escape(w)}(?!\w)", s, re.I) is not None
    hits = []
    for j, (s, e) in enumerate(zip(src_texts, en_texts)):
        if j == base_idx:
            continue
        if has(t, s) and not has(f, s):
            hits.append(dict(setting=j, en_unchanged=e.strip() == en_texts[base_idx].strip()))
    return dict(paraphrase_flipped=bool(hits), flipped_en_unchanged=any(h["en_unchanged"] for h in hits),
                hits=hits, base_has_from=has(f, src_texts[base_idx]))


@stage("steer")
def steer(cfg):
    import torch
    from blind.fisher import steer_greedy
    m, S = cfg["model"], cfg["set"]
    k = _layer(cfg)
    tok, model, pr, hooks = _ctx(cfg)
    items = _items(cfg)
    if S == "e2" and not cfg.get("items"):
        items = [i for i in items if i.id in ("es_duende", "es_duende_flamenco", "es_duende_cuento", "es_sobremesa",
                                              "es_gustar", "es_tomar_el_pelo", "es_tren")]
    rows = []
    mult = [-2, -1, 0, 1, 2]
    show = ["src", "en"] + (["it"] if S == "e1" else [])
    for it in items:
        if not (CACHE / m / "grads" / S / it.id / "src.npz").exists():
            continue
        ds, g = _direction(m, S, it.id, "en", k, n=1)
        if not ds:
            continue
        d = ds[0]
        P = g["src"][1]["P"]
        npos = P[1] - P[0]
        V = torch.tensor(d["v"].reshape(npos, -1), dtype=torch.float32)
        deltas = torch.stack([c * d["alpha"] * V for c in mult])
        texts = {}
        for ch in show:
            ids, _ = pr.build(it.text, instruction(ch, it.lang))
            texts[ch] = steer_greedy(model, tok, hooks, ids, tuple(P), k, deltas, max_new_tokens(ch, npos))
        ddir = CACHE / m / "dirs" / S
        ddir.mkdir(parents=True, exist_ok=True)
        np.savez(ddir / f"{it.id}_en_L{k}.npz", v=d["v"].astype(np.float32), alpha=d["alpha"], P=np.array(P))
        r = dict(item=it.id, group=it.group, text=it.text, layer=k, lam=d["lam"], above_null=d["above_null"],
                 n_above_null=d["n_above_null"], alpha=d["alpha"], settings=mult,
                 profile=sp.position_profile(d["v"], npos), tokens=g["src"][1]["tokens"],
                 **{f"{c}_texts": v for c, v in texts.items()}, flip=_flip_check(it, texts["src"], texts["en"]))
        rows.append(r)
        print(f"[steer] {it.id}: src {texts['src'][1]!r} | {texts['src'][2]!r} | {texts['src'][3]!r}", flush=True)
    p = res_dir(m) / f"steering_{S}.jsonl"
    write_jsonl(p, rows)
    allrows = []
    for q in sorted(res_dir(m).glob("steering_*.jsonl")):
        allrows += read_jsonl(q)
    write_jsonl(res_dir(m) / "steering.jsonl", allrows)
    return [p]


# ------------------------------------------------------------------ SAE
def load_sae(m, k, device):
    from sae_lens import SAE as _S
    rel, sid, _, _ = SAE[m]
    r = _S.from_pretrained(release=rel, sae_id=sid.format(k=k), device=str(device))
    return (r[0] if isinstance(r, tuple) else r).eval()


_NP = {}


def np_label(m, k, f, cache_path):
    import urllib.request
    if not _NP and cache_path.exists():
        _NP.update(load_json(cache_path))
    key = f"{k}:{int(f)}"
    if key in _NP:
        return _NP[key]
    _, _, npm, nps = SAE[m]
    url = f"https://www.neuronpedia.org/api/feature/{npm}/{nps.format(k=k)}/{int(f)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
        expl = [e.get("description") for e in d.get("explanations", []) if e.get("description")]
        _NP[key] = expl[0] if expl else None
    except Exception as e:
        return None                                    # not cached: retried next time
    save_json(cache_path, _NP)
    return _NP[key]


def sae_diff(sae, h, delta):
    """(|P|, n_features) activation change between h+delta and h-delta, plus base activations."""
    import torch
    with torch.no_grad():
        dt = sae.W_dec.dtype
        dev = sae.W_dec.device
        a = lambda x: sae.encode(x.to(dev, dt)).float().cpu()
        return a(h + delta) - a(h - delta), a(h)


@stage("sae")
def sae_stage(cfg):
    import torch
    from blind.fisher import states_at
    m, S = cfg["model"], cfg["set"]
    k = _layer(cfg)
    if m not in SAE:
        print(f"[sae] no SAEs for {m}; skipped")
        return []
    tok, model, pr, hooks = _ctx(cfg)
    sae = load_sae(m, k, dev(model))
    npc = res_dir(m) / "neuronpedia_cache.json"
    rows = []
    for r in read_jsonl(res_dir(m) / f"steering_{S}.jsonl"):
        z = np.load(CACHE / m / "dirs" / S / f"{r['item']}_en_L{k}.npz")
        P = tuple(int(x) for x in z["P"])
        it = next(i for i in load_set(S) if i.id == r["item"])
        ids, _ = pr.build(it.text, instruction("src", it.lang))
        h = states_at(model, hooks, ids, P, [k])[k]
        delta = torch.tensor(z["v"].reshape(P[1] - P[0], -1) * float(z["alpha"]))
        diff, base = sae_diff(sae, h, delta)
        tot = diff.abs().sum(0)
        top = torch.topk(tot, 20).indices.tolist()
        per_pos = [[dict(f=int(f), d=float(diff[j, f])) for f in torch.topk(diff[j].abs(), 5).indices.tolist()]
                   for j in range(diff.shape[0])]
        rows.append(dict(item=r["item"], layer=k, tokens=r["tokens"],
                         top=[dict(f=f, change=float(diff[:, f].sum()), abs_change=float(tot[f]),
                                   base=float(base[:, f].max()), label=np_label(m, k, f, npc)) for f in top],
                         per_position=per_pos))
        print(f"[sae] {r['item']}: " + "; ".join(f"{x['f']} {x['label']}" for x in rows[-1]["top"][:3]), flush=True)
    p = res_dir(m) / f"sae_{S}.jsonl"
    write_jsonl(p, rows)
    allrows = []
    for q in sorted(res_dir(m).glob("sae_*.jsonl")):
        allrows += read_jsonl(q)
    write_jsonl(res_dir(m) / "sae.jsonl", allrows)
    return [p]


# ------------------------------------------------------------------ E3
@stage("practices")
def practices(cfg):
    from blind.common import PRACTICES
    m = cfg["model"]
    S = "e3"
    k = _layer(cfg)
    idx = spectra_index(m, S)
    out = []
    for it in load_set(S):
        g = load_item_grads(m, S, it.id, [k])
        if "src" not in g or "en" not in g:
            continue
        others = [c for c in PRACTICES if c in g]
        cmp = sp.Comparator({c: sp.flat(g[c][0][k]) for c in ["src", "en", *others]})
        lens = {c: float(np.mean(g[c][1]["lens"])) for c in g}
        dirs = sp.held_out_rho(cmp, "src", "en", others, n_dirs=cfg.get("n_dirs", 5))
        row = idx.get((it.id, "en", k, EPS_HEAD, "whole"), {})
        nb = int(round(row.get("n_blind", {}).get("med", 0)))
        rec = dict(item=it.id, group=it.group, layer=k, n_blind_plain=nb, lens=lens, directions=dirs, practices={})
        for c in ["plain", *others]:
            ch = "en" if c == "plain" else c
            rho = np.array([d["rho"][c] for d in dirs])
            rho_ln = rho * lens["src"] / lens[ch]
            r2 = idx.get((it.id, ch, k, EPS_HEAD, "whole"), {})
            rec["practices"][c] = dict(
                U=r2.get("U", {}).get("med"), U_lennorm=r2.get("U_lennorm", {}).get("med"),
                U_null_hi=r2.get("U_null", {}).get("hi"),
                seen_frac=float(np.mean(np.minimum(1, rho))) if len(rho) else None,
                share_seen=float(np.mean(rho >= 0.5)) if len(rho) else None,
                seen_frac_lennorm=float(np.mean(np.minimum(1, rho_ln))) if len(rho) else None,
                share_seen_lennorm=float(np.mean(rho_ln >= 0.5)) if len(rho) else None)
        out.append(rec)
    p = res_dir(m) / "practices.json"
    save_json(p, out)
    return [p]


# ------------------------------------------------------------------ validation (E0 + E1 tests)
def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def _paired(Ua, Ub):
    from scipy.stats import wilcoxon
    common = sorted(set(Ua) & set(Ub))
    d = np.array([Ua[i] - Ub[i] for i in common])
    if len(d) == 0:
        return dict(n=0)
    try:
        p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = None
    return dict(n=len(d), median_diff=float(np.median(d)), ci95=_boot_ci(d), p_greater=p,
                n_positive=int((d > 0).sum()))


@stage("validate")
def validate(cfg):
    m = cfg["model"]
    b, types = load_builtin()
    grp = {i.id: i.group for i in b}
    rows = [r for r in read_jsonl(res_dir(m) / "spectra_e1.jsonl") if r["scope"] == "whole"]
    out = dict(model=m, headline_layer=_layer(cfg), eps=EPS_HEAD)
    e0p = res_dir(m) / "e0.json"
    if e0p.exists():
        e0 = load_json(e0p)
        out["e0"] = {k: e0[k] for k in ("identical_h_max_rel", "identical_h_pass", "grad_fd_best_corr", "grad_fd_pass")}

    def U(target, layer, eps=EPS_HEAD, groups=None, key="U"):
        return {r["item"]: r[key]["med"] for r in rows if r["target"] == target and r["layer"] == layer
                and r["eps"] == eps and (groups is None or grp[r["item"]] in groups)}

    def tests_at(layer, eps):
        t = {}
        gram = ["gender_adj", "plural_you", "formal_you"]
        t["t1_pooled"] = {f"en>{x}": _paired(U("en", layer, eps, gram), U(x, layer, eps, gram)) for x in ("it", "pt")}
        t["t1_by_type"] = {g: {f"en>{x}": _paired(U("en", layer, eps, [g]), U(x, layer, eps, [g])) for x in ("it", "pt", "fr")}
                           for g in gram}
        pro = ["profession"]
        t["t2"] = {"en>it": _paired(U("en", layer, eps, pro), U("it", layer, eps, pro)),
                   "fr>it": _paired(U("fr", layer, eps, pro), U("it", layer, eps, pro)),
                   "fr-en": _paired(U("fr", layer, eps, pro), U("en", layer, eps, pro))}
        t["t1_pass"] = all(v["p_greater"] is not None and v["p_greater"] < 0.05 for v in t["t1_pooled"].values())
        t["t2_pass"] = (all(t["t2"][k].get("p_greater") is not None and t["t2"][k]["p_greater"] < 0.05
                            for k in ("en>it", "fr>it")) and t["t2"]["fr-en"].get("ci95", [1, 1])[0] is not None
                        and t["t2"]["fr-en"]["ci95"][0] <= 0 <= t["t2"]["fr-en"]["ci95"][1])
        ctrl = [r for r in rows if r["layer"] == layer and r["eps"] == eps and grp[r["item"]] in ("ser_estar", "lexical")
                and r["target"] in ("en", "it", "fr", "pt")]
        near = [r["U"]["med"] <= r["U_null"]["hi"] for r in ctrl]
        t["t3"] = dict(n=len(near), share_near_null=float(np.mean(near)) if near else None,
                       by_lang={L: float(np.mean([r["U"]["med"] <= r["U_null"]["hi"] for r in ctrl if r["target"] == L]))
                                for L in ("en", "it", "fr", "pt") if any(r["target"] == L for r in ctrl)})
        t["t3_pass"] = bool(near) and np.mean(near) >= 0.75
        s2 = [r for r in rows if r["layer"] == layer and r["eps"] == eps and r["target"] == "src2"]
        nn = [r["U"]["med"] <= r["U_null"]["hi"] for r in s2]
        t["src2"] = dict(n=len(nn), share_near_null=float(np.mean(nn)) if nn else None,
                         median_U_src2=float(np.median([r["U"]["med"] for r in s2])) if s2 else None,
                         median_U_en=float(np.median(list(U("en", layer, eps).values()) or [np.nan])))
        t["src2_pass"] = bool(nn) and np.mean(nn) >= 0.75
        t["median_U_by_type"] = {g: {L: float(np.median(list(U(L, layer, eps, [g]).values()) or [np.nan]))
                                     for L in ("en", "it", "fr", "pt", "src2", "copy")} for g in types}
        t["U_tracenorm_t1"] = {f"en>{x}": _paired(U("en", layer, eps, gram, "U_tracenorm"),
                                                 U(x, layer, eps, gram, "U_tracenorm")) for x in ("it", "pt")}
        return t

    head = _layer(cfg)
    out["headline"] = tests_at(head, EPS_HEAD)
    out["robustness"] = {f"L{k}_eps{e:g}": {x: out_ for x, out_ in tests_at(k, e).items() if x.endswith("pass")}
                         for k in layers_for(m) for e in EPS_REL}
    st = read_jsonl(res_dir(m) / "steering_e1.jsonl")
    t4 = {}
    for g in ("gender_adj", "profession", "plural_you", "formal_you", "aspect", "ser_estar"):
        rs = [r for r in st if r["group"] == g and r.get("flip")]
        t4[g] = dict(n=len(rs), auto_flipped_en_unchanged=sum(r["flip"]["flipped_en_unchanged"] for r in rs),
                     auto_flipped=sum(r["flip"]["paraphrase_flipped"] for r in rs))
    out["t4_auto"] = t4
    out["t4_auto_pass"] = all(t4[g]["n"] and t4[g]["auto_flipped_en_unchanged"] >= t4[g]["n"] / 2
                              for g in ("gender_adj", "profession", "plural_you", "formal_you"))
    out["t4_note"] = "automatic pre-count from the flip markers; the final count is made by reading steering_e1.jsonl"
    surf = [r.get("surface_share") for r in rows if r["layer"] == head and r["eps"] == EPS_HEAD
            and r["target"] == "en" and r.get("surface_share") is not None]
    out["surface_share_median"] = float(np.median(surf)) if surf else None
    p = res_dir(m) / "validation.json"
    save_json(p, out)
    h = out["headline"]
    print(f"[validate {m}] t1 {h['t1_pass']}  t2 {h['t2_pass']}  t3 {h['t3_pass']}  src2 {h['src2_pass']}  "
          f"t4(auto) {out['t4_auto_pass']}", flush=True)
    return [p]


# ------------------------------------------------------------------ E4 atlas
@stage("atlas_dirs")
def atlas_dirs(cfg):
    import torch
    from blind.fisher import states_at, steer_greedy
    m = cfg["model"]
    S = "atlas"
    k = _layer(cfg)
    tok, model, pr, hooks = _ctx(cfg)
    sae = load_sae(m, k, dev(model)) if m in SAE else None
    rows, acts = [], []
    items = _items({**cfg, "set": S})
    for n_it, it in enumerate(items):
        g = load_item_grads(m, S, it.id, [k])
        if "src" not in g or "en" not in g:
            continue
        cmp = sp.Comparator({c: sp.flat(g[c][0][k]) for c in ("src", "en")})
        s = sp.summarise(sp.split_spectra(cmp, "src", "en", EPS_HEAD), keep_splits=False)
        nb = int(np.floor(s["n_blind"]["med"]))
        ds = sp.directions(cmp, "src", "en", min(3, nb))
        P = tuple(g["src"][1]["P"])
        npos = P[1] - P[0]
        ids_s, _ = pr.build(it.text, instruction("src", "es"))
        ids_e, _ = pr.build(it.text, instruction("en", "es"))
        h = states_at(model, hooks, ids_s, P, [k])[k]
        if sae is not None:
            with torch.no_grad():
                base = sae.encode(h.to(sae.W_dec.device, sae.W_dec.dtype)).float().cpu()
            tv, ti = torch.topk(base, 16, dim=1)
            acts.append(dict(item=it.id, tokens=g["src"][1]["tokens"], top=[[(int(f), round(float(v), 3))
                             for f, v in zip(ti[j].tolist(), tv[j].tolist()) if v > 0] for j in range(npos)]))
        for rank, d in enumerate(ds):
            a = sp.alpha_for(d["q_src"], d["q_tgt"])
            V = torch.tensor(d["v"].reshape(npos, -1), dtype=torch.float32)
            deltas = torch.stack([c * a * V for c in (-1, 0, 1)])
            src_t = steer_greedy(model, tok, hooks, ids_s, P, k, deltas, max_new_tokens("src", npos))
            en_t = steer_greedy(model, tok, hooks, ids_e, P, k, deltas, max_new_tokens("en", npos))
            r = dict(item=it.id, text=it.text, rank=rank, layer=k, lam=d["lam"], alpha=a, U=s["U"]["med"],
                     U_null_hi=s["U_null"]["hi"], n_blind=s["n_blind"]["med"], tokens=g["src"][1]["tokens"],
                     profile=sp.position_profile(d["v"], npos), src_texts=src_t, en_texts=en_t)
            if sae is not None:
                diff, _ = sae_diff(sae, h, a * V)
                tot = diff.abs().sum(0)
                r["sae_top"] = [int(f) for f in torch.topk(tot, 20).indices.tolist()]
                r["sae_change"] = [float(diff[:, f].sum()) for f in r["sae_top"]]
            ddir = CACHE / m / "dirs" / S
            ddir.mkdir(parents=True, exist_ok=True)
            np.savez(ddir / f"{it.id}_r{rank}.npz", v=d["v"].astype(np.float32), alpha=a, P=np.array(P))
            rows.append(r)
        print(f"[atlas_dirs] {n_it + 1}/{len(items)} {it.id}: U {s['U']['med']:.2f}, {len(ds)} directions", flush=True)
    p = res_dir(m) / "atlas_dirs.jsonl"
    write_jsonl(p, rows)
    write_jsonl(res_dir(m) / "atlas_acts.jsonl", acts)
    return [p]


@stage("atlas_langs")
def atlas_langs(cfg):
    """ρ_c = vᵀĜ_c v / vᵀĜ_src v for it/fr/pt (and en) on each atlas direction whose sentence has those channels."""
    m = cfg["model"]
    k = _layer(cfg)
    out = []
    for r in read_jsonl(res_dir(m) / "atlas_dirs.jsonl"):
        g = load_item_grads(m, "atlas", r["item"], [k])
        langs = [c for c in ("en", "it", "fr", "pt") if c in g]
        if len(langs) < 2:
            continue
        v = np.load(CACHE / m / "dirs" / "atlas" / f"{r['item']}_r{r['rank']}.npz")["v"].astype(np.float64)
        qs = sp.quad_v(sp.flat(g["src"][0][k]), v)
        out.append(dict(item=r["item"], rank=r["rank"],
                        rho={c: sp.quad_v(sp.flat(g[c][0][k]), v) / max(qs, 1e-300) for c in langs}))
    p = res_dir(m) / "atlas_langs.json"
    save_json(p, out)
    return [p]


@stage("atlas_cluster")
def atlas_cluster(cfg):
    from collections import Counter
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    m = cfg["model"]
    k = _layer(cfg)
    rows = [r for r in read_jsonl(res_dir(m) / "atlas_dirs.jsonl") if r.get("sae_top")]
    n_sent = len({r["item"] for r in read_jsonl(res_dir(m) / "atlas_dirs.jsonl")})
    n_total = len(load_set("atlas", n=cfg.get("n")))
    langs = {(x["item"], x["rank"]): x["rho"] for x in (load_json(res_dir(m) / "atlas_langs.json")
                                                         if (res_dir(m) / "atlas_langs.json").exists() else [])}
    acts = read_jsonl(res_dir(m) / "atlas_acts.jsonl")
    if len(rows) < 3:
        print("[atlas_cluster] fewer than 3 directions; nothing to cluster")
        return []
    sets = [set(r["sae_top"]) for r in rows]
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1 - len(sets[i] & sets[j]) / len(sets[i] | sets[j])
    lab = fcluster(linkage(squareform(D), "average"), t=cfg.get("cut", 0.85), criterion="distance")
    npc = res_dir(m) / "neuronpedia_cache.json"
    clusters = []
    for c in sorted(set(lab)):
        mem = [rows[i] for i in range(n) if lab[i] == c]
        items = {r["item"] for r in mem}
        if len(items) < cfg.get("min_sentences", 3):
            continue
        feats = Counter(f for r in mem for f in r["sae_top"]).most_common(8)
        feat_info = []
        for f, cnt in feats:
            ctx = []
            for a in acts:
                for j, tops in enumerate(a["top"]):
                    for ff, v in tops:
                        if ff == f:
                            ctx.append((v, a["item"], j))
            ctx = sorted(ctx, reverse=True)[:5]
            tokmap = {a["item"]: a["tokens"] for a in acts}
            feat_info.append(dict(f=f, n_members=cnt, label=np_label(m, k, f, npc) if m in SAE else None,
                                  contexts=["".join(tokmap[i][max(0, j - 6):j]) + "[[" + tokmap[i][j] + "]]"
                                            + "".join(tokmap[i][j + 1:j + 3]) for _, i, j in ctx]))
        ex = sorted(mem, key=lambda r: -r["lam"])[:3]
        rho = [langs[(r["item"], r["rank"])] for r in mem if (r["item"], r["rank"]) in langs]
        kept = {L: float(np.median([x[L] for x in rho if L in x])) for L in ("en", "it", "fr", "pt")
                if any(L in x for x in rho)}
        clusters.append(dict(cluster=int(c), n_directions=len(mem), n_sentences=len(items),
                             share_of_sentences=len(items) / n_total, features=feat_info,
                             examples=[dict(item=r["item"], text=r["text"], src_minus=r["src_texts"][0],
                                            src_base=r["src_texts"][1], src_plus=r["src_texts"][2],
                                            en_minus=r["en_texts"][0], en_base=r["en_texts"][1],
                                            en_plus=r["en_texts"][2]) for r in ex],
                             median_rho_by_lang=kept, label=None))
    clusters.sort(key=lambda c: -c["n_sentences"])
    out = dict(model=m, layer=k, n_sentences=n_total, n_sentences_with_blind=n_sent, n_directions=n,
               clusters=clusters)
    p = res_dir(m) / "atlas.json"
    save_json(p, out)
    md = [f"# Atlas of blind spots, Spanish → English ({m}, layer {k})", "",
          f"{n_sentences_line(n_sent, n_total)}  {n} blind directions above the null; {len(clusters)} clusters with "
          f"at least {cfg.get('min_sentences', 3)} sentences (SAE-signature Jaccard, average linkage, cut "
          f"{cfg.get('cut', 0.85)}).", "",
          "**Labels are to be written from the steering pairs only** (stop point 5); until then each says TODO.", ""]
    for c in clusters:
        md += [f"## Cluster {c['cluster']}: LABEL TODO", "",
               f"- sentences: {c['n_sentences']} ({100 * c['share_of_sentences']:.0f}% of {n_total}); directions: "
               f"{c['n_directions']}",
               "- kept by (median ρ, 1 = registered as strongly as the Spanish paraphrase): "
               + (", ".join(f"{L} {v:.2f}" for L, v in c["median_rho_by_lang"].items()) or "not measured"),
               "- recurring SAE features: " + "; ".join(f"{x['f']} ({x['label'] or 'no label'})" for x in c["features"][:5]),
               ""]
        for e in c["examples"]:
            md += [f"> *{e['text']}*", f">", f"> paraphrase −: {e['src_minus']}", f"> paraphrase +: {e['src_plus']}",
                   f"> English −: {e['en_minus']}", f"> English +: {e['en_plus']}", ""]
    (res_dir(m) / "atlas.md").write_text("\n".join(md))
    return [p, res_dir(m) / "atlas.md"]


def n_sentences_line(n_sent, n_total):
    return f"{n_sent} of {n_total} sentences have at least one blind direction above the null."
