"""N2: SAE features that don't cross over (HANDOFF_internals §4).  GPU, main model (gemma12b).

SAEs: Gemma Scope 2, google/gemma-scope-2-12b-it, resid_post, width 16k, L0 "medium" (sae_lens release
gemma-scope-2-12b-it-res, ids layer_<k>_width_16k_l0_medium).  These are the IT SAEs, and exactly the ones Neuronpedia hosts
(source "<k>-gemmascope-2-res-16k"), so its explanations apply.  Gemma Scope "layer k" = block k's output = our
hidden_states[k+1].  Layers used (§4.1): the SAE layer nearest the N1 peak-sharing layer, the one nearest the R1
detour layer, and the late layer 41.

Quantisation check (§4.1): FVE and L0 of each SAE on 200 English sentences under nf4 vs a bf16 reference (bf16 run with
CPU offload, since 8-bit 12B does not fit on 12 GB); nf4 is kept if its FVE is within 10% of bf16's.
Definitions (§4.2), over aligned es-en word pairs of the held-out corpus split (activations summed over subword tokens):
  language-identity features: AUROC (es words vs en words) > 0.95 (or < 0.05), excluded everywhere
  tau_f = sum_i min(a_f(x_i), a_f(y_i)) / sum_i a_f(x_i);  tau*_f = max(tau_f, max_g corr_i(a_f(x_i), a_g(y_i)))
  U_SAE(x) = sum_f a_f(x) (1 - tau*_f) / sum_f a_f(x), over scored non-language features
Link to N1 (§4.3): rho-bar_f = sum_k rho_k^2 c_k^2 / sum_k c_k^2 with c = A^T d_f (decoder direction in the es-en
whitened canonical coordinates at that layer); Spearman(rho-bar, tau*) across features.
Cards (§4.4): persist / vanish / appear features for duende and gusta (father's sentence and its greedy B1 translation),
the goblin-sense contrast sentence, a control, and a Hindi example.
"""
import json
import urllib.request
from collections import defaultdict

import numpy as np
import torch

from internals.cca import Basis
from internals.common import RESULTS, SAE_LAYERS, cache_dir, res_dir, stage
from ut.common import load_model

RELEASE = "gemma-scope-2-12b-it-res"
SAE_ID = "layer_{k}_width_16k_l0_medium"
NP_SOURCE = "{k}-gemmascope-2-res-16k"
NP_MODEL = "gemma-3-12b-it"
MIN_ACTIVE = 20
CONTRAST = ("El duende del cuento se escondió debajo de la cama.", "duende")
CARD_ITEMS = [("es_duende", "duende"), ("es_duende", "gusta"), ("contrast", "duende"), ("es_tren", "tren"),
              ("hi_jugaad", "जुगाड़")]


def load_sae(k, device):
    from sae_lens import SAE
    r = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID.format(k=k), device=str(device))
    sae = r[0] if isinstance(r, tuple) else r
    return sae.eval()


def encode(sae, H):
    with torch.no_grad():
        return sae.encode(H.to(sae.W_dec.dtype))


def choose_layers(m):
    meta = json.loads((cache_dir(m, "n1", "es-en") / "meta.json").read_text())
    avail = SAE_LAYERS[m]
    near = lambda hl: min(avail, key=lambda k: abs((k + 1) - hl))
    ls = [near(meta["peak_layer"])]
    if meta.get("detour_layer"):
        ls.append(near(meta["detour_layer"]))
    ls.append(41)
    return sorted(set(ls))


# ------------------------------------------------------------------ quantisation check
def quant_check(m, layers, sentences):
    from internals.collect import forward_states, prefix
    out = {}
    for mode in ("nf4", "bf16_offload"):
        if mode == "nf4":
            tok, model = load_model(m)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from ut.common import MODELS
            tok = AutoTokenizer.from_pretrained(MODELS[m]["hf_id"])
            model = AutoModelForCausalLM.from_pretrained(MODELS[m]["hf_id"], dtype=torch.bfloat16, device_map="auto",
                                                         max_memory={0: "9GiB", "cpu": "38GiB"}).eval()
        pre = prefix(tok)
        acts = defaultdict(list)
        for s in sentences:
            ids = pre + tok(s, add_special_tokens=False)["input_ids"]
            H, _ = forward_states(model, [ids], [k + 1 for k in layers])
            for k, h in zip(layers, H):
                acts[k].append(h[0, len(pre):].float().cpu())
        del model
        torch.cuda.empty_cache()
        res = {}
        for k in layers:
            sae = load_sae(k, "cuda" if torch.cuda.is_available() else "cpu")
            X = torch.cat(acts[k]).to(sae.W_dec.device)
            with torch.no_grad():
                a = encode(sae, X)
                rec = sae.decode(a).float()
            fve = float(1 - ((X - rec) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
            res[k] = dict(fve=fve, l0=float((a > 0).float().sum(-1).mean()), n_tokens=len(X))
            del sae
        out[mode] = res
        print(f"  quantisation {mode}: " + "  ".join(f"L{k} FVE {v['fve']:.3f} L0 {v['l0']:.0f}" for k, v in res.items()),
              flush=True)
    out["nf4_ok"] = {k: out["nf4"][k]["fve"] >= 0.9 * out["bf16_offload"][k]["fve"] for k in layers}
    return out


# ------------------------------------------------------------------ word-level activations
def word_acts(model, tok, sae_by_layer, texts, word_groups):
    """texts: list of strings; word_groups[i]: list of (word index, token positions) -> {layer: [n_words, F] sparse}."""
    from internals.collect import forward_states
    outs = {k: [] for k in sae_by_layer}
    for c in range(0, len(texts), 16):
        seqs = texts[c:c + 16]
        H, T = forward_states(model, seqs, [k + 1 for k in sae_by_layer])
        for (k, sae), h in zip(sae_by_layer.items(), H):
            for b, groups in enumerate(word_groups[c:c + 16]):
                if not groups:
                    continue
                pos = sorted({p for _, ps in groups for p in ps})
                a = encode(sae, h[b, pos].float())
                idx = {p: i for i, p in enumerate(pos)}
                for _, ps in groups:
                    outs[k].append(a[[idx[p] for p in ps]].sum(0).to_sparse().cpu())
    return {k: torch.stack(v) if v else None for k, v in outs.items()}


def auroc_cols(Xs, Ys):
    """AUROC per feature of es (Xs) vs en (Ys) word activations (dense float arrays [n, F])."""
    n1, n2 = len(Xs), len(Ys)
    allv = np.concatenate([Xs, Ys])
    from scipy.stats import rankdata
    out = np.empty(allv.shape[1])
    for f in range(allv.shape[1]):
        col = allv[:, f]
        if not col.any():
            out[f] = 0.5; continue
        r = rankdata(col)
        out[f] = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return out


def partner(Ax, Ay, device, chunk=1024):
    """For each feature f: max over g of corr(a_f(x), a_g(y)) across aligned pairs, and the argmax g."""
    X = torch.tensor(Ax, device=device, dtype=torch.float32)
    Y = torch.tensor(Ay, device=device, dtype=torch.float32)
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
    Y = (Y - Y.mean(0)) / Y.std(0).clamp_min(1e-6)
    best, arg = torch.full((X.shape[1],), -1.0), torch.zeros(X.shape[1], dtype=torch.long)
    for c in range(0, X.shape[1], chunk):
        C = X[:, c:c + chunk].T @ Y / len(X)
        v, i = C.max(1)
        best[c:c + chunk], arg[c:c + chunk] = v.cpu(), i.cpu()
    return best.numpy(), arg.numpy()


def np_label(k, f):
    """Neuronpedia's explanation (or None) and its top activation snippets, fetched for one feature."""
    url = f"https://www.neuronpedia.org/api/feature/{NP_MODEL}/{NP_SOURCE.format(k=k)}/{int(f)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
    except Exception as e:
        return dict(error=str(e))
    expl = [e.get("description") for e in d.get("explanations", []) if e.get("description")]
    snippets = []
    for a in (d.get("activations") or [])[:5]:
        toks, vals = a.get("tokens", []), a.get("values", [])
        if toks and vals:
            j = int(np.argmax(vals))
            snippets.append("".join(toks[max(0, j - 8):j]) + "[[" + toks[j] + "]]" + "".join(toks[j + 1:j + 4]))
    return dict(explanation=expl[0] if expl else None, top_snippets=snippets, frac_nonzero=d.get("frac_nonzero"))


@stage("sae")
def run(cfg):
    from internals.align import Aligner, words
    from internals.collect import prefix, tokenize_words
    from internals.data import read_corpus
    from internals.align import read_links
    from sent.data import load_sentences
    m = cfg.get("model", "gemma12b")
    layers = cfg.get("layers") or choose_layers(m)
    rd = res_dir(m)
    rows = [r for r in read_corpus("es-en") if r["split"] == "test"][:cfg.get("n_sentences", 2500)]
    qc = quant_check(m, layers, [r["y"] for r in rows[:200]])
    use_offload = not all(qc["nf4_ok"].values())
    if use_offload:
        print("  nf4 degrades the SAEs by more than 10% FVE: using the bf16 offloaded model for N2 (fewer sentences)", flush=True)
        rows = rows[:cfg.get("n_sentences_offload", 600)]
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ut.common import MODELS
        tok = AutoTokenizer.from_pretrained(MODELS[m]["hf_id"])
        model = AutoModelForCausalLM.from_pretrained(MODELS[m]["hf_id"], dtype=torch.bfloat16, device_map="auto",
                                                     max_memory={0: "9GiB", "cpu": "38GiB"}).eval()
    else:
        tok, model = load_model(m)
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    saes = {k: load_sae(k, dev_) for k in layers}
    pre = prefix(tok)
    links = read_links("es-en")
    xt, yt, xg, yg = [], [], [], []
    for r in rows:
        xi, xw = tokenize_words(tok, r["x"], pre); yi, yw = tokenize_words(tok, r["y"], pre)
        ok = [(i, j) for i, j in links.get(r["id"], []) if i < len(xw) and j < len(yw) and xw[i] and yw[j]]
        xt.append(xi); yt.append(yi)
        xg.append([(i, xw[i]) for i, _ in ok]); yg.append([(j, yw[j]) for _, j in ok])
    AX = word_acts(model, tok, saes, xt, xg)
    AY = word_acts(model, tok, saes, yt, yg)
    summary = dict(model=m, layers=layers, sae=dict(release=RELEASE, sae_id=SAE_ID), quantisation=qc,
                   used="bf16 offload" if use_offload else "nf4", n_sentences=len(rows), per_layer={})
    feats_rows = []
    tau_star_by_layer = {}
    langid_by_layer = {}
    for k in layers:
        Ax, Ay = AX[k].to_dense().numpy(), AY[k].to_dense().numpy()
        au = auroc_cols(Ax, Ay)
        langid = (au > 0.95) | (au < 0.05)
        active = (Ax > 0).sum(0)
        tau = np.minimum(Ax, Ay).sum(0) / np.maximum(Ax.sum(0), 1e-9)
        corr, arg = partner(Ax, Ay, dev_)
        tau_star = np.maximum(tau, corr)
        scored = (active >= MIN_ACTIVE) & ~langid
        # rho-bar from the N1 es-en basis at the matching layer
        bp = cache_dir(m, "n1", "es-en") / f"basis_L{k + 1}.npz"
        rho_bar = np.full(len(tau), np.nan)
        if bp.exists():
            B = Basis.load(bp)
            D = saes[k].W_dec.detach().float().to(B.A.device)                 # [F, d]
            c2 = ((D @ B.A) ** 2).cpu()
            rho2 = B.rho_ho.clamp(0, 1) ** 2
            rho_bar = ((c2 * rho2).sum(1) / c2.sum(1)).numpy()
        from scipy.stats import spearmanr
        ok = scored & ~np.isnan(rho_bar)
        sp = spearmanr(rho_bar[ok], tau_star[ok])[0] if ok.sum() > 10 else None
        # the 30 most frequent features with low rho-bar and low tau*
        lo = ok & (rho_bar <= np.nanquantile(rho_bar[ok], 0.25)) & (tau_star <= np.quantile(tau_star[ok], 0.25))
        top30 = np.argsort(-active * lo)[:30]
        top30 = [int(f) for f in top30 if lo[f]]
        labels = {f: np_label(k, f) for f in top30}
        # our own top-10 activating corpus contexts for those features (es side)
        ctx = {}
        word_index = [(ri, i) for ri, g in enumerate(xg) for i, _ in g]
        for f in top30:
            best = np.argsort(-Ax[:, f])[:10]
            ctx[f] = [dict(sentence=rows[word_index[b][0]]["x"], word=words(rows[word_index[b][0]]["x"])[word_index[b][1]],
                           activation=float(Ax[b, f])) for b in best if Ax[b, f] > 0]
        summary["per_layer"][k] = dict(n_features=len(tau), n_language_id=int(langid.sum()), n_scored=int(scored.sum()),
                                       spearman_rhobar_tau_star=sp, n_pairs=len(Ax),
                                       low_low_features=[dict(feature=f, active=int(active[f]), tau_star=float(tau_star[f]),
                                                              rho_bar=float(rho_bar[f]), neuronpedia=labels[f],
                                                              corpus_contexts=ctx[f]) for f in top30])
        tau_star_by_layer[k], langid_by_layer[k] = tau_star, langid
        for f in range(len(tau)):
            if active[f] > 0:
                feats_rows.append(dict(layer=k, feature=f, active=int(active[f]), tau=float(tau[f]), partner_corr=float(corr[f]),
                                       partner=int(arg[f]), tau_star=float(tau_star[f]), rho_bar=float(rho_bar[f]),
                                       langid_auroc=float(au[f]), language_id=bool(langid[f]), scored=bool(scored[f])))
        print(f"  L{k}: {int(langid.sum())} language-id features, {int(scored.sum())} scored, "
              f"Spearman(rho-bar, tau*) = {sp}", flush=True)
    import pandas as pd
    pq = rd / "n2_features.parquet"
    pd.DataFrame(feats_rows).to_parquet(pq)
    # ---- U_SAE for sentences.yaml spans, and the cards
    sents = {s.id: s for s in load_sentences(langs=["es", "hi"])}
    b1p = RESULTS / m / "replication_R2" / "candidates.jsonl"
    b1 = {c["sid"]: c["text"] for c in map(json.loads, open(b1p)) if c["rung"] == "B1" and c["sample"] < 0}
    from ut.common import chat, generate
    from sent.prompts import clean, rung_prompt
    from sent.data import Sentence
    cs = Sentence(id="contrast", lang="es", category="contrast", text=CONTRAST[0], hot=[CONTRAST[1]])
    b1["contrast"] = clean(generate(model, tok, [chat(tok, rung_prompt("B1", cs))], n=1, max_new_tokens=60, temperature=0)[0][0],
                           cs.text)
    sents["contrast"] = cs
    def span_feats(text, s0, e0):
        ids = pre + tok(text, add_special_tokens=False)["input_ids"]
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        pos = [len(pre) + i for i, (a, b) in enumerate(offs) if b > s0 and a < e0 and b > a]
        allpos = list(range(len(pre), len(ids)))
        return ids, pos, allpos
    from internals.collect import forward_states
    def acts_at(ids, pos, k):
        H, _ = forward_states(model, [ids], [k + 1])
        return encode(saes[k], H[0][0, pos].float()).sum(0).cpu().numpy()
    uspans = []
    for sid, s in sents.items():
        if sid == "contrast":
            continue
        for x in s.spans():
            ids, pos, _ = span_feats(s.text, x["start"], x["end"])
            for k in layers:
                a = acts_at(ids, pos, k)
                keep = ~langid_by_layer[k]
                ts = tau_star_by_layer[k]
                u = float((a[keep] * (1 - ts[keep])).sum() / max(a[keep].sum(), 1e-9))
                uspans.append(dict(sid=sid, span=x["text"], kind=x["kind"], category=s.category, layer=k, U_SAE=u))
    from ut.stats import auroc_ci
    for k in layers:
        h = [r["U_SAE"] for r in uspans if r["layer"] == k and r["kind"] == "hot" and sents[r["sid"]].lang == "es"]
        c = [r["U_SAE"] for r in uspans if r["layer"] == k and r["kind"] == "probe" and sents[r["sid"]].lang == "es"]
        hh = [r["U_SAE"] for r in uspans if r["layer"] == k and r["kind"] == "hot"]
        cc = [r["U_SAE"] for r in uspans if r["layer"] == k and r["kind"] == "probe"]
        summary["per_layer"][k]["U_SAE_auroc_hot_vs_probe_es"] = auroc_ci(h, c) if h and c else None
        summary["per_layer"][k]["U_SAE_auroc_hot_vs_probe_all"] = auroc_ci(hh, cc) if hh and cc else None
    al = Aligner()
    cards = []
    lab_cache = {}
    for sid, wd in CARD_ITEMS:
        s = sents.get(sid)
        if s is None or sid not in b1:
            continue
        st = s.text.find(wd)
        ids, pos, _ = span_feats(s.text, st, st + len(wd))
        tr = b1[sid]
        tids = pre + tok(tr, add_special_tokens=False)["input_ids"]
        link = al.align([words(s.text)], [words(tr)])[0]
        widx = next((i for i, (m0, w) in enumerate(zip(s.words(), words(s.text))) if m0[0] == st), None)
        tw = [j for i, j in link if i == widx]
        twords = list(__import__("sent.data", fromlist=["WORD"]).WORD.finditer(tr))
        toffs = tok(tr, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if tw:
            ts0, te0 = twords[tw[0]].start(), twords[tw[0]].end()
            tpos = [len(pre) + i for i, (a, b) in enumerate(toffs) if b > ts0 and a < te0 and b > a]
        else:
            tpos = list(range(len(pre), len(tids)))
        card = dict(sid=sid, sentence=s.text, word=wd, translation=tr, aligned_to=twords[tw[0]].group() if tw else None,
                    layers={})
        for k in layers:
            a = acts_at(ids, pos, k)
            b = acts_at(tids, tpos, k)
            keep = ~langid_by_layer[k]
            persist = [f for f in np.argsort(-np.minimum(a, b)) if keep[f] and a[f] > 0 and b[f] > 0][:8]
            vanish = [f for f in np.argsort(-a) if keep[f] and a[f] > 0 and b[f] == 0][:8]
            appear = [f for f in np.argsort(-b) if keep[f] and b[f] > 0 and a[f] == 0][:8]
            def lab(f):
                if (k, int(f)) not in lab_cache:
                    lab_cache[(k, int(f))] = np_label(k, f)
                return dict(feature=int(f), source_act=float(a[f]), target_act=float(b[f]),
                            tau_star=float(tau_star_by_layer[k][f]), neuronpedia=lab_cache[(k, int(f))])
            card["layers"][k] = dict(persist=[lab(f) for f in persist], vanish=[lab(f) for f in vanish],
                                     appear=[lab(f) for f in appear])
        cards.append(card)
    cj = rd / "n2_cards.json"
    cj.write_text(json.dumps(dict(cards=cards, u_spans=uspans), ensure_ascii=False, default=float))
    sj = rd / "n2_summary.json"
    sj.write_text(json.dumps(summary, ensure_ascii=False, default=float))
    return [pq, cj, sj]
