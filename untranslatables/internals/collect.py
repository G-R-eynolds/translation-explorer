"""N1 collection, GPU (HANDOFF_internals §3.2).

Word states: each sentence is fed raw (one prefix token -- BOS, or a newline for tokenizers without one -- then the
sentence, no prompt); a word's state at layer l is the mean of its subword tokens' hidden_states[l] (l = 0 embeddings,
l >= 1 output of block l-1; Gemma Scope's "layer_k" is block k's output = our layer k+1).  Pairs: SimAlign 1-to-1
links between the two sides of a corpus sentence pair.

  quick       a pass over 800 FIT + 300 TEST sentences at EVERY layer (states kept), PCA-256 CCA per layer with a
              mismatched-pair null -> how many shared directions each layer has; the peak picks a stored layer.
  collect     every FIT pair at the stored layers (every 4th, the quick peaks of both pairs, the R1 detour peak, and
              for gemma12b the Gemma Scope SAE layers): streaming sums of x, y, xx^T, yy^T, xy^T and a mismatched
              xy^T (y rolled by half a buffer) accumulated in float32 on the GPU per 8,192-pair buffer and added into
              float64 totals on the CPU, after subtracting a shift (the first buffer's mean) for numerical safety;
              the same sums for pairs where either word starts its sentence (robustness check (a)).  TEST pairs:
              up to 20,000 stored per layer in fp16, with metadata.
  evalstates  states of the evaluation items at the stored layers: data/sentences.yaml (es, hi; every token),
              the word-level contexts of data/words/es.yaml, FLORES words (es, en, hi, bho) at the hub layers, and the
              round-trip information weights of each sentences.yaml token (null and copy-oracle passes, as in sent/).
"""
import json

import numpy as np
import torch

from internals.common import PAIRS, RESULTS, SAE_LAYERS, cache_dir, res_dir, stage
from sent.data import WORD
from ut.arch import forward_hidden, hidden_size, n_layers
from ut.common import Progress, load_model

FLUSH = 8192
MAX_TEST_PAIRS = 20000
MAX_TOK = 256


# ------------------------------------------------------------------ tokens and word states
def prefix(tok):
    from sent.score import prefix_ids
    return prefix_ids(tok)


def tokenize_words(tok, text, pre):
    """ids (prefix + sentence) and, per WORD match, the positions of its subword tokens."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = (pre + enc["input_ids"])[:MAX_TOK]
    w2t = []
    for m in WORD.finditer(text):
        ws, we = m.start(), m.end()
        pos = [len(pre) + k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > ws and a < we and b > a]
        w2t.append([p for p in pos if p < MAX_TOK])
    return ids, w2t


def forward_states(model, seqs, layers):
    """[len(layers)] tensors [B, T, d] (model dtype) for right-padded seqs."""
    T = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), T), dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
    with torch.no_grad():
        hs = forward_hidden(model, input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                            output_hidden_states=True).hidden_states
    return [hs[l] for l in layers], T


def gather_means(H, T, groups):
    """groups: list of (batch row, token positions) -> [len(groups), d] float32 means."""
    flat, seg, w = [], [], []
    for g, (b, pos) in enumerate(groups):
        for p in pos:
            flat.append(b * T + p); seg.append(g); w.append(1.0 / len(pos))
    dev = H.device
    flat, seg, w = torch.tensor(flat, device=dev), torch.tensor(seg, device=dev), torch.tensor(w, device=dev)
    out = torch.zeros((len(groups), H.shape[-1]), device=dev)
    return out.index_add_(0, seg, H.reshape(-1, H.shape[-1])[flat].float() * w[:, None])


def pair_batches(rows, links, tok, pre, bs):
    """Yield (x seqs, y seqs, pairs) per batch; pairs = [(b, x positions, y positions, meta)]."""
    for c in range(0, len(rows), bs):
        chunk = rows[c:c + bs]
        xs, ys, pairs = [], [], []
        for b, r in enumerate(chunk):
            xi, xw = tokenize_words(tok, r["x"], pre)
            yi, yw = tokenize_words(tok, r["y"], pre)
            xs.append(xi); ys.append(yi)
            xm, ym = list(WORD.finditer(r["x"])), list(WORD.finditer(r["y"]))
            for i, j in links.get(r["id"], []):
                if i < len(xw) and j < len(yw) and xw[i] and yw[j]:
                    pairs.append((b, xw[i], yw[j], dict(sid=r["id"], i=i, j=j, xw=xm[i].group(), yw=ym[j].group(),
                                                        xs=xm[i].start(), xe=xm[i].end(), first=i == 0 or j == 0)))
        yield xs, ys, pairs


def batch_pair_states(model, xs, ys, pairs, layers):
    HX, TX = forward_states(model, xs, layers)
    HY, TY = forward_states(model, ys, layers)
    X = [gather_means(h, TX, [(b, px) for b, px, _, _ in pairs]) for h in HX]
    Y = [gather_means(h, TY, [(b, py) for b, _, py, _ in pairs]) for h in HY]
    return X, Y


# ------------------------------------------------------------------ streaming accumulator
class Acc:
    KEYS = ("xx", "yy", "xy", "xyn")

    def __init__(self, layers, d, device, first_layers=None):
        self.layers, self.d, self.device = layers, d, device
        self.first_layers = set(layers if first_layers is None else first_layers)
        z = lambda: torch.zeros((d, d), dtype=torch.float64)
        v = lambda: torch.zeros(d, dtype=torch.float64)
        self.A = {l: dict(**{k: z() for k in self.KEYS}, sx=v(), sy=v(), syn=v(), n=torch.tensor(0.0)) for l in layers}
        # first-word sums (robustness check (a)) only where they are used: the key layers
        self.F = {l: dict(xx=z(), yy=z(), xy=z(), sx=v(), sy=v(), n=torch.tensor(0.0)) for l in layers if l in self.first_layers}
        self.c = {}
        self.buf = {l: ([], []) for l in layers}
        self.fbuf = []

    def add(self, X, Y, first):
        for l, x, y in zip(self.layers, X, Y):
            self.buf[l][0].append(x.half().cpu()); self.buf[l][1].append(y.half().cpu())
        self.fbuf.append(torch.tensor(first, dtype=torch.bool))
        if sum(len(t) for t in self.buf[self.layers[0]][0]) >= FLUSH:
            self.flush()

    def flush(self):
        if not self.buf[self.layers[0]][0]:
            return
        first = torch.cat(self.fbuf).to(self.device)
        for l in self.layers:
            X = torch.cat(self.buf[l][0]).to(self.device).float()
            Y = torch.cat(self.buf[l][1]).to(self.device).float()
            if l not in self.c:
                self.c[l] = (X.mean(0).double().cpu(), Y.mean(0).double().cpu())
            X -= self.c[l][0].float().to(self.device); Y -= self.c[l][1].float().to(self.device)
            Yn = torch.roll(Y, len(Y) // 2, 0)
            a = self.A[l]
            for k, (P, Q) in dict(xx=(X, X), yy=(Y, Y), xy=(X, Y), xyn=(X, Yn)).items():
                a[k] += (P.T @ Q).double().cpu()
            a["sx"] += X.sum(0).double().cpu(); a["sy"] += Y.sum(0).double().cpu(); a["syn"] += Yn.sum(0).double().cpu()
            a["n"] += len(X)
            if first.any() and l in self.F:
                f, Xf, Yf = self.F[l], X[first], Y[first]
                for k, (P, Q) in dict(xx=(Xf, Xf), yy=(Yf, Yf), xy=(Xf, Yf)).items():
                    f[k] += (P.T @ Q).double().cpu()
                f["sx"] += Xf.sum(0).double().cpu(); f["sy"] += Yf.sum(0).double().cpu(); f["n"] += len(Xf)
            self.buf[l] = ([], [])
        self.fbuf = []

    def save(self, d):
        self.flush()
        for l in self.layers:
            cx, cy = self.c[l]
            torch.save(dict(self.A[l], cx=cx, cy=cy), d / f"acc_L{l}.pt")
            if l in self.F:
                torch.save(self.F[l], d / f"accfirst_L{l}.pt")


# ------------------------------------------------------------------ layer choice
def detour_peak(model_key):
    p = RESULTS / model_key / "replication_R1.json"
    if not p.exists():
        return None
    peaks = [v["peak_layer"] for v in json.loads(p.read_text())["pairs"].values()]
    return int(np.median(peaks))


def stored_layers(model_key, L):
    lay = set(range(4, L + 1, 4)) | {L}
    for pair in PAIRS:
        q = RESULTS / model_key / f"n1_quick_{pair}.json"
        if q.exists():
            lay.add(json.loads(q.read_text())["peak_layer"])
    dp = detour_peak(model_key)
    if dp:
        lay.add(dp)
    for k in SAE_LAYERS.get(model_key, []):
        lay.add(k + 1)
    return sorted(l for l in lay if 1 <= l <= L)


def key_layers(model_key, L, pair):
    k = set(SAE_LAYERS.get(model_key, []))
    k = {x + 1 for x in k}
    q = RESULTS / model_key / f"n1_quick_{pair}.json"
    if q.exists():
        k.add(json.loads(q.read_text())["peak_layer"])
    dp = detour_peak(model_key)
    if dp:
        k.add(dp)
    return sorted(k)


def peak_layer(model_key, pair):
    q = RESULTS / model_key / f"n1_quick_{pair}.json"
    return json.loads(q.read_text())["peak_layer"] if q.exists() else None


# ------------------------------------------------------------------ stages
def _rows_links(pair):
    from internals.align import read_links
    from internals.data import read_corpus
    return read_corpus(pair), read_links(pair)


@stage("quick")
def quick(cfg):
    """All layers on a subset; PCA-256 CCA per layer; peak = layer with most shared directions."""
    from internals.cca import TOP, eig, fit, heldout_rho
    m, pair = cfg["model"], cfg["pair"]
    rows, links = _rows_links(pair)
    fit_rows = [r for r in rows if r["split"] == "fit"][:cfg.get("n_fit", 800)]
    test_rows = [r for r in rows if r["split"] == "test"][:cfg.get("n_test", 300)]
    tok, model = load_model(m)
    L, pre = n_layers(model), prefix(tok)
    layers = list(range(L + 1))
    store = {}
    for split, rr in (("fit", fit_rows), ("test", test_rows)):
        Xs, Ys = [[] for _ in layers], [[] for _ in layers]
        prog = Progress(f"quick {pair} {split}", len(rr))
        for k, (xs, ys, pairs) in enumerate(pair_batches(rr, links, tok, pre, 16)):
            if not pairs:
                continue
            X, Y = batch_pair_states(model, xs, ys, pairs, layers)
            for l in layers:
                Xs[l].append(X[l].half().cpu()); Ys[l].append(Y[l].half().cpu())
            if k % 40 == 0:
                prog.tick(min((k + 1) * 16, len(rr)))
        store[split] = ([torch.cat(x) for x in Xs], [torch.cat(y) for y in Ys])
    del model
    torch.cuda.empty_cache()
    res = {}
    for l in layers[1:]:
        Xf, Yf = store["fit"][0][l].float(), store["fit"][1][l].float()
        Xt, Yt = store["test"][0][l].float(), store["test"][1][l].float()
        mx, my = Xf.mean(0), Yf.mean(0)
        Xf, Yf, Xt, Yt = Xf - mx, Yf - my, Xt - mx, Yt - my
        Px = eig(Xf.T.double() @ Xf.double() / len(Xf))[1][:, -min(256, Xf.shape[1]):].float()
        Py = eig(Yf.T.double() @ Yf.double() / len(Yf))[1][:, -min(256, Yf.shape[1]):].float()
        a, b = Xf @ Px, Yf @ Py
        Sxx, Syy, Sxy = a.T @ a / len(a), b.T @ b / len(b), a.T @ b / len(a)
        ex, ey = eig(Sxx), eig(Syy)
        f = fit(ex, ey, Sxy, 1e-2)
        rho = heldout_rho(f["A"], f["B"], Xt @ Px, Yt @ Py)
        bn = torch.roll(b, len(b) // 2, 0)
        fn = fit(ex, ey, a.T @ bn / len(a), 1e-2)
        rn = heldout_rho(fn["A"], fn["B"], Xt @ Px, torch.roll(Yt @ Py, len(Yt) // 2, 0))
        thr = float(np.quantile(rn.numpy(), 0.99))
        res[l] = dict(n_shared=int((rho > thr).sum()), sum_top50=float(rho[:TOP].sum()), null_q99=thr)
        print(f"  quick {pair} L{l:>2}: shared {res[l]['n_shared']:>3}/256  top50 {res[l]['sum_top50']:.2f}", flush=True)
    peak = max(res, key=lambda l: (res[l]["n_shared"], res[l]["sum_top50"]))
    out = res_dir(m) / f"n1_quick_{pair}.json"
    out.write_text(json.dumps(dict(model=m, pair=pair, pca=256, n_fit_pairs=len(store["fit"][0][0]),
                                   n_test_pairs=len(store["test"][0][0]), layers=res, peak_layer=peak)))
    print(f"  peak layer {peak}", flush=True)
    return [out]


@stage("collect")
def collect(cfg):
    m, pair = cfg["model"], cfg["pair"]
    rows, links = _rows_links(pair)
    tok, model = load_model(m)
    L, d, pre = n_layers(model), hidden_size(model), prefix(tok)
    layers = cfg.get("layers") or stored_layers(m, L)
    print(f"  stored layers: {layers}", flush=True)
    cd = cache_dir(m, "n1", pair)
    acc = Acc(layers, d, model.device, first_layers=key_layers(m, L, pair))
    fit_rows = [r for r in rows if r["split"] == "fit"][:cfg.get("n_fit", 10 ** 9)]
    test_rows = [r for r in rows if r["split"] == "test"]
    bs = cfg.get("batch", 32)
    prog = Progress(f"collect {pair} fit", len(fit_rows))
    n_pairs = n_words = 0
    for k, (xs, ys, pairs) in enumerate(pair_batches(fit_rows, links, tok, pre, bs)):
        n_words += sum(len(WORD.findall(r["x"])) for r in fit_rows[k * bs:(k + 1) * bs])
        if not pairs:
            continue
        X, Y = batch_pair_states(model, xs, ys, pairs, layers)
        acc.add(X, Y, [p[3]["first"] for p in pairs])
        n_pairs += len(pairs)
        if k % 100 == 0:
            prog.tick(min((k + 1) * bs, len(fit_rows)))
    acc.save(cd)
    del acc
    # held-out pairs
    tx = {l: np.lib.format.open_memmap(cd / f"test_x_L{l}.npy", "w+", np.float16, (MAX_TEST_PAIRS, d)) for l in layers}
    ty = {l: np.lib.format.open_memmap(cd / f"test_y_L{l}.npy", "w+", np.float16, (MAX_TEST_PAIRS, d)) for l in layers}
    meta, n = [], 0
    for xs, ys, pairs in pair_batches(test_rows, links, tok, pre, bs):
        if not pairs or n >= MAX_TEST_PAIRS:
            continue
        pairs = pairs[:MAX_TEST_PAIRS - n]
        X, Y = batch_pair_states(model, xs, ys, pairs, layers)
        for l, x, y in zip(layers, X, Y):
            tx[l][n:n + len(pairs)] = x.half().cpu().numpy(); ty[l][n:n + len(pairs)] = y.half().cpu().numpy()
        meta += [p[3] for p in pairs]
        n += len(pairs)
    for l in layers:                                    # trim to the pairs actually stored
        for nm, mm in (("x", tx[l]), ("y", ty[l])):
            mm.flush()
            arr = np.load(cd / f"test_{nm}_L{l}.npy", mmap_mode="r")[:n].copy()
            np.save(cd / f"test_{nm}_L{l}.npy", arr)
    info = dict(model=m, pair=pair, layers=layers, key_layers=key_layers(m, L, pair), peak_layer=peak_layer(m, pair),
                detour_layer=detour_peak(m), n_fit_sentences=len(fit_rows), n_fit_pairs=n_pairs,
                aligned_share_fit=n_pairs / max(n_words, 1), n_test_pairs=n, test_pairs=meta, d=d, n_layers=L)
    (cd / "meta.json").write_text(json.dumps(info, ensure_ascii=False))
    return [cd / "meta.json"]


@stage("evalstates")
def eval_states(cfg):
    """States of every evaluation item at the stored layers (see module docstring)."""
    from internals.data import read_flores
    from sent.data import load_sentences
    from sent.score import run_rt
    from ut.represent import unmark
    from ut.common import load_yaml, DATA
    m = cfg["model"]
    langs = cfg.get("langs", ["es", "hi"])
    tok, model = load_model(m)
    L, d, pre = n_layers(model), hidden_size(model), prefix(tok)
    layers = cfg.get("layers") or stored_layers(m, L)
    cd = cache_dir(m, "n1", "eval")
    outs = []
    # (1) sentences.yaml: every token, all stored layers, plus round-trip information weights
    sents = [s for s in load_sentences(langs=langs)]
    rt = run_rt(model, tok, sents, [], batch=8)
    toks, index = [], {}
    n = 0
    for s in sents:
        enc = tok(s.text, add_special_tokens=False, return_offsets_mapping=True)
        H, _ = forward_states(model, [pre + enc["input_ids"]], layers)
        toks.append(torch.stack([h[0, len(pre):] for h in H]).half().cpu())
        T = len(enc["input_ids"])
        index[s.id] = dict(start=n, end=n + T, offsets=enc["offset_mapping"], lang=s.lang,
                           lp_null=rt["sentences"][s.id]["lp_null"], lp_oracle=rt["sentences"][s.id]["lp_oracle"])
        n += T
    np.save(cd / "sent_tokens.npy", torch.cat(toks, 1).numpy())
    (cd / "sent_index.json").write_text(json.dumps(dict(layers=layers, sentences=index), ensure_ascii=False))
    outs += [cd / "sent_tokens.npy", cd / "sent_index.json"]
    # (2) word-level contexts (data/words/es.yaml): the marked word's mean state
    if "es" in langs:
        items, groups_text = [], []
        for w in load_yaml(DATA / "words" / "es.yaml")["words"]:
            for k, c in enumerate(w.get("contexts", [])):
                text, span = unmark(c)
                items.append(dict(word=w["word"], category=w["category"], k=k, text=text, span=span))
        S = []
        for c in range(0, len(items), 16):
            chunk = items[c:c + 16]
            seqs, groups = [], []
            for b, it in enumerate(chunk):
                enc = tok(it["text"], add_special_tokens=False, return_offsets_mapping=True)
                seqs.append(pre + enc["input_ids"])
                s0, e0 = it["span"]
                groups.append((b, [len(pre) + k for k, (a, bb) in enumerate(enc["offset_mapping"]) if bb > s0 and a < e0 and bb > a]))
            H, T = forward_states(model, seqs, layers)
            S.append(torch.stack([gather_means(h, T, groups) for h in H], 1).half().cpu())
        np.save(cd / "words_es.npy", torch.cat(S).numpy())                       # [n_items, n_layers, d]
        (cd / "words_es.json").write_text(json.dumps(dict(layers=layers, items=items), ensure_ascii=False))
        outs += [cd / "words_es.npy"]
    # (3) FLORES words at the hub layers (peaks of both pairs + detour), all four languages
    hub = sorted({x for x in (peak_layer(m, "es-en"), peak_layer(m, "hi-en"), detour_peak(m)) if x} & set(layers))
    if cfg.get("flores", True) and hub:
        rows = read_flores()
        for lg in cfg.get("flores_langs", ["es", "en", "hi", "bho"]):
            S, idx = [], []
            for c in range(0, len(rows), 32):
                chunk = rows[c:c + 32]
                seqs, groups = [], []
                for b, r in enumerate(chunk):
                    ids, w2t = tokenize_words(tok, r[lg], pre)
                    seqs.append(ids)
                    for wi, pos in enumerate(w2t):
                        if pos:
                            groups.append((b, pos)); idx.append((r["id"], wi))
                H, T = forward_states(model, seqs, hub)
                S.append(torch.stack([gather_means(h, T, groups) for h in H], 0).half().cpu())
            np.save(cd / f"flores_{lg}.npy", torch.cat(S, 1).numpy())               # [n_hub, n_words, d]
            (cd / f"flores_{lg}.json").write_text(json.dumps(dict(layers=hub, index=idx)))
            outs.append(cd / f"flores_{lg}.npy")
            print(f"  flores {lg}: {len(idx)} words", flush=True)
    return outs
