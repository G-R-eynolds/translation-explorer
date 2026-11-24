# Reload earlier stages' results, so this stage also works in a fresh kernel (after 00_setup.py).
A = load_json("A_detour.json") if have("A_detour.json") else None
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
summary = load_json("B_summary.json")["per_sentence"] if have("B_summary.json") else {}


def pick_layers():
    f = [0.25, 0.4, 0.5, 0.6, 0.75]
    ls = {max(1, min(N_LAYERS - 1, round(x * N_LAYERS))) for x in f}
    if A and A.get("detour_peak_layer"): ls.add(int(min(A["detour_peak_layer"], N_LAYERS - 1)))
    return sorted(ls)
C_LAYERS = pick_layers(); print("Part C layers:", C_LAYERS)

def load_pairs(lang, n):
    """n parallel sentence pairs (x, en, source); a mix of domains so 'conventional practice' isn't one register."""
    from datasets import load_dataset
    # Europarl first: if it fails to load, OPUS-100 takes the whole quota
    sources = ([("Helsinki-NLP/europarl", "en-es")] if lang == "es" else []) + [("Helsinki-NLP/opus-100", f"en-{lang}")]
    want, pairs, seen = int(n * 1.3), [], set()
    for k, (name, cfg) in enumerate(sources):
        quota = (want - len(pairs)) // (len(sources) - k)
        try: ds = load_dataset(name, cfg, split="train", streaming=True)
        except Exception as e: print(f"  could not load {name}/{cfg}: {e}"); continue
        got = 0
        for ex in ds:
            tr = ex.get("translation", {}); en, xx = tr.get("en"), tr.get(lang)
            if not en or not xx: continue
            ne, nx = len(en.split()), len(xx.split())
            if not (8 <= ne <= 40 and 8 <= nx <= 40) or abs(ne - nx) > 0.5 * max(ne, nx): continue
            if sum(ch.isdigit() for ch in xx) > 6 or xx in seen: continue
            seen.add(xx); pairs.append((xx.strip(), en.strip(), name)); got += 1
            if got >= quota: break
        print(f"  {name}/{cfg}: {got} pairs")
    random.Random(SEED).shuffle(pairs)
    return pairs[:n]

def words_for(lang, texts):
    """Word tokens (text, start, end, morph, pos). Spanish uses spaCy when available, for morphology."""
    if lang == "es":
        try:
            import spacy
            nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            out = []
            for doc in nlp.pipe(texts, batch_size=256):
                out.append([(t.text, t.idx, t.idx + len(t.text), str(t.morph), t.pos_) for t in doc
                            if WORD_RE.fullmatch(t.text)])
            return out
        except Exception as e:
            print("  spaCy unavailable, using regex words (no morphology probes):", e)
    return [[(w, a, b, "", "") for w, a, b in words_with_spans(t)] for t in texts]

_AL = None
def align_pairs(src_words, tgt_words, batch=64, layer=8):
    """1-to-1 word links (i, j) per sentence pair: mutual nearest neighbours of XLM-R (layer 8) word embeddings,
    i.e. SimAlign's high-precision 'argmax' method, implemented here to avoid a fragile dependency."""
    global _AL
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _AL is None:
        _AL = (AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base"),
               AutoModel.from_pretrained("FacebookAI/xlm-roberta-base").to(dev).eval())
    atok, amod = _AL
    def embed(word_lists):
        wl_ = [[w[0] for w in wl] or ["."] for wl in word_lists]
        e = atok(wl_, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            H = amod(**{k: v.to(dev) for k, v in e.items()}, output_hidden_states=True).hidden_states[layer]
        out = []
        for b, wl in enumerate(wl_):
            M = torch.zeros(len(wl), H.shape[1], device=dev)
            for p, w in enumerate(e.word_ids(b)):
                if w is not None: M[w, p] = 1.0
            cnt = M.sum(1, keepdim=True)
            out.append((torch.nn.functional.normalize((M @ H[b].float()) / cnt.clamp_min(1), dim=-1), cnt[:, 0] > 0))
        return out
    links = []
    for i in range(0, len(src_words), batch):
        es, et = embed(src_words[i:i + batch]), embed(tgt_words[i:i + batch])
        for (a, va), (b, vb), sw, tw in zip(es, et, src_words[i:i + batch], tgt_words[i:i + batch]):
            if not sw or not tw: links.append([]); continue
            S = a @ b.T; S[~va] = -2; S[:, ~vb] = -2
            fwd, bwd = S.argmax(1), S.argmax(0)
            links.append([(k, int(fwd[k])) for k in range(len(sw)) if int(bwd[fwd[k]]) == k and float(S[k, fwd[k]]) > -1
                          and not any(ch.isdigit() for ch in sw[k][0] + tw[int(fwd[k])][0])])
    return links

# ---- collection ----

class CovAcc:
    def __init__(self, d, dev):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.n, self.sx, self.sy = 0, z(d), z(d)
        self.Sxx, self.Syy, self.Sxy = z(d, d), z(d, d), z(d, d)
    def add(self, X, Y):
        X, Y = X.double(), Y.double()
        self.n += X.shape[0]; self.sx += X.sum(0); self.sy += Y.sum(0)
        self.Sxx += X.T @ X; self.Syy += Y.T @ Y; self.Sxy += X.T @ Y
    def finish(self):
        n = self.n; mx, my = self.sx / n, self.sy / n
        return dict(n=n, mx=mx.cpu(), my=my.cpu(),
                    Cxx=((self.Sxx - n * torch.outer(mx, mx)) / (n - 1)).cpu(),
                    Cyy=((self.Syy - n * torch.outer(my, my)) / (n - 1)).cpu(),
                    Cxy=((self.Sxy - n * torch.outer(mx, my)) / (n - 1)).cpu())

def word_states(texts, word_lists, layers, batch=32):
    """For each text: dict layer -> [n_words, d] float32 (word = mean of its tokens), plus a validity mask
    (words containing the first content token, an attention sink, are invalid)."""
    out = []
    for i in range(0, len(texts), batch):
        enc_ = [encode_raw(t) for t in texts[i:i + batch]]
        L = max(len(e[0]) for e in enc_)
        ids = torch.full((len(enc_), L), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
        for b, (s, _) in enumerate(enc_): ids[b, :len(s)] = torch.tensor(s); am[b, :len(s)] = 1
        st = capture(ids.to(INPUT_DEV), am.to(INPUT_DEV), layers)
        for b, (s, offs) in enumerate(enc_):
            wl = word_lists[i + b]; first = len(BOS)
            toks = [tokens_in_span(offs, w[1], w[2]) for w in wl]
            valid = np.array([len(t) > 0 and first not in t for t in toks], bool)
            M = torch.zeros(len(wl), len(s))
            for k, t in enumerate(toks):
                if t: M[k, t] = 1.0 / len(t)
            M = M.to(st[layers[0]].device)
            out.append(({l: M @ st[l][b, :len(s)].float() for l in layers}, valid))
        del st
    return out

def collect_pair(lang):
    tag = f"C_{lang}"
    cov_paths = {l: os.path.join(CACHE, f"{tag}_cov_L{l}.pt") for l in C_LAYERS}
    held_path = os.path.join(CACHE, f"{tag}_held.pt")
    if all(os.path.exists(p) for p in cov_paths.values()) and os.path.exists(held_path): return
    with Timer(f"Part C [{lang}]: corpus"):
        pairs = load_pairs(lang, N_SENT_FIT + N_SENT_HELD)
        xs, ens = [p[0] for p in pairs], [p[1] for p in pairs]
        wx, we = words_for(lang, xs), words_for("en", ens)
    with Timer(f"Part C [{lang}]: word alignment ({len(pairs)} pairs)"):
        links = align_pairs(wx, we)
    n_fit = len(pairs) - N_SENT_HELD
    acc = {l: CovAcc(D_MODEL, INPUT_DEV) for l in C_LAYERS}
    held = {l: ([], []) for l in C_LAYERS}; held_meta = []
    with Timer(f"Part C [{lang}]: collecting states at layers {C_LAYERS}"):
        B_ = 32
        for i in range(0, len(pairs), B_):
            sx = word_states(xs[i:i + B_], wx[i:i + B_], C_LAYERS)
            sy = word_states(ens[i:i + B_], we[i:i + B_], C_LAYERS)
            for b in range(len(sx)):
                k = i + b; (rx, vx), (ry, vy) = sx[b], sy[b]
                lk = [(a, c) for a, c in links[k] if vx[a] and vy[c]]
                if not lk: continue
                ia = torch.tensor([a for a, _ in lk]); ic = torch.tensor([c for _, c in lk])
                for l in C_LAYERS:
                    X, Y = rx[l][ia.to(rx[l].device)], ry[l][ic.to(ry[l].device)]
                    if k < n_fit: acc[l].add(X, Y)
                    else: held[l][0].append(X.half().cpu()); held[l][1].append(Y.half().cpu())
                if k >= n_fit:
                    held_meta += [dict(sent=k, x=wx[k][a][0], y=we[k][c][0], morph=wx[k][a][3], pos=wx[k][a][4]) for a, c in lk]
            if (i // B_) % 100 == 0: print(f"    {i}/{len(pairs)} sentences, {acc[C_LAYERS[0]].n} fit pairs")
    for l in C_LAYERS: torch.save(acc[l].finish(), cov_paths[l])
    torch.save(dict(X={l: torch.cat(held[l][0]) for l in C_LAYERS}, Y={l: torch.cat(held[l][1]) for l in C_LAYERS},
                    meta=held_meta, n_fit_pairs=acc[C_LAYERS[0]].n, n_sent=len(pairs)), held_path)
    del acc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

if RUN_C:
    for lang in PAIRS: collect_pair(lang)

pack()
