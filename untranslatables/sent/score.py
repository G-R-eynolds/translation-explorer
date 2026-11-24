"""Step 2 (GPU): raw scores for every candidate in both meaning spaces (handoff §2).  The CPU analysis
(sent.analyze) turns them into leftovers, curves and validation, so tau, delta and the layer can be changed
without re-running the model.

Rows scored, per sentence:
  cand      every ladder candidate
  masked    the candidate with source-language words (rare in English) replaced by an ellipsis, if it had any
            (the copying / spelling check of §2.1 and §2.2)
  mismatch  controls only: the source against every OTHER control's greedy B1 translation (layer choice, §6.1)
  spell     controls only: a nonsense string sharing the source words' first syllables (spelling control, §6.1)
Round trip: log p(s_i | s_<i, back-translation prompt with t) for every source token, plus the empty-text
baseline and the copy oracle per sentence.  Concept layer: at EVERY layer, lse_i = logsumexp_j(-C_ij/eps_ot)
and the softmax-weighted cost, for each source token i against all target tokens j (see sent.measure).

    python -m sent.score --model 3b                       # RT with the 3B as back-translator, plus CL
    python -m sent.score --model 3b --bt qwen7b --rt-only  # second back-translator on the 3B's candidates (§2.2)
Outputs: results/sent/<model>/raw/{rows.jsonl, rt_<bt>.json, cl_<model>.npz + .json, centroids_<model>.npz}
"""
import argparse
import re

import numpy as np

from ut.arch import forward_hidden, logits as arch_logits
from ut.common import MODELS, Progress, Timer, chat, describe, load_json, load_model, n_layers, save_json
from sent.data import CENTROID_LANG, WORD, centring_texts, fold, load_sentences, sent_dir
from sent.ladder import read_jsonl, write_jsonl
from sent.measure import EPS_OT
from sent.prompts import backtranslate_prompt, copy_prompt

MAX_T_TOKENS = 448
SUFFIXES = ["ton", "ly", "er", "ment", "ing", "ble", "ish", "ard"]
VOWELS = "aeiouyáéíóúàèìòùâêîôûäöüãõ"


# ------------------------------------------------------------------ rows
def spelling_control(text):
    """'El tren sale a las ocho' -> 'elton trenly saler ament lasing ...': each word's first syllable
    (onset + vowels + one consonant) plus an English-looking suffix.  Shares spelling, not meaning."""
    out = []
    for k, m in enumerate(WORD.finditer(text)):
        w = m.group().lower()
        syl = re.match(rf"[^{VOWELS}]*[{VOWELS}]+[^{VOWELS}]?", w)
        out.append((syl.group() if syl else w[:3]) + SUFFIXES[k % len(SUFFIXES)])
    return " ".join(out).capitalize() + "."


def mask_source_words(english, sent, max_en_zipf=3.0):
    """Replace words of `english` that are also words of the source (accent-folded, >= 3 letters) and are rare
    in English (wordfreq Zipf < 3) with '…'.  'duende', 'saudade', 'Gemütlichkeit' are masked; 'hospital',
    'Tokyo' and 'chat' are not.  Returns the masked text (unchanged text = nothing to mask)."""
    from wordfreq import zipf_frequency
    src = {fold(w) for _, _, w in sent.words() if len(w) >= 3}
    def sub(m):
        w = m.group()
        return "…" if fold(w) in src and zipf_frequency(w.lower(), "en") < max_en_zipf else w
    return WORD.sub(sub, english)


PLACEHOLDER = "[·]"


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mask_cognates(english, sent, max_dist=0.3, min_len=4):
    """The internals run's masking (HANDOFF_internals §2 R2.1): replace every word of `english` that is a source
    word (accent-folded, >= 3 letters, not an English stopword) or a close cognate of one (normalised edit distance
    <= 0.3 to a source word of >= 4 letters) with the neutral placeholder '[·]'."""
    from sent.data import stopwords
    en_sw = stopwords("en")
    src = [fold(w) for _, _, w in sent.words() if len(w) >= 3]
    def sub(m):
        f = fold(m.group())
        if len(f) < 3 or f in en_sw:
            return m.group()
        for w in src:
            if f == w or (len(w) >= min_len and _edit_distance(f, w) / max(len(f), len(w)) <= max_dist):
                return PLACEHOLDER
        return m.group()
    return WORD.sub(sub, english)


def build_rows(cands, sents, mask_rule="zipf", cl_rows=True):
    by = {s.id: s for s in sents}
    rows = []
    for c in cands:
        if c["sid"] not in by:
            continue
        rows.append(dict(rid=c["cid"], sid=c["sid"], role="cand", parent=None, text=c["text"]))
        m = (mask_cognates if mask_rule == "cognate" else mask_source_words)(c["text"], by[c["sid"]])
        if m != c["text"]:
            rows.append(dict(rid=c["cid"] + "|masked", sid=c["sid"], role="masked", parent=c["cid"], text=m))
    if not cl_rows:
        return rows
    ctrl = [s for s in sents if s.is_control]
    b1 = {c["sid"]: c["text"] for c in cands if c["rung"] == "B1" and c["sample"] < 0}
    for s in ctrl:
        for o in ctrl:
            if o.id != s.id and o.id in b1:
                rows.append(dict(rid=f"{s.id}|mismatch|{o.id}", sid=s.id, role="mismatch", parent=o.id, text=b1[o.id]))
        rows.append(dict(rid=f"{s.id}|spell", sid=s.id, role="spell", parent=None, text=spelling_control(s.text)))
    return rows


# ------------------------------------------------------------------ tokens
def prefix_ids(tok):
    """One token in front of every text whose state is discarded (the attention sink): BOS, or a newline for
    tokenizers without one (Qwen)."""
    return [tok.bos_token_id] if tok.bos_token_id is not None else tok("\n", add_special_tokens=False)["input_ids"]


def source_tokens(tok, sent):
    enc = tok(sent.text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]


def _pad(seqs, pad_id):
    import torch
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
    return ids, att


# ------------------------------------------------------------------ round trip
def rt_logprobs(model, tok, jobs, batch=32, label="round trip"):
    """jobs: [(prefix_ids, target_ids)] -> [log p of each target token given prefix + earlier targets]."""
    import torch
    out = [None] * len(jobs)
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][0]) + len(jobs[k][1]))
    prog = Progress(label, len(jobs))
    for c in range(0, len(order), batch):
        ks = order[c:c + batch]
        ids, att = _pad([jobs[k][0] + jobs[k][1] for k in ks], tok.pad_token_id)
        with torch.no_grad():
            h = forward_hidden(model, input_ids=ids.to(model.device), attention_mask=att.to(model.device)).last_hidden_state
            for b, k in enumerate(ks):
                P, n = len(jobs[k][0]), len(jobs[k][1])
                lg = arch_logits(model, h[b, P - 1:P - 1 + n])
                tgt = torch.tensor(jobs[k][1], device=lg.device)[:, None]
                out[k] = torch.log_softmax(lg, -1).gather(1, tgt)[:, 0].cpu().numpy()
        if (c // batch) % 10 == 0 or c + batch >= len(order):
            prog.tick(min(c + batch, len(order)))
    return out


def run_rt(model, tok, sents, rows, batch=32):
    def assistant_prefix(user):
        return tok(chat(tok, user), add_special_tokens=False)["input_ids"]
    src = {s.id: source_tokens(tok, s) for s in sents}
    by = {s.id: s for s in sents}
    jobs, keys = [], []
    for s in sents:
        ids = src[s.id][0]
        jobs += [(assistant_prefix(backtranslate_prompt(s.lang, "")), ids),
                 (assistant_prefix(copy_prompt(s.lang, s.text)), ids)]
        keys += [("null", s.id), ("oracle", s.id)]
    for r in rows:
        jobs.append((assistant_prefix(backtranslate_prompt(by[r["sid"]].lang, r["text"])), src[r["sid"]][0]))
        keys.append(("row", r["rid"]))
    lps = rt_logprobs(model, tok, jobs, batch=batch)
    sent_out = {s.id: dict(tokens=[tok.decode([i]) for i in src[s.id][0]], ids=src[s.id][0], offsets=src[s.id][1])
                for s in sents}
    rows_out = {}
    for (kind, key), lp in zip(keys, lps):
        if kind == "row":
            rows_out[key] = lp.round(4).tolist()
        else:
            sent_out[key]["lp_" + kind] = lp.round(4).tolist()
    return dict(sentences=sent_out, rows=rows_out)


# ------------------------------------------------------------------ concept layer
def hidden_states(model, seqs):
    """seqs: token-id lists -> tensor [L+1, B, T, d] (model dtype) and lengths."""
    import torch
    ids, att = _pad(seqs, 0)
    with torch.no_grad():
        hs = forward_hidden(model, input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                         output_hidden_states=True).hidden_states
    return torch.stack(hs, 0), [len(s) for s in seqs]


def centroids(model, tok, langs, cache_path, batch=16):
    """Mean token state (all layers) of each language's centring texts, prefix token excluded."""
    import torch
    if cache_path.exists():
        z = np.load(cache_path)
        if set(langs) <= set(z.files):
            return {l: torch.tensor(z[l], device=model.device) for l in langs}
    pre = prefix_ids(tok)
    out = {}
    for l in langs:
        texts = centring_texts(l)
        seqs = [pre + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        tot, n = None, 0
        for c in range(0, len(seqs), batch):
            H, lens = hidden_states(model, seqs[c:c + batch])
            for b, T in enumerate(lens):
                x = H[:, b, len(pre):T].float().sum(1)
                tot = x if tot is None else tot + x
                n += T - len(pre)
        out[l] = tot / n
        print(f"  centroid {l}: {len(texts)} texts, {n} tokens", flush=True)
    np.savez(cache_path, **{l: v.cpu().numpy() for l, v in out.items()})
    return out


def _unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def source_states(model, tok, sents, mu):
    """Centred, unit-norm states of each sentence's own tokens: {sid: tensor [L+1, n, d]}."""
    pre = prefix_ids(tok)
    out = {}
    for s in sents:
        ids, _ = source_tokens(tok, s)
        H, _ = hidden_states(model, [pre + ids])
        out[s.id] = _unit(H[:, 0, len(pre):].float() - mu[CENTROID_LANG.get(s.lang, s.lang)][:, None])
    return out


def run_cl(model, tok, sents, rows, mu, batch=12, eps=EPS_OT):
    """For each row: lse [L+1, n] and softcost [L+1, n] against the row's English text."""
    import torch
    pre = prefix_ids(tok)
    S = source_states(model, tok, sents, mu)
    seqs = [pre + tok(r["text"], add_special_tokens=False)["input_ids"][:MAX_T_TOKENS] for r in rows]
    order = sorted(range(len(rows)), key=lambda k: len(seqs[k]))
    lse, sc = [None] * len(rows), [None] * len(rows)
    prog = Progress("concept layer", len(rows))
    for c in range(0, len(order), batch):
        ks = order[c:c + batch]
        H, lens = hidden_states(model, [seqs[k] for k in ks])
        for b, k in enumerate(ks):
            Sk = S[rows[k]["sid"]]
            if lens[b] <= len(pre):                                      # empty translation: nothing carried
                lse[k] = np.full(Sk.shape[:2], -1e4, np.float32); sc[k] = np.ones(Sk.shape[:2], np.float32)
                continue
            G = _unit(H[:, b, len(pre):lens[b]].float() - mu["en"][:, None])
            C = 1 - torch.einsum("lnd,lmd->lnm", Sk, G)
            Z = -C / eps
            lse[k] = torch.logsumexp(Z, -1).cpu().numpy().astype(np.float32)
            sc[k] = (torch.softmax(Z, -1) * C).sum(-1).cpu().numpy().astype(np.float32)
        if (c // batch) % 10 == 0 or c + batch >= len(order):
            prog.tick(min(c + batch, len(order)))
    return lse, sc


def save_cl(d, model_key, rows, lse, sc, sents, tok, n_layers_):
    np.savez_compressed(d / f"cl_{model_key}.npz", **{f"lse{k}": x for k, x in enumerate(lse)},
                        **{f"sc{k}": x for k, x in enumerate(sc)})
    save_json(d / f"cl_{model_key}.json", dict(model=describe(model_key), eps_ot=EPS_OT, n_layers=n_layers_,
                                               rids=[r["rid"] for r in rows],
                                               tokens={s.id: source_tokens(tok, s) for s in sents}))


def load_cl(d, model_key):
    meta = load_json(d / f"cl_{model_key}.json")
    z = np.load(d / f"cl_{model_key}.npz")
    return {rid: (z[f"lse{k}"], z[f"sc{k}"]) for k, rid in enumerate(meta["rids"])}, meta


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS, help="the model whose candidates are scored")
    ap.add_argument("--bt", default=None, choices=MODELS, help="back-translator (default: --model)")
    ap.add_argument("--rt-only", action="store_true")
    ap.add_argument("--cl-only", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cl-batch", type=int, default=12)
    ap.add_argument("--dialect", action="store_true")
    ap.add_argument("--mask-rule", default="zipf", choices=["zipf", "cognate"],
                    help="zipf: the 19 Nov rule (source words rare in English); cognate: the internals run's default")
    a = ap.parse_args()
    bt = a.bt or a.model
    d = sent_dir(a.model, "raw")
    cands = read_jsonl(sent_dir(a.model) / "candidates.jsonl")
    sids = {c["sid"] for c in cands}
    sents = [s for s in load_sentences(dialect=True) if s.id in sids]
    rows = build_rows(cands, sents, mask_rule=a.mask_rule, cl_rows=not a.rt_only)
    write_jsonl(d / "rows.jsonl", rows)
    from collections import Counter
    print(f"{len(sents)} sentences, {len(rows)} rows: {dict(Counter(r['role'] for r in rows))}", flush=True)
    if not a.cl_only:
        tok, model = load_model(bt)
        with Timer(f"round trip ({bt})"):
            rt = run_rt(model, tok, sents, rows, batch=a.batch)
        save_json(d / f"rt_{bt}.json", dict(model=describe(bt), candidates_from=describe(a.model), **rt))
        if bt != a.model or a.rt_only:
            return
    else:
        tok, model = load_model(a.model)
    langs = sorted({CENTROID_LANG.get(s.lang, s.lang) for s in sents} | {"en"})
    with Timer("concept layer"):
        mu = centroids(model, tok, langs, d / f"centroids_{a.model}.npz")
        lse, sc = run_cl(model, tok, sents, rows, mu, batch=a.cl_batch)
    save_cl(d, a.model, rows, lse, sc, sents, tok, n_layers(model))
    print("wrote", d)


if __name__ == "__main__":
    main()
