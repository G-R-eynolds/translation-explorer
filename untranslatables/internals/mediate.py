"""N3: causal mediation, what the model knows but does not translate (HANDOFF_internals §5).  GPU, main model.

Decomposition (§5.1).  From the N1 basis at layer l (the es-en or hi-en peak-sharing layer, and the R1 detour layer as a
second run): the top-r shared canonical directions a_k (held-out rho above the null), with Sxx(lam) a_k stored; for a
centred source state c = h - mu:  c_sh = P c with P = sum_k Sxx(lam) a_k a_k^T (an oblique projection, idempotent
because a_k^T Sxx(lam) a_j = delta_kj; checked numerically), c_pr = c - c_sh.
Interventions act on hidden_states[l] (the output of block l-1) at the source span's token positions only:
  ablate shared:  h -> mu + c_pr      ablate private: h -> mu + c_sh      ablate whole: h -> mu
(mean-ablation: each component set to its corpus mean, which is 0 after centring; mu is added back).

Effects (§5.2), per span and component V, each normalised by the whole-state effect:
  E_trans  change in the mean per-token log-prob of the model's own greedy standard (B1) translation, teacher-forced in
           the B1 translation prompt; plus the greedy translation re-generated under the intervention
  E_mono   (a) change in the mean log-prob of the sentence's own tokens after the span (raw sentence)
           (b) change in the mean log-prob of the source-language sense gloss after
               "En esta frase, '<span>' significa:" (Hindi: "इस वाक्य में '<span>' का अर्थ है:")
Interchange (§5.3) on data/sense_pairs.yaml: the word's private / shared / whole state from context 2 patched into the run
on context 1; log-odds of the two English translations and of the two source-language glosses, and the greedy output.
Gate (§7.3 and §7.1): N1's AUROC gate, the projection checks, and a whole-ablation effect floor; if the N1 gate failed
the stage writes n3_skipped.json and stops (cfg override_gate=true forces it).
"""
import contextlib
import json
from collections import defaultdict

import numpy as np
import torch

from internals.cca import Basis
from internals.common import RESULTS, cache_dir, res_dir, stage
from sent.data import load_sentences
from sent.prompts import rung_prompt
from ut.arch import forward_hidden, layers as arch_layers, logits as arch_logits
from ut.common import chat, load_model, load_yaml, stop_ids
from ut.represent import unmark

WHOLE_FLOOR = 0.02            # nats/token: below this the whole-state ablation barely matters (sanity floor, §7.3)
SENSE_Q = {"es": "En esta frase, '{span}' significa:", "hi": "इस वाक्य में '{span}' का अर्थ है:"}


# ------------------------------------------------------------------ the projection
class Components:
    def __init__(self, basis, device):
        self.mu = basis.mu_x.float().to(device)
        A = basis.A_shared_t.float().to(device)          # [d, r]
        S = basis.SxxA.float().to(device)                # [d, r]
        self.P = S @ A.T                                 # x_sh = P x
        self.r = A.shape[1]

    def check(self):
        P = self.P.double()
        return dict(r=self.r, idempotence_rel_err=float((P @ P - P).norm() / P.norm()),
                    trace=float(torch.trace(P)), rank_expected=self.r)

    def split(self, h):
        c = h.float() - self.mu
        sh = c @ self.P.T
        return c, sh, c - sh

    def transform(self, h, kind, donor=None):
        """kind: clean | ablate_shared | ablate_private | ablate_whole | patch_private | patch_shared | patch_whole."""
        if kind == "clean":
            return h
        c, sh, pr = self.split(h)
        if kind == "ablate_shared":
            new = pr
        elif kind == "ablate_private":
            new = sh
        elif kind == "ablate_whole":
            new = torch.zeros_like(c)
        else:
            dc, dsh, dpr = self.split(donor)
            new = {"patch_private": sh + dpr, "patch_shared": dsh + pr, "patch_whole": dc}[kind]
        return (new + self.mu).to(h.dtype)


@contextlib.contextmanager
def intervene(model, layer, comp, rows):
    """rows[b] = (positions in row b, kind, donor states [len(positions), d] or None).  Prefill only (decoding steps,
    sequence length 1, are left alone so generation continues from the intervened prompt)."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:
            return out
        h = h.clone()
        for b, (pos, kind, donor) in enumerate(rows):
            if kind == "clean" or not pos:
                continue
            p = torch.tensor(pos, device=h.device)
            d_ = None if donor is None else donor.to(h.device)
            h[b, p] = comp.transform(h[b, p], kind, d_)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hd = arch_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


# ------------------------------------------------------------------ text + positions
def encode_with_span(tok, text, span_text_start, span_text_end, prefix_text=""):
    """Token ids of prefix_text + text and the positions of the tokens overlapping [start, end) of `text`."""
    full = prefix_text + text
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    s0, e0 = len(prefix_text) + span_text_start, len(prefix_text) + span_text_end
    pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > s0 and a < e0 and b > a]
    return enc["input_ids"], pos


def chat_with_span(tok, user, sentence, s0, e0):
    """Chat-formatted prompt ids, and the positions of the span inside the LAST occurrence of `sentence` in it."""
    txt = chat(tok, user)
    at = txt.rfind(sentence)
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > at + s0 and a < at + e0 and b > a]
    return enc["input_ids"], pos


def teacher_lp(model, prompts, targets, rows, layer, comp, batch=8):
    """Mean log-prob per target token for each (prompt ids, target ids) under row-wise interventions."""
    out = []
    for c in range(0, len(prompts), batch):
        P, Tg, R = prompts[c:c + batch], targets[c:c + batch], rows[c:c + batch]
        seqs = [p + t for p, t in zip(P, Tg)]
        Tm = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), Tm), dtype=torch.long); att = torch.zeros_like(ids)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        with torch.no_grad(), intervene(model, layer, comp, R):
            h = forward_hidden(model, input_ids=ids.to(model.device), attention_mask=att.to(model.device)).last_hidden_state
            for b in range(len(seqs)):
                n0, n = len(P[b]), len(Tg[b])
                lg = arch_logits(model, h[b, n0 - 1:n0 - 1 + n])
                lp = torch.log_softmax(lg, -1).gather(1, torch.tensor(Tg[b], device=lg.device)[:, None])[:, 0]
                out.append(lp.cpu().numpy())
    return out


def greedy(model, tok, prompts, rows, layer, comp, max_new=60, batch=8):
    """Greedy continuations under row-wise interventions (left padding; positions shifted accordingly)."""
    texts = []
    stops = stop_ids(tok)
    for c in range(0, len(prompts), batch):
        P, R = prompts[c:c + batch], rows[c:c + batch]
        Tm = max(len(p) for p in P)
        ids = torch.full((len(P), Tm), tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(ids)
        shifted = []
        for b, p in enumerate(P):
            ids[b, Tm - len(p):] = torch.tensor(p); att[b, Tm - len(p):] = 1
            pos, kind, donor = R[b]
            shifted.append(([q + Tm - len(p) for q in pos], kind, donor))
        with torch.no_grad(), intervene(model, layer, comp, shifted):
            g = model.generate(input_ids=ids.to(model.device), attention_mask=att.to(model.device), max_new_tokens=max_new,
                               do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                               eos_token_id=stops)[:, Tm:].tolist()
        for s in g:
            cut = next((i for i, t in enumerate(s) if t in stops), len(s))
            texts.append(tok.decode(s[:cut], skip_special_tokens=True).strip())
    return texts


# ------------------------------------------------------------------ setup
def load_components(m, pair, layer, device):
    B = Basis.load(cache_dir(m, "n1", pair) / f"basis_L{layer}.npz", device=device)
    spec = json.loads((res_dir(m) / f"n1_spectrum_{pair}_L{layer}.json").read_text())
    rho = np.asarray(spec["rho_heldout"])
    shared = np.nonzero(rho > spec["rho_null_heldout_q99"])[0]
    order = np.argsort(-rho)[:len(shared)]
    B.A_shared_t = B.A[:, torch.tensor(order, device=B.A.device)]
    # SxxA was stored for the same held-out-ranked top r directions
    assert B.SxxA.shape[1] == len(order), (B.SxxA.shape, len(order))
    return Components(B, device)


def mediation_layers(m, pair):
    meta = json.loads((cache_dir(m, "n1", pair) / "meta.json").read_text())
    ls = [meta["peak_layer"]]
    if meta.get("detour_layer") and meta["detour_layer"] in meta["layers"] and meta["detour_layer"] != ls[0]:
        ls.append(meta["detour_layer"])
    return ls


def gate(m):
    p = RESULTS / m / "n1_spans_summary.json"
    if not p.exists():
        return False, "no N1 span summary"
    g = json.loads(p.read_text()).get("gate_N1", {})
    return bool(g.get("passed")), f"N1 gate: AUROC {g.get('auroc')}"


def b1_translations(m):
    p = RESULTS / m / "replication_R2" / "candidates.jsonl"
    return {c["sid"]: c["text"] for c in map(json.loads, open(p)) if c["rung"] == "B1" and c["sample"] < 0}


@stage("mediate")
def run(cfg):
    m = cfg.get("model", "gemma12b")
    ok, why = gate(m)
    if not ok and not cfg.get("override_gate"):
        p = res_dir(m) / "n3_skipped.json"
        p.write_text(json.dumps(dict(skipped=True, reason=why)))
        print("N3 skipped:", why, flush=True)
        return [p]
    tok, model = load_model(m)
    dev_ = model.device
    b1 = b1_translations(m)
    rows_out, checks = [], {}
    kinds = ["clean", "ablate_shared", "ablate_private", "ablate_whole"]
    for lang, pair in (("es", "es-en"), ("hi", "hi-en")):
        if not (cache_dir(m, "n1", pair) / "meta.json").exists():
            continue
        sents = [s for s in load_sentences(langs=[lang]) if s.id in b1]
        items = []                                     # (sentence, span dict, unit kind)
        for s in sents:
            for x in s.spans():
                items.append((s, x, x["kind"]))
            if s.is_control:                          # every other content word of a control sentence: E_trans, E_mono(a)
                sp = {(x["start"], x["end"]) for x in s.spans()}
                for ws, we, w in s.content_words():
                    if (ws, we) not in sp:
                        items.append((s, dict(text=w, start=ws, end=we, kind="control_word", sense_src=None), "control_word"))
        for layer in mediation_layers(m, pair):
            comp = load_components(m, pair, layer, dev_)
            checks[f"{pair}_L{layer}"] = comp.check()
            for s, x, unit in items:
                rec = dict(pair=pair, layer=layer, sid=s.id, lang=lang, category=s.category, span=x["text"], unit=unit,
                           start=x["start"], end=x["end"])
                # E_trans: the model's own B1 translation, teacher-forced
                pid, ppos = chat_with_span(tok, rung_prompt("B1", s), s.text, x["start"], x["end"])
                tgt = tok(b1[s.id], add_special_tokens=False)["input_ids"]
                lps = teacher_lp(model, [pid] * 4, [tgt] * 4, [(ppos, k, None) for k in kinds], layer, comp)
                rec["trans"] = {k: float(v.mean()) for k, v in zip(kinds, lps)}
                rec["trans_greedy"] = dict(zip(kinds[1:], greedy(model, tok, [pid] * 3, [(ppos, k, None) for k in kinds[1:]],
                                                                 layer, comp)))
                # E_mono (a): the sentence's own continuation after the span
                from internals.collect import prefix
                pre = prefix(tok)
                ids, pos = encode_with_span(tok, s.text, x["start"], x["end"])
                after = [k for k, (a, _) in enumerate(tok(s.text, add_special_tokens=False, return_offsets_mapping=True)
                                                      ["offset_mapping"]) if a >= x["end"]]
                if after:
                    cut = after[0]
                    pr_ids, tg_ids = pre + ids[:cut], ids[cut:]
                    lps = teacher_lp(model, [pr_ids] * 4, [tg_ids] * 4, [([len(pre) + p for p in pos], k, None) for k in kinds],
                                     layer, comp)
                    rec["mono_cont"] = {k: float(v.mean()) for k, v in zip(kinds, lps)}
                # E_mono (b): the source-language sense gloss
                if x.get("sense_src"):
                    q = "\n" + SENSE_Q[lang].format(span=x["text"]) + " "
                    ids_s, pos_s = encode_with_span(tok, s.text + q, x["start"], x["end"])
                    g = tok(x["sense_src"], add_special_tokens=False)["input_ids"]
                    lps = teacher_lp(model, [pre + ids_s] * 4, [g] * 4, [([len(pre) + p for p in pos_s], k, None) for k in kinds],
                                     layer, comp)
                    rec["mono_sense"] = {k: float(v.mean()) for k, v in zip(kinds, lps)}
                for key in ("trans", "mono_cont", "mono_sense"):
                    if key in rec:
                        d = rec[key]
                        whole = d["ablate_whole"] - d["clean"]
                        rec[f"E_{key}"] = {V: (d[f"ablate_{V}"] - d["clean"]) / whole if abs(whole) > 1e-6 else None
                                           for V in ("shared", "private")}
                        rec[f"whole_{key}"] = whole
                rows_out.append(rec)
            print(f"  {pair} L{layer}: {len(items)} spans done", flush=True)
    rd = res_dir(m)
    p = rd / "n3_ablation.jsonl"
    with open(p, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarise(rows_out, checks)
    q = rd / "n3_summary.json"
    q.write_text(json.dumps(summ, default=float, ensure_ascii=False))
    return [p, q]


def summarise(rows, checks):
    from ut.stats import boot_ci
    out = dict(projection_checks=checks, groups={})
    floor = {}
    for r in rows:
        floor.setdefault((r["pair"], r["layer"]), []).append(abs(r["whole_trans"]))
    out["whole_ablation_trans_mean_abs"] = {f"{k[0]}_L{k[1]}": float(np.mean(v)) for k, v in floor.items()}
    out["sanity_floor_passed"] = {f"{k[0]}_L{k[1]}": bool(np.mean(v) >= WHOLE_FLOOR) for k, v in floor.items()}
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp = "hot" if r["unit"] == "hot" else "control"
        for key in ("E_trans", "E_mono_cont", "E_mono_sense"):
            if key in r:
                for V, v in r[key].items():
                    if v is not None:
                        g[(r["pair"], r["layer"], grp)][f"{key}_{V}"].append(v)
        if "E_trans" in r and "E_mono_sense" in r and r["E_trans"]["private"] is not None and r["E_mono_sense"]["private"] is not None:
            g[(r["pair"], r["layer"], grp)]["dissociation_private"].append(r["E_mono_sense"]["private"] - r["E_trans"]["private"])
    for (pair, layer, grp), d in g.items():
        out["groups"][f"{pair}_L{layer}_{grp}"] = {k: dict(n=len(v), mean=float(np.mean(v)), ci=boot_ci(v)) for k, v in d.items()}
    return out


# ------------------------------------------------------------------ §5.3 interchange
@stage("interchange")
def interchange(cfg):
    m = cfg.get("model", "gemma12b")
    ok, why = gate(m)
    if not ok and not cfg.get("override_gate"):
        p = res_dir(m) / "n3_interchange_skipped.json"
        p.write_text(json.dumps(dict(skipped=True, reason=why)))
        return [p]
    from internals.collect import forward_states, prefix
    from sent.data import Sentence
    tok, model = load_model(m)
    pre = prefix(tok)
    pairs = load_yaml(cfg.get("path", "data/sense_pairs.yaml"))["pairs"]
    rows = []
    kinds = ["clean", "patch_private", "patch_shared", "patch_whole"]
    for lang, pair_key in (("es", "es-en"), ("hi", "hi-en")):
        P = [p for p in pairs if p["lang"] == lang]
        if not P or not (cache_dir(m, "n1", pair_key) / "meta.json").exists():
            continue
        for layer in mediation_layers(m, pair_key):
            comp = load_components(m, pair_key, layer, model.device)
            for p in P:
                directions = [(0, 1)] + ([(1, 0)] if p.get("same_frame") else [])
                for a, b in directions:
                    ta, sa = unmark(p["ctx"][a]); tb, sb = unmark(p["ctx"][b])
                    # donor: the word's states in context b at this layer (raw sentence)
                    ids_b, pos_b = encode_with_span(tok, tb, *sb)
                    H, _ = forward_states(model, [pre + ids_b], [layer])
                    donor = H[0][0, [len(pre) + q for q in pos_b]].float()
                    sent_a = Sentence(id=p["id"], lang=lang, category="pair", text=ta)
                    pid, ppos = chat_with_span(tok, rung_prompt("B1", sent_a), ta, *sa)
                    if len(ppos) != len(donor):
                        donor = donor.mean(0, keepdim=True).expand(len(ppos), -1)
                    en_a, en_b = (p["en"][0], p["en"][1]) if a == 0 else (p["en"][1], p["en"][0])
                    gl_a, gl_b = (p["gloss"][0], p["gloss"][1]) if a == 0 else (p["gloss"][1], p["gloss"][0])
                    R = [(ppos, k, None if k == "clean" else donor) for k in kinds]
                    tA = tok(en_a, add_special_tokens=False)["input_ids"]; tB = tok(en_b, add_special_tokens=False)["input_ids"]
                    lpA = teacher_lp(model, [pid] * 4, [tA] * 4, R, layer, comp)
                    lpB = teacher_lp(model, [pid] * 4, [tB] * 4, R, layer, comp)
                    lo_en = {k: float(b_.sum() - a_.sum()) for k, a_, b_ in zip(kinds, lpA, lpB)}
                    q = "\n" + SENSE_Q[lang].format(span=ta[sa[0]:sa[1]]) + " "
                    ids_s, pos_s = encode_with_span(tok, ta + q, *sa)
                    Rs = [([len(pre) + x for x in pos_s], k, None if k == "clean" else
                           (donor if len(pos_s) == len(donor) else donor.mean(0, keepdim=True).expand(len(pos_s), -1)))
                          for k in kinds]
                    gA = tok(gl_a, add_special_tokens=False)["input_ids"]; gB = tok(gl_b, add_special_tokens=False)["input_ids"]
                    sA = teacher_lp(model, [pre + ids_s] * 4, [gA] * 4, Rs, layer, comp)
                    sB = teacher_lp(model, [pre + ids_s] * 4, [gB] * 4, Rs, layer, comp)
                    lo_src = {k: float(b_.mean() - a_.mean()) for k, a_, b_ in zip(kinds, sA, sB)}
                    gens = dict(zip(kinds, greedy(model, tok, [pid] * 4, R, layer, comp, max_new=40)))
                    norm = lambda lo: {k: (lo[k] - lo["clean"]) / (lo["patch_whole"] - lo["clean"])
                                       if abs(lo["patch_whole"] - lo["clean"]) > 1e-6 else None for k in kinds[1:]}
                    rows.append(dict(id=p["id"], type=p["type"], lang=lang, pair=pair_key, layer=layer, direction=f"{a}<-{b}",
                                     context=ta, donor_context=tb, en=[en_a, en_b], gloss=[gl_a, gl_b],
                                     logodds_en=lo_en, logodds_src=lo_src, normalised_en=norm(lo_en),
                                     normalised_src=norm(lo_src), generations=gens))
                    print(f"  {p['id']:18s} L{layer} {a}<-{b}: dLO_en private {lo_en['patch_private'] - lo_en['clean']:+.2f} "
                          f"whole {lo_en['patch_whole'] - lo_en['clean']:+.2f} | dLO_src private "
                          f"{lo_src['patch_private'] - lo_src['clean']:+.2f} whole {lo_src['patch_whole'] - lo_src['clean']:+.2f}"
                          f" | {gens['patch_private']!r}", flush=True)
    pth = res_dir(m) / "n3_interchange.jsonl"
    with open(pth, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [pth]
