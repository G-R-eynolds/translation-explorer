"""Hidden states (M2, B1), LEACE stance erasure, the layer sweep (§7) and the logit lens.

    python layers.py --model 3b --extract          # GPU: mean-pooled residual states for every rendering
    python layers.py --model 3b --lens             # GPU: logit lens on the untranslatable source spans
    python layers.py --model 3b --analyze          # M2/B1 distances per layer, stance probes, separation, figure

Context is held fixed: every rendering of line i is embedded after the SAME prefix, the translation
prompt (source poem, neutral stance) followed by the central-route lines 1..i-1 (runs/<model>/spine.json,
the medoid of the largest M1r cluster on each line).  States are mean-pooled over the rendering's tokens.
"""
import argparse
from collections import defaultdict

import numpy as np

from backtranslate import build_units
from common import DATA, MODELS, Timer, chat, encode, load_json, load_model, run_dir, save_json
from sample import SAMPLE_PROMPT, fill

NEUTRAL = "Faithful English, line for line"
SPANS = [  # (line index, source substring); the four from the handoff, then two more branch points
    (3, "ffora a cchi ttoca"), (4, "filastrocca"), (5, "a ppecorone"), (13, "bbona sera"),
    (7, "bbiocca"), (11, "sonajjera"),
]


def prefix_for(tok, src, spine_lines, i):
    head = "".join(f"{k + 1}. {spine_lines[k]}\n" for k in range(i)) + f"{i + 1}."
    return chat(tok, fill(SAMPLE_PROMPT, src, desc=NEUTRAL), assistant_prefix=head)


def all_items(model_key):
    """Every rendering we need states for: lattice, demo and (non-demo) control items."""
    rd = run_dir(model_key)
    out = {"lattice": build_units(model_key, "lattice"), "demo": build_units(model_key, "demo")}
    if (rd / "controls.json").exists():
        out["controls"] = build_units(model_key, "controls")
    return out


def extract(a):
    import torch
    rd = run_dir(a.model)
    src = load_json(DATA / "source.json")
    spine = load_json(rd / "spine.json")
    tok, model = load_model(a.model)
    tok.padding_side = "right"
    for name, units in all_items(a.model).items():
        H = None
        by_line = defaultdict(list)
        for k, u in enumerate(units):
            by_line[u["line"]].append(k)
        with Timer(f"extract {name} ({len(units)})"):
            for i, ks in sorted(by_line.items()):
                pre = encode(tok, prefix_for(tok, src, spine["lines"], i))
                for c in range(0, len(ks), a.batch):
                    chunk = ks[c:c + a.batch]
                    seqs = [pre + encode(tok, " " + units[k]["text"]) for k in chunk]
                    Lm = max(map(len, seqs))
                    ids = torch.full((len(seqs), Lm), tok.pad_token_id)
                    att = torch.zeros((len(seqs), Lm), dtype=torch.long)
                    rmask = torch.zeros((len(seqs), Lm))
                    for b, s in enumerate(seqs):
                        ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1; rmask[b, len(pre):len(s)] = 1
                    with torch.no_grad():
                        hs = model.model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states  # trunk only: no 128k-vocab logits
                    rm = rmask.cuda()[:, :, None]
                    pooled = torch.stack([(h.float() * rm).sum(1) / rm.sum(1) for h in hs], 1)  # (B, L+1, d)
                    if H is None:
                        H = np.zeros((len(units),) + tuple(pooled.shape[1:]), dtype=np.float16)
                    H[chunk] = pooled.cpu().numpy().astype(np.float16)
        np.savez(rd / "results" / f"hidden_{name}.npz", H=H, uids=np.array([u["uid"] for u in units]),
                 stance=np.array([u["stance"] for u in units]), line=np.array([u["line"] for u in units]))
    save_json(rd / "results" / "hidden_meta.json", dict(prefix_example=prefix_for(tok, src, spine["lines"], 3),
                                                        pooling="mean over rendering tokens", spine=spine))


def lens(a):
    """Logit lens: decode each layer's residual through the final norm + unembedding.
    View A: at the last token of each untranslatable span, inside the translation prompt.
    View B (Wendler et al.-style cloze): after the poem, '"<span>" means, in English: "', at the last
    position, i.e. the layer-wise guess for the first English word of the gloss."""
    import torch
    rd = run_dir(a.model)
    src = load_json(DATA / "source.json")
    tok, model = load_model(a.model)
    norm, head = model.model.norm, model.lm_head
    res = []

    def decode_layers(hs, pos):
        out = []
        for l, h in enumerate(hs):
            x = h[0, pos].float()
            x = x if l == len(hs) - 1 else norm(x.to(h.dtype)).float()  # last state is already normed
            p = torch.softmax(head(x.to(h.dtype)).float(), -1)
            top = torch.topk(p, 5)
            ent = float(-(p * torch.log(p + 1e-12)).sum())
            out.append(dict(layer=l, top=[(tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(top.values, top.indices)],
                            entropy=round(ent, 3)))
        return out

    prompt_a = chat(tok, fill(SAMPLE_PROMPT, src, desc=NEUTRAL))
    enc = tok(prompt_a, add_special_tokens=False, return_offsets_mapping=True)
    with torch.no_grad():
        hs_a = model.model(input_ids=torch.tensor([enc["input_ids"]]).cuda(), output_hidden_states=True).hidden_states
    for line, span in SPANS:
        anchor = prompt_a.index(f"{line + 1}. {src['lines'][line]}")
        end = prompt_a.index(span, anchor) + len(span)
        pos = max(k for k, (s, e) in enumerate(enc["offset_mapping"]) if s < end)
        cloze = (f"Here is a sonnet in Romanesco (19th-century Roman dialect) by G. G. Belli:\n\n{chr(10).join(src['lines'])}\n\n"
                 f"In line {line + 1}, what does \"{span}\" mean? Answer in English in a few words.")
        prompt_b = chat(tok, cloze, assistant_prefix=f"In line {line + 1}, \"{span}\" means, in English: \"")
        ids_b = encode(tok, prompt_b)
        with torch.no_grad():
            hs_b = model.model(input_ids=torch.tensor([ids_b]).cuda(), output_hidden_states=True).hidden_states
            gen = model.generate(torch.tensor([ids_b]).cuda(), max_new_tokens=20, do_sample=False,
                                 pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        res.append(dict(line=line, span=span, source_token=tok.decode([enc["input_ids"][pos]]),
                        view_a=decode_layers(hs_a, pos), view_b=decode_layers(hs_b, len(ids_b) - 1),
                        view_b_greedy=tok.decode(gen[0, len(ids_b):], skip_special_tokens=True)))
        print(span, "| greedy gloss:", res[-1]["view_b_greedy"])
        for r in res[-1]["view_b"][::3]:
            print(f"   L{r['layer']:2d} H={r['entropy']:5.2f}", r["top"][:3])
    save_json(rd / "results" / "logit_lens.json", dict(prompt_view_a=prompt_a, spans=res))


# ------------------------------------------------------------------ analysis
def centre_cos(X, mu):
    Xc = X - mu
    Xc = Xc / (np.linalg.norm(Xc, axis=-1, keepdims=True) + 1e-8)
    return Xc


def onehot(labels, classes):
    import torch
    return torch.nn.functional.one_hot(torch.tensor([classes.index(s) for s in labels]), len(classes)).float()


def fit_leace(X, stances, classes):
    import torch
    from concept_erasure import LeaceEraser
    return LeaceEraser.fit(torch.tensor(X, dtype=torch.float32), onehot(stances, classes))


def apply(er, X):
    import torch
    return er(torch.tensor(X, dtype=torch.float32)).numpy()


def probe():
    """Linear stance probe: standardised features, L2 logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))


def analyze(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rd = run_dir(a.model)
    z = {n: np.load(rd / "results" / f"hidden_{n}.npz") for n in ("lattice", "demo", "controls")
         if (rd / "results" / f"hidden_{n}.npz").exists()}
    lat = z["lattice"]
    H = lat["H"].astype(np.float32)          # (n, L+1, d)
    stance, line, uids = list(lat["stance"]), lat["line"], [str(u) for u in lat["uids"]]
    sample_of = np.array([u.split(":")[0] for u in uids])
    classes = sorted(set(stance))
    nL = H.shape[1]
    m1 = load_json(rd / "clusters_m1r.json") if (rd / "clusters_m1r.json").exists() else None
    lab_m1 = {}
    if m1:
        for Ln in m1["lines"]:
            for c in Ln["clusters"]:
                for s in c["members"]:
                    lab_m1[(Ln["line"], s)] = c["id"]
    sweep = defaultdict(list)
    dist = {}   # name -> per-layer distances
    ctrl = load_json(rd / "controls.json") if (rd / "controls.json").exists() else None
    for l in range(nL):
        X = H[:, l]
        mu = X.mean(0)
        er = fit_leace(X, stance, classes)
        Xe = apply(er, X)
        mue = Xe.mean(0)
        reps = {"raw": (X, mu, None), "erased": (Xe, mue, er)}
        # distances: lattice per line, demo per line, control pairs
        for rname, (Xr, m_, er_) in reps.items():
            Zl = centre_cos(Xr, m_)
            for i in range(14):
                idx = np.where(line == i)[0]
                dist[(f"{rname}", "lattice", i, l)] = 1 - Zl[idx] @ Zl[idx].T
            for setname in ("demo", "controls"):
                if setname not in z:
                    continue
                Y = z[setname]["H"][:, l].astype(np.float32)
                Y = apply(er_, Y) if er_ is not None else Y
                Zs = centre_cos(Y, m_)
                if setname == "demo":
                    ln = z["demo"]["line"]
                    for i in range(14):
                        idx = np.where(ln == i)[0]
                        dist[(rname, "demo", i, l)] = 1 - Zs[idx] @ Zs[idx].T
                else:
                    pos = {str(u): k for k, u in enumerate(z["controls"]["uids"])}
                    dist[(rname, "controls", None, l)] = {
                        p["id"]: float(1 - Zs[pos[p["a"]]] @ Zs[pos[p["b"]]]) for p in ctrl["pairs"]
                        if p["set"] != "demo"}
        # style decodability (stance probe), grouped CV by sample; erased version re-fits LEACE per fold
        acc_raw, acc_er = [], []
        for tr, te in GroupKFold(n_splits=4).split(X, groups=sample_of):
            yt, ye = [stance[k] for k in tr], [stance[k] for k in te]
            acc_raw.append(probe().fit(X[tr], yt).score(X[te], ye))
            erf = fit_leace(X[tr], yt, classes)   # eraser fitted on the training fold only
            acc_er.append(probe().fit(apply(erf, X[tr]), yt).score(apply(erf, X[te]), ye))
        sweep["stance_probe_raw"].append(float(np.mean(acc_raw)))
        sweep["stance_probe_erased"].append(float(np.mean(acc_er)))
        # meaning separation with the M1r clusters as labels: between / within mean cosine distance
        for rname in ("raw", "erased"):
            ratios = []
            if m1:
                for i in range(14):
                    idx = np.where(line == i)[0]
                    lab = np.array([lab_m1[(i, sample_of[k])] for k in idx])
                    if len(set(lab)) < 2:
                        continue
                    Dm = dist[(rname, "lattice", i, l)]
                    same = lab[:, None] == lab[None, :]
                    off = ~np.eye(len(idx), dtype=bool)
                    if (same & off).any():
                        ratios.append(Dm[~same].mean() / Dm[same & off].mean())
            sweep[f"meaning_sep_{rname}"].append(float(np.mean(ratios)) if ratios else None)
        print(f"layer {l:2d}: probe raw {sweep['stance_probe_raw'][-1]:.2f} erased {sweep['stance_probe_erased'][-1]:.2f}"
              f"  sep raw {sweep['meaning_sep_raw'][-1]} erased {sweep['meaning_sep_erased'][-1]}", flush=True)
    sweep["chance"] = 1 / len(classes)
    save_json(rd / "results" / "layer_sweep.json", sweep)
    # pack distances
    out = {}
    for (rname, setname, i, l), v in dist.items():
        if setname == "controls":
            out[f"{rname}|controls|{l}|ids"] = np.array(list(v.keys()))
            out[f"{rname}|controls|{l}|d"] = np.array(list(v.values()))
        else:
            out[f"{rname}|{setname}|{i}|{l}"] = v.astype(np.float32)
    np.savez_compressed(rd / "results" / "m2_distances.npz", **out)
    print("wrote m2_distances.npz and layer_sweep.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--lens", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    if a.extract:
        extract(a)
    if a.lens:
        lens(a)
    if a.analyze:
        analyze(a)
