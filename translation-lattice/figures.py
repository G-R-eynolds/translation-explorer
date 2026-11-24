"""Layer-sweep figure (§7) and the markdown tables for results.md.

    python figures.py --model 3b      -> runs/3b/results/layer_sweep.png and tables.md
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import MODELS, load_json, run_dir

RAW, ERASED = "#2a78d6", "#eb6834"          # categorical slots 1, 2 (raw / LEACE-erased in every panel)
SPAN_COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=8.5, color=INK)
    ax.plot([x[-1]], [y[-1]], "o", ms=4, color=color, mec="white", mew=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    a = ap.parse_args()
    rd = run_dir(a.model) / "results"
    sw = load_json(rd / "layer_sweep.json")
    val = load_json(rd / "validation.json")
    lens = load_json(rd / "logit_lens.json")
    L = np.arange(len(sw["stance_probe_raw"]))

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    ax = axs[0, 0]
    style(ax, "Stance is only weakly decodable; LEACE takes it to chance", "stance probe accuracy (grouped 4-fold CV)")
    for key, col, lab in (("stance_probe_raw", RAW, "raw"), ("stance_probe_erased", ERASED, "LEACE-erased")):
        ax.plot(L, sw[key], lw=2, color=col); label_end(ax, L, sw[key], lab, col)
    ax.axhline(sw["chance"], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate("chance", (0, sw["chance"]), xytext=(2, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.02)

    ax = axs[0, 1]
    style(ax, "Meaning separation (M1r clusters as labels)", "between / within-cluster cosine distance")
    for key, col, lab, dy in (("meaning_sep_raw", RAW, "raw", 8), ("meaning_sep_erased", ERASED, "LEACE-erased", -8)):
        y = np.array([np.nan if v is None else v for v in sw[key]])
        ax.plot(L, y, lw=2, color=col); label_end(ax, L, y, lab, col, dy)
    ax.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate("no separation", (0, 1), xytext=(2, 4), textcoords="offset points", fontsize=8, color=MUTED)

    ax = axs[1, 0]
    style(ax, "Style vs meaning control pairs, per layer", "AUROC (meaning edits vs restyles)")
    for m, col, lab, dy in (("b1", RAW, "raw (B1)", -7), ("m2", ERASED, "LEACE-erased (M2)", 7)):
        y = np.array([np.nan if v is None else v for v in val["layer_auroc"][m]["per_layer"]])
        ax.plot(L, y, lw=2, color=col); label_end(ax, L, y, lab, col, dy)
    for m, ls in (("b2", "-"), ("m1", "-"), ("m1r", "-"), ("b0", (0, (1, 2)))):
        v = val["validation"][m]["auroc_controls"]
        ax.axhline(v, color=MUTED, lw=1, ls=ls)
        ax.annotate(f"{dict(b2='B2 NLI', m1='M1', m1r='M1r', b0='B0 lexical')[m]} {v:.2f}", (0, v), xytext=(2, 4 if m != 'm1r' else -11),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate("chance", (28, 0.5), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.set_ylim(0.2, 1.02); ax.set_xlabel("layer (0 = embeddings)", fontsize=9, color=MUTED)

    ax = axs[1, 1]
    style(ax, "Source side: logit-lens entropy of the gloss's first word", "entropy (nats), cloze view")
    for s, col in zip(lens["spans"], SPAN_COLS):
        y = np.array([r["entropy"] for r in s["view_b"]])
        ax.plot(L[:len(y)], y, lw=1.6, color=col, label=f"{s['span']} ({s['line'] + 1})")
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2, labelcolor=INK)
    ax.annotate("layers 0-2: the lens just echoes the input token", (0, 0), xytext=(0, -2), textcoords="offset points",
                fontsize=7.5, color=MUTED, va="top")
    ax.set_xlabel("layer (0 = embeddings)", fontsize=9, color=MUTED)

    fig.suptitle(f"Layer sweep: {MODELS[a.model]['hf_id']}", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    fig.savefig(rd / "layer_sweep.png", dpi=150)
    print("wrote", rd / "layer_sweep.png")
    return a.model


def tables(model_key):
    rd = run_dir(model_key) / "results"
    v = load_json(rd / "validation.json")
    f = lambda x, d=2: "–" if x is None else f"{x:.{d}f}"
    out = ["### Validation\n",
           "| metric | AUROC controls | AUROC own-context restyles | AUROC demo (weak) | tau | meaning edits > tau | ARI hand (tau-cut) | ARI hand (hand k) | ARI clusters~stance |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m, r in v["validation"].items():
        name = v["metrics"][m] + (f", layer {r['layer']}" if "layer" in r else "")
        auc = f(r["auroc_controls"]) + (f" (cross-fit {f(r['auroc_controls_crossfit_layer'])})" if "layer" in r else "")
        out.append(f"| {name} | {auc} | {f(r['auroc_controls_ownctx'])} | {f(r['auroc_demo'])} | {f(r['tau'], 3)} | "
                   f"{f(r['meaning_above_tau'])} | {f(r['ari_hand_own_k'])} | {f(r['ari_hand_hand_k'])} | {f(r['ari_clusters_vs_stance'])} |")
    ms = list(v["validation"])
    out += ["\n### Per line (k = clusters after the tau cut; shared = clusters with >= 2 samples; top = largest cluster of 48)\n",
            "| line | " + " | ".join(f"{m} KLE / H bits / k (shared, top)" for m in ms) + " |",
            "|---|" + "---|" * len(ms)]
    for i, r in sorted(v["per_line"].items(), key=lambda kv: int(kv[0])):
        out.append(f"| {int(i) + 1} | " + " | ".join(
            f"{r[m]['kle']:.2f} / {r[m]['sem_entropy_bits']:.2f} / {r[m]['k']} ({r[m]['n_shared_clusters']}, {r[m]['largest_cluster']})"
            for m in ms) + " |")
    out += ["\n### Agreement among metrics on the lattice (Spearman of pairwise distances; mean ARI of clusterings)\n",
            "| pair | Spearman | ARI |", "|---|---|---|"]
    out += [f"| {k} | {r['spearman']:.2f} | {r['mean_ari']:.2f} |" for k, r in v["agreement"].items()]
    (rd / "tables.md").write_text("\n".join(out) + "\n")
    print("wrote", rd / "tables.md")


if __name__ == "__main__":
    tables(main())
