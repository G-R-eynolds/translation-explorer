"""Clean quantitative visualisation of translation directions for duende_flamenco.

Reads:
  untranslatables/results/explorer/g3_4b_pt/samples/duende_flamenco.json
  untranslatables/results/explorer/g3_4b_pt/samples/duende_flamenco_emb.npy
  untranslatables/results/explorer/g3_4b_pt/branches/duende_flamenco.json
  untranslatables/results/explorer/g3_4b_pt/steering/duende_flamenco.jsonl

Writes:
  figures_for_dad/duende_directions_numbers.png
  figures_for_dad/duende_directions_numbers.svg
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "untranslatables" / "results" / "explorer" / "g3_4b_pt"
OUT = Path(__file__).resolve().parent

# Warm editorial palette matching duende_five_ways.html
PAPER = "#f7f6f3"
PANEL = "#fdfcf9"
INK = "#1d1b1f"
MUTED = "#67636b"
RULE = "#dedbe0"
WINE = "#8a2343"

READING_META = {
    1: dict(name="magic · fairy · genie", short="magic", color="#2d5f8b"),
    3: dict(name="spirit · soul", short="spirit", color="#2e7d5b"),
    2: dict(name="duende · flame · spark", short="duende", color="#8a2343"),
    0: dict(name="possessed · gift · rhythm", short="gift / possessed", color="#b86b24"),
    4: dict(name="charm · charisma · whimsy", short="charm", color="#6d4c91"),
}


def load_data():
    S = json.loads((EXP / "samples" / "duende_flamenco.json").read_text())
    emb = np.load(EXP / "samples" / "duende_flamenco_emb.npy")
    B = json.loads((EXP / "branches" / "duende_flamenco.json").read_text())
    steer = [
        json.loads(line)
        for line in (EXP / "steering" / "duende_flamenco.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return S, emb, B, steer


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(RULE)
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def main():
    S, emb, B, steer = load_data()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )

    fig = plt.figure(figsize=(15.5, 9.6), dpi=220)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.18, 1.0],
        hspace=0.36,
        wspace=0.24,
        left=0.055,
        right=0.97,
        top=0.855,
        bottom=0.08,
    )

    # Header
    fig.text(
        0.055,
        0.955,
        "DIRECTIONS ON TRANSLATION  ·  GEMMA 3 4B BASE  ·  64 TRANSLATIONS",
        fontsize=8.5,
        color=MUTED,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.920,
        "Branching around “duende” at the number level",
        fontsize=19,
        color=INK,
        fontweight="bold",
        family="DejaVu Serif",
    )
    fig.text(
        0.055,
        0.888,
        "Source: “Cuando canta Camarón, tiene duende; me gusta mucho el duende.”   "
        f"(57 kept in 5 readings, 7 low-probability tail dropped; H_readings = {S['entropy_bits']:.2f} bits, silhouette = {S['silhouette']:.2f})",
        fontsize=9.5,
        color=MUTED,
    )

    # ------------------------------------------------------------------
    # PANEL A (Top, spanning full width): Token-level branching profile
    # ------------------------------------------------------------------
    ax_tok = fig.add_subplot(gs[0, :])
    style_ax(ax_tok)

    tokens = [t.replace("\n", "\\n") for t in B["ref_tokens"]]
    ent = np.array(B["entropy"], dtype=float)
    fish = np.array(B["fisher_top"], dtype=float)
    t_fork = B["t"]
    xs = np.arange(len(tokens))

    # Highlight fork band
    ax_tok.axvspan(t_fork - 0.45, t_fork + 0.45, color="#f2dfe5", zorder=0)

    bars = ax_tok.bar(
        xs,
        ent,
        width=0.62,
        color=[WINE if i == t_fork else "#b7b2bd" for i in xs],
        edgecolor="none",
        zorder=2,
        label="Next-token entropy H (nats)",
    )

    for i, val in enumerate(ent):
        ax_tok.text(
            i,
            val + 0.08,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if i == t_fork else "normal",
            color=WINE if i == t_fork else MUTED,
        )

    ax_tok.set_xticks(xs)
    ax_tok.set_xticklabels([t.strip() for t in tokens], fontsize=9.5)
    for idx, lbl in enumerate(ax_tok.get_xticklabels()):
        if idx == t_fork:
            lbl.set_color(WINE)
            lbl.set_fontweight("bold")
        elif idx < t_fork:
            lbl.set_color(INK)
        else:
            lbl.set_color(MUTED)

    ax_tok.set_ylabel("Next-token entropy H (nats)", fontsize=9.5, color=INK)
    ax_tok.set_ylim(0, 3.55)
    ax_tok.set_xlim(-0.7, len(tokens) - 0.3)
    ax_tok.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7, zorder=1)

    # Secondary axis: output-Fisher top eigenvalue
    ax_f = ax_tok.twinx()
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["left"].set_visible(False)
    ax_f.spines["bottom"].set_visible(False)
    ax_f.spines["right"].set_color(RULE)
    ax_f.plot(
        xs,
        fish,
        color="#2d5f8b",
        marker="o",
        markersize=5,
        linewidth=1.7,
        zorder=4,
        label="Output-Fisher top eigenvalue λ₁",
    )
    ax_f.set_ylabel("Output-Fisher λ₁ (state sensitivity)", fontsize=9.5, color="#2d5f8b")
    ax_f.tick_params(axis="y", colors="#2d5f8b", labelsize=8.5)
    ax_f.set_ylim(0, 0.46)

    # Fork annotation
    ax_tok.annotate(
        f"FORK (t = {t_fork})\nH = {ent[t_fork]:.2f} nats · λ₁ = {fish[t_fork]:.3f}\nspecificity = {B['specificity']:.3f}",
        xy=(t_fork + 0.32, ent[t_fork] - 0.05),
        xytext=(t_fork + 1.55, 2.75),
        fontsize=8.5,
        color=WINE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL, edgecolor=WINE, linewidth=1.0),
        arrowprops=dict(arrowstyle="->", color=WINE, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    ax_tok.set_title(
        "A. Where the 64 translations branch: token-by-token entropy and Fisher sensitivity along the reference",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=10,
    )

    # Combined legend
    h1, l1 = ax_tok.get_legend_handles_labels()
    h2, l2 = ax_f.get_legend_handles_labels()
    ax_tok.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=True,
        facecolor=PANEL,
        edgecolor=RULE,
        fontsize=8.5,
    )

    # ------------------------------------------------------------------
    # PANEL B (Bottom-left): 64 translations in semantic direction space
    # ------------------------------------------------------------------
    ax_pca = fig.add_subplot(gs[1, 0])
    style_ax(ax_pca)

    labels = np.array(S["labels"])
    logp = np.array(S["logp"])
    kept_mask = labels != -1

    # Centre on kept samples and project onto top-2 PCs
    mu = emb[kept_mask].mean(axis=0)
    X_all = emb - mu
    U, s, Vt = np.linalg.svd(X_all[kept_mask], full_matrices=False)
    coords = X_all @ Vt[:2].T
    var_expl = (s[:2] ** 2) / (s ** 2).sum()

    # Crosshairs at origin (shared prefix / mean)
    ax_pca.axhline(0, color=RULE, lw=0.9, ls="--", zorder=1)
    ax_pca.axvline(0, color=RULE, lw=0.9, ls="--", zorder=1)
    ax_pca.scatter([0], [0], s=55, color=INK, zorder=5)

    # Dropped 7 outliers
    drop_xy = coords[~kept_mask]
    ax_pca.scatter(
        drop_xy[:, 0],
        drop_xy[:, 1],
        s=36,
        marker="x",
        color="#a9a4af",
        linewidths=1.2,
        zorder=2,
        label="Dropped tail (7 / 64)",
    )

    # Plot each reading cluster + direction vector from origin to centroid
    # Explicit non-overlapping callout positions in data coordinates
    callout_pos = {
        1: (-0.52, 0.46, "left"),    # magic · fairy · genie
        0: (-0.52, -0.30, "left"),   # possessed · gift · rhythm
        4: (-0.12, 0.58, "center"),  # charm · charisma · whimsy
        2: (0.28, 0.14, "left"),     # duende · flame · spark
        3: (0.14, -0.52, "left"),    # spirit · soul
    }

    for r in S["readings"]:
        meta = READING_META[r]
        mask = labels == r
        pts = coords[mask]
        c_xy = pts.mean(axis=0)
        n_r = int(mask.sum())
        pct = 100.0 * n_r / int(kept_mask.sum())
        med_lp = float(np.median(logp[mask]))

        ax_pca.scatter(
            pts[:, 0],
            pts[:, 1],
            s=46,
            color=meta["color"],
            alpha=0.72,
            edgecolors=PANEL,
            linewidths=0.6,
            zorder=3,
        )

        # Direction arrow from origin to reading centroid
        arr = FancyArrowPatch(
            (0, 0),
            (c_xy[0], c_xy[1]),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2.2,
            color=meta["color"],
            zorder=4,
        )
        ax_pca.add_patch(arr)
        ax_pca.scatter(
            [c_xy[0]],
            [c_xy[1]],
            s=75,
            color=meta["color"],
            edgecolors=INK,
            linewidths=0.9,
            zorder=5,
        )

        tx, ty, ha = callout_pos[r]
        ax_pca.annotate(
            f"{meta['name']}\n{n_r}/57 ({pct:.1f}%) · med log p = {med_lp:.1f}",
            xy=(c_xy[0], c_xy[1]),
            xytext=(tx, ty),
            fontsize=8.0,
            fontweight="bold",
            color=meta["color"],
            ha=ha,
            va="center",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PANEL, edgecolor=meta["color"], alpha=0.95, lw=0.85),
            arrowprops=dict(arrowstyle="-", color=meta["color"], lw=0.9, alpha=0.75, shrinkA=4, shrinkB=6),
            zorder=6,
        )

    ax_pca.set_xlabel(f"PC1 ({var_expl[0]*100:.1f}% variance of centred 768-d embeddings)", fontsize=9, color=INK)
    ax_pca.set_ylabel(f"PC2 ({var_expl[1]*100:.1f}% variance)", fontsize=9, color=INK)
    ax_pca.set_xlim(-0.58, 0.72)
    ax_pca.set_ylim(-0.64, 0.72)
    ax_pca.legend(loc="lower left", frameon=True, facecolor=PANEL, edgecolor=RULE, fontsize=8)
    ax_pca.set_title(
        "B. Five translation directions radiating from the fork (all 64 samples)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=10,
    )

    # ------------------------------------------------------------------
    # PANEL C (Bottom-right): Steering along translation directions
    # 3 stacked mini-axes for top 3 readings: R1 (magic), R3 (spirit), R2 (duende)
    # ------------------------------------------------------------------
    gs_steer = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.38)
    targets = [1, 3, 2]
    kls = [0.05, 0.1, 0.25, 0.5]
    x_pos = np.arange(len(kls))

    for idx, target_r in enumerate(targets):
        ax_s = fig.add_subplot(gs_steer[idx, 0])
        style_ax(ax_s)
        meta = READING_META[target_r]

        # Collect natural-cursor rows for target A = target_r
        by_B = defaultdict(dict)
        base_share = None
        for row in steer:
            if row["variant"] == "natural" and row["mode"] == "cursor" and row["A"] == target_r:
                by_B[row["B"]][row["kl"]] = (row["success_rate"] * 100.0, row["alpha"])
                base_share = row["baseline_share_A"] * 100.0

        # Baseline dashed line
        ax_s.axhline(base_share, color=MUTED, ls="--", lw=1.1, zorder=1)
        ax_s.text(
            0.03,
            base_share + 5.0 if base_share < 50 else base_share - 14.0,
            f"unsteered: {base_share:.0f}%",
            fontsize=7.8,
            color=MUTED,
            ha="left",
        )

        styles = [
            dict(ls="-", marker="o", lw=2.0, alpha=0.95),
            dict(ls=":", marker="s", lw=1.8, alpha=0.85),
        ]
        for j, (src_B, kl_map) in enumerate(sorted(by_B.items())):
            ys = [kl_map[k][0] for k in kls]
            alphas = [kl_map[k][1] for k in kls]
            st = styles[j % len(styles)]
            ax_s.plot(
                x_pos,
                ys,
                color=meta["color"] if j == 0 else "#5a5562",
                markersize=5.5,
                label=f"from {READING_META[src_B]['short']} → {meta['short']}",
                zorder=3,
                **st,
            )
            # Label peak point with clean non-overlapping offset
            best_i = int(np.argmax(ys))
            y_off = 9 if j == 0 and ys[best_i] <= 85 else (-17 if j == 1 else -15)
            if target_r == 3 and j == 0:
                best_i = 1  # also 100% at KL=0.1, avoids colliding with j=1 at x=0
                y_off = 8
            elif target_r == 3 and j == 1:
                y_off = -17
            x_align = "left" if best_i == 0 else ("right" if best_i == len(kls) - 1 else "center")
            ax_s.annotate(
                f"{ys[best_i]:.0f}% (α={alphas[best_i]:.0f})",
                xy=(best_i, ys[best_i]),
                xytext=(best_i, min(max(ys[best_i] + y_off, 6), 107)),
                ha=x_align,
                fontsize=7.6,
                fontweight="bold" if j == 0 else "normal",
                color=meta["color"] if j == 0 else "#5a5562",
            )

        ax_s.set_ylim(-5, 118)
        ax_s.set_yticks([0, 50, 100])
        ax_s.set_yticklabels(["0%", "50%", "100%"])
        ax_s.set_xticks(x_pos)
        if idx == len(targets) - 1:
            ax_s.set_xticklabels([f"{k:g}" for k in kls], fontsize=8.5)
            ax_s.set_xlabel("Steering strength: mean per-token KL along reference (nats)", fontsize=9, color=INK)
        else:
            ax_s.set_xticklabels([])

        ax_s.set_ylabel(f"→ {meta['short']}", fontsize=9, fontweight="bold", color=meta["color"])
        ax_s.grid(axis="y", color=RULE, lw=0.6, alpha=0.7, zorder=0)
        ax_s.legend(loc="lower right", frameon=True, facecolor=PANEL, edgecolor=RULE, fontsize=7.5)

        if idx == 0:
            ax_s.set_title(
                "C. Steering along Fisher-whitened directions (cursor mode)",
                loc="left",
                fontsize=11,
                fontweight="bold",
                color=INK,
                pad=10,
            )

    fig.text(
        0.055,
        0.018,
        "Data: untranslatables/results/explorer/g3_4b_pt/{samples,branches,steering}/duende_flamenco.*  ·  "
        "Steering adds α·v at layer 17 from fork token t=7 onward; fluent success = lands in target reading within 1.5 nats/tok of reading median.",
        fontsize=8,
        color=MUTED,
    )

    png_path = OUT / "duende_directions_numbers.png"
    svg_path = OUT / "duende_directions_numbers.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    print("Wrote", png_path)
    print("Wrote", svg_path)


if __name__ == "__main__":
    main()
