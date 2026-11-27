"""Generate Dad-ready figures from Track B unalignability_diag.json across Gemma 3 4B, 12B, 27B.

Produces:
  - figures_for_dad/5_untranslatability_vs_ignorance.png
  - figures_for_dad/6_scale_trajectories_4b_to_27b.png
  - figures_for_dad/7_cycle_consistency_and_rig.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ut.common import ROOT

GROUP_STYLE = {
    "ordinary": {"color": "#186456", "marker": "o", "label": "Ordinary equivalents (camisa, domani, tristeza)"},
    "stylistic": {"color": "#23527c", "marker": "s", "label": "Stylistic / polysemous (volverán, bank)"},
    "untranslatable": {"color": "#a63227", "marker": "D", "label": "Genuine untranslatables (duende, saudade, sprezzatura...)"},
    "ignorance": {"color": "#965710", "marker": "^", "label": "Model ignorance (Romanesco, Welsh, Basque, Yami)"},
}


def _load_diag(model_key: str) -> dict | None:
    p = ROOT / "results" / "explorer" / model_key / "unalignability_diag.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plot_all(models: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {m: _load_diag(m) for m in models}
    loaded = {k: v for k, v in loaded.items() if v is not None}
    if not loaded:
        raise FileNotFoundError(f"No unalignability_diag.json found for {models}")

    plt.rcParams.update({
        "figure.facecolor": "#fafcfa",
        "axes.facecolor": "#fafcfa",
        "axes.edgecolor": "#b7c6b9",
        "axes.labelcolor": "#17231e",
        "xtick.color": "#465850",
        "ytick.color": "#465850",
        "text.color": "#17231e",
        "font.family": "sans-serif",
        "font.size": 10.5,
    })

    # 1. Quadrant Map on largest available model (27B -> 12B -> 4B)
    primary_key = loaded and list(loaded.keys())[-1]
    rows = loaded[primary_key]["rows"]

    fig, ax = plt.subplots(figsize=(10.2, 6.8), dpi=180)
    ax.grid(True, linestyle=":", color="#d3ddd4", linewidth=0.9, zorder=0)

    xs = [r["source_sharpness"] for r in rows]
    ys = [0.55 * r["en_leftover"] + 0.45 * r["en_semantic_spread"] for r in rows]
    x_med = float(np.median(xs))
    y_med = float(np.median(ys))
    ax.axvline(x_med, color="#b7c6b9", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(y_med, color="#b7c6b9", linestyle="--", linewidth=1.0, zorder=1)

    for grp, st in GROUP_STYLE.items():
        grp_rows = [r for r in rows if r["group"] == grp]
        if not grp_rows:
            continue
        gx = [r["source_sharpness"] for r in grp_rows]
        gy = [0.55 * r["en_leftover"] + 0.45 * r["en_semantic_spread"] for r in grp_rows]
        sizes = [90 + 180 * (1.0 - r["rig_proxy"]) for r in grp_rows]
        ax.scatter(
            gx, gy, s=sizes, c=st["color"], marker=st["marker"],
            edgecolors="#fafcfa", linewidths=1.2, alpha=0.92, label=st["label"], zorder=3,
        )
        for r, rx, ry in zip(grp_rows, gx, gy):
            ax.annotate(
                r["word"],
                (rx, ry),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9.2,
                fontweight="semibold" if grp == "untranslatable" else "normal",
                color=st["color"],
                zorder=4,
            )

    ax.set_xlabel("Source-Language Conceptual Sharpness (contextual micro-shifts × fluency × semantic precision)", labelpad=8)
    ax.set_ylabel("Cross-Lingual Dispersion & English Leftover", labelpad=8)
    ax.set_title(
        f"Untranslatability vs. Model Ignorance ({primary_key})\n"
        "Marker size = Directional Steerability (inverse RIG: larger = cleaner steering)",
        loc="left", fontsize=11.8, fontweight="bold", pad=12,
    )
    ax.legend(frameon=True, facecolor="#f2f6f1", edgecolor="#d3ddd4", fontsize=8.8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "5_untranslatability_vs_ignorance.png", dpi=180)
    plt.close(fig)

    # 2. Scale Trajectories (4B -> 27B) if >= 2 models available
    keys_ordered = list(loaded.keys())
    if len(keys_ordered) >= 2:
        m0, m1 = keys_ordered[0], keys_ordered[-1]
        r0 = {r["id"]: r for r in loaded[m0]["rows"]}
        r1 = {r["id"]: r for r in loaded[m1]["rows"]}
        common = [k for k in r0 if k in r1]

        fig, ax = plt.subplots(figsize=(10.0, 6.4), dpi=180)
        ax.grid(True, linestyle=":", color="#d3ddd4", linewidth=0.9, zorder=0)
        for k in common:
            a, b = r0[k], r1[k]
            st = GROUP_STYLE[a["group"]]
            ax_x, ax_y = a["source_sharpness"], 0.55 * a["en_leftover"] + 0.45 * a["en_semantic_spread"]
            bx_x, bx_y = b["source_sharpness"], 0.55 * b["en_leftover"] + 0.45 * b["en_semantic_spread"]
            ax.annotate(
                "", xy=(bx_x, bx_y), xytext=(ax_x, ax_y),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35", color=st["color"], lw=1.6, alpha=0.8),
                zorder=2,
            )
            ax.scatter([ax_x], [ax_y], s=42, color=st["color"], alpha=0.4, zorder=2)
            ax.scatter([bx_x], [bx_y], s=85, color=st["color"], marker=st["marker"], edgecolors="#fafcfa", lw=1.0, zorder=3)
            ax.annotate(b["word"], (bx_x, bx_y), xytext=(5, 4), textcoords="offset points", fontsize=8.8, color=st["color"], zorder=4)

        for grp, st in GROUP_STYLE.items():
            ax.scatter([], [], c=st["color"], marker=st["marker"], s=70, label=st["label"])
        ax.set_xlabel("Source-Language Conceptual Sharpness")
        ax.set_ylabel("Cross-Lingual Dispersion & English Leftover")
        ax.set_title(
            f"Scale Trajectories: {m0} → {m1}\n"
            "Genuine untranslatables sharpen rightward while remaining dispersed; ignorance lags behind",
            loc="left", fontsize=11.8, fontweight="bold", pad=12,
        )
        ax.legend(frameon=True, facecolor="#f2f6f1", edgecolor="#d3ddd4", fontsize=8.8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / "6_scale_trajectories_4b_to_27b.png", dpi=180)
        plt.close(fig)

    # 3. Hsieh & Lian (2610.08303) Cycle-Consistency (CC) vs Directional RIG
    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=180)
    ax.grid(True, linestyle=":", color="#d3ddd4", linewidth=0.9, zorder=0)
    for grp, st in GROUP_STYLE.items():
        grp_rows = [r for r in rows if r["group"] == grp]
        if not grp_rows:
            continue
        gx = [r["cycle_consistency_cc"] for r in grp_rows]
        gy = [r["rig_proxy"] for r in grp_rows]
        ax.scatter(gx, gy, s=110, c=st["color"], marker=st["marker"], edgecolors="#fafcfa", lw=1.2, label=st["label"], zorder=3)
        for r, rx, ry in zip(grp_rows, gx, gy):
            ax.annotate(r["word"], (rx, ry), xytext=(6, 4), textcoords="offset points", fontsize=9.0, color=st["color"], zorder=4)

    ax.set_xlabel("Cycle-Consistency Distortion CC = 1 − Spearman(d_src, d_roundtrip)  [Hsieh & Lian 2026]", labelpad=8)
    ax.set_ylabel("Directional RIG Proxy (higher = harder to steer)", labelpad=8)
    ax.set_title(
        f"Testing Hsieh & Lian (arXiv:2610.08303) on {primary_key}\n"
        "Steerable untranslatables (high CC, lower RIG) vs. low-resource ignorance (high RIG)",
        loc="left", fontsize=11.8, fontweight="bold", pad=12,
    )
    ax.legend(
        frameon=True, facecolor="#f2f6f1", edgecolor="#d3ddd4", fontsize=8.5,
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "7_cycle_consistency_and_rig.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Track B figures -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["g3_4b_pt", "g3_12b_pt", "g3_27b_pt"])
    ap.add_argument("--out", default=str(ROOT.parent / "figures_for_dad"))
    args = ap.parse_args()
    plot_all(args.models, Path(args.out))


if __name__ == "__main__":
    main()
