"""Live streaming, fork-burst probing, and offline replay engines for the Spectral Translation Workbench.

Two engines share a clean JSON-serialisable contract for the HTTP/SSE server and browser UI:

  ReplayEngine(results_dir)
    Loads saved Phase-1 explorer runs (`samples/`, `branches/`, `steering/`), builds a centred 2D PCA
    compass per text, auto-extracts reading keywords, and maps any cursor `(x, y)` to smooth reading
    weights plus the closest real steered or sampled translation.

  LiveTranslationEngine(model_key="tiny", layer=None, global_axes_path=None)
    Lazy-loads a causal LM (or uses injected `(tok, model)`), streams greedy KV-cached translations
    with live next-token entropy and optional output-Fisher top eigenvalue, probes any clicked token
    cursor with a fast ancestral burst + gradient clustering to construct a local 2D spectral plane,
    and steers continuations in real time via `CursorHook`.
"""
from __future__ import annotations

import collections
import json
import math
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch

from blind.fisher import seq_logp
from explorer import readings as R
from explorer.directions import alpha_for, calibrate, grads_at
from explorer.frame import STOPS, build_prompt, cut, load
from explorer.sample import generate, steered
from explorer.steer import CursorHook
from ut import arch
from ut.common import ROOT

DEFAULT_RESULTS_DIR = ROOT / "results" / "explorer" / "g3_4b_pt"

# Warm editorial palette aligned with figures_for_dad/duende_five_ways.html
READING_PALETTE = [
    "#8a2343",  # wine (primary)
    "#2c5d72",  # slate teal
    "#9c6520",  # warm ochre
    "#476b4c",  # moss green
    "#6b467a",  # plum
    "#a84a36",  # terracotta
    "#4c5c8a",  # indigo
    "#7a6a38",  # olive
]

DEFAULT_GLOBAL_AXES = [
    {"id": "literal_vs_idiomatic", "alias": "literal_idiomatic", "neg": "Literal", "pos": "Idiomatic"},
    {"id": "plain_vs_poetic", "alias": "plain_poetic", "neg": "Plain", "pos": "Poetic"},
    {"id": "foreignising_vs_domesticating", "alias": "foreign_domestic", "neg": "Foreignising", "pos": "Domesticating"},
    {"id": "concise_vs_glossed", "alias": "concise_glossed", "neg": "Concise", "pos": "Glossed"},
]

_AXIS_ALIASES = {
    "literal_idiomatic": "literal_vs_idiomatic",
    "plain_poetic": "plain_vs_poetic",
    "foreign_domestic": "foreignising_vs_domesticating",
    "concise_glossed": "concise_vs_glossed",
    "literal_vs_idiomatic": "literal_idiomatic",
    "plain_vs_poetic": "plain_poetic",
    "foreignising_vs_domesticating": "foreign_domestic",
    "concise_vs_glossed": "concise_glossed",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "this", "that", "these", "those", "very", "much", "lot", "really", "so", "as", "when", "while",
    "there", "here", "not", "no", "yes", "what", "which", "who", "whom", "if", "then", "than", "into",
    "about", "up", "out", "just", "also", "too", "more", "most", "some", "any", "all", "one", "two",
    "like", "love", "things", "thing", "good", "got", "get", "gets", "make", "makes", "side",
    "s", "t", "re", "ve", "ll", "d", "m", "spanish", "english",
}


def _extract_reading_keywords(
    texts: list[str],
    labels: list[int] | np.ndarray,
    reading_ids: list[int],
    top: int = 3,
) -> dict[int, list[str]]:
    """Auto-extract distinctive human-friendly keywords for each reading cluster."""
    per_r: dict[int, collections.Counter] = {int(r): collections.Counter() for r in reading_ids}
    total: collections.Counter = collections.Counter()
    doc_freq: collections.Counter = collections.Counter()
    n_kept = 0
    for txt, lab in zip(texts, labels):
        lab_i = int(lab)
        if lab_i not in per_r:
            continue
        n_kept += 1
        first_line = txt.split("\n")[0].lower()
        words = set(re.findall(r"[a-zA-Záéíóúüñ]+(?:'[a-z]+)?", first_line))
        for w in words:
            if len(w) <= 1 or w in _STOPWORDS:
                continue
            per_r[lab_i][w] += 1
            total[w] += 1
            doc_freq[w] += 1

    sizes = {int(r): sum(1 for lab in labels if int(lab) == int(r)) for r in reading_ids}
    out: dict[int, list[str]] = {}
    for r in reading_ids:
        ri = int(r)
        nr = max(sizes[ri], 1)
        n_other = max(n_kept - nr, 1)
        scored: list[tuple[float, int, str]] = []
        for w, cnt in per_r[ri].items():
            if n_kept >= 4 and doc_freq[w] / n_kept >= 0.78:
                continue
            p_in = cnt / nr
            p_out = (total[w] - cnt) / n_other
            lift = p_in - p_out
            if lift > 0.04:
                scored.append((lift * (0.5 + p_in), cnt, w))
        scored.sort(reverse=True)
        kws = [w for _, _, w in scored[:top]]
        if not kws:
            kws = [w for w, _ in per_r[ri].most_common(top)]
        if not kws:
            # Final fallback: short snippet from the first member text
            first_mem = next((t.split("\n")[0].strip() for t, l in zip(texts, labels) if int(l) == ri and t.strip()), "")
            kws = [first_mem[:24]] if first_mem else [f"reading {ri}"]
        out[ri] = kws
    return out


def _rbf_reading_shares(
    x: float,
    y: float,
    reading_ids: list[int],
    anchors: np.ndarray,
    sigma: float | None = None,
) -> dict[int, float]:
    """Smooth Gaussian-RBF softmax weights over 2D reading anchors.

    Strictly maximises reading `r` whenever `(x, y)` sits at or near `anchors[r]`.
    """
    if len(reading_ids) == 0:
        return {}
    if len(reading_ids) == 1:
        return {int(reading_ids[0]): 1.0}
    p = np.array([float(x), float(y)], dtype=np.float64)
    dists = np.linalg.norm(anchors - p[None, :], axis=1)
    if sigma is None:
        pdist = [
            float(np.linalg.norm(anchors[i] - anchors[j]))
            for i in range(len(anchors))
            for j in range(i + 1, len(anchors))
        ]
        med_d = float(np.median(pdist)) if pdist else 0.8
        sigma = max(0.22, min(0.45, med_d * 0.38))
    logits = -(dists ** 2) / (2.0 * sigma * sigma)
    logits -= logits.max()
    w = np.exp(logits)
    w /= max(float(w.sum()), 1e-30)
    return {int(r): float(wi) for r, wi in zip(reading_ids, w)}


def _power_iter_fisher_top(W: torch.Tensor, p: torch.Tensor, iters: int = 3, seed: int = 0) -> float:
    """Top eigenvalue of output Fisher W^T (diag(p) - p p^T) W via matrix-vector products."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(W.shape[1], generator=g).to(W.device)
    v = v / v.norm().clamp_min(1e-30)
    lam = 0.0
    for _ in range(max(1, int(iters))):
        Wv = (W @ v.to(W.dtype)).float()
        u = p * Wv - p * (p @ Wv)
        Fv = (W.T @ u.to(W.dtype)).float()
        lam = float(Fv @ v)
        v = Fv / Fv.norm().clamp_min(1e-30)
    return lam


def _load_global_axes(path: str | Path | None, layer: int | None = None) -> tuple[dict[str, np.ndarray], Any]:
    """Load optional precomputed global steering axes from GlobalAxisBank .npz, flat .npz, .pt, or .json.

    Returns ``(flat_axes_for_layer, bank_or_None)``.
    """
    if not path:
        return {}, None
    p = Path(path)
    if not p.exists():
        return {}, None
    out: dict[str, np.ndarray] = {}
    bank = None
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        if "meta" in data.files or "__meta__" in data.files:
            from explorer.global_axes import GlobalAxisBank
            bank = GlobalAxisBank.load(p)
            for ax_name, by_layer in bank.vectors.items():
                if not by_layer:
                    continue
                chosen_l = layer if (layer is not None and layer in by_layer) else sorted(by_layer)[len(by_layer) // 2]
                scale = float(bank.scales.get(ax_name, 1.0))
                out[ax_name] = np.asarray(by_layer[chosen_l], dtype=np.float32) * scale
            return out, bank
        for k in data.files:
            arr = np.asarray(data[k], dtype=np.float32)
            if arr.ndim == 1:
                out[k] = arr
    elif p.suffix in (".pt", ".pth"):
        obj = torch.load(p, map_location="cpu", weights_only=True)
        src = obj.get("axes", obj) if isinstance(obj, dict) else {}
        for k, v in src.items():
            arr = np.asarray(v.float().cpu().numpy() if torch.is_tensor(v) else v, dtype=np.float32)
            if arr.ndim == 1:
                out[str(k)] = arr
    elif p.suffix == ".json":
        obj = json.loads(p.read_text())
        src = obj.get("axes", obj) if isinstance(obj, dict) else {}
        for k, v in src.items():
            if isinstance(v, list):
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1:
                    out[str(k)] = arr
    return out, bank


class ReplayEngine:
    """Zero-GPU replay engine backed by saved `results/explorer/<model>/` runs."""

    def __init__(self, results_dir: str | Path | None = None):
        self.results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        self._records: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._load_all()

    def _load_all(self) -> None:
        samples_dir = self.results_dir / "samples"
        if not samples_dir.exists():
            return
        for sp in sorted(samples_dir.glob("*.json")):
            tid = sp.stem
            emb_path = samples_dir / f"{tid}_emb.npy"
            if not emb_path.exists():
                continue
            self._records[tid] = self._build_text_record(tid, sp, emb_path)
            self._order.append(tid)
        # Put duende_flamenco first if present for the best default landing experience
        if "duende_flamenco" in self._order:
            self._order.remove("duende_flamenco")
            self._order.insert(0, "duende_flamenco")

    def _build_text_record(self, tid: str, sp: Path, emb_path: Path) -> dict[str, Any]:
        S = json.loads(sp.read_text())
        E = np.load(emb_path).astype(np.float64)
        labels = np.asarray(S["labels"], dtype=int)
        reading_ids = [int(r) for r in S["readings"]]
        texts = list(S["texts"])
        logp = np.asarray(S["logp"], dtype=np.float64)
        lens = np.maximum(np.asarray(S.get("lens", [1] * len(texts)), dtype=np.float64), 1.0)
        lpt = logp / lens
        shares_raw = {int(k): float(v) for k, v in S.get("shares", {}).items()}

        keep = labels >= 0
        mu = E[keep].mean(axis=0) if keep.any() else E.mean(axis=0)
        Z = R.centre(E, mu)
        Z_keep = Z[keep] if keep.any() else Z

        # Top-2 PCA plane on kept centred embeddings
        if len(Z_keep) >= 2:
            _, svals, Vt = np.linalg.svd(Z_keep, full_matrices=False)
            if Vt.shape[0] >= 2:
                axes = Vt[:2]
                var_expl = float((svals[:2] ** 2).sum() / max(float((svals ** 2).sum()), 1e-30))
            else:
                axes = np.vstack([Vt[0], np.zeros_like(Vt[0])])
                var_expl = 1.0
        else:
            axes = np.zeros((2, Z.shape[1]), dtype=np.float64)
            axes[0, 0] = 1.0
            if Z.shape[1] > 1:
                axes[1, 1] = 1.0
            var_expl = 1.0

        proj = Z @ axes.T  # (n, 2)
        if reading_ids:
            raw_anchors = np.array([proj[labels == r].mean(axis=0) for r in reading_ids], dtype=np.float64)
            scale = max(float(np.linalg.norm(raw_anchors, axis=1).max()), 1e-8)
            anchors = raw_anchors / scale
            coords = proj / scale
        else:
            anchors = np.zeros((0, 2), dtype=np.float64)
            scale = max(float(np.linalg.norm(proj, axis=1).max()), 1e-8)
            coords = proj / scale

        # Branch timeline
        bp = self.results_dir / "branches" / f"{tid}.json"
        if bp.exists():
            B = json.loads(bp.read_text())
        else:
            B = {
                "ref": texts[0] if texts else "",
                "ref_tokens": [],
                "entropy": [],
                "fisher_top": [],
                "t": 0,
                "specificity": None,
                "why": "branch file not available",
            }
        ref_tokens = list(B.get("ref_tokens", []))
        fork_t = int(B.get("t", 0))
        fork_token = ref_tokens[fork_t] if 0 <= fork_t < len(ref_tokens) else ""
        prefix_text = "".join(ref_tokens[:fork_t]).strip()

        # Steering trajectories & baseline
        stp = self.results_dir / "steering" / f"{tid}.jsonl"
        steering_rows = [json.loads(line) for line in stp.read_text().splitlines() if line.strip()] if stp.exists() else []
        base_p = self.results_dir / "steering" / f"{tid}_baseline.json"
        baseline = json.loads(base_p.read_text()) if base_p.exists() else {"assigned": [], "texts": []}

        # Keywords & reading metadata
        kw_map = _extract_reading_keywords(texts, labels, reading_ids, top=3)
        reading_meta: list[dict[str, Any]] = []
        anchor_map: dict[int, np.ndarray] = {}
        color_map: dict[int, str] = {}
        label_map: dict[int, str] = {}

        # Per-reading steering lift stats (natural/cursor)
        lift_by_r: dict[int, dict[str, Any]] = {}
        for r in reading_ids:
            nc_rows = [row for row in steering_rows if row.get("A") == r and row.get("variant") == "natural" and row.get("mode") == "cursor"]
            if nc_rows:
                best_row = max(nc_rows, key=lambda row: row.get("success_rate", 0.0))
                lift_by_r[r] = {
                    "baseline_share": round(float(nc_rows[0].get("baseline_share_A", 0.0)), 4),
                    "best_success_rate": round(float(best_row.get("success_rate", 0.0)), 4),
                    "best_kl": float(best_row.get("kl", 0.0)),
                    "best_example": best_row["texts"][0] if best_row.get("texts") else "",
                }

        for idx, r in enumerate(reading_ids):
            color = READING_PALETTE[idx % len(READING_PALETTE)]
            color_map[r] = color
            kws = kw_map.get(r, [f"reading {r}"])
            rlabel = " · ".join(kws)
            label_map[r] = rlabel
            anchor_map[r] = anchors[idx]
            mem_idx = np.where(labels == r)[0]
            mem_sorted = sorted(mem_idx.tolist(), key=lambda i: float(lpt[i]), reverse=True)
            seen_txt: set[str] = set()
            exemplars: list[dict[str, Any]] = []
            for i in mem_sorted:
                t_clean = texts[i].split("\n")[0].strip()
                if t_clean in seen_txt:
                    continue
                seen_txt.add(t_clean)
                exemplars.append({
                    "i": int(i),
                    "text": texts[i],
                    "logp_per_tok": round(float(lpt[i]), 4),
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                })
                if len(exemplars) >= 6:
                    break
            reading_meta.append({
                "id": int(r),
                "label": rlabel,
                "keywords": kws,
                "share": round(float(shares_raw.get(r, len(mem_idx) / max(int(keep.sum()), 1))), 4),
                "count": int(len(mem_idx)),
                "median_logp": round(float(np.median(lpt[mem_idx])) if len(mem_idx) else 0.0, 4),
                "color": color,
                "anchor": [round(float(anchors[idx, 0]), 4), round(float(anchors[idx, 1]), 4)],
                "exemplars": exemplars,
                "steering_lift": lift_by_r.get(r),
            })

        dots = [
            {
                "i": int(i),
                "reading": int(labels[i]),
                "kept": bool(labels[i] >= 0),
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "text": texts[i],
                "logp": round(float(logp[i]), 4),
                "logp_per_tok": round(float(lpt[i]), 4),
                "color": color_map.get(int(labels[i]), "#b5b0ba"),
            }
            for i in range(len(texts))
        ]

        # Build compact steering trajectory points for UI & replay lookup
        steering_points: list[dict[str, Any]] = []
        for row in steering_rows:
            A, Br = int(row["A"]), int(row["B"])
            if A not in anchor_map or Br not in anchor_map:
                continue
            # Best fluent text in this steering row
            txts = row.get("texts", [])
            ass = row.get("assigned", [])
            flu = row.get("fluency", [])
            succ = row.get("success", [])
            best_i = 0
            if txts:
                if succ and succ[0]:
                    best_i = 0
                else:
                    good = [i for i in range(len(txts)) if i < len(succ) and succ[i]]
                    if not good:
                        good = [i for i in range(len(txts)) if i < len(ass) and int(ass[i]) == A]
                    if good:
                        best_i = max(good, key=lambda i: flu[i] if i < len(flu) else -1e9)
                    elif flu:
                        best_i = int(np.argmax(flu))
            steering_points.append({
                "A": A,
                "B": Br,
                "variant": row.get("variant", "natural"),
                "mode": row.get("mode", "cursor"),
                "kl": float(row.get("kl", 0.0)),
                "kl_level": int(row.get("kl_level", 0)),
                "alpha": round(float(row.get("alpha", 0.0)), 4),
                "success_rate": round(float(row.get("success_rate", 0.0)), 4),
                "baseline_share_A": round(float(row.get("baseline_share_A", 0.0)), 4),
                "text": txts[best_i] if txts else "",
                "fluency": round(float(flu[best_i]), 4) if best_i < len(flu) else 0.0,
                "assigned": int(ass[best_i]) if best_i < len(ass) else A,
                "greedy_success": bool(row.get("greedy_success", False)),
            })

        default_text = B.get("ref") or (baseline["texts"][0] if baseline.get("texts") else (texts[0] if texts else ""))
        default_fluency = float(np.median(lpt[keep])) if keep.any() else 0.0

        bundle = {
            "id": tid,
            "text": S["text"],
            "kind": S.get("kind", "hard"),
            "n_samples": len(texts),
            "n_kept": int(keep.sum()),
            "n_readings": len(reading_ids),
            "entropy_bits": round(float(S.get("entropy_bits", 0.0)), 4),
            "silhouette": round(float(S.get("silhouette", 0.0)), 4),
            "variance_explained": round(var_expl, 4),
            "ref_text": B.get("ref", default_text),
            "ref_tokens": ref_tokens,
            "entropy": [round(float(v), 4) for v in B.get("entropy", [])],
            "fisher_top": [round(float(v), 4) for v in B.get("fisher_top", [])],
            "fork_t": fork_t,
            "fork_token": fork_token,
            "specificity": round(float(B["specificity"]), 4) if B.get("specificity") is not None else None,
            "why": B.get("why", ""),
            "prefix_text": prefix_text,
            "readings": reading_meta,
            "anchors": {str(r): [round(float(anchors[i, 0]), 4), round(float(anchors[i, 1]), 4)] for i, r in enumerate(reading_ids)},
            "dots": dots,
            "steering_points": steering_points,
            "baseline_texts": baseline.get("texts", [])[:6],
            "default_translation": default_text,
            "default_fluency": round(default_fluency, 4),
        }

        return {
            "bundle": bundle,
            "reading_ids": reading_ids,
            "anchors_np": anchors,
            "anchor_map": anchor_map,
            "label_map": label_map,
            "coords_np": coords,
            "labels_np": labels,
            "texts": texts,
            "lpt_np": lpt,
            "steering_points": steering_points,
            "default_text": default_text,
            "default_fluency": default_fluency,
        }

    def list_texts(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tid in self._order:
            b = self._records[tid]["bundle"]
            out.append({
                "id": b["id"],
                "text": b["text"],
                "kind": b["kind"],
                "n_samples": b["n_samples"],
                "n_readings": b["n_readings"],
                "fork_t": b["fork_t"],
                "fork_token": b["fork_token"],
                "entropy_bits": b["entropy_bits"],
            })
        return out

    def get_text_bundle(self, text_id: str) -> dict[str, Any]:
        if text_id not in self._records:
            raise KeyError(f"Unknown text_id: {text_id!r}. Available: {self._order}")
        return self._records[text_id]["bundle"]

    def steer_replay(
        self,
        text_id: str,
        x: float,
        y: float,
        global_coords: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute smooth reading weights at `(x, y)` and return the best real saved translation."""
        if text_id not in self._records:
            raise KeyError(f"Unknown text_id: {text_id!r}")
        rec = self._records[text_id]
        fx, fy = float(x), float(y)
        p = np.array([fx, fy], dtype=np.float64)
        radius = float(np.linalg.norm(p))
        rids = rec["reading_ids"]
        anchors_np = rec["anchors_np"]

        shares_int = _rbf_reading_shares(fx, fy, rids, anchors_np)
        dom_r = max(shares_int, key=shares_int.get) if shares_int else -1
        dom_label = rec["label_map"].get(dom_r, str(dom_r))

        chosen_text = rec["default_text"]
        chosen_fluency = rec["default_fluency"]
        source = "baseline"
        condition: dict[str, Any] = {"mode": "baseline", "note": "Unsteered reference at origin"}

        # Near exact origin, return baseline reference (unless an anchor sits virtually at origin)
        anchor_dists = np.linalg.norm(anchors_np - p[None, :], axis=1) if len(anchors_np) else np.array([99.0])
        min_anchor_d = float(anchor_dists.min())
        if radius < 0.03 and min_anchor_d > 0.04:
            pass
        else:
            # Prefer real steered continuations (natural/cursor) targeting dominant reading `dom_r`
            cand_rows = [
                sp for sp in rec["steering_points"]
                if sp["A"] == dom_r and sp["variant"] == "natural" and sp["mode"] == "cursor" and sp["text"]
            ]
            if not cand_rows:
                cand_rows = [sp for sp in rec["steering_points"] if sp["A"] == dom_r and sp["text"]]

            if cand_rows:
                # Map radial distance from origin to KL level (0..3)
                avail_levels = sorted({sp["kl_level"] for sp in cand_rows})
                if radius < 0.40:
                    want_lvl = 0
                elif radius < 0.80:
                    want_lvl = 1
                elif radius < 1.20:
                    want_lvl = 2
                else:
                    want_lvl = 3
                best_lvl = min(avail_levels, key=lambda lv: abs(lv - want_lvl))
                lvl_rows = [sp for sp in cand_rows if sp["kl_level"] == best_lvl]

                # Match direction away from B toward A against cursor direction p
                a_dom = rec["anchor_map"][dom_r]
                p_dir = p / max(radius, 1e-8)

                def _row_score(sp: dict[str, Any]) -> tuple[float, float]:
                    a_b = rec["anchor_map"].get(sp["B"], np.zeros(2))
                    dv = a_dom - a_b
                    dv_norm = float(np.linalg.norm(dv))
                    align = float(np.dot(p_dir, dv / max(dv_norm, 1e-8))) if radius > 1e-4 else 0.0
                    # Combine directional alignment with steering success & fluency
                    return (align + 0.35 * sp["success_rate"], sp["fluency"])

                best_row = max(lvl_rows, key=_row_score)
                chosen_text = best_row["text"]
                chosen_fluency = best_row["fluency"]
                source = "steered"
                condition = {
                    "A": best_row["A"],
                    "B": best_row["B"],
                    "variant": best_row["variant"],
                    "mode": best_row["mode"],
                    "kl": best_row["kl"],
                    "kl_level": best_row["kl_level"],
                    "alpha": best_row["alpha"],
                    "success_rate": best_row["success_rate"],
                    "baseline_share_A": best_row["baseline_share_A"],
                }
            else:
                # Reading outside top-3 steered pairs -> pick nearest kept sample in `dom_r`
                idxs = np.where(rec["labels_np"] == dom_r)[0]
                if len(idxs) == 0:
                    idxs = np.where(rec["labels_np"] >= 0)[0]
                if len(idxs) > 0:
                    d_samp = np.linalg.norm(rec["coords_np"][idxs] - p[None, :], axis=1)
                    # Balance 2D proximity with fluency so degenerate samples aren't surfaced
                    lpt_sub = rec["lpt_np"][idxs]
                    score = -d_samp + 0.08 * (lpt_sub - float(np.median(lpt_sub)))
                    best_i = int(idxs[int(np.argmax(score))])
                    chosen_text = rec["texts"][best_i]
                    chosen_fluency = float(rec["lpt_np"][best_i])
                    source = "sample"
                    condition = {
                        "sample_index": best_i,
                        "distance_2d": round(float(np.linalg.norm(rec["coords_np"][best_i] - p)), 4),
                        "note": "Nearest saved sample for reading (outside top-3 steered grid)",
                    }

        return {
            "text_id": text_id,
            "x": round(fx, 4),
            "y": round(fy, 4),
            "radius": round(radius, 4),
            "text": chosen_text,
            "dominant_reading": int(dom_r),
            "dominant_label": dom_label,
            "shares": {str(k): round(float(v), 4) for k, v in shares_int.items()},
            "fluency": round(float(chosen_fluency), 4),
            "source": source,
            "condition": condition,
            "global_coords_applied": False,
        }


class LiveTranslationEngine:
    """Real-time KV-cached streaming, fork-burst probing, and 2D spectral steering engine."""

    def __init__(
        self,
        model_key: str = "tiny",
        layer: int | None = None,
        global_axes_path: str | Path | None = None,
        fork_head_path: str | Path | None = None,
        reft_path: str | Path | None = None,
        lora_path: str | Path | None = None,
        tok: Any = None,
        model: Any = None,
    ):
        self.model_key = model_key
        self._requested_layer = layer
        self.layer: int | None = layer
        self.global_axes_path = Path(global_axes_path) if global_axes_path else None
        self.fork_head_path = Path(fork_head_path) if fork_head_path else None
        self.reft_path = Path(reft_path) if reft_path else None
        self.lora_path = Path(lora_path) if lora_path else None
        self.global_axes, self.global_bank = _load_global_axes(self.global_axes_path, layer=layer)
        self.tok = tok
        self.model = model
        self.hook: CursorHook | None = None
        self.fork_scorer: Any = None
        self.reft_controller: Any = None
        self._lora_loaded = False
        self.forks: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        from explorer.session import CoTranslationManager
        self.sessions = CoTranslationManager(self)
        if self.tok is not None and self.model is not None:
            self._init_hook()

    def _init_hook(self) -> None:
        if self.layer is None:
            self.layer = arch.n_layers(self.model) // 2
        if self.global_axes_path and not self.global_axes:
            self.global_axes, self.global_bank = _load_global_axes(self.global_axes_path, layer=self.layer)
        if self.lora_path and self.lora_path.exists() and not self._lora_loaded:
            from explorer.adapters import load_lora
            load_lora(self.model, self.lora_path)
            self._lora_loaded = True
        if self.reft_path and self.reft_path.exists() and self.reft_controller is None:
            from explorer.adapters import SpectralReFTController
            self.reft_controller = SpectralReFTController.from_checkpoint(self.model, self.reft_path)
        if self.hook is None:
            self.hook = CursorHook(self.model, self.layer)
        if self.fork_head_path and self.fork_head_path.exists() and self.fork_scorer is None:
            from explorer.fork_head import ForkHead, LiveForkScorer
            head = ForkHead.load(self.fork_head_path, map_location=self._dev())
            meta = getattr(head, "meta", {})
            fh_layer = int(meta.get("layer", self.layer)) if isinstance(meta, dict) else self.layer
            self.fork_scorer = LiveForkScorer(head, self.model, fh_layer)

    def ensure_loaded(self) -> tuple[Any, Any]:
        with self._lock:
            if self.tok is None or self.model is None:
                self.tok, self.model = load(self.model_key)
            self._init_hook()
            return self.tok, self.model

    def close(self) -> None:
        with self._lock:
            if self.hook is not None:
                self.hook.remove()
                self.hook = None
            if self.fork_scorer is not None:
                self.fork_scorer.remove()
                self.fork_scorer = None
            if self.reft_controller is not None:
                self.reft_controller.remove()
                self.reft_controller = None

    def _dev(self) -> torch.device:
        return arch.text_model(self.model).embed_tokens.weight.device

    def _resolve_coords(self, global_coords: dict[str, float] | None, target_keys: set[str]) -> dict[str, float]:
        if not global_coords:
            return {}
        resolved: dict[str, float] = {}
        for k, val in global_coords.items():
            fv = float(val)
            if abs(fv) <= 1e-9:
                continue
            if k in target_keys:
                resolved[k] = fv
            elif k in _AXIS_ALIASES and _AXIS_ALIASES[k] in target_keys:
                resolved[_AXIS_ALIASES[k]] = fv
        return resolved

    def _global_vector(self, global_coords: dict[str, float] | None) -> np.ndarray | None:
        if not global_coords or not self.global_axes:
            return None
        d = arch.hidden_size(self.model)
        vec = np.zeros(d, dtype=np.float32)
        used = False
        resolved = self._resolve_coords(global_coords, set(self.global_axes.keys()))
        for k, fv in resolved.items():
            ax = self.global_axes[k]
            if ax.shape[0] == d:
                vec += fv * ax
                used = True
        return vec if used else None

    def status(self) -> dict[str, Any]:
        loaded = self.model is not None
        dev = str(self._dev()) if loaded else ("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "model_key": self.model_key,
            "loaded": loaded,
            "device": dev,
            "layer": self.layer,
            "n_layers": arch.n_layers(self.model) if loaded else None,
            "hidden_size": arch.hidden_size(self.model) if loaded else None,
            "active_forks": len(self.forks),
            "global_axes_loaded": list(self.global_axes.keys()),
            "global_axes_schema": DEFAULT_GLOBAL_AXES,
            "fork_head_loaded": self.fork_scorer is not None,
            "reft_loaded": self.reft_controller is not None,
            "lora_loaded": self._lora_loaded,
        }

    def stream_translate(
        self,
        source_text: str,
        max_new: int = 48,
        entropy_threshold: float = 1.5,
        compute_fisher: bool = False,
        fisher_iters: int = 3,
        global_coords: dict[str, float] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Manual KV-cached greedy decode loop yielding per-token uncertainty and fork signals."""
        tok, model = self.ensure_loaded()
        ids, start = build_prompt(tok, source_text)
        dev = self._dev()
        gvec = self._global_vector(global_coords)

        with self._lock:
            if gvec is not None:
                self.hook.set(torch.tensor(gvec, dtype=torch.float32), b=start - 1, only_at_b=False)
            else:
                self.hook.set(None)
            if self.reft_controller is not None:
                rc = self._resolve_coords(global_coords, set(self.reft_controller.axis_names))
                self.reft_controller.set_coords(rc or None, start_pos=start - 1)
            if self.fork_scorer is not None:
                self.fork_scorer.reset()

            gen_ids: list[int] = []
            pkv = None
            x = torch.tensor([ids], device=dev)
            W = model.lm_head.weight if compute_fisher else None

            try:
                for t in range(int(max_new)):
                    with torch.no_grad():
                        out = arch.forward_hidden(model, input_ids=x, past_key_values=pkv, use_cache=True)
                        pkv = out.past_key_values
                        h = out.last_hidden_state[0, -1]
                        lg = arch.logits(model, h[None])[0]
                        p = lg.softmax(-1)
                        ent = float(-(p * p.clamp_min(1e-30).log()).sum())
                        lam = _power_iter_fisher_top(W, p, iters=fisher_iters, seed=t) if compute_fisher else None
                        k_top = min(5, int(p.numel()))
                        top_vals, top_ids = torch.topk(p, k=k_top)
                        next_id = int(lg.argmax().item())
                        fh_latest = self.fork_scorer.latest() if self.fork_scorer is not None else None

                    if next_id in (tok.eos_token_id, tok.pad_token_id):
                        break

                    gen_ids.append(next_id)
                    decoded = tok.decode(gen_ids)
                    if any(st in decoded for st in STOPS):
                        break

                    top_tokens = [
                        {
                            "token": tok.decode([int(tid)]),
                            "token_id": int(tid),
                            "prob": round(float(val), 4),
                        }
                        for val, tid in zip(top_vals.tolist(), top_ids.tolist())
                    ]
                    event = {
                        "t": t,
                        "token_id": next_id,
                        "token": tok.decode([next_id]),
                        "text_so_far": decoded,
                        "entropy": round(ent, 4),
                        "fisher_top": round(lam, 4) if lam is not None else None,
                        "is_fork": bool(
                            ent >= float(entropy_threshold)
                            or (fh_latest is not None and fh_latest["fork_prob"] >= 0.5)
                        ),
                        "top_tokens": top_tokens,
                    }
                    if fh_latest is not None:
                        event["fork_head_prob"] = round(float(fh_latest["fork_prob"]), 4)
                        event["fork_head_entropy"] = round(float(fh_latest["pred_entropy"]), 4)
                        event["fork_head_fisher"] = round(float(fh_latest["pred_fisher_top"]), 4)
                    yield event
                    x = torch.tensor([[next_id]], device=dev)
            finally:
                self.hook.set(None)
                if self.reft_controller is not None:
                    self.reft_controller.set_coords(None)

    def probe_fork(
        self,
        source_text: str,
        prefix_token_ids: list[int] | None = None,
        n_samples: int = 16,
        lookahead_tokens: int = 16,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Burst-sample continuations at a fork cursor, cluster by centred gradients, and build a 2D compass."""
        tok, model = self.ensure_loaded()
        ids, _ = build_prompt(tok, source_text)
        pref_toks = [int(i) for i in (prefix_token_ids or [])]
        prefix = ids + pref_toks
        b = len(prefix) - 1
        prefix_text = tok.decode(pref_toks)
        n_samples = max(4, int(n_samples))
        lookahead_tokens = max(2, int(lookahead_tokens))

        with self._lock:
            self.hook.set(None)
            conts = generate(
                model,
                tok,
                prefix,
                n=n_samples,
                max_new=lookahead_tokens,
                temperature=float(temperature),
                seed=int(seed),
                batch=min(32, n_samples),
            )
            cont_texts = [tok.decode(c).strip() for c in conts]
            full_texts = [tok.decode(pref_toks + c).strip() for c in conts]

            # Per-sample gradients at cursor b (zero sentence-transformer download needed)
            G = grads_at(model, prefix, conts, self.layer, b, batch=min(8, n_samples))
            g_mu = G.mean(axis=0)
            G_c = G - g_mu
            g_norms = np.linalg.norm(G_c, axis=1, keepdims=True)
            if float(g_norms.max()) < 1e-12:
                rng = np.random.default_rng(int(seed))
                G_c = rng.normal(scale=1e-4, size=G.shape)
                g_norms = np.linalg.norm(G_c, axis=1, keepdims=True)
            X = G_c / g_norms.clip(1e-12)

            kmax = min(5, max(2, n_samples // 3))
            cl = R.cluster(X, min_share=0.05, kmax=kmax, sil_min=0.08)
            labels = np.asarray(cl["labels"], dtype=int)
            rd = [int(r) for r in cl["readings"]]
            sil = float(cl["silhouette"])

            # Graceful k=2 top-PC split fallback if silhouette is low so the 2D compass is always explorable
            fallback_split = False
            if len(rd) < 2:
                fallback_split = True
                _, _, Vt_g = np.linalg.svd(G_c, full_matrices=False)
                proj1 = G_c @ Vt_g[0]
                med = float(np.median(proj1))
                labels = (proj1 >= med).astype(int)
                if len(set(labels.tolist())) < 2:
                    labels = (np.arange(n_samples) % 2).astype(int)
                counts = np.bincount(labels, minlength=2)
                rd = [int(r) for r in np.argsort(-counts)]

            # Build orthonormal 2D steering plane (u1, u2)
            M = np.array([G[labels == r].mean(axis=0) for r in rd], dtype=np.float64)
            c = M.mean(axis=0)
            d = G.shape[1]

            def _orth_vec(base: np.ndarray, mat: np.ndarray) -> np.ndarray:
                orth = mat - np.outer(mat @ base, base)
                _, sv, Vt_o = np.linalg.svd(orth, full_matrices=False)
                if Vt_o.shape[0] > 0 and sv[0] > 1e-12:
                    u = Vt_o[0]
                else:
                    e = np.zeros(d, dtype=np.float64)
                    e[int(np.argmin(np.abs(base)))] = 1.0
                    u = e - float(e @ base) * base
                return u / max(float(np.linalg.norm(u)), 1e-12)

            if len(rd) >= 3:
                _, svals, Vt = np.linalg.svd(M - c, full_matrices=False)
                u1 = Vt[0] / max(float(np.linalg.norm(Vt[0])), 1e-12)
                if Vt.shape[0] >= 2 and svals[1] > 1e-12:
                    u2 = Vt[1] - float(Vt[1] @ u1) * u1
                    u2 /= max(float(np.linalg.norm(u2)), 1e-12)
                else:
                    u2 = _orth_vec(u1, G_c)
                axes = np.vstack([u1, u2])
                var_expl = float((svals[:2] ** 2).sum() / max(float((svals ** 2).sum()), 1e-30))
            else:
                diff = M[0] - M[1]
                if float(np.linalg.norm(diff)) < 1e-12:
                    _, _, Vt_g = np.linalg.svd(G_c, full_matrices=False)
                    u1 = Vt_g[0] / max(float(np.linalg.norm(Vt_g[0])), 1e-12)
                else:
                    u1 = diff / float(np.linalg.norm(diff))
                u2 = _orth_vec(u1, G_c)
                axes = np.vstack([u1, u2])
                total_var = float((G_c ** 2).sum())
                proj_var = float(((G_c @ axes.T) ** 2).sum())
                var_expl = proj_var / max(total_var, 1e-30)

            raw_anchors = (M - c) @ axes.T
            scale = max(float(np.linalg.norm(raw_anchors, axis=1).max()), 1e-8)
            anchors = raw_anchors / scale
            dot_coords = ((G - c) @ axes.T) / scale

            # Fast 4-step calibration (or Fisher-norm scale fallback)
            far = int(np.argmax(np.linalg.norm(anchors, axis=1)))
            v_far = anchors[far] @ axes
            v_far_norm = max(float(np.linalg.norm(v_far)), 1e-12)
            v_unit = v_far / v_far_norm
            a0 = alpha_for(G, v_unit, 0.25)
            if not math.isfinite(a0) or a0 <= 0.0 or a0 > 1e4:
                a0 = 1.0
            ref_cont = next((ct for ct in conts if len(ct) > 0), [])
            if ref_cont:
                a_far = calibrate(model, self.hook, prefix + ref_cont, b, v_unit, target=0.25, mode="cursor", a0=a0, iters=4)
                if not math.isfinite(a_far) or a_far <= 0.0:
                    a_far = a0
            else:
                a_far = a0
            alpha_unit = float(a_far / v_far_norm)

        # Human-friendly labels & representative candidates per reading
        kw_map = _extract_reading_keywords(cont_texts, labels, rd, top=3)
        reading_meta: list[dict[str, Any]] = []
        cand_list: list[tuple[int, list[int]]] = []
        color_map: dict[int, str] = {}

        for idx, r in enumerate(rd):
            color = READING_PALETTE[idx % len(READING_PALETTE)]
            color_map[r] = color
            mem_idx = np.where(labels == r)[0].tolist()
            kws = kw_map.get(r, [f"reading {r}"])
            rlabel = " · ".join(kws)
            seen_c: set[tuple[int, ...]] = set()
            exemplars: list[dict[str, Any]] = []
            for i in mem_idx:
                key = tuple(conts[i])
                if conts[i] and key not in seen_c and len([x for x in cand_list if x[0] == r]) < 2:
                    cand_list.append((r, conts[i]))
                    seen_c.add(key)
                exemplars.append({
                    "i": int(i),
                    "text": full_texts[i],
                    "continuation": cont_texts[i],
                    "x": round(float(dot_coords[i, 0]), 4),
                    "y": round(float(dot_coords[i, 1]), 4),
                })
            reading_meta.append({
                "id": int(r),
                "label": rlabel,
                "keywords": kws,
                "share": round(float(len(mem_idx) / n_samples), 4),
                "count": int(len(mem_idx)),
                "color": color,
                "anchor": [round(float(anchors[idx, 0]), 4), round(float(anchors[idx, 1]), 4)],
                "exemplars": exemplars[:5],
            })

        dots = [
            {
                "i": int(i),
                "reading": int(labels[i]),
                "kept": True,
                "x": round(float(dot_coords[i, 0]), 4),
                "y": round(float(dot_coords[i, 1]), 4),
                "text": full_texts[i],
                "continuation": cont_texts[i],
                "color": color_map.get(int(labels[i]), "#b5b0ba"),
            }
            for i in range(n_samples)
        ]

        fork_id = f"fork_{uuid.uuid4().hex[:10]}"
        self.forks[fork_id] = {
            "fork_id": fork_id,
            "source_text": source_text,
            "ids": ids,
            "prefix_token_ids": pref_toks,
            "prefix": prefix,
            "prefix_text": prefix_text,
            "b": b,
            "layer": self.layer,
            "axes": axes,
            "anchors": anchors,
            "reading_ids": rd,
            "reading_meta": reading_meta,
            "alpha_unit": alpha_unit,
            "cand_list": cand_list,
        }

        return {
            "fork_id": fork_id,
            "source_text": source_text,
            "prefix_text": prefix_text,
            "prefix_token_ids": pref_toks,
            "b": b,
            "layer": self.layer,
            "n_samples": n_samples,
            "silhouette": round(sil, 4),
            "fallback_split": fallback_split,
            "variance_explained": round(var_expl, 4),
            "alpha_unit": round(alpha_unit, 6),
            "readings": reading_meta,
            "anchors": {str(r): [round(float(anchors[i, 0]), 4), round(float(anchors[i, 1]), 4)] for i, r in enumerate(rd)},
            "dots": dots,
            "samples": full_texts,
        }

    def steer_at(
        self,
        fork_id: str,
        x: float,
        y: float,
        global_coords: dict[str, float] | None = None,
        max_new: int = 32,
        temperature: float = 0.0,
        only_at_b: bool = False,
    ) -> dict[str, Any]:
        """Steer continuation at `(x, y)` on a probed fork plane (+ optional global tone axes)."""
        if fork_id not in self.forks:
            raise KeyError(f"Unknown fork_id: {fork_id!r}")
        tok, model = self.ensure_loaded()
        fork = self.forks[fork_id]
        fx, fy = float(x), float(y)

        vec_np = fork["alpha_unit"] * (fx * fork["axes"][0] + fy * fork["axes"][1])
        gvec = self._global_vector(global_coords)
        if gvec is not None:
            vec_np = vec_np + gvec.astype(np.float64)
        vec = torch.tensor(vec_np, dtype=torch.float32)

        reft_applied = False
        with self._lock:
            if self.reft_controller is not None:
                rc = self._resolve_coords(global_coords, set(self.reft_controller.axis_names))
                if rc:
                    self.reft_controller.set_coords(rc, start_pos=fork["b"])
                    reft_applied = True
                else:
                    self.reft_controller.set_coords(None)
            try:
                cont_ids = steered(
                    model,
                    tok,
                    self.hook,
                    fork["prefix"],
                    vec,
                    fork["b"],
                    int(max_new),
                    n=1,
                    temperature=float(temperature),
                    only_at_b=bool(only_at_b),
                )[0]

                # Reading affinity scores under the steered model
                cand_list = fork["cand_list"]
                rids = fork["reading_ids"]
                if cand_list:
                    self.hook.set(vec, fork["b"], only_at_b=bool(only_at_b))
                    try:
                        with torch.no_grad():
                            lps = seq_logp(model, fork["prefix"], [c for _, c in cand_list]).float().cpu().numpy()
                    finally:
                        self.hook.set(None)
                    lens = np.maximum(np.array([len(c) for _, c in cand_list], dtype=np.float64), 1.0)
                    norm_lp = lps / lens
                    norm_lp = norm_lp - float(norm_lp.max())
                    probs = np.exp(norm_lp)
                    probs /= max(float(probs.sum()), 1e-30)
                    shares_raw: dict[int, float] = {int(r): 0.0 for r in rids}
                    for (r, _), pr in zip(cand_list, probs):
                        shares_raw[int(r)] += float(pr)
                    total_s = sum(shares_raw.values())
                    shares = {str(r): round(shares_raw[r] / max(total_s, 1e-30), 4) for r in rids}
                else:
                    rbf = _rbf_reading_shares(fx, fy, rids, fork["anchors"])
                    shares = {str(r): round(float(v), 4) for r, v in rbf.items()}
            finally:
                if self.reft_controller is not None:
                    self.reft_controller.set_coords(None)

        cont_text = tok.decode(cont_ids).strip()
        full_text = tok.decode(fork["prefix_token_ids"] + cont_ids).strip()
        return {
            "fork_id": fork_id,
            "text": full_text,
            "prefix_text": fork["prefix_text"],
            "continuation": cont_text,
            "x": round(fx, 4),
            "y": round(fy, 4),
            "shares": shares,
            "tokens": [tok.decode([tid]) for tid in cont_ids],
            "token_ids": [int(tid) for tid in cont_ids],
            "global_coords_applied": (gvec is not None) or reft_applied,
        }
