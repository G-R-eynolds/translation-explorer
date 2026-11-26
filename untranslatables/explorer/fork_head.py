"""Amortised real-time fork & blind-spot tension side-head (Phase 4).

Predicts interpretive fork probability, next-token entropy, output-Fisher top eigenvalue λ₁,
and an optional orthonormalized local sensitivity subspace directly from layer-ℓ hidden states
in <1 ms per decode step with zero extra model forward passes.

CLI:
    python -m explorer.fork_head --model g3_4b_pt [--layer 17] [--epochs 80]
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from explorer.frame import build_prompt, load_texts
from ut import arch
from ut.common import ROOT


def _gram_schmidt(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable re-orthogonalized Modified Gram-Schmidt along dim=-2 for (..., k, d)."""
    k = v.shape[-2]
    basis: list[torch.Tensor] = []
    for i in range(k):
        u = v[..., i, :]
        for q in basis:
            u = u - (u * q).sum(dim=-1, keepdim=True) * q
        for q in basis:
            u = u - (u * q).sum(dim=-1, keepdim=True) * q
        norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
        basis.append(u / norm)
    return torch.stack(basis, dim=-2)


def _subspace_alignment_loss(pred_dirs: torch.Tensor, target_dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign- and basis-rotation-invariant Grassmannian cosine subspace loss.

    pred_dirs:   (B, k, d) orthonormal rows
    target_dirs: (B, k', d) target direction vectors (rows with ~0 norm are ignored)
    Returns (loss, mean_overlap) where overlap = ||Q U^T||_F^2 / min(k, k') in [0, 1].
    """
    if target_dirs.ndim == 2:
        target_dirs = target_dirs.unsqueeze(-2)
    valid = target_dirs.norm(dim=-1).max(dim=-1).values > 1e-6
    if not bool(valid.any()):
        zero = pred_dirs.sum() * 0.0
        return zero, pred_dirs.new_ones(())
    q = pred_dirs[valid]
    u = _gram_schmidt(target_dirs[valid])
    gram = torch.matmul(q, u.transpose(-1, -2))  # (B_v, k, k')
    denom = float(min(q.shape[-2], u.shape[-2]))
    overlap = (gram * gram).sum(dim=(-1, -2)) / denom
    overlap = overlap.clamp(0.0, 1.0)
    mean_overlap = overlap.mean()
    return 1.0 - mean_overlap, mean_overlap


class ForkHead(nn.Module):
    """Lightweight residual MLP side-head operating on layer-ℓ residual stream states.

    Architecture:
        LayerNorm(d_model) -> Linear(d_model, hidden_dim) -> GELU -> Dropout(0.1)
        -> Linear(hidden_dim, hidden_dim) -> GELU + residual

    Outputs:
        fork_logit : (...,)              raw logit for interpretive fork probability
        entropy    : (...,)              predicted next-token entropy (softplus, >= 0)
        fisher_top : (...,)              predicted output-Fisher top eigenvalue λ₁ (softplus, >= 0)
        dirs       : (..., n_dirs, d)    orthonormalized local sensitivity subspace (or None if disabled)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        n_dirs: int = 2,
        dropout: float = 0.1,
        predict_dirs: bool | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.n_dirs = int(n_dirs)
        self.dropout_p = float(dropout)
        self.predict_dirs = (self.n_dirs > 0) if predict_dirs is None else (bool(predict_dirs) and self.n_dirs > 0)

        self.norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, self.hidden_dim)
        self.drop = nn.Dropout(self.dropout_p)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scalar_head = nn.Linear(self.hidden_dim, 3)
        self.dir_head = nn.Linear(self.hidden_dim, self.n_dirs * self.d_model) if self.predict_dirs else None
        self.meta: dict[str, Any] = {}

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor | None]:
        x = self.norm(h.to(dtype=self.norm.weight.dtype))
        z = F.gelu(self.fc1(x))
        z = z + F.gelu(self.fc2(self.drop(z)))
        s = self.scalar_head(z)
        out: dict[str, torch.Tensor | None] = {
            "fork_logit": s[..., 0],
            "entropy": F.softplus(s[..., 1]),
            "fisher_top": F.softplus(s[..., 2]),
            "dirs": None,
        }
        if self.dir_head is not None:
            raw = self.dir_head(z).view(*z.shape[:-1], self.n_dirs, self.d_model)
            out["dirs"] = _gram_schmidt(raw)
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "n_dirs": self.n_dirs,
            "dropout": self.dropout_p,
            "predict_dirs": self.predict_dirs,
        }

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "meta": dict(self.meta if meta is None else meta),
        }
        torch.save(payload, p)
        return p

    @property
    def recommended_alpha(self) -> float | None:
        val = self.meta.get("recommended_alpha")
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if (np.isfinite(f) and f > 0.0) else None

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ForkHead":
        ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
        head = cls(**ckpt["config"])
        head.load_state_dict(ckpt["state_dict"])
        head.meta = dict(ckpt.get("meta", {}))
        head.eval()
        return head


def load_fork_head(path: str | Path, map_location: str | torch.device = "cpu") -> ForkHead:
    """Load a saved ForkHead checkpoint from disk."""
    return ForkHead.load(path, map_location=map_location)


def _extract_local_fisher_dirs(
    model,
    x: torch.Tensor,
    layer: int,
    fork_abs_positions: list[int],
    n_dirs: int,
    top_k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Extract top-`n_dirs` orthonormal local output-Fisher steering directions at `fork_abs_positions`.

    Runs a single forward pass per call with a float32 leaf perturbation `z` injected at `layer`
    via an out-of-place `index_add` forward hook (works on frozen/NF4/INT8 models with zero `.grad`
    leakage on model weights), slices `last_hidden_state` at fork positions before `lm_head`, forms
    the top-K logit-space Fisher eigenspace, and pulls back top eigen-directions via `autograd.grad`.
    """
    d_model = arch.hidden_size(model)
    m = len(fork_abs_positions)
    if m == 0 or n_dirs <= 0:
        return (
            torch.zeros((m, max(int(n_dirs), 0), d_model), dtype=torch.float32),
            torch.zeros((m,), dtype=torch.bool),
            [0.0] * m,
        )

    try:
        dev = x.device
        idx_t = torch.tensor(fork_abs_positions, dtype=torch.long, device=dev)
        z = torch.zeros((m, d_model), dtype=torch.float32, device=dev, requires_grad=True)
        blk = arch.layers(model)[layer]

        def _inject_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            dz = z.to(dtype=h.dtype)
            if h.ndim == 3:
                h_new = h.index_add(1, idx_t, dz.unsqueeze(0))
            else:
                h_new = h.index_add(0, idx_t, dz)
            return (h_new, *out[1:]) if isinstance(out, tuple) else h_new

        dirs_out = torch.zeros((m, n_dirs, d_model), dtype=torch.float32)
        mask_out = torch.zeros((m,), dtype=torch.bool)
        lam1s = [0.0] * m

        handle = blk.register_forward_hook(_inject_hook)
        try:
            with torch.enable_grad():
                out = arch.forward_hidden(model, input_ids=x, attention_mask=torch.ones_like(x))
                h_fork_last = out.last_hidden_state[0, idx_t, :]  # (m, d_model)
                lg_fork = arch.logits(model, h_fork_last).float()  # (m, V)

                V = int(lg_fork.shape[-1])
                K = min(int(top_k), V)
                if K < 2:
                    return dirs_out, mask_out, lam1s

                # Precompute top-K Fisher pullback scalars for all fork positions
                tasks: list[tuple[int, float, list[torch.Tensor]]] = []
                for k in range(m):
                    lg_k = lg_fork[k]
                    with torch.no_grad():
                        p_full = lg_k.softmax(-1)
                        top_p, top_idx = torch.topk(p_full, k=K)
                        mass_k = float(top_p.sum().item())
                        p_hat = top_p.double() / max(mass_k, 1e-12)
                        F_k = torch.diag(p_hat) - torch.outer(p_hat, p_hat)
                        F_k = 0.5 * (F_k + F_k.T)
                        evals, evecs = torch.linalg.eigh(F_k)
                        evals = evals.flip(0)
                        evecs = evecs.flip(1)
                        pos_mask = evals > 1e-12
                        n_pos = int(pos_mask.sum().item())
                    if n_pos == 0:
                        continue
                    R = min(max(int(n_dirs) + 2, 4), K - 1, n_pos)
                    z_top = lg_k.index_select(0, top_idx)
                    scalars: list[torch.Tensor] = []
                    for r in range(R):
                        w_r = (torch.sqrt(evals[r]) * evecs[:, r]).to(dtype=z_top.dtype, device=z_top.device)
                        scalars.append((w_r * z_top).sum())
                    tasks.append((k, mass_k, scalars))

                total_pulls = sum(len(sc) for _, _, sc in tasks)
                pull_idx = 0
                for k, mass_k, scalars in tasks:
                    rows: list[np.ndarray] = []
                    for s_r in scalars:
                        pull_idx += 1
                        retain = pull_idx < total_pulls
                        (gz,) = torch.autograd.grad(s_r, (z,), retain_graph=retain, allow_unused=False)
                        rows.append(gz[k].detach().cpu().numpy().astype(np.float64))
                    G_k = np.vstack(rows)  # (R, d_model)
                    _, S_vals, Vh = np.linalg.svd(G_k, full_matrices=False)
                    if len(S_vals) > 0:
                        lam1s[k] = float((S_vals[0] ** 2) * mass_k)
                    if len(S_vals) >= n_dirs and float(S_vals[n_dirs - 1]) > 1e-8:
                        dirs_out[k] = torch.from_numpy(Vh[:n_dirs].copy()).float()
                        mask_out[k] = True
        finally:
            handle.remove()
        return dirs_out, mask_out, lam1s
    except torch.cuda.OutOfMemoryError:
        if m <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = m // 2
        d1, m1, l1 = _extract_local_fisher_dirs(model, x, layer, fork_abs_positions[:mid], n_dirs, top_k=top_k)
        d2, m2, l2 = _extract_local_fisher_dirs(model, x, layer, fork_abs_positions[mid:], n_dirs, top_k=top_k)
        return torch.cat([d1, d2], dim=0), torch.cat([m1, m2], dim=0), l1 + l2


def _fallback_alpha_from_hidden(H: torch.Tensor) -> float:
    if H is not None and H.numel() > 0:
        med_norm = float(H.float().norm(dim=-1).median().item())
        if np.isfinite(med_norm) and med_norm > 0.0:
            return float(np.clip(0.05 * med_norm, 1e-3, 1e4))
    return 1.0


def _calibrate_recommended_alpha(
    model,
    layer: int,
    calib_samples: list[tuple[list[int], int, np.ndarray, float]],
    H: torch.Tensor,
    target_kl: float = 0.25,
    max_samples: int = 5,
    iters: int = 6,
) -> float:
    """Calibrate baseline cursor-mode steering alpha (for target_kl nats) via cached-p0 log-bisection."""
    if not calib_samples:
        return _fallback_alpha_from_hidden(H)

    from explorer.steer import CursorHook

    dev = arch.text_model(model).embed_tokens.weight.device
    hook = CursorHook(model, layer)
    alphas: list[float] = []
    try:
        for seq_ids, b_abs, unit_dir, lam1 in calib_samples[:max_samples]:
            if b_abs < 0 or b_abs >= len(seq_ids):
                continue
            v = np.asarray(unit_dir, dtype=np.float64)
            v_norm = float(np.linalg.norm(v))
            if not np.isfinite(v_norm) or v_norm <= 1e-12:
                continue
            v = v / v_norm
            x = torch.tensor([seq_ids], dtype=torch.long, device=dev)

            hook.set(None)
            with torch.no_grad():
                h0 = arch.forward_hidden(model, input_ids=x).last_hidden_state[0, b_abs:]
                logp0 = arch.logits(model, h0).float().log_softmax(-1)
                p0 = logp0.exp()

            def _kl(a_val: float) -> float:
                vec = torch.tensor(a_val * v, dtype=torch.float32, device=dev)
                hook.set(vec, b_abs, only_at_b=False)
                try:
                    with torch.no_grad():
                        hs = arch.forward_hidden(model, input_ids=x).last_hidden_state[0, b_abs:]
                        logps = arch.logits(model, hs).float().log_softmax(-1)
                    val = float((p0 * (logp0 - logps)).sum(-1).mean().item())
                    return max(0.0, val) if np.isfinite(val) else 0.0
                finally:
                    hook.set(None)

            a0 = float(np.clip(np.sqrt(2.0 * target_kl / max(lam1, 1e-6)), 1e-2, 1e2))
            k0 = _kl(a0)
            if k0 <= 0.0:
                continue

            if k0 < target_kl:
                lo, hi = a0, a0
                while hi < 1e3 and _kl(hi) < target_kl:
                    lo, hi = hi, min(hi * 4.0, 1e3)
            else:
                lo, hi = a0, a0
                while lo > 1e-3 and _kl(lo) > target_kl:
                    hi, lo = lo, max(lo * 0.25, 1e-3)

            best_a = hi if (k0 < target_kl and _kl(hi) < target_kl) else (lo if (k0 > target_kl and _kl(lo) > target_kl) else a0)
            if lo < hi and _kl(lo) <= target_kl <= _kl(hi):
                for _ in range(int(iters)):
                    mid = float(np.sqrt(lo * hi))
                    if _kl(mid) < target_kl:
                        lo = mid
                    else:
                        hi = mid
                best_a = float(np.sqrt(lo * hi))

            if np.isfinite(best_a) and best_a > 0.0:
                alphas.append(float(np.clip(best_a, 1e-3, 1e4)))
    finally:
        hook.remove()

    if not alphas:
        return _fallback_alpha_from_hidden(H)
    return float(np.clip(float(np.median(alphas)), 1e-3, 1e4))


def _resolve_ref_ids(tok, branch: dict[str, Any], sample: dict[str, Any] | None) -> list[int]:
    """Recover the exact reference token ID sequence matching a saved branches/<id>.json record."""
    n_ent = len(branch.get("entropy", []))
    ref_str = str(branch.get("ref", "")).strip()

    if sample is not None and sample.get("ys"):
        ys = sample["ys"]
        labels = sample.get("labels")
        readings = sample.get("readings")
        logp = sample.get("logp")
        if labels is not None and readings and logp is not None:
            idx_r0 = [i for i, lab in enumerate(labels) if lab == readings[0]]
            if idx_r0:
                ref_i = max(idx_r0, key=lambda i: logp[i])
                cand = ys[ref_i]
                if len(cand) == n_ent:
                    return list(cand)
        # Fallback: scan samples for matching text & token length
        for y in ys:
            if len(y) == n_ent and (not ref_str or tok.decode(y).strip() == ref_str):
                return list(y)
        for y in ys:
            if len(y) == n_ent:
                return list(y)

    if branch.get("ref_token_ids"):
        return list(branch["ref_token_ids"])
    if branch.get("ref_tokens"):
        return list(tok("".join(branch["ref_tokens"]), add_special_tokens=False)["input_ids"])
    if ref_str:
        return list(tok(ref_str, add_special_tokens=False)["input_ids"])
    return []


def extract_fork_dataset(
    model,
    tok,
    explorer_dir: str | Path,
    layer: int | None = None,
    entropy_fork_thresh: float = 1.5,
    prompt_style: str = "legacy",
    n_dirs: int = 2,
) -> dict[str, Any]:
    """Extract (hidden_state, fork_label, entropy, fisher_top[, y_dirs, y_dirs_mask]) from saved explorer runs.

    Scans `explorer_dir/branches/*.json` and matching `explorer_dir/samples/*.json`, runs one
    no-grad forward pass per text through `ut.arch.forward_hidden(..., output_hidden_states=True)`,
    slices `h[start + t - 1]` at `layer` (default `n_layers // 2`), and when `n_dirs > 0` extracts
    top-`n_dirs` orthonormal local Fisher steering directions on fork tokens plus a calibrated
    `meta["recommended_alpha"]`.
    """
    from explorer.frame import build_doc_prompt

    explorer_dir = Path(explorer_dir)
    branches_dir = explorer_dir / "branches"
    samples_dir = explorer_dir / "samples"
    if not branches_dir.is_dir():
        raise FileNotFoundError(f"Branches directory not found: {branches_dir}")

    n_l = arch.n_layers(model)
    layer = (n_l // 2) if layer is None else (int(layer) % n_l)
    dev = arch.text_model(model).embed_tokens.weight.device
    n_dirs = int(n_dirs)
    d_model = arch.hidden_size(model)

    try:
        text_lookup = {t["id"]: t["text"] for t in load_texts()}
    except Exception:
        text_lookup = {}

    H_chunks: list[torch.Tensor] = []
    dirs_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    calib_samples: list[tuple[list[int], int, np.ndarray, float]] = []
    y_fork: list[float] = []
    y_entropy: list[float] = []
    y_fisher: list[float] = []
    token_text_ids: list[str] = []
    token_positions: list[int] = []
    sources: list[str] = []

    for bp in sorted(branches_dir.glob("*.json")):
        B = json.loads(bp.read_text())
        tid = str(B.get("id", bp.stem))
        sp = samples_dir / f"{tid}.json"
        if not sp.exists():
            sp = samples_dir / bp.name
        S = json.loads(sp.read_text()) if sp.exists() else None

        src_text = (S.get("text") if S else None) or B.get("text") or text_lookup.get(tid)
        if not src_text:
            continue

        y_ref = _resolve_ref_ids(tok, B, S)
        ent = B.get("entropy", [])
        fis = B.get("fisher_top", [])
        n_tok = min(len(y_ref), len(ent), len(fis))
        if n_tok == 0:
            continue

        ids, start = build_doc_prompt(tok, src_text) if prompt_style == "doc" else build_prompt(tok, src_text)
        seq_ids = ids + y_ref[:n_tok]
        x = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = arch.forward_hidden(
                model,
                input_ids=x,
                attention_mask=torch.ones_like(x),
                output_hidden_states=True,
            )
            # hidden_states[0] is embeddings; hidden_states[layer + 1] is output of decoder block `layer`
            h_layer = out.hidden_states[layer + 1][0]
            h_slice = h_layer[start - 1 : start - 1 + n_tok].detach().float().cpu()

        fork_t = B.get("t", -1)
        H_chunks.append(h_slice)
        sources.append(tid)
        fork_local_idxs: list[int] = []
        for t in range(n_tok):
            e_t = float(ent[t])
            f_t = float(fis[t])
            is_fork = (t == fork_t or e_t >= entropy_fork_thresh)
            y_fork.append(1.0 if is_fork else 0.0)
            if is_fork:
                fork_local_idxs.append(t)
            y_entropy.append(e_t)
            y_fisher.append(f_t)
            token_text_ids.append(tid)
            token_positions.append(t)

        if n_dirs > 0:
            d_txt = torch.zeros((n_tok, n_dirs, d_model), dtype=torch.float32)
            m_txt = torch.zeros((n_tok,), dtype=torch.bool)
            if fork_local_idxs:
                fork_abs = [start - 1 + t for t in fork_local_idxs]
                d_f, m_f, l1_f = _extract_local_fisher_dirs(model, x, layer, fork_abs, n_dirs=n_dirs)
                for idx_local, dk, mk, l1k in zip(fork_local_idxs, d_f, m_f, l1_f):
                    if bool(mk.item()):
                        d_txt[idx_local] = dk
                        m_txt[idx_local] = True
                        if len(calib_samples) < 5:
                            calib_samples.append((seq_ids, start - 1 + idx_local, dk[0].numpy(), l1k))
            dirs_chunks.append(d_txt)
            mask_chunks.append(m_txt)

    if not H_chunks:
        raise ValueError(f"No usable branch/sample entries found under {explorer_dir}")

    H = torch.cat(H_chunks, dim=0)
    median_hidden_norm = float(H.float().norm(dim=-1).median().item()) if len(H) else 0.0
    recommended_alpha = _calibrate_recommended_alpha(model, layer, calib_samples, H)

    ds: dict[str, Any] = {
        "H": H,
        "y_fork": torch.tensor(y_fork, dtype=torch.float32),
        "y_entropy": torch.tensor(y_entropy, dtype=torch.float32),
        "y_fisher": torch.tensor(y_fisher, dtype=torch.float32),
        "meta": {
            "layer": layer,
            "d_model": int(H.shape[1]),
            "n_texts": len(sources),
            "n_tokens": int(H.shape[0]),
            "n_dirs": int(n_dirs),
            "entropy_fork_thresh": float(entropy_fork_thresh),
            "prompt_style": prompt_style,
            "recommended_alpha": float(recommended_alpha),
            "median_hidden_norm": float(median_hidden_norm),
            "sources": sources,
            "text_ids": token_text_ids,
            "positions": token_positions,
        },
    }
    if n_dirs > 0:
        ds["y_dirs"] = torch.cat(dirs_chunks, dim=0)
        ds["y_dirs_mask"] = torch.cat(mask_chunks, dim=0)
    return ds


def extract_self_supervised_fork_dataset(
    model,
    tok,
    layer: int | None = None,
    entropy_fork_thresh: float = 1.25,
    prompt_style: str = "doc",
    max_new_per_text: int = 64,
    n_dirs: int = 2,
) -> dict[str, Any]:
    """Build a ForkHead dataset directly from `texts.yaml` + `tone_axes.yaml` in one pass per text.

    Runs greedy KV-cached decoding on each source text, capturing layer-`layer` hidden states,
    exact next-token entropy, power-iteration output-Fisher λ₁, and (when `n_dirs > 0`) local
    output-Fisher steering subspaces `y_dirs` + calibrated `meta["recommended_alpha"]`.
    """
    from explorer.frame import DOC_STOPS, STOPS, build_doc_prompt
    from explorer.global_axes import load_tone_axes_dataset
    from explorer.live_engine import _power_iter_fisher_top

    n_l = arch.n_layers(model)
    layer = (n_l // 2) if layer is None else (int(layer) % n_l)
    dev = arch.text_model(model).embed_tokens.weight.device
    W = model.lm_head.weight
    stops = DOC_STOPS if prompt_style == "doc" else STOPS
    n_dirs = int(n_dirs)
    d_model = arch.hidden_size(model)

    corpus: list[tuple[str, str, str]] = []
    seen_src: set[str] = set()
    for t in load_texts():
        if t["text"] not in seen_src:
            corpus.append((t["id"], t["text"], t.get("lang", "Spanish")))
            seen_src.add(t["text"])
    for item in load_tone_axes_dataset()["items"]:
        if item["source"] not in seen_src:
            corpus.append((item["id"], item["source"], item.get("lang", "Spanish")))
            seen_src.add(item["source"])

    H_chunks: list[torch.Tensor] = []
    dirs_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    calib_samples: list[tuple[list[int], int, np.ndarray, float]] = []
    y_fork: list[float] = []
    y_entropy: list[float] = []
    y_fisher: list[float] = []
    token_text_ids: list[str] = []
    token_positions: list[int] = []
    sources: list[str] = []
    eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

    for tid, src, src_lang in corpus:
        ids, start = build_doc_prompt(tok, src, src_lang=src_lang) if prompt_style == "doc" else build_prompt(tok, src)
        gen_ids: list[int] = []
        pkv = None
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            for _ in range(int(max_new_per_text)):
                out = arch.forward_hidden(model, input_ids=x, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                lg = arch.logits(model, out.last_hidden_state[0, -1][None])[0]
                nxt = int(lg.argmax().item())
                if nxt in eos_ids:
                    break
                gen_ids.append(nxt)
                if any(st in tok.decode(gen_ids) for st in stops):
                    break
                x = torch.tensor([[nxt]], dtype=torch.long, device=dev)
        if not gen_ids:
            continue

        seq_ids = ids + gen_ids
        x_full = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = arch.forward_hidden(model, input_ids=x_full, output_hidden_states=True)
            h_layer = out.hidden_states[layer + 1][0, start - 1 : start - 1 + len(gen_ids)].detach().float().cpu()
            h_last = out.last_hidden_state[0, start - 1 : start - 1 + len(gen_ids)]
            lg_all = arch.logits(model, h_last)
            p_all = lg_all.softmax(-1)
            ents = (-(p_all * p_all.clamp_min(1e-30).log()).sum(dim=-1)).cpu().tolist()
            fises = [_power_iter_fisher_top(W, p_all[j], iters=3, seed=j) for j in range(len(gen_ids))]

        H_chunks.append(h_layer)
        sources.append(tid)
        fork_local_idxs: list[int] = []
        for j, (e_t, f_t) in enumerate(zip(ents, fises)):
            is_fork = float(e_t) >= entropy_fork_thresh
            y_fork.append(1.0 if is_fork else 0.0)
            if is_fork:
                fork_local_idxs.append(j)
            y_entropy.append(float(e_t))
            y_fisher.append(float(f_t))
            token_text_ids.append(tid)
            token_positions.append(j)

        if n_dirs > 0:
            n_tok = len(gen_ids)
            d_txt = torch.zeros((n_tok, n_dirs, d_model), dtype=torch.float32)
            m_txt = torch.zeros((n_tok,), dtype=torch.bool)
            if fork_local_idxs:
                fork_abs = [start - 1 + j for j in fork_local_idxs]
                d_f, m_f, l1_f = _extract_local_fisher_dirs(model, x_full, layer, fork_abs, n_dirs=n_dirs)
                for idx_local, dk, mk, l1k in zip(fork_local_idxs, d_f, m_f, l1_f):
                    if bool(mk.item()):
                        d_txt[idx_local] = dk
                        m_txt[idx_local] = True
                        if len(calib_samples) < 5:
                            calib_samples.append((seq_ids, start - 1 + idx_local, dk[0].numpy(), l1k))
            dirs_chunks.append(d_txt)
            mask_chunks.append(m_txt)

    if not H_chunks:
        raise ValueError("No usable tokens generated during self-supervised fork extraction")

    H = torch.cat(H_chunks, dim=0)
    median_hidden_norm = float(H.float().norm(dim=-1).median().item()) if len(H) else 0.0
    recommended_alpha = _calibrate_recommended_alpha(model, layer, calib_samples, H)

    ds: dict[str, Any] = {
        "H": H,
        "y_fork": torch.tensor(y_fork, dtype=torch.float32),
        "y_entropy": torch.tensor(y_entropy, dtype=torch.float32),
        "y_fisher": torch.tensor(y_fisher, dtype=torch.float32),
        "meta": {
            "layer": layer,
            "d_model": int(H.shape[1]),
            "n_texts": len(sources),
            "n_tokens": int(H.shape[0]),
            "n_dirs": int(n_dirs),
            "entropy_fork_thresh": float(entropy_fork_thresh),
            "prompt_style": prompt_style,
            "recommended_alpha": float(recommended_alpha),
            "median_hidden_norm": float(median_hidden_norm),
            "sources": sources,
            "text_ids": token_text_ids,
            "positions": token_positions,
        },
    }
    if n_dirs > 0:
        ds["y_dirs"] = torch.cat(dirs_chunks, dim=0)
        ds["y_dirs_mask"] = torch.cat(mask_chunks, dim=0)
    return ds


def save_dataset(dataset: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, p)
    return p


def load_dataset(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def _train_val_split(y_fork: torch.Tensor, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(y_fork.shape[0])
    if val_frac <= 0.0 or n < 2:
        idx = torch.arange(n, dtype=torch.long)
        return idx, idx

    g = torch.Generator().manual_seed(seed)
    pos = torch.where(y_fork >= 0.5)[0]
    neg = torch.where(y_fork < 0.5)[0]

    # Stratified split when both classes have at least 2 examples
    if len(pos) >= 2 and len(neg) >= 2:
        pos_perm = pos[torch.randperm(len(pos), generator=g)]
        neg_perm = neg[torch.randperm(len(neg), generator=g)]
        n_val_pos = max(1, min(len(pos) - 1, int(round(len(pos) * val_frac))))
        n_val_neg = max(1, min(len(neg) - 1, int(round(len(neg) * val_frac))))
        val_idx = torch.cat([pos_perm[:n_val_pos], neg_perm[:n_val_neg]])
        tr_idx = torch.cat([pos_perm[n_val_pos:], neg_perm[n_val_neg:]])
        return tr_idx[torch.randperm(len(tr_idx), generator=g)], val_idx[torch.randperm(len(val_idx), generator=g)]

    perm = torch.randperm(n, generator=g)
    n_val = max(1, min(n - 1, int(round(n * val_frac))))
    return perm[n_val:], perm[:n_val]


def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def train_fork_head(
    dataset: dict[str, Any],
    hidden_dim: int = 256,
    n_dirs: int = 0,
    epochs: int = 60,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a ForkHead with imbalanced BCE + Huber (entropy & Fisher) + optional subspace loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    H = torch.as_tensor(dataset["H"], dtype=torch.float32)
    dev = H.device
    y_fork = torch.as_tensor(dataset["y_fork"], dtype=torch.float32, device=dev)
    y_ent = torch.as_tensor(dataset["y_entropy"], dtype=torch.float32, device=dev)
    y_fis = torch.as_tensor(dataset["y_fisher"], dtype=torch.float32, device=dev)
    y_dirs_raw = dataset.get("y_dirs")
    y_dirs = torch.as_tensor(y_dirs_raw, dtype=torch.float32, device=dev) if y_dirs_raw is not None else None
    y_dirs_mask_raw = dataset.get("y_dirs_mask")
    if y_dirs is not None and y_dirs_mask_raw is not None:
        y_dirs_mask = torch.as_tensor(y_dirs_mask_raw, dtype=torch.bool, device=dev)
        y_dirs = y_dirs.clone()
        y_dirs[~y_dirs_mask] = 0.0

    d_model = int(H.shape[-1])
    head = ForkHead(d_model=d_model, hidden_dim=hidden_dim, n_dirs=n_dirs).to(dev)
    use_dirs = y_dirs is not None and head.predict_dirs

    tr_idx, val_idx = _train_val_split(y_fork.cpu(), val_frac, seed)
    tr_idx, val_idx = tr_idx.to(dev), val_idx.to(dev)

    y_fork_tr = y_fork[tr_idx]
    n_pos = float((y_fork_tr >= 0.5).sum().item())
    n_neg = float(len(y_fork_tr) - n_pos)
    pw_val = min(n_neg / n_pos, 20.0) if (n_pos > 0 and n_neg > 0) else 1.0
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_val], device=dev, dtype=torch.float32))

    def _batch_loss(idx: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        pred = head(H[idx])
        loss = (
            bce_fn(pred["fork_logit"], y_fork[idx])
            + F.huber_loss(pred["entropy"], y_ent[idx], delta=1.0)
            + F.huber_loss(pred["fisher_top"], y_fis[idx], delta=1.0)
        )
        if use_dirs:
            dir_loss, _ = _subspace_alignment_loss(pred["dirs"], y_dirs[idx])
            loss = loss + dir_loss
        return loss, pred

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = min(128, len(tr_idx))
    g = torch.Generator().manual_seed(seed)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    train_loss_last = 0.0

    for _ in range(int(epochs)):
        head.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g).to(dev)]
        ep_loss, n_seen = 0.0, 0
        for start_i in range(0, len(perm), batch_size):
            b_idx = perm[start_i : start_i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(b_idx)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(b_idx)
            n_seen += len(b_idx)
        train_loss_last = ep_loss / max(n_seen, 1)

        head.eval()
        with torch.no_grad():
            v_loss, _ = _batch_loss(val_idx)
            v_loss_f = float(v_loss.item())
        if v_loss_f < best_val:
            best_val = v_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    head.eval()

    with torch.no_grad():
        val_loss_t, pred_val = _batch_loss(val_idx)
        val_loss_f = float(val_loss_t.item())
        probs = torch.sigmoid(pred_val["fork_logit"]).detach().cpu().numpy()
        y_true = (y_fork[val_idx] >= 0.5).detach().cpu().numpy().astype(int)
        acc = float(((probs >= 0.5) == y_true).mean())
        auroc = _compute_auroc(y_true, probs)
        ent_mae = float((pred_val["entropy"] - y_ent[val_idx]).abs().mean().item())
        fis_mae = float((pred_val["fisher_top"] - y_fis[val_idx]).abs().mean().item())
        val_metrics: dict[str, Any] = {
            "auroc_or_acc": float(auroc) if auroc is not None else acc,
            "accuracy": acc,
            "auroc": auroc,
            "entropy_mae": ent_mae,
            "fisher_mae": fis_mae,
        }
        if use_dirs:
            val_td = y_dirs[val_idx]
            if val_td.ndim == 2:
                val_td = val_td.unsqueeze(-2)
            has_valid_dirs = bool((val_td.norm(dim=-1).max(dim=-1).values > 1e-6).any())
            if has_valid_dirs:
                _, mean_ov = _subspace_alignment_loss(pred_val["dirs"], y_dirs[val_idx])
                val_metrics["dir_subspace_sim"] = float(mean_ov.item())
            else:
                val_metrics["dir_subspace_sim"] = None

    ds_meta = dataset.get("meta", {}) or {}
    med_norm = ds_meta.get("median_hidden_norm")
    if med_norm is None:
        med_norm = float(H.float().norm(dim=-1).median().item()) if len(H) else 0.0
    rec_alpha = ds_meta.get("recommended_alpha")
    if rec_alpha is None or not (np.isfinite(float(rec_alpha)) and float(rec_alpha) > 0.0):
        rec_alpha = _fallback_alpha_from_hidden(H)

    head.meta = {
        "layer": ds_meta.get("layer"),
        "recommended_alpha": float(rec_alpha),
        "median_hidden_norm": float(med_norm),
        "train_loss": train_loss_last,
        "val_loss": val_loss_f,
        "val_metrics": val_metrics,
    }

    saved: str | None = None
    if save_path is not None:
        saved = str(head.save(save_path, meta=head.meta))

    return {
        "head": head,
        "train_loss": train_loss_last,
        "val_loss": val_loss_f,
        "val_metrics": val_metrics,
        "saved": saved,
    }


class LiveForkScorer:
    """Zero-extra-pass forward hook recording ForkHead predictions at every prefill/decode step."""

    def __init__(self, head: ForkHead, model, layer: int):
        self.head = head.eval()
        self.layer = int(layer) % arch.n_layers(model)
        blk = arch.layers(model)[self.layer]
        first_param = next(blk.parameters(), None)
        if first_param is not None:
            self.head = self.head.to(device=first_param.device)
        self._history: list[dict[str, Any]] = []
        self._step = 0
        self._pause_depth = 0
        self._handle = blk.register_forward_hook(self._hook)

    @property
    def is_paused(self) -> bool:
        return self._pause_depth > 0

    @property
    def recommended_alpha(self) -> float | None:
        return self.head.recommended_alpha

    @contextlib.contextmanager
    def paused(self):
        """Re-entrant context manager disabling hook execution during lookahead/compass/calibration."""
        self._pause_depth += 1
        try:
            yield self
        finally:
            self._pause_depth = max(0, self._pause_depth - 1)

    def _hook(self, mod, inp, out):
        if self._pause_depth > 0:
            return None
        h = out[0] if isinstance(out, tuple) else out
        h_last = h[:, -1, :] if h.ndim == 3 else h[-1:]
        head_param = next(self.head.parameters())
        h_in = h_last.detach().to(device=head_param.device, dtype=head_param.dtype)
        with torch.no_grad():
            pred = self.head(h_in)

        fork_logit = pred["fork_logit"]
        fork_prob = torch.sigmoid(fork_logit)
        ent = pred["entropy"]
        fis = pred["fisher_top"]
        dirs = pred["dirs"]

        seq_len = int(h.shape[1]) if h.ndim == 3 else int(h.shape[0])
        if h_in.shape[0] == 1:
            rec: dict[str, Any] = {
                "step": self._step,
                "seq_len": seq_len,
                "fork_prob": float(fork_prob[0].item()),
                "fork_logit": float(fork_logit[0].item()),
                "pred_entropy": float(ent[0].item()),
                "pred_fisher_top": float(fis[0].item()),
                "dirs": dirs[0].detach().cpu() if dirs is not None else None,
                "recommended_alpha": self.recommended_alpha,
            }
        else:
            rec = {
                "step": self._step,
                "seq_len": seq_len,
                "fork_prob": [float(v) for v in fork_prob.detach().cpu().tolist()],
                "fork_logit": [float(v) for v in fork_logit.detach().cpu().tolist()],
                "pred_entropy": [float(v) for v in ent.detach().cpu().tolist()],
                "pred_fisher_top": [float(v) for v in fis.detach().cpu().tolist()],
                "dirs": dirs.detach().cpu() if dirs is not None else None,
                "recommended_alpha": self.recommended_alpha,
            }
        self._history.append(rec)
        self._step += 1
        return None

    def latest(self) -> dict[str, Any] | None:
        return self._history[-1] if self._history else None

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._step = 0

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "LiveForkScorer":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Extract fork dataset and train real-time ForkHead.")
    ap.add_argument("--model", default="g3_4b_pt")
    ap.add_argument("--explorer-dir", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-dirs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--entropy-thresh", type=float, default=1.5)
    ap.add_argument("--prompt-style", choices=("legacy", "doc"), default="legacy")
    ap.add_argument("--from-texts", action="store_true", help="Self-supervised extraction directly from texts.yaml + tone_axes.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args(argv)

    explorer_dir = Path(args.explorer_dir) if args.explorer_dir else (ROOT / "results" / "explorer" / args.model)
    out_path = Path(args.out) if args.out else (explorer_dir / "fork_head.pt")
    ds_path = Path(args.dataset) if args.dataset else None

    if ds_path is not None and ds_path.exists():
        ds = load_dataset(ds_path)
    else:
        from explorer.frame import load
        tok, model = load(args.model)
        use_self_sup = args.from_texts or not (explorer_dir / "branches").is_dir()
        if use_self_sup:
            ds = extract_self_supervised_fork_dataset(
                model,
                tok,
                layer=args.layer,
                entropy_fork_thresh=args.entropy_thresh,
                prompt_style=args.prompt_style,
                n_dirs=args.n_dirs,
            )
        else:
            ds = extract_fork_dataset(
                model,
                tok,
                explorer_dir,
                layer=args.layer,
                entropy_fork_thresh=args.entropy_thresh,
                prompt_style=args.prompt_style,
                n_dirs=args.n_dirs,
            )
        if ds_path is not None:
            save_dataset(ds, ds_path)

    res = train_fork_head(
        ds,
        hidden_dim=args.hidden_dim,
        n_dirs=args.n_dirs,
        epochs=args.epochs,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        save_path=out_path,
    )
    summary = {
        "model": args.model,
        "layer": ds.get("meta", {}).get("layer"),
        "n_tokens": int(ds["H"].shape[0]),
        "n_dirs": int(args.n_dirs),
        "recommended_alpha": res["head"].recommended_alpha,
        "params": res["head"].param_count(),
        "train_loss": res["train_loss"],
        "val_loss": res["val_loss"],
        "val_metrics": res["val_metrics"],
        "saved": res["saved"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
