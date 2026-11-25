"""Global tone/practice translational axes and blind-spot bridge.

Extracts Fisher-whitened natural contrast directions across a multi-reference
parallel dataset (tone_axes.yaml) at the translation boundary b = start - 1,
and bridges blind/spectrum.py Comparator eigenvectors into the same
GlobalAxisBank interface so both global practice sliders and blind-spot axes
can be composed and steered live in the workbench.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from blind import spectrum as blind_spectrum
from blind.fisher import seq_logp
from explorer.directions import alpha_for, natural
from explorer.frame import HERE, build_prompt
from ut import arch

DEFAULT_DATASET = HERE / "data" / "tone_axes.yaml"


def load_tone_axes_dataset(path: str | Path | None = None) -> dict:
    """Load the curated multi-reference tone/practice parallel dataset."""
    p = Path(path) if path is not None else DEFAULT_DATASET
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "axes" not in data or "items" not in data:
        raise ValueError(f"Invalid tone_axes dataset at {p}: expected 'axes' and 'items'")
    return data


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    nrm = float(np.linalg.norm(v))
    return v / max(nrm, 1e-300)


def default_layers(model) -> list[int]:
    """Choose a compact set of middle-to-upper decoder layers for steering."""
    L = arch.n_layers(model)
    if L <= 4:
        return list(range(L))
    return sorted({int(round(f * (L - 1))) for f in (0.25, 0.5, 0.75)})


class GlobalAxisBank:
    """Bank of unit-norm steering axes across decoder layers with calibration scales.

    Attributes:
        vectors:      {axis_name: {layer: unit_vec (d,) float64}}
        scales:       {axis_name: float} calibration multiplier at slider value 1.0
        descriptions: {axis_name: {"neg": str, "pos": str}}
        meta:         arbitrary metadata dict (model, layers, eps, ...)
    """

    def __init__(
        self,
        vectors: dict[str, dict[int, np.ndarray]],
        scales: dict[str, float] | None = None,
        descriptions: dict[str, dict[str, str]] | None = None,
        meta: dict | None = None,
    ):
        self.vectors: dict[str, dict[int, np.ndarray]] = {
            name: {int(l): np.asarray(v, dtype=np.float64).copy() for l, v in lmap.items()}
            for name, lmap in vectors.items()
        }
        self.scales: dict[str, float] = {
            name: float((scales or {}).get(name, 1.0)) for name in self.vectors
        }
        self.descriptions: dict[str, dict[str, str]] = {
            name: dict((descriptions or {}).get(name, {})) for name in self.vectors
        }
        self.meta: dict = dict(meta or {})
        inferred_layers = sorted({l for lmap in self.vectors.values() for l in lmap})
        self.meta.setdefault("layers", inferred_layers)

    @property
    def axis_names(self) -> list[str]:
        return list(self.vectors.keys())

    @property
    def layers(self) -> list[int]:
        if self.vectors:
            return sorted({l for lmap in self.vectors.values() for l in lmap})
        return [int(l) for l in self.meta.get("layers", [])]

    @property
    def model(self) -> str | None:
        return self.meta.get("model")

    @property
    def hidden_size(self) -> int:
        for lmap in self.vectors.values():
            for v in lmap.values():
                return int(v.shape[0])
        return 0

    def compose(self, coords: dict[str, float]) -> dict[int, np.ndarray]:
        """Linearly combine active slider values z_k in [-1, 1] into {layer: vec (d,)}."""
        d = self.hidden_size
        out = {layer: np.zeros(d, dtype=np.float64) for layer in self.layers}
        for name, lmap in self.vectors.items():
            z = float(coords.get(name, 0.0))
            if z == 0.0:
                continue
            scale = float(self.scales.get(name, 1.0))
            coeff = z * scale
            for layer, vec in lmap.items():
                out[layer] = out[layer] + coeff * vec
        return out

    def merge(self, other: GlobalAxisBank) -> GlobalAxisBank:
        """Return a new GlobalAxisBank combining axes from self and other."""
        vecs = {k: {l: v.copy() for l, v in lmap.items()} for k, lmap in self.vectors.items()}
        scales = dict(self.scales)
        descs = {k: dict(v) for k, v in self.descriptions.items()}
        for k, lmap in other.vectors.items():
            vecs[k] = {l: v.copy() for l, v in lmap.items()}
            scales[k] = other.scales.get(k, 1.0)
            descs[k] = dict(other.descriptions.get(k, {}))
        meta = dict(self.meta)
        meta["layers"] = sorted({l for lmap in vecs.values() for l in lmap})
        return GlobalAxisBank(vecs, scales=scales, descriptions=descs, meta=meta)

    def save(self, path: str | Path) -> Path:
        """Save axis vectors, scales, descriptions, and metadata to a single .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        axis_names = self.axis_names
        payload_meta = {
            "axis_names": axis_names,
            "axis_layers": {name: sorted(self.vectors[name].keys()) for name in axis_names},
            "scales": self.scales,
            "descriptions": self.descriptions,
            "meta": self.meta,
        }
        arrs: dict[str, np.ndarray] = {
            "meta": np.frombuffer(json.dumps(payload_meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        }
        for idx, name in enumerate(axis_names):
            for layer, vec in self.vectors[name].items():
                arrs[f"v__{idx}__{int(layer)}"] = np.asarray(vec, dtype=np.float64)
        with open(p, "wb") as f:
            np.savez(f, **arrs)
        return p

    @classmethod
    def load(cls, path: str | Path) -> GlobalAxisBank:
        """Load a GlobalAxisBank saved with .save(path)."""
        p = Path(path)
        with np.load(p, allow_pickle=False) as z:
            payload = json.loads(z["meta"].tobytes().decode("utf-8"))
            axis_names = payload["axis_names"]
            axis_layers = payload["axis_layers"]
            vectors: dict[str, dict[int, np.ndarray]] = {}
            for idx, name in enumerate(axis_names):
                vectors[name] = {
                    int(layer): np.asarray(z[f"v__{idx}__{int(layer)}"], dtype=np.float64)
                    for layer in axis_layers[name]
                }
        return cls(
            vectors=vectors,
            scales=payload.get("scales", {}),
            descriptions=payload.get("descriptions", {}),
            meta=payload.get("meta", {}),
        )


def grads_at_layers(
    model,
    prefix: list[int],
    conts: list[list[int]],
    layers: list[int],
    b: int,
    batch: int = 8,
) -> dict[int, np.ndarray]:
    """Per-sample gradients of log p(cont | prefix) w.r.t. block-`l` outputs at position b.

    Computes all requested layers simultaneously in a single forward+backward pass per batch
    by attaching independent zero leaves at position `b` on each block.
    """
    layers = [int(l) for l in layers]
    d = arch.hidden_size(model)
    dev = arch.text_model(model).embed_tokens.weight.device
    out: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    state: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(mod, inp, o):
            h = o[0] if isinstance(o, tuple) else o
            z = state.get(layer_idx)
            if z is None or h.shape[1] <= b:
                return None
            oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
            oh[b] = 1
            new = h + z.to(h.dtype)[:, None, :] * oh[None, :, None]
            return (new, *o[1:]) if isinstance(o, tuple) else new
        return hook

    handles = [arch.layers(model)[l].register_forward_hook(_make_hook(l)) for l in layers]
    try:
        with torch.enable_grad():
            cur_batch = max(1, int(batch))
            c = 0
            while c < len(conts):
                part = [y if y else [prefix[-1]] for y in conts[c:c + cur_batch]]
                try:
                    for l in layers:
                        state[l] = torch.zeros(len(part), d, device=dev, dtype=torch.float32, requires_grad=True)
                    lp = seq_logp(model, prefix, part)
                    grads = torch.autograd.grad(lp.sum(), [state[l] for l in layers])
                    for l, g in zip(layers, grads):
                        out[l].append(g.detach().float().cpu().numpy())
                    state.clear()
                    c += len(part)
                except torch.cuda.OutOfMemoryError:
                    state.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if cur_batch == 1:
                        raise
                    cur_batch = max(1, cur_batch // 2)
    finally:
        for hd in handles:
            hd.remove()

    return {l: np.concatenate(out[l], axis=0).astype(np.float64) for l in layers}


def extract_global_axes(
    model,
    tok,
    dataset: dict | None = None,
    layers: list[int] | None = None,
    eps: float = 1e-2,
    max_items: int | None = None,
    batch: int = 8,
    target_kl: float = 0.5,
    pole_threshold: float = 0.5,
) -> GlobalAxisBank:
    """Extract Fisher-whitened natural contrast directions for all global axes.

    For each item, computes gradients at the start-of-translation boundary b = start - 1
    across `layers`, pools positive-pole (coord >= 0.5) and negative-pole (coord <= -0.5)
    gradients across all sentences, and solves (G_hat + lam I)^{-1} (g_pos - g_neg).
    """
    if dataset is None:
        dataset = load_tone_axes_dataset()
    if layers is None:
        layers = default_layers(model)
    layers = [int(l) for l in layers]

    axes_meta: dict[str, dict[str, str]] = dataset["axes"]
    axis_names = list(axes_meta.keys())
    items = dataset["items"][:max_items] if max_items is not None else dataset["items"]

    pos_pool: dict[str, dict[int, list[np.ndarray]]] = {a: {l: [] for l in layers} for a in axis_names}
    neg_pool: dict[str, dict[int, list[np.ndarray]]] = {a: {l: [] for l in layers} for a in axis_names}

    for item in items:
        prefix, start = build_prompt(tok, item["source"])
        b = start - 1
        trs = item["translations"]
        conts = [tok(" " + tr["text"], add_special_tokens=False)["input_ids"] for tr in trs]
        G_by_layer = grads_at_layers(model, prefix, conts, layers, b, batch=batch)

        for idx, tr in enumerate(trs):
            coords = tr.get("coords", {})
            for axis in axis_names:
                val = float(coords.get(axis, 0.0))
                if val >= pole_threshold:
                    for l in layers:
                        pos_pool[axis][l].append(G_by_layer[l][idx])
                elif val <= -pole_threshold:
                    for l in layers:
                        neg_pool[axis][l].append(G_by_layer[l][idx])

    vectors: dict[str, dict[int, np.ndarray]] = {}
    scales: dict[str, float] = {}
    for axis in axis_names:
        vectors[axis] = {}
        layer_alphas: list[float] = []
        for l in layers:
            if not pos_pool[axis][l] or not neg_pool[axis][l]:
                raise ValueError(
                    f"Axis '{axis}' at layer {l} lacks contrast examples "
                    f"(pos={len(pos_pool[axis][l])}, neg={len(neg_pool[axis][l])})"
                )
            GA = np.stack(pos_pool[axis][l], axis=0)
            GB = np.stack(neg_pool[axis][l], axis=0)
            v = natural(GA, GB, eps=eps)
            vectors[axis][l] = v
            Z = np.vstack([GA, GB])
            q = float(np.mean((Z @ v) ** 2))
            if q > 1e-12:
                layer_alphas.append(alpha_for(Z, v, target_kl))
            else:
                layer_alphas.append(1.0)
        scales[axis] = float(np.median(layer_alphas)) / float(np.sqrt(max(len(layers), 1)))

    meta = {
        "model": getattr(model, "key", None),
        "layers": layers,
        "eps": float(eps),
        "n_items": len(items),
        "target_kl": float(target_kl),
    }
    return GlobalAxisBank(vectors=vectors, scales=scales, descriptions=axes_meta, meta=meta)


def from_blind_comparator(
    comparator: blind_spectrum.Comparator,
    src: str,
    tgt: str,
    layer: int,
    axis_name: str = "blind_spot_1",
    top_k: int = 1,
    eps: float = 1e-2,
    d_model: int | None = None,
    target_kl: float = 0.5,
    seen: bool = False,
) -> GlobalAxisBank:
    """Bridge blind/spectrum.py Comparator generalized eigenvectors into a GlobalAxisBank.

    If the comparator vectors are flattened over multiple source positions (D = n_pos * d_model),
    pass `d_model` to average position-wise components into a single d-dimensional steering
    direction (falling back to the highest-energy position if the mean cancels).
    """
    dirs = blind_spectrum.directions(comparator, src, tgt, n_dirs=top_k, eps=eps, seen=seen)
    if not dirs:
        raise ValueError(f"Comparator returned no directions for src={src!r}, tgt={tgt!r}")

    if len(dirs) == 1:
        names = [axis_name]
    elif axis_name.endswith("_1"):
        stem = axis_name[:-2]
        names = [f"{stem}_{i + 1}" for i in range(len(dirs))]
    else:
        names = [f"{axis_name}_{i + 1}" for i in range(len(dirs))]

    vectors: dict[str, dict[int, np.ndarray]] = {}
    scales: dict[str, float] = {}
    descriptions: dict[str, dict[str, str]] = {}
    eigenvalues: dict[str, float] = {}

    layer = int(layer)
    for name, entry in zip(names, dirs):
        v = np.asarray(entry["v"], dtype=np.float64)
        if d_model is not None and v.shape[0] != d_model:
            if v.shape[0] % d_model != 0:
                raise ValueError(f"Eigenvector length {v.shape[0]} is not divisible by d_model={d_model}")
            v2d = v.reshape(-1, d_model)
            v_mean = v2d.mean(axis=0)
            if float(np.linalg.norm(v_mean)) > 1e-12:
                v = _unit(v_mean)
            else:
                best = int(np.argmax((v2d ** 2).sum(axis=1)))
                v = _unit(v2d[best])
        else:
            v = _unit(v)

        vectors[name] = {layer: v}
        q_max = max(float(entry["q_src"]), float(entry["q_tgt"]))
        scales[name] = blind_spectrum.alpha_for(entry["q_src"], entry["q_tgt"], target_kl=target_kl) if q_max > 1e-12 else 1.0
        descriptions[name] = {
            "neg": f"{tgt} (blind-spot -)",
            "pos": f"{src} (blind-spot +)",
        }
        eigenvalues[name] = float(entry["lam"])

    meta = {
        "source": "blind_comparator",
        "src": src,
        "tgt": tgt,
        "layers": [layer],
        "eps": float(eps),
        "seen": bool(seen),
        "eigenvalues": eigenvalues,
    }
    return GlobalAxisBank(vectors=vectors, scales=scales, descriptions=descriptions, meta=meta)
