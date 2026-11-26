"""Pure-PyTorch LoRA adapters and continuous Spectral ReFT controllers (no peft dependency).

Components:
  LoRALinear             wraps nn.Linear with frozen base weights, Kaiming A, exact-zero B
  inject_lora            replaces target linear projections inside decoder blocks with LoRALinear
  lora_disabled          context manager temporarily toggling LoRA off (for base-model KL anchor)
  save_lora / load_lora  checkpoint persistence for LoRA config + weights
  SpectralReFTBlock      low-rank gated residual edit Δh = ((h Rᵀ) ⊙ gate(z) + bias(z)) R with
                         exact Δh(z=0) == 0 and smooth norm clamp max_delta_norm * tanh(||Δh||/M)
  SpectralReFTController attaches SpectralReFTBlocks to decoder blocks via forward hooks
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ut import arch


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear.

    Initializes A with Kaiming uniform and B with exact zeros so the initial
    forward output is bit-identical to `base_linear`.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_linear).__name__}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.enabled = True

        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        x_drop = self.dropout(x).to(self.lora_A.dtype)
        delta = F.linear(F.linear(x_drop, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta.to(base_out.dtype)

    @torch.no_grad()
    def merge(self, inplace: bool = False) -> nn.Linear:
        """Fuse LoRA weights into an nn.Linear (returns a new Linear unless inplace=True)."""
        _check_mergeable_base(self.base, self.out_features, self.in_features)
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        delta_w = delta_w.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        if inplace:
            self.base.weight.add_(delta_w)
            self.lora_B.zero_()
            return self.base
        fused = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        fused.weight.copy_(self.base.weight + delta_w)
        if self.base.bias is not None:
            fused.bias.copy_(self.base.bias)
        return fused

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "lora_A": self.lora_A.detach().cpu().clone(),
            "lora_B": self.lora_B.detach().cpu().clone(),
        }


def _check_mergeable_base(base: nn.Module, out_features: int, in_features: int) -> None:
    """Validate that `base` is a dense floating-point linear layer safe for LoRA weight merging."""
    w = getattr(base, "weight", None)
    if w is None or not isinstance(w, torch.Tensor):
        raise RuntimeError(
            f"Cannot merge LoRA into {type(base).__name__}: base layer has no Tensor weight."
        )
    quant_detected = (
        type(base).__name__ in ("Linear4bit", "Linear8bitLt")
        or getattr(w, "quant_state", None) is not None
        or getattr(base, "quant_state", None) is not None
        or getattr(w, "CB", None) is not None
        or getattr(w, "SCB", None) is not None
        or bool(getattr(w, "bnb_quantized", False))
    )
    if quant_detected:
        raise RuntimeError(
            f"Cannot merge LoRA weights into quantized layer {type(base).__name__} "
            f"(weight dtype={w.dtype}, shape={tuple(w.shape)}). Keep LoRA unmerged during inference "
            "or dequantize the base weight first."
        )
    if not w.is_floating_point():
        raise RuntimeError(
            f"Cannot merge LoRA weights into non-floating-point weight (dtype={w.dtype}) "
            f"on {type(base).__name__}."
        )
    expected_shape = (int(out_features), int(in_features))
    if tuple(w.shape) != expected_shape:
        raise RuntimeError(
            f"Cannot merge LoRA weights into {type(base).__name__}: expected weight shape "
            f"{expected_shape}, got {tuple(w.shape)} (possibly packed/quantized)."
        )


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable LoRA parameters in `model`."""
    params: list[nn.Parameter] = []
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            params.extend([mod.lora_A, mod.lora_B])
    return params


@contextmanager
def lora_disabled(model: nn.Module):
    """Temporarily disable all LoRALinear modules in `model` (for KL anchoring to base)."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, state in zip(mods, prev):
            m.enabled = state


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Iterable[str] = ("q_proj", "v_proj", "o_proj"),
    layers: Iterable[int] | None = None,
    dropout: float = 0.0,
) -> list[str]:
    """Replace matching nn.Linear projections inside decoder layers with LoRALinear.

    Freezes all non-LoRA parameters in `model` and returns the list of injected
    submodule paths (relative to decoder layers, e.g. 'layers.0.self_attn.q_proj').
    """
    model.requires_grad_(False)
    targets = set(target_modules)
    blocks = arch.layers(model)
    selected = list(range(len(blocks))) if layers is None else [int(l) for l in layers]
    injected: list[str] = []

    for l_idx in selected:
        block = blocks[l_idx]
        # Collect matches first so we don't mutate the module tree while iterating.
        to_replace: list[tuple[nn.Module, str, str, nn.Linear]] = []
        for sub_name, sub_mod in block.named_modules():
            if not sub_name:
                continue
            leaf = sub_name.split(".")[-1]
            if leaf in targets and isinstance(sub_mod, nn.Linear):
                parent_path = sub_name.rsplit(".", 1)[0] if "." in sub_name else ""
                parent = block.get_submodule(parent_path) if parent_path else block
                if isinstance(parent, LoRALinear):
                    continue
                to_replace.append((parent, leaf, sub_name, sub_mod))

        for parent, leaf, sub_name, sub_mod in to_replace:
            setattr(parent, leaf, LoRALinear(sub_mod, rank=rank, alpha=alpha, dropout=dropout))
            injected.append(f"layers.{l_idx}.{sub_name}")

    model._lora_config = {
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": sorted(targets),
        "layers": selected,
        "injected": list(injected),
    }
    return injected


def save_lora(model: nn.Module, path: str | Path) -> Path:
    """Save LoRA config and adapter weights via torch.save."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            weights[name] = mod.lora_state_dict()
    if not weights:
        raise ValueError("No LoRALinear modules found in model to save")
    config = dict(getattr(model, "_lora_config", {}))
    if not config:
        first = next(m for m in model.modules() if isinstance(m, LoRALinear))
        config = {
            "rank": first.rank,
            "alpha": first.alpha,
            "dropout": 0.0,
            "target_modules": sorted({n.split(".")[-1] for n in weights}),
            "layers": None,
        }
    torch.save({"config": config, "weights": weights}, p)
    return p


def load_lora(model: nn.Module, path: str | Path) -> list[str]:
    """Load saved LoRA weights into `model` (injecting LoRA automatically if needed)."""
    p = Path(path)
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(p, map_location="cpu")

    config = ckpt["config"]
    weights = ckpt["weights"]

    existing = {name: mod for name, mod in model.named_modules() if isinstance(mod, LoRALinear)}
    if not all(name in existing for name in weights):
        inject_lora(
            model,
            rank=config["rank"],
            alpha=config["alpha"],
            target_modules=config["target_modules"],
            layers=config.get("layers"),
            dropout=config.get("dropout", 0.0),
        )
        existing = {name: mod for name, mod in model.named_modules() if isinstance(mod, LoRALinear)}

    loaded: list[str] = []
    with torch.no_grad():
        for name, state in weights.items():
            if name not in existing:
                raise KeyError(f"Saved LoRA module '{name}' not found in model")
            mod = existing[name]
            mod.lora_A.copy_(state["lora_A"].to(device=mod.lora_A.device, dtype=mod.lora_A.dtype))
            mod.lora_B.copy_(state["lora_B"].to(device=mod.lora_B.device, dtype=mod.lora_B.dtype))
            loaded.append(name)
    return loaded


# ------------------------------------------------------------------ Spectral ReFT
class SpectralReFTBlock(nn.Module):
    """Continuous low-rank gated representation intervention block.

    Computes:
        raw   = ((h @ R^T) * gate(z) + bias(z)) @ R
        delta = raw * (M * tanh(r / M) / r),  r = sqrt(||raw||^2 + eps^2)

    Because `gate` and `bias` are bias-free Linear(n_axes, rank) projections:
      - At z == 0, gate(0) == 0 and bias(0) == 0, so delta == 0 identically.
      - For any z, ||delta|| <= max_delta_norm strictly, with smooth NaN-free gradients.
    """

    def __init__(
        self,
        d_model: int,
        n_axes: int,
        rank: int = 8,
        max_delta_norm: float = 5.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if n_axes <= 0:
            raise ValueError(f"n_axes must be > 0, got {n_axes}")

        self.d_model = int(d_model)
        self.n_axes = int(n_axes)
        self.rank = int(rank)
        self.max_delta_norm = float(max_delta_norm)

        self.R = nn.Parameter(torch.empty(self.rank, self.d_model, dtype=torch.float32))
        nn.init.orthogonal_(self.R)

        self.gate = nn.Linear(self.n_axes, self.rank, bias=False, dtype=torch.float32)
        self.bias = nn.Linear(self.n_axes, self.rank, bias=False, dtype=torch.float32)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.bias.weight, mean=0.0, std=0.05)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z_b = z.unsqueeze(0).expand(h.shape[0], -1)
        else:
            z_b = z
        z_b = z_b.to(device=h.device, dtype=self.R.dtype)
        h_f = h.to(self.R.dtype)

        proj = torch.matmul(h_f, self.R.t())            # (B, T, rank)
        g = self.gate(z_b).unsqueeze(1)                 # (B, 1, rank)
        b = self.bias(z_b).unsqueeze(1)                 # (B, 1, rank)
        delta = torch.matmul(proj * g + b, self.R)      # (B, T, d_model)

        if self.max_delta_norm > 0.0:
            r = torch.sqrt((delta * delta).sum(dim=-1, keepdim=True) + 1e-16)
            scale = (self.max_delta_norm * torch.tanh(r / self.max_delta_norm)) / r
            delta = delta * scale

        return delta.to(h.dtype)


class SpectralReFTController:
    """Manages SpectralReFTBlock forward hooks across selected decoder layers."""

    def __init__(
        self,
        model: nn.Module,
        layers: Iterable[int],
        axis_names: Iterable[str],
        rank: int = 8,
        max_delta_norm: float = 5.0,
    ):
        self.model = model
        self.layers = [int(l) for l in layers]
        self.axis_names = list(axis_names)
        self.rank = int(rank)
        self.max_delta_norm = float(max_delta_norm)

        d_model = arch.hidden_size(model)
        dev = arch.text_model(model).embed_tokens.weight.device
        self.blocks = nn.ModuleDict(
            {
                str(l): SpectralReFTBlock(
                    d_model=d_model,
                    n_axes=len(self.axis_names),
                    rank=self.rank,
                    max_delta_norm=self.max_delta_norm,
                ).to(dev)
                for l in self.layers
            }
        )

        self._z: torch.Tensor | None = None
        self._start_pos: int = 0
        self._pos: dict[int, int] = {l: 0 for l in self.layers}
        self._explicit_pos: dict[int, int | None] = {l: None for l in self.layers}
        self._handles = [
            arch.layers(model)[l].register_forward_hook(self._make_hook(l))
            for l in self.layers
        ]

    def sync_pos(self, pos: int) -> None:
        """Explicitly set the absolute start position of the next forward pass across all hooked layers."""
        p = int(pos)
        self._pos = {l: p for l in self.layers}
        self._explicit_pos = {l: p for l in self.layers}

    def _make_hook(self, layer_idx: int):
        block = self.blocks[str(layer_idx)]

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            T = h.shape[1]
            exp = self._explicit_pos.get(layer_idx)
            if exp is not None:
                start = exp
                self._explicit_pos[layer_idx] = None
                self._pos[layer_idx] = start + T
            elif T > 1:
                start = 0
                self._pos[layer_idx] = T
            else:
                start = self._pos[layer_idx]
                self._pos[layer_idx] += 1

            if self._z is None:
                return None
            idx = torch.arange(start, start + T, device=h.device)
            mask = idx >= self._start_pos
            if not bool(mask.any()):
                return None

            delta = block(h, self._z)
            new = h + delta * mask[None, :, None].to(h.dtype)
            return (new, *out[1:]) if isinstance(out, tuple) else new

        return hook

    def set_coords(
        self,
        z: Mapping[str, float] | torch.Tensor | Iterable[float] | None,
        start_pos: int = 0,
    ) -> None:
        """Set active continuous slider coordinates (dict, tensor/sequence, or None)."""
        self._start_pos = int(start_pos)
        self._pos = {l: 0 for l in self.layers}
        self._explicit_pos = {l: None for l in self.layers}
        if z is None:
            self._z = None
            return
        if isinstance(z, Mapping):
            vals = [float(z.get(name, 0.0)) for name in self.axis_names]
            self._z = torch.tensor(vals, dtype=torch.float32)
        elif isinstance(z, torch.Tensor):
            self._z = z.to(dtype=torch.float32)
        else:
            self._z = torch.tensor(list(z), dtype=torch.float32)

        if self._z.shape[-1] != len(self.axis_names):
            raise ValueError(
                f"Expected coordinate vector of length {len(self.axis_names)}, got shape {tuple(self._z.shape)}"
            )

    def parameters(self):
        return self.blocks.parameters()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.blocks.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.blocks.load_state_dict(state_dict)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "layers": self.layers,
                "axis_names": self.axis_names,
                "rank": self.rank,
                "max_delta_norm": self.max_delta_norm,
            },
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.blocks.state_dict().items()},
        }
        torch.save(payload, p)
        return p

    def load(self, path: str | Path) -> SpectralReFTController:
        """Load saved SpectralReFT weights into this controller."""
        p = Path(path)
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(p, map_location="cpu")
        cfg = ckpt["config"]
        if list(cfg["layers"]) != self.layers or list(cfg["axis_names"]) != self.axis_names:
            raise ValueError(
                f"Checkpoint config mismatch: expected layers={self.layers}, axes={self.axis_names}, "
                f"got layers={cfg['layers']}, axes={cfg['axis_names']}"
            )
        self.blocks.load_state_dict(ckpt["state_dict"])
        return self

    @classmethod
    def from_checkpoint(cls, model: nn.Module, path: str | Path) -> SpectralReFTController:
        """Instantiate and attach a SpectralReFTController directly from a saved file."""
        p = Path(path)
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(p, map_location="cpu")
        cfg = ckpt["config"]
        ctrl = cls(
            model=model,
            layers=cfg["layers"],
            axis_names=cfg["axis_names"],
            rank=cfg["rank"],
            max_delta_norm=cfg.get("max_delta_norm", 5.0),
        )
        ctrl.blocks.load_state_dict(ckpt["state_dict"])
        return ctrl

    def remove(self) -> None:
        for hd in self._handles:
            hd.remove()
        self._handles.clear()
        self._z = None
