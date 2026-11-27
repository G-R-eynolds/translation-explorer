"""Stateful KV-cached Live Co-Translation Engine (v2).

Designed for interactive token/word-by-word translation of long poems, multi-stanza
verse, and formatted paragraphs on quantized Gemma 3 12B / 27B (and smaller models).

Key capabilities:
  - Persistent mutable KV cache per session (prefill once; ~15 ms forward step;
    instant rewind via negative-delta transformers 5.x cache cropping with safe
    pre/post validation for Gemma 3 DynamicSlidingWindowLayer caches).
  - Tier-1 instant frontier on every step (zero backward passes):
      * Adaptive regime detection (`linear` | `subtle` | `fork`) combining Shannon
        entropy, post-dedup branch spread, and optional amortised `ForkHead`.
      * Surface-duplicate collapsing + inline 1-step function-word peek-ahead so forks
        hiding behind `"a"`/`"the"`/`"of"` surface in a single KV clone.
      * Batched KV-cached lookahead across up to 6 distinct starters, semantically
        and lexically deduplicated into genuinely distinct continuation ribbons, with
        zero-extra-forward depth-2 `sub_branches` previews.
  - Fast auto-glide SSE generator: streams whole words quickly through linear passages
    (skipping heavy lookahead until ambiguous/paused) and releases the engine lock
    between steps so concurrent requests never starve.
  - Tier-2 on-demand & async-precomputed 2D Spectral Compass at the active cursor
    (<10 ms zero-backward-pass amortised `ForkHead(n_dirs=2)` subspace with branch
    projection, or OOM-safe `DOC_STOPS` gradient-space probing).
  - Non-destructive explored-path history tree with true `redo()` / forward stepping
    and nested branching (`n0 -> n1 -> n2`), plus lightweight source↔target alignment.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
import re
import threading
import time
import uuid
from typing import Any, Generator

import numpy as np
import torch

from blind.fisher import seq_logp
from explorer import readings as R
from explorer.directions import alpha_for, calibrate
from explorer.frame import DOC_STOPS, _partial_stop_pos, build_doc_prompt, cut_doc
from explorer.live_engine import READING_PALETTE, _STOPWORDS
from explorer.recipe_live import branch_semantic_dispersion
from ut import arch

# Common English function words that often sit immediately before an interpretive fork
# (e.g. "he has [a] [magic / spirit / touch of duende]").
_PEEK_FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "into", "as", "his", "her", "its", "their", "my", "your", "our", "this", "that",
    "he", "she", "it", "i", "we", "they", "is", "are", "was", "were", "has", "have",
    "had", "be", "and", "or", "but", "not", "very", "so",
}

_PUNCT_ONLY = re.compile(r"^[\s.,;:!?…—–\-\"'()\[\]{}«»¿¡]+$")
_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…»\"])\s+")
_SENT_END_RE = re.compile(r"[.!?…»\"]")

_NUMBER_WORDS = {
    "zero": "0",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS.keys()) + r")\b")


# ------------------------------------------------------------------ KV cache helpers
def _cache_seq_len(pkv: Any) -> int:
    if pkv is None:
        return 0
    if hasattr(pkv, "get_seq_length"):
        try:
            return int(pkv.get_seq_length())
        except Exception:
            pass
    if hasattr(pkv, "key_cache") and pkv.key_cache:
        return int(pkv.key_cache[0].shape[-2])
    if hasattr(pkv, "layers") and pkv.layers:
        k = getattr(pkv.layers[0], "keys", None)
        if k is not None:
            return int(k.shape[-2])
    if isinstance(pkv, (tuple, list)) and pkv and isinstance(pkv[0], (tuple, list)):
        return int(pkv[0][0].shape[-2])
    return 0


def _cache_has_sliding_layers(pkv: Any) -> bool:
    """Return True if `pkv` contains any sliding-window cache layer (e.g. Gemma 3)."""
    layers = getattr(pkv, "layers", None)
    if layers is None:
        return False
    return any(
        bool(getattr(layer, "is_sliding", False)) or getattr(layer, "sliding_window", None) is not None
        for layer in layers
    )


def _cache_can_crop(pkv: Any, length: int) -> bool:
    """Read-only check whether `pkv` can be safely cropped in-place to `length` without corruption.

    Pre-validates EVERY layer before any mutation occurs so hybrid caches (Gemma 3 alternating
    DynamicSlidingWindowLayer + DynamicLayer) never suffer partial-layer mutation or SDPA crashes.
    """
    if pkv is None:
        return False
    target = int(length)
    cur = _cache_seq_len(pkv)
    if target < 0 or target > cur:
        return False
    if target == cur:
        return True
    delta = cur - target

    layers = getattr(pkv, "layers", None)
    if layers is not None:
        for layer in layers:
            if not getattr(layer, "is_initialized", True):
                continue
            keys = getattr(layer, "keys", None)
            if keys is None:
                continue
            avail = int(keys.shape[-2])
            w = getattr(layer, "sliding_window", None)
            is_sliding = bool(getattr(layer, "is_sliding", False) or w is not None)
            if is_sliding:
                if w is None or int(w) <= 0:
                    return False
                w_int = int(w)
                l_cur = int(layer.get_seq_length()) if hasattr(layer, "get_seq_length") else cur
                if l_cur != cur:
                    return False
                if l_cur >= w_int:
                    if not getattr(layer, "record_past", False):
                        return False
                    needed = min(target, max(w_int - 1, 0))
                    if avail < needed + delta:
                        return False
                else:
                    if avail < target + delta:
                        return False
            else:
                if avail < target + delta:
                    return False
        return True

    if hasattr(pkv, "key_cache") and pkv.key_cache:
        for k in pkv.key_cache:
            if k is not None and int(k.shape[-2]) < target + delta:
                return False
        return True

    if isinstance(pkv, tuple):
        for layer in pkv:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                return False
            if int(layer[0].shape[-2]) < target + delta:
                return False
        return True

    return hasattr(pkv, "crop")


def _cache_crop(pkv: Any, length: int) -> tuple[Any, bool]:
    """Attempt safe in-place crop of KV cache to `length`. Returns `(pkv, ok)`."""
    if pkv is None:
        return None, False
    target = int(length)
    cur = _cache_seq_len(pkv)
    if target < 0 or target > cur:
        return pkv, False
    if target == cur:
        return pkv, True
    if not _cache_can_crop(pkv, target):
        return pkv, False

    delta = cur - target
    if hasattr(pkv, "crop"):
        try:
            # transformers 5.x expects negative delta: crop(-(cur - target))
            pkv.crop(-delta)
        except ValueError:
            # transformers 4.x fallback (only if non-sliding and cache was not partially mutated)
            if not _cache_has_sliding_layers(pkv) and _cache_seq_len(pkv) == cur:
                try:
                    pkv.crop(target)
                except Exception:
                    return pkv, False
            else:
                return pkv, False
        except Exception:
            return pkv, False

        # Post-crop invariant verification across all layers
        if _cache_seq_len(pkv) != target:
            return pkv, False
        layers = getattr(pkv, "layers", None)
        if layers is not None:
            for layer in layers:
                if not getattr(layer, "is_initialized", True):
                    continue
                keys = getattr(layer, "keys", None)
                if keys is None:
                    continue
                w = getattr(layer, "sliding_window", None)
                is_sliding = bool(getattr(layer, "is_sliding", False) or w is not None)
                if is_sliding:
                    w_int = int(w)
                    l_len = int(layer.get_seq_length()) if hasattr(layer, "get_seq_length") else target
                    expected_keys = min(target, max(w_int - 1, 0))
                    if l_len != target or int(keys.shape[-2]) != expected_keys:
                        return pkv, False
                else:
                    if int(keys.shape[-2]) != target:
                        return pkv, False
        return pkv, True

    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        try:
            pkv.key_cache = [k[:, :, :target, :].contiguous() if k is not None else None for k in pkv.key_cache]
            pkv.value_cache = [v[:, :, :target, :].contiguous() if v is not None else None for v in pkv.value_cache]
            return pkv, _cache_seq_len(pkv) == target
        except Exception:
            return pkv, False

    if isinstance(pkv, tuple):
        try:
            cropped = tuple(
                (layer[0][:, :, :target, :].contiguous(), layer[1][:, :, :target, :].contiguous())
                for layer in pkv
            )
            return cropped, True
        except Exception:
            return pkv, False
    return pkv, False


def _repeat_kv_tensor(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size == 1:
        return t.clone()
    if t.shape[0] == 1:
        return t.repeat(batch_size, 1, 1, 1)
    return t.repeat_interleave(batch_size, dim=0)


def _cache_clone_batch(pkv: Any, batch_size: int) -> Any:
    """Return an independent KV cache expanded to `batch_size` in a single allocation without mutating `pkv`.

    Returns None if cloning is unsupported for this cache type (caller falls back to prefill).
    """
    if pkv is None or batch_size <= 0:
        return None
    B = int(batch_size)
    try:
        if isinstance(pkv, tuple):
            return tuple(
                (_repeat_kv_tensor(layer[0], B), _repeat_kv_tensor(layer[1], B))
                for layer in pkv
            )

        # transformers 5.x `DynamicCache` / `Cache` with `.layers`
        layers = getattr(pkv, "layers", None)
        if layers is not None and all(
            hasattr(l, "keys")
            and hasattr(l, "values")
            and not hasattr(l, "conv_states")
            and not hasattr(l, "recurrent_states")
            for l in layers
        ):
            cloned = copy.copy(pkv)
            new_layers = []
            for layer in layers:
                nl = copy.copy(layer)
                clen = getattr(nl, "cumulative_length", None)
                if isinstance(clen, torch.Tensor):
                    nl.cumulative_length = clen.clone()
                sw_t = getattr(nl, "_sliding_window_tensor", None)
                if isinstance(sw_t, torch.Tensor):
                    nl._sliding_window_tensor = sw_t.clone()
                if layer.keys is not None:
                    nl.keys = _repeat_kv_tensor(layer.keys, B)
                if layer.values is not None:
                    nl.values = _repeat_kv_tensor(layer.values, B)
                new_layers.append(nl)
            cloned.layers = new_layers
            return cloned

        # transformers 4.x `DynamicCache` with `.key_cache` / `.value_cache`
        if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
            cloned = copy.copy(pkv)
            cloned.key_cache = [_repeat_kv_tensor(k, B) if k is not None else None for k in pkv.key_cache]
            cloned.value_cache = [_repeat_kv_tensor(v, B) if v is not None else None for v in pkv.value_cache]
            return cloned

        # Generic fallback for non-standard SSM/hybrid caches
        cloned = copy.deepcopy(pkv)
        if B == 1:
            return cloned
        if hasattr(cloned, "batch_repeat_interleave"):
            cloned.batch_repeat_interleave(B)
            return cloned
    except Exception:
        return None
    return None


def _cache_select_batch(pkv: Any, indices: list[int], device: torch.device) -> Any:
    """Re-index batch rows of an existing cloned KV cache (e.g. expanding row 0 during function-word peek)."""
    if pkv is None or not indices:
        return None
    idx_t = torch.tensor([int(i) for i in indices], dtype=torch.long, device=device)
    try:
        if hasattr(pkv, "batch_select_indices"):
            pkv.batch_select_indices(idx_t)
            return pkv
    except Exception:
        pass
    try:
        if isinstance(pkv, tuple):
            return tuple(
                (
                    layer[0].index_select(0, idx_t.to(layer[0].device)),
                    layer[1].index_select(0, idx_t.to(layer[1].device)),
                )
                for layer in pkv
            )
        layers = getattr(pkv, "layers", None)
        if layers is not None:
            for layer in layers:
                if getattr(layer, "keys", None) is not None:
                    layer.keys = layer.keys.index_select(0, idx_t.to(layer.keys.device))
                if getattr(layer, "values", None) is not None:
                    layer.values = layer.values.index_select(0, idx_t.to(layer.values.device))
            return pkv
        if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
            pkv.key_cache = [
                k.index_select(0, idx_t.to(k.device)) if k is not None else None for k in pkv.key_cache
            ]
            pkv.value_cache = [
                v.index_select(0, idx_t.to(v.device)) if v is not None else None for v in pkv.value_cache
            ]
            return pkv
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ Text & dedup helpers
def _is_word_boundary(prev_text: str, tok_text: str) -> bool:
    """Return True if `tok_text` begins a new word/line/punctuation unit after `prev_text`."""
    if not prev_text:
        return False
    if not tok_text:
        return False
    if "\n" in prev_text or "\n" in tok_text:
        return True
    if tok_text[0].isspace():
        return True
    if _PUNCT_ONLY.match(tok_text) and not _PUNCT_ONLY.match(prev_text):
        return True
    if _PUNCT_ONLY.match(prev_text) and not _PUNCT_ONLY.match(tok_text):
        return True
    return False


def _extract_first_word_ids(tok: Any, token_ids: list[int], max_subwords: int = 6) -> list[int]:
    """Slice the leading whole word (or line break / punctuation token) from `token_ids`."""
    if not token_ids:
        return []
    out = [int(token_ids[0])]
    first_str = tok.decode([out[0]])
    # Standalone newline or whitespace-only token stands on its own
    if "\n" in first_str or _PUNCT_ONLY.match(first_str):
        return out
    prev = first_str
    for tid in token_ids[1:max_subwords]:
        s = tok.decode([int(tid)])
        if _is_word_boundary(prev, s):
            break
        out.append(int(tid))
        prev = s
    return out


def _normalize_ribbon_text(text: str) -> str:
    """Orthographic, contraction, and numeral/time normalization for semantic branch deduplication."""
    s = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u00b4", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
        .replace("\u2014", " - ")
        .replace("\u2013", " - ")
        .lower()
    )
    # Expand common English contractions
    s = re.sub(r"\bwon't\b", "will not", s)
    s = re.sub(r"\bcan't\b", "can not", s)
    s = re.sub(r"\bcannot\b", "can not", s)
    s = re.sub(r"\bshan't\b", "shall not", s)
    s = re.sub(r"\b([a-z]+)n't\b", r"\1 not", s)
    s = re.sub(r"\bi'm\b", "i am", s)
    s = re.sub(r"\b([a-z]+)'re\b", r"\1 are", s)
    s = re.sub(r"\b([a-z]+)'ve\b", r"\1 have", s)
    s = re.sub(r"\b([a-z]+)'ll\b", r"\1 will", s)
    s = re.sub(r"\b([a-z]+)'d\b", r"\1 would", s)
    s = re.sub(r"\b(he|she|it|that|there|here|what|who)'s\b", r"\1 is", s)
    # Normalize exact-hour time expressions & o'clock
    s = re.sub(r"\b(\d{1,2}):00\b", r"\1", s)
    s = re.sub(r"\bo'?clock\b", "", s)
    s = re.sub(r"\b(\d{1,2})\s*a\.?m\.?(?=\W|$)", r"\1 morning", s)
    s = re.sub(r"\b(\d{1,2})\s*p\.?m\.?(?=\W|$)", r"\1 pm", s)
    # Normalize cardinal number words -> digits
    s = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], s)
    return re.sub(r"\s+", " ", s).strip()


def _content_words(text: str) -> list[str]:
    norm = _normalize_ribbon_text(text)
    words = [w.strip("'") for w in _WORD_RE.findall(norm)]
    return [w for w in words if (len(w) >= 2 or w.isdigit()) and w not in _STOPWORDS]


def _ribbon_similarity(a: str, b: str) -> float:
    """Fast lexical/orthographic similarity for collapsing near-duplicate continuation ribbons."""
    na = _normalize_ribbon_text(a)
    nb = _normalize_ribbon_text(b)
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    if na == nb:
        return 1.0
    ca = _content_words(na)
    cb = _content_words(nb)
    if not ca or not cb:
        wa = set(_WORD_RE.findall(na))
        wb = set(_WORD_RE.findall(nb))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)
    if ca == cb:
        return 1.0
    # Prefix-truncation match only when one normalized ribbon is actually a prefix of the other
    if (nb.startswith(na) or na.startswith(nb)) and (len(ca) == 1 or len(cb) == 1):
        return 0.80
    sa, sb = set(ca[:6]), set(cb[:6])
    return len(sa & sb) / max(len(sa | sb), 1)


def _distinctive_word(ribbon: str, others: list[str]) -> str:
    """Pick the most informative word distinguishing `ribbon` from competing branches."""
    cw = _content_words(ribbon)
    other_sets = [set(_content_words(o)) for o in others]
    for w in cw:
        if all(w not in os_ for os_ in other_sets):
            return w
    if cw:
        return cw[0]
    raw = _WORD_RE.findall(ribbon.strip())
    return raw[0] if raw else ribbon.strip()[:16]


def _is_tag_starter(s: str) -> bool:
    """Return True if a decoded token begins a closing/boundary XML tag."""
    sl = s.lstrip()
    return sl.startswith(("</", "<source", "<translation"))


def _has_doc_stop(text: str) -> bool:
    return any(st in text for st in DOC_STOPS)


def _strip_doc_stop(text: str) -> str:
    best = len(text)
    for st in DOC_STOPS:
        idx = text.find(st)
        if idx != -1 and idx < best:
            best = idx
    if best == len(text):
        p = _partial_stop_pos(text, DOC_STOPS)
        if p != -1 and p < best:
            best = p
    return text[:best]


def _branch_embed(model: Any, token_ids: list[int]) -> np.ndarray | None:
    """Compute a unit-normalized mean token embedding from the model's native embed_tokens."""
    if not token_ids:
        return None
    try:
        emb = arch.text_model(model).embed_tokens
        n_vocab = int(emb.weight.shape[0])
        ids = [int(i) for i in token_ids[:6] if 0 <= int(i) < n_vocab]
        if not ids:
            return None
        dev = emb.weight.device
        with torch.no_grad():
            t = torch.tensor(ids, dtype=torch.long, device=dev)
            v = emb(t).detach().float().mean(dim=0).cpu().numpy()
        norm = float(np.linalg.norm(v))
        return (v / norm) if norm > 1e-12 else None
    except Exception:
        return None


def _cosine_np(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ------------------------------------------------------------------ Source↔Target Alignment
def _parse_source_lines(source_text: str) -> list[dict[str, Any]]:
    """Parse `source_text` into indexed stanza-aware lines (or sentences for single-line prose)."""
    raw = source_text.splitlines()
    if not raw:
        return []
    if len(raw) == 1:
        line = raw[0]
        parts = [p.strip() for p in _SENT_SPLIT_RE.split(line.strip()) if p.strip()]
        if len(parts) >= 2:
            return [{"index": i, "text": p, "stanza": 0} for i, p in enumerate(parts)]
        return [{"index": 0, "text": line, "stanza": 0}]

    out: list[dict[str, Any]] = []
    stanza = 0
    seen_in_stanza = False
    for line in raw:
        if not line.strip():
            if seen_in_stanza:
                stanza += 1
                seen_in_stanza = False
            continue
        out.append({"index": len(out), "text": line, "stanza": stanza})
        seen_in_stanza = True
    return out


def _grads_at_oom_safe(model: Any, prefix: list[int], conts: list[list[int]], layer: int, b: int, batch: int = 4) -> np.ndarray:
    """OOM-safe per-sample gradients at cursor `b` (automatically halves batch size on CUDA OOM)."""
    d = arch.hidden_size(model)
    dev = arch.text_model(model).embed_tokens.weight.device
    out: list[np.ndarray] = []
    state: dict[str, torch.Tensor] = {}

    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        if "z" not in state or h.shape[1] <= b:
            return None
        oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
        oh[b] = 1
        new = h + state["z"].to(h.dtype)[:, None, :] * oh[None, :, None]
        return (new, *o[1:]) if isinstance(o, tuple) else new

    hd = arch.layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            cur_batch = max(1, int(batch))
            c = 0
            while c < len(conts):
                part = [y if y else [prefix[-1]] for y in conts[c : c + cur_batch]]
                try:
                    state["z"] = torch.zeros(len(part), d, device=dev, dtype=torch.float32, requires_grad=True)
                    lp = seq_logp(model, prefix, part)
                    (g,) = torch.autograd.grad(lp.sum(), [state["z"]])
                    out.append(g.detach().float().cpu().numpy())
                    state.pop("z", None)
                    c += len(part)
                except torch.cuda.OutOfMemoryError:
                    state.pop("z", None)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if cur_batch == 1:
                        raise
                    cur_batch = max(1, cur_batch // 2)
    finally:
        hd.remove()
    return np.concatenate(out, axis=0).astype(np.float64)


# ------------------------------------------------------------------ CoTranslationSession
class CoTranslationSession:
    """Stateful, KV-cached interactive co-translation session for arbitrary-length texts."""

    def __init__(
        self,
        engine: Any,
        source_text: str,
        src_lang: str = "Spanish",
        tgt_lang: str = "English",
        global_coords: dict[str, float] | None = None,
        lookahead_tokens: int = 9,
        max_branches: int = 6,
    ):
        self.session_id = f"sess_{uuid.uuid4().hex[:10]}"
        self.engine = engine
        self.source_text = source_text.strip("\r\n")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lookahead_tokens = max(4, int(lookahead_tokens))
        self.max_branches = max(2, min(8, int(max_branches)))
        self.global_coords: dict[str, float] = dict(global_coords or {})
        self._source_lines = _parse_source_lines(self.source_text)

        tok, _ = engine.ensure_loaded()
        self.prompt_ids, self.start = build_doc_prompt(tok, self.source_text, src_lang=src_lang, tgt_lang=tgt_lang)
        n_src_toks = len(tok(self.source_text, add_special_tokens=False)["input_ids"])
        self.max_total_tokens = max(256, 4 * n_src_toks + 128)

        self.committed_ids: list[int] = []
        self.committed_meta: list[dict[str, Any]] = []
        self.pkv: Any = None
        self.last_logits: torch.Tensor | None = None
        self.last_hidden_state: torch.Tensor | None = None
        self.last_hidden_norm: float = 0.0
        self.last_fh: dict[str, Any] | None = None
        self.last_frontier: dict[str, Any] | None = None
        self.active_compass: dict[str, Any] | None = None

        # Async Tier-2 compass precomputation state (NEVER hold _precompute_meta_lock while taking engine._lock)
        self._cursor_version: int = 0
        self._precompute_cancel = threading.Event()
        self._precompute_meta_lock = threading.Lock()
        self._precompute_thread: threading.Thread | None = None
        self._precompute_target_version: int = -1
        self.auto_precompute_compass: bool = True
        self._precompute_delay_s: float = 0.05

        # Lightweight non-destructive history tree of explored paths
        self.root_node_id = "n0"
        self.active_node_id = self.root_node_id
        self.history: dict[str, dict[str, Any]] = {
            self.root_node_id: {
                "id": self.root_node_id,
                "parent_id": None,
                "fork_token_idx": 0,
                "token_ids": [],
                "meta": [],
                "label": "Main path",
                "created_at": time.time(),
            }
        }
        self._node_seq = 1
        self._source_word_vecs: dict[str, np.ndarray] = {}
        self._source_recipe_cache: dict[str, dict[str, Any]] = {}

        # Initial prefill of the source prompt
        self._prefill_full()

    def _extract_source_word_vecs(self, tok: Any, layer_states: torch.Tensor) -> None:
        """Cache pooled hidden vectors for each content word in `self.source_text` (zero extra passes)."""
        n_src = len(tok(self.source_text, add_special_tokens=False)["input_ids"])
        tail_ids = tok(f"\n</source>\n<translation lang=\"{self.tgt_lang}\">\n", add_special_tokens=False)["input_ids"]
        src_end = max(1, len(self.prompt_ids) - len(tail_ids))
        src_start = max(0, src_end - n_src)
        src_ids = self.prompt_ids[src_start:src_end]
        if not src_ids or layer_states.shape[0] < src_end:
            return
        prev_decoded = ""
        cur_word = ""
        cur_vecs: list[np.ndarray] = []
        for idx in range(len(src_ids)):
            cur_decoded = tok.decode(src_ids[: idx + 1])
            piece = cur_decoded[len(prev_decoded) :]
            prev_decoded = cur_decoded
            if not piece:
                continue
            vec = layer_states[src_start + idx].detach().float().cpu().numpy()
            if piece[0].isspace() or _PUNCT_ONLY.match(piece):
                if cur_word and cur_vecs:
                    clean = cur_word.strip(" .,;:!?¡¿\"'()[]{}—–-").lower()
                    if len(clean) >= 3:
                        self._source_word_vecs[clean] = np.mean(cur_vecs, axis=0)
                cur_word = piece.lstrip()
                cur_vecs = [vec] if cur_word and not _PUNCT_ONLY.match(cur_word) else []
            else:
                cur_word += piece
                cur_vecs.append(vec)
        if cur_word and cur_vecs:
            clean = cur_word.strip(" .,;:!?¡¿\"'()[]{}—–-").lower()
            if len(clean) >= 3:
                self._source_word_vecs[clean] = np.mean(cur_vecs, axis=0)

    def get_source_recipe(self, word: str | None = None) -> dict[str, Any] | None:
        """Return live English-basis NNOMP ingredients + leftover for a source word (or active line)."""
        dec = getattr(self.engine, "en_decomposer", None)
        if dec is None or not self._source_word_vecs:
            return None
        target_word = (word or "").strip(" .,;:!?¡¿\"'()[]{}—–-").lower()
        if not target_word or target_word not in self._source_word_vecs:
            # Pick the longest / most distinctive content word on the active source line
            al_idx = 0
            if self.last_frontier is not None:
                al = self._compute_alignment([""] * len(self.committed_ids))
                al_idx = int(al.get("active_source_line", 0))
            line_txt = ""
            for sl in self._source_lines:
                if sl["index"] == al_idx:
                    line_txt = sl["text"]
                    break
            cands = [m.group(0).lower() for m in _WORD_RE.finditer(line_txt) if m.group(0).lower() in self._source_word_vecs]
            if not cands:
                cands = list(self._source_word_vecs.keys())
            if not cands:
                return None
            # Prefer longer words not in basic stopwords
            cands.sort(key=lambda w: (w in _STOPWORDS, -len(w)))
            target_word = cands[0]
        if target_word in self._source_recipe_cache:
            return self._source_recipe_cache[target_word]
        rec = dec.decompose(target_word, self._source_word_vecs[target_word], k=5)
        self._source_recipe_cache[target_word] = rec
        return rec

    # ---------------------------------------------------------- Steering & ForkHead pause helpers
    def _pause_fork_scorer(self):
        scorer = getattr(self.engine, "fork_scorer", None)
        paused_fn = getattr(scorer, "paused", None) if scorer is not None else None
        return paused_fn() if callable(paused_fn) else nullcontext()

    def _apply_global_steering(
        self,
        start_pos: int | None = None,
        local_vec: torch.Tensor | np.ndarray | None = None,
        local_b: int = 0,
        local_only_at_b: bool = False,
    ) -> None:
        b_global = (self.start - 1) if start_pos is None else int(start_pos)
        self.engine.apply_steering(
            global_coords=self.global_coords,
            global_start=b_global,
            local_vec=local_vec,
            local_b=local_b,
            local_only_at_b=local_only_at_b,
        )

    def _sync_hooks_pos(self, pos: int) -> None:
        self.engine.sync_steering_pos(pos)

    def _clear_steering(self) -> None:
        self.engine.clear_steering()

    def _record_last_hidden(self, h_last: torch.Tensor) -> None:
        h_cpu = h_last.detach().float().cpu()
        self.last_hidden_state = h_cpu
        self.last_hidden_norm = float(h_cpu.norm().item())

    def _invalidate_cursor(self) -> None:
        """Bump cursor version, cancel any in-flight compass precompute, and drop stale compass."""
        self._cursor_version += 1
        self._precompute_cancel.set()
        self.active_compass = None

    def cancel_precompute(self, wait: bool = False) -> None:
        """Cancel any active background compass precomputation without requiring engine._lock."""
        self._precompute_cancel.set()
        self._cursor_version += 1
        th = None
        with self._precompute_meta_lock:
            th = self._precompute_thread
        if wait and th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=2.0)

    def _is_compass_ready(self) -> bool:
        return (
            self.active_compass is not None
            and self.active_compass.get("cursor_len") == len(self.committed_ids)
            and self.active_compass.get("cursor_version") == self._cursor_version
        )

    def _prefill_full(self) -> None:
        """Full prefill of `prompt_ids + committed_ids` with current global steering."""
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        seq = self.prompt_ids + self.committed_ids
        x = torch.tensor([seq], dtype=torch.long, device=dev)

        dec = getattr(self.engine, "en_decomposer", None)
        want_src_vecs = dec is not None and not self._source_word_vecs

        self._apply_global_steering()
        self._sync_hooks_pos(0)
        if self.engine.fork_scorer is not None:
            self.engine.fork_scorer.reset()
        try:
            with torch.no_grad():
                out = arch.forward_hidden(model, input_ids=x, use_cache=True, output_hidden_states=want_src_vecs)
                self.pkv = out.past_key_values
                h_last = out.last_hidden_state[0, -1]
                self._record_last_hidden(h_last)
                self.last_logits = arch.logits(model, h_last[None])[0].detach().cpu()
                self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
                if want_src_vecs and getattr(out, "hidden_states", None) is not None:
                    hs_list = out.hidden_states
                    layer_idx = min(len(hs_list) - 1, dec.layer + 1)
                    self._extract_source_word_vecs(tok, hs_list[layer_idx][0])
        finally:
            self._clear_steering()

    def _advance_cache(self, new_token_ids: list[int]) -> None:
        """Incrementally advance the live KV cache by `new_token_ids`."""
        if not new_token_ids:
            return
        _, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        cache_pos = len(self.prompt_ids) + len(self.committed_ids) - len(new_token_ids)
        if self.pkv is None or _cache_seq_len(self.pkv) != cache_pos:
            self._prefill_full()
            return

        x = torch.tensor([new_token_ids], dtype=torch.long, device=dev)
        self._apply_global_steering()
        self._sync_hooks_pos(cache_pos)
        if self.engine.fork_scorer is not None:
            self.engine.fork_scorer.reset()
        try:
            with torch.no_grad():
                out = arch.forward_hidden(model, input_ids=x, past_key_values=self.pkv, use_cache=True)
                self.pkv = out.past_key_values
                h_last = out.last_hidden_state[0, -1]
                self._record_last_hidden(h_last)
                self.last_logits = arch.logits(model, h_last[None])[0].detach().cpu()
                self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
        except Exception:
            # Safe fallback if sliding-window cache or hook encountered an edge case
            self._prefill_full()
        finally:
            self._clear_steering()

    def _rewind_cache_to(self, cache_target: int, last_tok: int) -> None:
        """Crop `self.pkv` to `cache_target - 1` and run a single 1-token forward on `last_tok`."""
        if cache_target >= 1:
            cropped, ok = _cache_crop(self.pkv, cache_target - 1)
            if ok:
                self.pkv = cropped
                _, model = self.engine.ensure_loaded()
                dev = self.engine._dev()
                x = torch.tensor([[int(last_tok)]], dtype=torch.long, device=dev)
                self._apply_global_steering()
                self._sync_hooks_pos(cache_target - 1)
                if self.engine.fork_scorer is not None:
                    self.engine.fork_scorer.reset()
                try:
                    with torch.no_grad():
                        out = arch.forward_hidden(model, input_ids=x, past_key_values=self.pkv, use_cache=True)
                        self.pkv = out.past_key_values
                        h_last = out.last_hidden_state[0, -1]
                        self._record_last_hidden(h_last)
                        self.last_logits = arch.logits(model, h_last[None])[0].detach().cpu()
                        self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
                    return
                except Exception:
                    pass
                finally:
                    self._clear_steering()
        self._prefill_full()

    # ---------------------------------------------------------- Unified Batched KV-Cached Lookahead
    def _batched_lookahead(
        self,
        starters: list[list[int]],
        lookahead_steps: int,
        extra_vec: torch.Tensor | None = None,
        extra_b: int | None = None,
    ) -> list[list[int]]:
        """Run parallel greedy rollouts from arbitrarily mixed-length `starters` in ONE clone and ONE loop."""
        if not starters:
            return []
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        B = len(starters)
        prefix = self.prompt_ids + self.committed_ids
        prefix_len = len(prefix)
        max_st = max(len(s) for s in starters)
        total_steps = max(int(lookahead_steps), max_st - 1)

        local_b = extra_b if extra_b is not None else (self.start - 1)
        self._apply_global_steering(
            start_pos=self.start - 1,
            local_vec=extra_vec,
            local_b=local_b,
            local_only_at_b=False,
        )

        use_cloned_cache = (extra_vec is None)
        pkv_batch = _cache_clone_batch(self.pkv, B) if use_cloned_cache else None

        seqs = [[int(s[0])] for s in starters]
        alive = [True] * B
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

        try:
            with self._pause_fork_scorer(), torch.no_grad():
                if pkv_batch is not None:
                    x0 = torch.tensor([[int(s[0])] for s in starters], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(prefix_len)
                    out = arch.forward_hidden(model, input_ids=x0, past_key_values=pkv_batch, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos = prefix_len + 1
                else:
                    full_in = [prefix + [int(s[0])] for s in starters]
                    x0 = torch.tensor(full_in, dtype=torch.long, device=dev)
                    self._sync_hooks_pos(0)
                    out = arch.forward_hidden(model, input_ids=x0, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos = prefix_len + 1

                lg = arch.logits(model, out.last_hidden_state[:, -1, :])
                nxt = lg.argmax(dim=-1).tolist()

                for step_i in range(total_steps):
                    step_tokens: list[int] = []
                    for i in range(B):
                        if not alive[i]:
                            step_tokens.append(int(nxt[i]))
                            continue
                        if step_i + 1 < len(starters[i]):
                            tid = int(starters[i][step_i + 1])
                        else:
                            tid = int(nxt[i])
                        step_tokens.append(tid)
                        if tid in eos_ids:
                            alive[i] = False
                            continue
                        seqs[i].append(tid)
                        if _has_doc_stop(tok.decode(seqs[i])):
                            alive[i] = False
                    if not any(alive) or step_i == total_steps - 1:
                        break
                    x_step = torch.tensor([[t] for t in step_tokens], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(cur_pos)
                    out = arch.forward_hidden(model, input_ids=x_step, past_key_values=pkv_batch, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos += 1
                    lg = arch.logits(model, out.last_hidden_state[:, -1, :])
                    nxt = lg.argmax(dim=-1).tolist()
        finally:
            self._clear_steering()

        return [cut_doc(tok, s) for s in seqs]

    def _rollout_frontier(
        self,
        starters_list: list[dict[str, Any]],
        can_peek: bool,
        entropy: float,
        fh_prob: float | None,
    ) -> tuple[
        list[dict[str, Any]],
        list[list[int]],
        list[np.ndarray | None],
        list[list[tuple[int, float]] | None],
        list[int],
        float,
    ]:
        """Single-clone unified rollout with inline function-word peek, hidden-delta capture, and depth-2 sub-dists."""
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        prefix = self.prompt_ids + self.committed_ids
        prefix_len = len(prefix)
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

        trivially_linear_raw = (
            len(starters_list) == 1
            or (
                not can_peek
                and starters_list[0]["prob"] >= 0.84
                and entropy < 0.65
                and (fh_prob is None or fh_prob < 0.45)
            )
        )
        n_init = 1 if trivially_linear_raw else min(self.max_branches, len(starters_list))
        selected = list(starters_list[:n_init])
        B = len(selected)
        peeked_entropy = entropy

        self._apply_global_steering(start_pos=self.start - 1)
        pkv_batch = _cache_clone_batch(self.pkv, B)

        seqs: list[list[int]] = [[int(st["ids"][0])] for st in selected]
        starter_lens: list[int] = [len(st["ids"]) for st in selected]
        alive: list[bool] = [True] * B
        h_sum: list[torch.Tensor | None] = [None] * B
        h_cnt: list[int] = [0] * B
        sub_dists: list[list[tuple[int, float]] | None] = [None] * B
        fw_lens: list[int] = [0] * B

        def _capture_sub_dist(row_idx: int, lg_row: torch.Tensor) -> None:
            p_row = lg_row.float().softmax(-1)
            k_sub = min(12, int(p_row.numel()))
            vals, ids_t = torch.topk(p_row, k=k_sub)
            sub_dists[row_idx] = [(int(tid), float(v)) for tid, v in zip(ids_t.tolist(), vals.tolist())]
            fw_lens[row_idx] = len(seqs[row_idx])

        try:
            with self._pause_fork_scorer(), torch.no_grad():
                if pkv_batch is not None:
                    x0 = torch.tensor([[int(st["ids"][0])] for st in selected], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(prefix_len)
                    out = arch.forward_hidden(model, input_ids=x0, past_key_values=pkv_batch, use_cache=True)
                    pkv_batch = out.past_key_values
                else:
                    full_in = [prefix + [int(st["ids"][0])] for st in selected]
                    x0 = torch.tensor(full_in, dtype=torch.long, device=dev)
                    self._sync_hooks_pos(0)
                    out = arch.forward_hidden(model, input_ids=x0, use_cache=True)
                    pkv_batch = out.past_key_values
                cur_pos = prefix_len + 1

                h_step = out.last_hidden_state[:, -1, :]
                lg = arch.logits(model, h_step)
                h_cpu = h_step.detach().float().cpu()
                for i in range(B):
                    h_sum[i] = h_cpu[i].clone()
                    h_cnt[i] = 1

                # Inline 1-step function-word peek directly from row 0's logits (0 extra forwards!)
                forced_step1: list[int | None] = [None] * B
                if can_peek:
                    p2 = lg[0].float().softmax(-1)
                    ent2 = float(-(p2 * p2.clamp_min(1e-30).log()).sum().item())
                    peeked_entropy = max(entropy, ent2)
                    if ent2 >= 0.95:
                        vals2, ids2 = torch.topk(p2, k=min(14, int(p2.numel())))
                        fn_id = int(selected[0]["ids"][0])
                        fn_prob = float(selected[0]["prob"])
                        seen2: dict[str, dict[str, Any]] = {}
                        for tid2, v2 in zip(ids2.tolist(), vals2.tolist()):
                            if tid2 in eos_ids or v2 < 0.03:
                                continue
                            s2 = tok.decode([tid2])
                            if _is_tag_starter(s2):
                                continue
                            key2 = s2.strip().lower() or f"__W_{tid2}"
                            jp = float(fn_prob * v2)
                            if key2 in seen2:
                                seen2[key2]["prob"] += jp
                            else:
                                seen2[key2] = {"ids": [fn_id, int(tid2)], "prob": jp, "raw_text": s2}
                        expanded = sorted(seen2.values(), key=lambda x: -x["prob"])[:4]
                        if len(expanded) >= 2:
                            alt_rows = [
                                (r_idx, selected[r_idx])
                                for r_idx in range(1, B)
                                if selected[r_idx]["prob"] >= 0.06
                            ]
                            cap_alts = max(0, self.max_branches - len(expanded))
                            alt_rows = alt_rows[:cap_alts]
                            # Combine expanded (from row 0) + kept direct alts, sorted by prob
                            combined_entries: list[tuple[int, dict[str, Any], int | None]] = []
                            for ex in expanded:
                                combined_entries.append((0, ex, int(ex["ids"][1])))
                            for r_idx, alt_st in alt_rows:
                                combined_entries.append((r_idx, alt_st, None))
                            combined_entries.sort(key=lambda item: -item[1]["prob"])
                            combined_entries = combined_entries[: self.max_branches]

                            select_indices = [item[0] for item in combined_entries]
                            reindexed = _cache_select_batch(pkv_batch, select_indices, dev)
                            if reindexed is not None:
                                pkv_batch = reindexed
                                idx_t = torch.tensor(select_indices, dtype=torch.long, device=dev)
                                lg = lg.index_select(0, idx_t)
                                selected = [item[1] for item in combined_entries]
                                forced_step1 = [item[2] for item in combined_entries]
                                B = len(selected)
                                seqs = [[int(st["ids"][0])] for st in selected]
                                starter_lens = [len(st["ids"]) for st in selected]
                                alive = [True] * B
                                h_sum = [h_cpu[src_r].clone() for src_r in select_indices]
                                h_cnt = [1] * B
                                sub_dists = [None] * B
                                fw_lens = [0] * B

                # If peek confirmed linear & we don't need extra rows, trim to 1 row cleanly
                if (
                    B > 1
                    and selected[0]["prob"] >= 0.84
                    and peeked_entropy < 0.65
                    and (fh_prob is None or fh_prob < 0.45)
                ):
                    reindexed = _cache_select_batch(pkv_batch, [0], dev)
                    if reindexed is not None:
                        pkv_batch = reindexed
                        lg = lg[:1]
                        selected = selected[:1]
                        forced_step1 = forced_step1[:1]
                        B = 1
                        seqs = seqs[:1]
                        starter_lens = starter_lens[:1]
                        alive = alive[:1]
                        h_sum = h_sum[:1]
                        h_cnt = h_cnt[:1]
                        sub_dists = sub_dists[:1]
                        fw_lens = fw_lens[:1]

                nxt = lg.argmax(dim=-1).tolist()
                total_steps = max(int(self.lookahead_tokens), max(starter_lens) - 1)

                for step_i in range(total_steps):
                    step_tokens: list[int] = []
                    for i in range(B):
                        if not alive[i]:
                            step_tokens.append(int(nxt[i]))
                            continue
                        if step_i == 0 and forced_step1[i] is not None:
                            tid = int(forced_step1[i])
                        else:
                            tid = int(nxt[i])
                            # Capture depth-2 distribution at the FIRST word boundary after starter
                            if sub_dists[i] is None and len(seqs[i]) >= starter_lens[i]:
                                prev_s = tok.decode([seqs[i][-1]])
                                nxt_s = tok.decode([tid])
                                first_tok_s = tok.decode([seqs[i][0]])
                                crossed = (
                                    (len(seqs[i]) == 1 and ("\n" in first_tok_s or bool(_PUNCT_ONLY.match(first_tok_s))))
                                    or _is_word_boundary(prev_s, nxt_s)
                                    or len(seqs[i]) >= 6
                                )
                                if crossed:
                                    _capture_sub_dist(i, lg[i])
                        step_tokens.append(tid)
                        if tid in eos_ids:
                            alive[i] = False
                            continue
                        seqs[i].append(tid)
                        if _has_doc_stop(tok.decode(seqs[i])):
                            alive[i] = False

                    if not any(alive) or step_i == total_steps - 1:
                        break

                    x_step = torch.tensor([[t] for t in step_tokens], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(cur_pos)
                    out = arch.forward_hidden(model, input_ids=x_step, past_key_values=pkv_batch, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos += 1
                    h_step = out.last_hidden_state[:, -1, :]
                    lg = arch.logits(model, h_step)
                    nxt = lg.argmax(dim=-1).tolist()
                    h_cpu = h_step.detach().float().cpu()
                    for i in range(B):
                        if alive[i]:
                            h_sum[i] = h_cpu[i].clone() if h_sum[i] is None else (h_sum[i] + h_cpu[i])
                            h_cnt[i] += 1
        finally:
            self._clear_steering()

        ribbons = [cut_doc(tok, s) for s in seqs]
        h_deltas: list[np.ndarray | None] = []
        base_h = self.last_hidden_state
        for i in range(B):
            if h_sum[i] is not None and h_cnt[i] > 0:
                h_mean = h_sum[i] / float(h_cnt[i])
                delta = (h_mean - base_h) if (base_h is not None and base_h.shape == h_mean.shape) else h_mean
                h_deltas.append(delta.numpy().astype(np.float32))
            else:
                h_deltas.append(None)

        return selected, ribbons, h_deltas, sub_dists, fw_lens, peeked_entropy

    def _build_sub_branches(
        self,
        tok: Any,
        branch_entry: dict[str, Any],
        eos_ids: set[int],
    ) -> list[dict[str, Any]]:
        """Build depth-2 nested sub-fork continuations from captured word-boundary distribution."""
        sub_dist = branch_entry.get("sub_dist")
        if not sub_dist:
            return []
        fw_len = int(branch_entry.get("fw_len", 0))
        r_ids = branch_entry.get("preview_ids", [])
        greedy_tid = int(r_ids[fw_len]) if 0 <= fw_len < len(r_ids) else None

        seen: dict[str, dict[str, Any]] = {}
        for tid, prob in sub_dist:
            if tid in eos_ids:
                continue
            if prob < 0.025 and len(seen) >= 1:
                continue
            tok_s = _strip_doc_stop(tok.decode([tid]))
            if not tok_s or _is_tag_starter(tok_s):
                continue
            key = tok_s.strip().lower() or f"__NL_{tid}"
            if key in seen:
                seen[key]["prob"] += float(prob)
                continue
            if greedy_tid is not None and tid == greedy_tid:
                tail_ids = r_ids[fw_len:]
                sub_fw_ids = _extract_first_word_ids(tok, tail_ids)
                p_txt = _strip_doc_stop(tok.decode(tail_ids)) or tok_s
                fw_txt = tok.decode(sub_fw_ids) or tok_s
            else:
                p_txt = tok_s
                fw_txt = tok_s
            seen[key] = {
                "preview_text": p_txt,
                "first_word_text": fw_txt,
                "prob": float(prob),
            }

        items = sorted(seen.values(), key=lambda x: -x["prob"])[:3]
        if not items:
            return []
        mass = sum(it["prob"] for it in items) or 1.0
        return [
            {
                "preview_text": it["preview_text"],
                "first_word_text": it["first_word_text"],
                "prob": round(float(it["prob"]), 4),
                "share": round(float(it["prob"] / mass), 4),
            }
            for it in items
        ]

    # ---------------------------------------------------------- Tier-1 Instant Frontier
    def inspect_frontier(self, _schedule_precompute: bool = True) -> dict[str, Any]:
        """Compute the live uncertainty regime, ghost preview, and semantic branch fan at the current cursor."""
        tok, model = self.engine.ensure_loaded()
        if self.last_logits is None:
            self._prefill_full()

        lg = self.last_logits
        p = lg.softmax(-1)
        entropy = float(-(p * p.clamp_min(1e-30).log()).sum().item())
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

        # Check end-of-translation conditions
        committed_text = tok.decode(self.committed_ids)
        at_end = (
            len(self.committed_ids) >= self.max_total_tokens
            or _has_doc_stop(committed_text)
            or _partial_stop_pos(committed_text, DOC_STOPS) != -1
        )
        top_vals, top_ids = torch.topk(p, k=min(28, int(p.numel())))
        top_pairs = [(int(tid), float(val)) for tid, val in zip(top_ids.tolist(), top_vals.tolist())]

        # Detect if top token is EOS or closing tag `</translation>`
        top1_id, top1_prob = top_pairs[0]
        top1_str = tok.decode([top1_id])
        if (top1_id in eos_ids or _is_tag_starter(top1_str)) and top1_prob >= 0.45:
            at_end = True

        fh_prob = float(self.last_fh["fork_prob"]) if self.last_fh is not None else None
        fh_ent = float(self.last_fh["pred_entropy"]) if self.last_fh is not None else None
        fh_fis = float(self.last_fh["pred_fisher_top"]) if self.last_fh is not None else None

        def _make_end_frontier() -> dict[str, Any]:
            fr = {
                "at_end": True,
                "regime": "linear",
                "fork_kind": "linear",
                "semantic_dispersion": 0.0,
                "source_recipe": None,
                "fork_score": 0.0,
                "entropy": round(entropy, 4),
                "peeked_entropy": round(entropy, 4),
                "top1_prob": round(top1_prob, 4),
                "fork_head_prob": round(fh_prob, 4) if fh_prob is not None else None,
                "fork_head_entropy": round(fh_ent, 4) if fh_ent is not None else None,
                "fork_head_fisher": round(fh_fis, 4) if fh_fis is not None else None,
                "ghost": {"text": "", "next_word_text": "", "next_word_ids": [], "full_ids": []},
                "branches": [],
            }
            self.last_frontier = fr
            return fr

        if at_end:
            return _make_end_frontier()

        # 1. Surface-deduplicated next-token starters (collapse case / leading space duplicates)
        starters_map: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []
        for tid, prob in top_pairs:
            if tid in eos_ids:
                continue
            if prob < 0.012 and len(ordered_keys) >= 2:
                continue
            s = tok.decode([tid])
            if _is_tag_starter(s):
                continue
            if "\n" in s:
                key = "__NL__" if "\n\n" not in s else "__STANZA__"
            else:
                norm = s.strip().lower()
                key = norm if norm else f"__WS_{tid}"
            if key in starters_map:
                starters_map[key]["prob"] += prob
            else:
                starters_map[key] = {"ids": [tid], "prob": prob, "raw_text": s}
                ordered_keys.append(key)

        if not ordered_keys:
            if _is_tag_starter(top1_str) or top1_id in eos_ids:
                return _make_end_frontier()
            starters_list = [{"ids": [top1_id], "prob": top1_prob, "raw_text": top1_str}]
        else:
            starters_list = sorted(starters_map.values(), key=lambda x: -x["prob"])

        # 2 & 3. Unified single-clone rollout with inline 1-step function-word peek
        lead = starters_list[0]
        lead_norm = lead["raw_text"].strip().lower()
        can_peek = (
            lead_norm in _PEEK_FUNCTION_WORDS
            and lead["prob"] >= 0.42
            and len(lead["ids"]) == 1
        )
        selected, ribbons, h_deltas, sub_dists, fw_lens, peeked_entropy = self._rollout_frontier(
            starters_list,
            can_peek=can_peek,
            entropy=entropy,
            fh_prob=fh_prob,
        )

        # Split closing-tag protection: if the lead rollout was purely a closing tag (e.g. ['<', '/', 'translation', '>'])
        # and had dominant probability, treat cursor as cleanly at_end rather than leaking '<'!
        if selected and (not ribbons[0] or not _strip_doc_stop(tok.decode(ribbons[0]))):
            if selected[0]["prob"] >= 0.45:
                return _make_end_frontier()

        # 4. Semantic + lexical deduplication of lookahead ribbons
        merged: list[dict[str, Any]] = []
        for st, r_ids, h_d, s_dist, fw_l in zip(selected, ribbons, h_deltas, sub_dists, fw_lens):
            if not r_ids:
                continue
            r_text = _strip_doc_stop(tok.decode(r_ids))
            if not r_text:
                continue
            emb = _branch_embed(model, r_ids)
            placed = False
            for m in merged:
                lex = _ribbon_similarity(r_text, m["preview_text"])
                if lex >= 0.72:
                    m["prob"] += st["prob"]
                    placed = True
                    break
                emb_cos = _cosine_np(emb, m["emb"])
                if emb_cos >= 0.88 and lex >= 0.25:
                    m["prob"] += st["prob"]
                    placed = True
                    break
                h_cos = _cosine_np(h_d, m["h_delta"])
                if h_cos >= 0.94 and emb_cos >= 0.80:
                    m["prob"] += st["prob"]
                    placed = True
                    break
            if not placed:
                fw_ids = _extract_first_word_ids(tok, r_ids)
                merged.append({
                    "preview_ids": r_ids,
                    "preview_text": r_text,
                    "first_word_ids": fw_ids,
                    "first_word_text": tok.decode(fw_ids),
                    "prob": float(st["prob"]),
                    "raw_prob": float(st["prob"]),
                    "emb": emb,
                    "h_delta": h_d,
                    "sub_dist": s_dist,
                    "fw_len": fw_l,
                })

        if not merged:
            return _make_end_frontier()

        merged.sort(key=lambda x: -x["prob"])
        total_mass = sum(m["prob"] for m in merged) or 1.0
        top_merged = merged[: self.max_branches]

        # 5. Classify uncertainty regime honestly (linear vs subtle vs fork)
        eff_ent = max(entropy, peeked_entropy)
        top_share = float(top_merged[0]["prob"] / total_mass) if top_merged else 1.0
        second_prob = float(top_merged[1]["prob"]) if len(top_merged) >= 2 else 0.0
        n_distinct = len(top_merged)

        ent_score = min(1.0, max(0.0, (eff_ent - 0.35) / 1.65))
        spread_score = min(1.0, max(0.0, (1.0 - top_share) * 1.45))
        fh_term = float(fh_prob) if fh_prob is not None else ent_score
        fork_score = round(float(0.45 * ent_score + 0.35 * spread_score + 0.20 * fh_term), 4)

        if n_distinct <= 1 or (top_share >= 0.80 and eff_ent < 0.85 and (fh_prob is None or fh_prob < 0.50)):
            regime = "linear"
        elif (
            eff_ent >= 1.25
            or (fh_prob is not None and fh_prob >= 0.55 and n_distinct >= 2)
            or (n_distinct >= 3 and top_share < 0.65)
            or (n_distinct >= 2 and second_prob >= 0.18)
        ):
            regime = "fork"
        else:
            regime = "subtle"

        # Normalise display shares & attach depth-2 sub_branches (on fork/subtle) and has_newline
        all_texts = [m["preview_text"] for m in top_merged]
        branches: list[dict[str, Any]] = []
        for idx, m in enumerate(top_merged):
            others = [t for j, t in enumerate(all_texts) if j != idx]
            sub_brs = self._build_sub_branches(tok, m, eos_ids) if regime in ("fork", "subtle") else []
            branches.append({
                "index": idx,
                "preview_text": m["preview_text"],
                "preview_ids": m["preview_ids"],
                "first_word_text": m["first_word_text"],
                "first_word_ids": m["first_word_ids"],
                "prob": round(float(m["prob"]), 4),
                "share": round(float(m["prob"] / total_mass), 4),
                "distinctive": _distinctive_word(m["preview_text"], others),
                "color": READING_PALETTE[idx % len(READING_PALETTE)],
                "has_newline": ("\n" in m["preview_text"]) or ("\n" in m["first_word_text"]),
                "sub_branches": sub_brs,
            })

        # Ghost continuation = top branch
        b0 = branches[0]
        ghost = {
            "text": b0["preview_text"],
            "next_word_text": b0["first_word_text"],
            "next_word_ids": b0["first_word_ids"],
            "full_ids": b0["preview_ids"],
        }

        sem_disp, fork_kind = branch_semantic_dispersion(top_merged, regime)
        src_recipe = self.get_source_recipe() if fork_kind == "conceptual" else None
        # If English leftover is very high (>= 0.60) on a fork, upgrade to conceptual gap even if surface ribbons overlap
        if src_recipe is not None and regime == "fork" and src_recipe.get("leftover", 0.0) >= 0.60:
            fork_kind = "conceptual"

        frontier = {
            "at_end": False,
            "regime": regime,
            "fork_kind": fork_kind,
            "semantic_dispersion": sem_disp,
            "source_recipe": src_recipe,
            "fork_score": fork_score,
            "entropy": round(entropy, 4),
            "peeked_entropy": round(peeked_entropy, 4),
            "top1_prob": round(top1_prob, 4),
            "fork_head_prob": round(fh_prob, 4) if fh_prob is not None else None,
            "fork_head_entropy": round(fh_ent, 4) if fh_ent is not None else None,
            "fork_head_fisher": round(fh_fis, 4) if fh_fis is not None else None,
            "ghost": ghost,
            "branches": branches,
        }
        self.last_frontier = frontier

        if _schedule_precompute and regime == "fork" and self.auto_precompute_compass:
            self._schedule_compass_precompute()

        return frontier

    def _lightweight_frontier(self) -> tuple[dict[str, Any], bool]:
        """0-forward lightweight frontier for fast linear gliding; returns `(frontier, needs_full_inspect)`."""
        tok, _ = self.engine.ensure_loaded()
        if self.last_logits is None:
            self._prefill_full()
        lg = self.last_logits
        p = lg.softmax(-1)
        entropy = float(-(p * p.clamp_min(1e-30).log()).sum().item())
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

        top_vals, top_ids = torch.topk(p, k=min(4, int(p.numel())))
        top1_id = int(top_ids[0].item())
        top1_prob = float(top_vals[0].item())
        top1_str = tok.decode([top1_id])

        committed_text = tok.decode(self.committed_ids)
        at_end = (
            len(self.committed_ids) >= self.max_total_tokens
            or _has_doc_stop(committed_text)
            or _partial_stop_pos(committed_text, DOC_STOPS) != -1
            or ((top1_id in eos_ids or _is_tag_starter(top1_str)) and top1_prob >= 0.45)
        )

        fh_prob = float(self.last_fh["fork_prob"]) if self.last_fh is not None else None
        fh_ent = float(self.last_fh["pred_entropy"]) if self.last_fh is not None else None
        fh_fis = float(self.last_fh["pred_fisher_top"]) if self.last_fh is not None else None

        lead_norm = top1_str.strip().lower()
        needs_full = (not at_end) and (
            entropy >= 0.65
            or top1_prob < 0.84
            or (fh_prob is not None and fh_prob >= 0.45)
            or (lead_norm in _PEEK_FUNCTION_WORDS and top1_prob >= 0.42)
        )
        ent_score = min(1.0, max(0.0, (entropy - 0.35) / 1.65))
        fh_term = float(fh_prob) if fh_prob is not None else ent_score
        fork_score = round(float(0.60 * ent_score + 0.40 * fh_term), 4) if not at_end else 0.0

        ghost = (
            {"text": "", "next_word_text": "", "next_word_ids": [], "full_ids": []}
            if at_end
            else {
                "text": top1_str,
                "next_word_text": top1_str,
                "next_word_ids": [top1_id],
                "full_ids": [top1_id],
            }
        )
        frontier = {
            "at_end": at_end,
            "regime": "linear",
            "fork_kind": "linear",
            "semantic_dispersion": 0.0,
            "source_recipe": None,
            "fork_score": fork_score,
            "entropy": round(entropy, 4),
            "peeked_entropy": round(entropy, 4),
            "top1_prob": round(top1_prob, 4),
            "fork_head_prob": round(fh_prob, 4) if fh_prob is not None else None,
            "fork_head_entropy": round(fh_ent, 4) if fh_ent is not None else None,
            "fork_head_fisher": round(fh_fis, 4) if fh_fis is not None else None,
            "ghost": ghost,
            "branches": [],
        }
        return frontier, needs_full

    def _peek_through_function_word(self, fn_id: int, fn_prob: float) -> tuple[float, list[dict[str, Any]]] | None:
        """Peek 1 cached token past a high-prob function word to detect immediate downstream forks."""
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        pkv_1 = _cache_clone_batch(self.pkv, 1)
        if pkv_1 is None:
            return None

        prefix_len = len(self.prompt_ids) + len(self.committed_ids)
        self._apply_global_steering()
        self._sync_hooks_pos(prefix_len)
        try:
            with self._pause_fork_scorer(), torch.no_grad():
                x = torch.tensor([[int(fn_id)]], dtype=torch.long, device=dev)
                out = arch.forward_hidden(model, input_ids=x, past_key_values=pkv_1, use_cache=False)
                lg = arch.logits(model, out.last_hidden_state[0, -1][None])[0].float().cpu()
        except Exception:
            return None
        finally:
            self._clear_steering()

        p2 = lg.softmax(-1)
        ent2 = float(-(p2 * p2.clamp_min(1e-30).log()).sum().item())
        if ent2 < 0.95:
            return ent2, []

        vals, ids = torch.topk(p2, k=min(14, int(p2.numel())))
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        seen: dict[str, dict[str, Any]] = {}
        for tid, v in zip(ids.tolist(), vals.tolist()):
            if tid in eos_ids or v < 0.03:
                continue
            s = tok.decode([tid])
            if _is_tag_starter(s):
                continue
            key = s.strip().lower() or f"__W_{tid}"
            joint_p = float(fn_prob * v)
            if key in seen:
                seen[key]["prob"] += joint_p
            else:
                seen[key] = {"ids": [int(fn_id), int(tid)], "prob": joint_p, "raw_text": s}
        expanded = sorted(seen.values(), key=lambda x: -x["prob"])[:4]
        return ent2, expanded

    # ---------------------------------------------------------- Non-destructive History & Alignment
    def _record_active_node(self) -> None:
        tok, _ = self.engine.ensure_loaded()
        node = self.history[self.active_node_id]
        existing = node["token_ids"]
        # Non-destructive invariant: NEVER truncate a saved suffix while rewound inside this node
        if len(self.committed_ids) < len(existing) and existing[: len(self.committed_ids)] == self.committed_ids:
            return
        node["token_ids"] = list(self.committed_ids)
        node["meta"] = [dict(m) for m in self.committed_meta]
        txt = _strip_doc_stop(tok.decode(self.committed_ids))
        node["text"] = txt
        tail = txt.strip().splitlines()[-1] if txt.strip() else ""
        node["snippet"] = tail[-44:] if len(tail) > 44 else tail

    def _split_history_node(self, fork_at: int, label_hint: str) -> None:
        """Create a child/sibling history node when diverging after a rewind without mutating parent suffix."""
        fork_idx = int(fork_at)
        pid = self.active_node_id
        # If branching again at or before the current node's own fork point, attach to the shared ancestor
        while (
            self.history[pid]["parent_id"] is not None
            and fork_idx <= int(self.history[pid]["fork_token_idx"])
        ):
            anc_id = self.history[pid]["parent_id"]
            anc_ids = self.history[anc_id]["token_ids"]
            if len(anc_ids) >= fork_idx and anc_ids[:fork_idx] == self.committed_ids[:fork_idx]:
                pid = anc_id
            else:
                break

        nid = f"n{self._node_seq}"
        self._node_seq += 1
        self.history[nid] = {
            "id": nid,
            "parent_id": pid,
            "fork_token_idx": fork_idx,
            "token_ids": list(self.committed_ids),
            "meta": [dict(m) for m in self.committed_meta],
            "label": label_hint.strip()[:28] or f"Branch {self._node_seq - 1}",
            "created_at": time.time(),
        }
        self.active_node_id = nid
        self._record_active_node()

    def _redo_remaining(self) -> int:
        node_ids = self.history[self.active_node_id]["token_ids"]
        cur_len = len(self.committed_ids)
        if cur_len < len(node_ids) and node_ids[:cur_len] == self.committed_ids:
            return len(node_ids) - cur_len
        return 0

    def _compute_alignment(self, tok_strs: list[str]) -> dict[str, Any]:
        """Compute lightweight O(N) source line/stanza ↔ committed target span alignment metadata."""
        src_lines = self._source_lines
        if not src_lines:
            return {"source_lines": [], "active_source_line": 0, "spans": []}
        if not tok_strs:
            return {"source_lines": [dict(sl) for sl in src_lines], "active_source_line": 0, "spans": []}

        n_toks = len(tok_strs)
        n_src = len(src_lines)
        spans: list[dict[str, Any]] = []

        if "\n" in self.source_text:
            # Partition committed tokens into non-empty target line spans with stanza indices
            target_spans: list[tuple[int, int, str, int]] = []
            cur_start: int | None = None
            cur_chars: list[str] = []
            cur_stanza = 0
            seen_in_stanza = False
            ended_with_nl = False
            ended_with_stanza = False

            for i, s in enumerate(tok_strs):
                if "\n" not in s:
                    if cur_start is None:
                        cur_start = i
                    cur_chars.append(s)
                    seen_in_stanza = True
                    ended_with_nl = False
                    ended_with_stanza = False
                    continue

                nl_count = s.count("\n")
                parts = s.split("\n")
                # Leading non-empty fragment belongs to current line
                if parts[0]:
                    if cur_start is None:
                        cur_start = i
                    cur_chars.append(parts[0])
                    seen_in_stanza = True
                    target_spans.append((cur_start, i + 1, "".join(cur_chars), cur_stanza))
                    cur_start = None
                    cur_chars = []
                elif cur_start is not None:
                    target_spans.append((cur_start, i, "".join(cur_chars), cur_stanza))
                    cur_start = None
                    cur_chars = []

                # Blank line / stanza break detection
                if nl_count >= 2 or (not parts[0] and not parts[-1] and not seen_in_stanza):
                    if seen_in_stanza:
                        cur_stanza += 1
                        seen_in_stanza = False
                        ended_with_stanza = True
                elif nl_count == 1 and not parts[0] and not parts[-1] and i > 0 and tok_strs[i - 1].endswith("\n"):
                    if seen_in_stanza:
                        cur_stanza += 1
                        seen_in_stanza = False
                        ended_with_stanza = True

                # Trailing non-empty fragment after newline starts a new line at token `i`
                if parts[-1]:
                    cur_start = i
                    cur_chars = [parts[-1]]
                    seen_in_stanza = True
                    ended_with_nl = False
                    ended_with_stanza = False
                else:
                    ended_with_nl = True

            if cur_start is not None:
                target_spans.append((cur_start, n_toks, "".join(cur_chars), cur_stanza))

            src_by_stanza: dict[int, list[int]] = {}
            for sl in src_lines:
                src_by_stanza.setdefault(int(sl["stanza"]), []).append(int(sl["index"]))

            # Fallback for unwrapped single-line translation of multi-line poem
            if (
                len(target_spans) == 1
                and n_src > 1
                and len(target_spans[0][2]) > 1.8 * max(len(src_lines[0]["text"]), 1)
                and (target_spans[0][1] - target_spans[0][0]) >= n_src
            ):
                t0, t1, _, _ = target_spans[0]
                span_len = t1 - t0
                weights = [max(len(sl["text"]), 1) for sl in src_lines]
                total_w = float(sum(weights))
                cursor = t0
                cum_w = 0.0
                for si, w in enumerate(weights):
                    cum_w += w
                    nxt_c = t1 if si == n_src - 1 else min(t1, max(cursor + 1, t0 + int(round(span_len * cum_w / total_w))))
                    if cursor < nxt_c:
                        spans.append({
                            "target_start": cursor,
                            "target_end": nxt_c,
                            "source_line": si,
                            "confidence": 0.55,
                        })
                        cursor = nxt_c
            else:
                stanza_ord: dict[int, int] = {}
                for g_idx, (t_s, t_e, t_txt, t_st) in enumerate(target_spans):
                    k = stanza_ord.get(t_st, 0)
                    stanza_ord[t_st] = k + 1
                    st_lines = src_by_stanza.get(t_st)
                    if st_lines is not None and k < len(st_lines):
                        src_idx = st_lines[k]
                        src_len = max(len(src_lines[src_idx]["text"].strip()), 1)
                        tgt_len = max(len(t_txt.strip()), 1)
                        ratio = min(src_len, tgt_len) / max(src_len, tgt_len)
                        conf = round(0.75 + 0.20 * ratio, 3)
                    else:
                        src_idx = min(g_idx, n_src - 1)
                        conf = 0.62
                    spans.append({
                        "target_start": int(t_s),
                        "target_end": int(t_e),
                        "source_line": int(src_idx),
                        "confidence": float(conf),
                    })

            if not spans:
                active_src = 0
            elif ended_with_stanza and cur_stanza in src_by_stanza:
                active_src = src_by_stanza[cur_stanza][0]
            elif ended_with_nl:
                active_src = min(spans[-1]["source_line"] + 1, n_src - 1)
            else:
                active_src = spans[-1]["source_line"]
        else:
            # Prose mode: single sentence or multi-sentence segmentation
            if n_src == 1:
                spans.append({
                    "target_start": 0,
                    "target_end": n_toks,
                    "source_line": 0,
                    "confidence": 0.95,
                })
                active_src = 0
            else:
                seg_start = 0
                seg_idx = 0
                last_ended = False
                for i, s in enumerate(tok_strs):
                    if _SENT_END_RE.search(s):
                        src_idx = min(seg_idx, n_src - 1)
                        conf = 0.80 if seg_idx < n_src else 0.60
                        spans.append({
                            "target_start": seg_start,
                            "target_end": i + 1,
                            "source_line": src_idx,
                            "confidence": conf,
                        })
                        seg_start = i + 1
                        seg_idx += 1
                        last_ended = (i == n_toks - 1)
                if seg_start < n_toks:
                    src_idx = min(seg_idx, n_src - 1)
                    conf = 0.80 if seg_idx < n_src else 0.60
                    spans.append({
                        "target_start": seg_start,
                        "target_end": n_toks,
                        "source_line": src_idx,
                        "confidence": conf,
                    })
                    last_ended = False
                if not spans:
                    active_src = 0
                elif last_ended:
                    active_src = min(spans[-1]["source_line"] + 1, n_src - 1)
                else:
                    active_src = spans[-1]["source_line"]

        return {
            "source_lines": [dict(sl) for sl in src_lines],
            "active_source_line": int(max(0, min(active_src, n_src - 1))),
            "spans": spans,
        }

    def state_dict(self, include_frontier: bool = True) -> dict[str, Any]:
        tok, _ = self.engine.ensure_loaded()
        frontier = self.inspect_frontier() if include_frontier else self.last_frontier
        text = _strip_doc_stop(tok.decode(self.committed_ids))

        # Group committed tokens into clickable word spans with word_start_idx for clean word-level rewind
        tok_strs = [tok.decode([int(tid)]) for tid in self.committed_ids]
        tokens_out: list[dict[str, Any]] = []
        cur_word_start = 0
        for i, (tid, s) in enumerate(zip(self.committed_ids, tok_strs)):
            if i > 0 and (_is_word_boundary(tok_strs[i - 1], s) or bool(self.committed_meta[i].get("step_boundary"))):
                cur_word_start = i
            tokens_out.append({
                "idx": i,
                "token_id": int(tid),
                "token": s,
                "word_start_idx": cur_word_start,
                **self.committed_meta[i],
            })

        history_summary = [
            {
                "id": n["id"],
                "parent_id": n["parent_id"],
                "fork_token_idx": n["fork_token_idx"],
                "label": n["label"],
                "n_tokens": len(n["token_ids"]),
                "snippet": n.get("snippet", ""),
                "active": n["id"] == self.active_node_id,
            }
            for n in self.history.values()
        ]

        rem = self._redo_remaining()
        compass_ready = self._is_compass_ready()

        return {
            "session_id": self.session_id,
            "source_text": self.source_text,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "committed_text": text,
            "committed_token_ids": list(self.committed_ids),
            "n_tokens": len(self.committed_ids),
            "tokens": tokens_out,
            "global_coords": dict(self.global_coords),
            "frontier": frontier,
            "has_compass": compass_ready,
            "compass_ready": compass_ready,
            "can_redo": rem > 0,
            "redo_remaining": int(rem),
            "alignment": self._compute_alignment(tok_strs),
            "history": history_summary,
            "active_node_id": self.active_node_id,
        }

    # ---------------------------------------------------------- Step / Redo / Fast Glide / Rewind
    def step(
        self,
        mode: str = "word",
        branch_index: int | None = None,
        commit_full_branch: bool = False,
        custom_text: str | None = None,
        custom_token_ids: list[int] | None = None,
        source_tag: str | None = None,
    ) -> dict[str, Any]:
        """Advance the live translation by a word, token, chosen branch, or steered/typed text."""
        tok, _ = self.engine.ensure_loaded()
        frontier = self.last_frontier or self.inspect_frontier()
        if frontier.get("at_end") and custom_text is None and custom_token_ids is None:
            return self.state_dict(include_frontier=True)

        # If we rewound earlier and are now diverging from the active node's stored suffix, fork a child/sibling node
        cur_node_ids = self.history[self.active_node_id]["token_ids"]
        diverging = len(cur_node_ids) > len(self.committed_ids)

        chosen_ids: list[int] = []
        tag = source_tag or "step"

        if custom_token_ids is not None:
            chosen_ids = [int(i) for i in custom_token_ids]
            tag = source_tag or "steered"
        elif custom_text is not None:
            chosen_ids = tok(str(custom_text), add_special_tokens=False)["input_ids"]
            tag = source_tag or "typed"
        elif branch_index is not None:
            branches = frontier.get("branches", [])
            if 0 <= int(branch_index) < len(branches):
                br = branches[int(branch_index)]
                chosen_ids = list(br["preview_ids"] if commit_full_branch else br["first_word_ids"])
                tag = "branch"
            else:
                raise IndexError(f"branch_index {branch_index} out of range (0..{len(branches) - 1})")
        else:
            ghost = frontier.get("ghost", {})
            if mode == "token":
                ids_src = ghost.get("next_word_ids") or ghost.get("full_ids") or []
                chosen_ids = [int(ids_src[0])] if ids_src else []
            else:
                chosen_ids = list(ghost.get("next_word_ids") or [])
            tag = "step"

        if not chosen_ids:
            return self.state_dict(include_frontier=True)

        # Split history node ONLY if chosen tokens actually diverge from the saved suffix prefix
        if diverging:
            old_next = cur_node_ids[len(self.committed_ids) : len(self.committed_ids) + len(chosen_ids)]
            common = min(len(old_next), len(chosen_ids))
            if old_next[:common] != chosen_ids[:common]:
                self._split_history_node(len(self.committed_ids), tok.decode(chosen_ids))

        ent = float(frontier.get("entropy", 0.0))
        fscore = float(frontier.get("fork_score", 0.0))
        is_fk = frontier.get("regime") == "fork"

        is_word_commit = (
            mode == "word"
            or custom_token_ids is not None
            or custom_text is not None
            or branch_index is not None
        )
        for idx, tid in enumerate(chosen_ids):
            self.committed_ids.append(int(tid))
            meta_row: dict[str, Any] = {
                "entropy": round(ent, 3) if idx == 0 else 0.0,
                "fork_score": round(fscore, 3) if idx == 0 else 0.0,
                "is_fork": bool(is_fk) if idx == 0 else False,
                "source": tag,
            }
            if idx == 0 and is_word_commit:
                meta_row["step_boundary"] = True
            self.committed_meta.append(meta_row)

        self._invalidate_cursor()
        self._advance_cache(chosen_ids)
        self._record_active_node()
        return self.state_dict(include_frontier=True)

    def redo(self, mode: str = "word") -> dict[str, Any]:
        """Advance forward along the active history node's saved rewound suffix without creating duplicate branches."""
        if mode not in ("word", "token"):
            raise ValueError(f"Invalid redo mode: {mode!r} (expected 'word' or 'token')")
        rem = self._redo_remaining()
        if rem == 0:
            return self.state_dict(include_frontier=True)

        tok, _ = self.engine.ensure_loaded()
        node = self.history[self.active_node_id]
        cur_len = len(self.committed_ids)
        remaining_ids = node["token_ids"][cur_len:]
        saved_meta = node.get("meta", [])
        if mode == "token":
            adv_ids = [int(remaining_ids[0])]
        else:
            if cur_len < len(saved_meta) and saved_meta[cur_len].get("step_boundary"):
                end = cur_len + 1
                prev_s = tok.decode([int(node["token_ids"][cur_len])])
                while end < len(node["token_ids"]) and (end - cur_len) < 16:
                    if end < len(saved_meta) and saved_meta[end].get("step_boundary"):
                        break
                    cur_s = tok.decode([int(node["token_ids"][end])])
                    if _is_word_boundary(prev_s, cur_s):
                        break
                    prev_s = cur_s
                    end += 1
                adv_ids = [int(i) for i in node["token_ids"][cur_len:end]]
            else:
                adv_ids = _extract_first_word_ids(tok, remaining_ids)
        adv_ids = adv_ids[: len(remaining_ids)]

        for offset, tid in enumerate(adv_ids):
            idx = cur_len + offset
            if idx < len(saved_meta):
                m = dict(saved_meta[idx])
            else:
                m = {"entropy": 0.0, "fork_score": 0.0, "is_fork": False, "source": "redo"}
            self.committed_ids.append(int(tid))
            self.committed_meta.append(m)

        self._invalidate_cursor()
        self._advance_cache(adv_ids)
        return self.state_dict(include_frontier=True)

    def _fast_step_word(self, max_subwords: int = 6) -> bool:
        """Commit one whole greedy word directly on `self.pkv` with 0 cache clones and 0 lookahead."""
        tok, _ = self.engine.ensure_loaded()
        if self.last_logits is None:
            self._prefill_full()

        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        lg0 = self.last_logits
        p0 = lg0.softmax(-1)
        ent0 = float(-(p0 * p0.clamp_min(1e-30).log()).sum().item())
        top_vals, top_ids = torch.topk(p0, k=min(8, int(p0.numel())))
        tid0: int | None = None
        for v, tid_t in zip(top_vals.tolist(), top_ids.tolist()):
            tid = int(tid_t)
            if tid in eos_ids:
                if tid == int(top_ids[0].item()) and v >= 0.45:
                    return False
                continue
            s = tok.decode([tid])
            if _is_tag_starter(s):
                if tid == int(top_ids[0].item()) and v >= 0.45:
                    return False
                continue
            tid0 = tid
            break
        if tid0 is None:
            return False

        fh_prob0 = float(self.last_fh["fork_prob"]) if self.last_fh is not None else None
        ent_score0 = min(1.0, max(0.0, (ent0 - 0.35) / 1.65))
        fscore0 = round(float(0.60 * ent_score0 + 0.40 * (fh_prob0 if fh_prob0 is not None else ent_score0)), 3)
        is_fk0 = bool(ent0 >= 1.25 or (fh_prob0 is not None and fh_prob0 >= 0.55))

        word_start = len(self.committed_ids)
        cur_node_ids = list(self.history[self.active_node_id]["token_ids"])
        word_ids: list[int] = [tid0]
        first_str = tok.decode([tid0])

        self._invalidate_cursor()
        self.committed_ids.append(tid0)
        self.committed_meta.append({
            "entropy": round(ent0, 3),
            "fork_score": fscore0,
            "is_fork": is_fk0,
            "source": "step",
            "step_boundary": True,
        })
        self._advance_cache([tid0])

        if "\n" not in first_str and not _PUNCT_ONLY.match(first_str):
            prev_s = first_str
            for _ in range(1, max_subwords):
                if len(self.committed_ids) >= self.max_total_tokens:
                    break
                nxt_id = int(self.last_logits.argmax().item())
                if nxt_id in eos_ids:
                    break
                nxt_s = tok.decode([nxt_id])
                if _is_tag_starter(nxt_s) or _is_word_boundary(prev_s, nxt_s):
                    break
                cand_text = tok.decode(word_ids + [nxt_id])
                if _has_doc_stop(cand_text) or _partial_stop_pos(cand_text, DOC_STOPS) != -1:
                    break
                word_ids.append(nxt_id)
                self.committed_ids.append(nxt_id)
                self.committed_meta.append({
                    "entropy": 0.0,
                    "fork_score": 0.0,
                    "is_fork": False,
                    "source": "step",
                })
                self._advance_cache([nxt_id])
                prev_s = nxt_s

        # Respect F4 non-destructive history if we were rewound inside an existing node
        if len(cur_node_ids) > word_start:
            old_next = cur_node_ids[word_start : word_start + len(word_ids)]
            common = min(len(old_next), len(word_ids))
            if old_next[:common] != word_ids[:common]:
                # Temporarily reconstruct prefix at word_start for accurate parent fork_token_idx
                saved_ids = list(self.committed_ids)
                saved_meta = list(self.committed_meta)
                self.committed_ids = saved_ids[:word_start]
                self.committed_meta = saved_meta[:word_start]
                self._split_history_node(word_start, tok.decode(word_ids))
                self.committed_ids = saved_ids
                self.committed_meta = saved_meta

        self._record_active_node()
        return True

    def _glide_next(
        self,
        steps_taken: int,
        max_words: int,
        stop_on_fork: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Execute a single glide step; skips heavy lookahead on trivially linear glided words."""
        events: list[dict[str, Any]] = []
        if steps_taken == 0:
            light0, _ = self._lightweight_frontier()
            if light0["at_end"]:
                self.inspect_frontier(_schedule_precompute=False)
                st0 = self.state_dict(include_frontier=False)
                events.append({"event": "paused", "reason": "end", "steps": 0, "state": st0})
                return events, True

        advanced = self._fast_step_word()
        if not advanced:
            self.inspect_frontier(_schedule_precompute=False)
            st_end = self.state_dict(include_frontier=False)
            events.append({"event": "paused", "reason": "end", "steps": steps_taken, "state": st_end})
            return events, True

        steps_taken += 1
        light_f, needs_full = self._lightweight_frontier()
        at_limit = steps_taken >= max_words
        need_inspect = light_f["at_end"] or at_limit or (stop_on_fork and needs_full)
        if need_inspect:
            # Schedule compass precompute only if we actually pause on fork (handled below)
            next_f = self.inspect_frontier(_schedule_precompute=False)
        else:
            next_f = light_f
            self.last_frontier = light_f

        st = self.state_dict(include_frontier=False)
        events.append({
            "event": "step",
            "steps": steps_taken,
            "committed_text": st["committed_text"],
            "n_tokens": st["n_tokens"],
            "regime": next_f["regime"],
            "fork_score": next_f["fork_score"],
            "state": st,
        })

        if next_f.get("at_end"):
            events.append({"event": "paused", "reason": "end", "steps": steps_taken, "state": st})
            return events, True
        if stop_on_fork and next_f.get("regime") == "fork":
            if self.auto_precompute_compass:
                self._schedule_compass_precompute()
            events.append({"event": "paused", "reason": "fork", "steps": steps_taken, "state": st})
            return events, True
        if at_limit:
            if next_f.get("regime") == "fork" and self.auto_precompute_compass:
                self._schedule_compass_precompute()
            events.append({"event": "paused", "reason": "limit", "steps": steps_taken, "state": st})
            return events, True
        return events, False

    def glide(
        self,
        max_words: int = 32,
        stop_on_fork: bool = True,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream words off the persistent KV cache, auto-pausing when a complex fork opens."""
        steps_taken = 0
        limit = max(1, int(max_words))
        while True:
            events, done = self._glide_next(steps_taken, limit, stop_on_fork=stop_on_fork)
            for ev in events:
                if ev.get("event") == "step":
                    steps_taken = int(ev["steps"])
                yield ev
            if done:
                return

    def rewind(self, to_token_index: int) -> dict[str, Any]:
        """Rewind the cursor to `to_token_index`, non-destructively preserving the explored path in history."""
        target = max(0, min(int(to_token_index), len(self.committed_ids)))
        if target == len(self.committed_ids):
            return self.state_dict(include_frontier=True)

        self._record_active_node()

        self.committed_ids = self.committed_ids[:target]
        self.committed_meta = self.committed_meta[:target]
        self._invalidate_cursor()

        cache_target = len(self.prompt_ids) + target
        last_tok = self.committed_ids[-1] if target > 0 else self.prompt_ids[-1]
        self._rewind_cache_to(cache_target, last_tok)
        return self.state_dict(include_frontier=True)

    def checkout_node(self, node_id: str) -> dict[str, Any]:
        """Switch to any previously explored branch node in `self.history` using LCP KV-cache reuse."""
        if node_id not in self.history:
            raise KeyError(f"Unknown history node: {node_id!r}")
        self._record_active_node()
        node = self.history[node_id]

        old_ids = list(self.committed_ids)
        new_ids = list(node["token_ids"])
        lcp = 0
        for a, b in zip(old_ids, new_ids):
            if a != b:
                break
            lcp += 1
        full_lcp = len(self.prompt_ids) + lcp

        self.active_node_id = node_id
        self.committed_ids = new_ids
        self.committed_meta = [dict(m) for m in node["meta"]]
        self._invalidate_cursor()

        cache_valid = self.pkv is not None and _cache_seq_len(self.pkv) == len(self.prompt_ids) + len(old_ids)
        if not cache_valid:
            self._prefill_full()
        elif old_ids == new_ids and self.last_logits is not None:
            pass
        elif lcp == len(old_ids):
            self._advance_cache(new_ids[lcp:])
        elif lcp < len(new_ids):
            cropped, ok = _cache_crop(self.pkv, full_lcp)
            if ok:
                self.pkv = cropped
                self._advance_cache(new_ids[lcp:])
            else:
                self._prefill_full()
        else:
            last_tok = new_ids[-1] if new_ids else self.prompt_ids[-1]
            self._rewind_cache_to(full_lcp, last_tok)

        return self.state_dict(include_frontier=True)

    def set_global_coords(self, global_coords: dict[str, float]) -> dict[str, Any]:
        """Update global practice/tone sliders and refresh cache from `start - 1` (avoiding full prompt prefill)."""
        self.global_coords = {str(k): float(v) for k, v in (global_coords or {}).items()}
        self._invalidate_cursor()

        crop_pos = self.start - 1
        expected_len = len(self.prompt_ids) + len(self.committed_ids)
        if crop_pos >= 1 and self.pkv is not None and _cache_seq_len(self.pkv) == expected_len:
            cropped, ok = _cache_crop(self.pkv, crop_pos)
            if ok:
                self.pkv = cropped
                _, model = self.engine.ensure_loaded()
                dev = self.engine._dev()
                tail = [int(self.prompt_ids[-1])] + [int(t) for t in self.committed_ids]
                x = torch.tensor([tail], dtype=torch.long, device=dev)
                self._apply_global_steering()
                self._sync_hooks_pos(crop_pos)
                if self.engine.fork_scorer is not None:
                    self.engine.fork_scorer.reset()
                try:
                    with torch.no_grad():
                        out = arch.forward_hidden(model, input_ids=x, past_key_values=self.pkv, use_cache=True)
                        self.pkv = out.past_key_values
                        h_last = out.last_hidden_state[0, -1]
                        self._record_last_hidden(h_last)
                        self.last_logits = arch.logits(model, h_last[None])[0].detach().cpu()
                        self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
                    return self.state_dict(include_frontier=True)
                except Exception:
                    pass
                finally:
                    self._clear_steering()

        self._prefill_full()
        return self.state_dict(include_frontier=True)

    # ---------------------------------------------------------- Tier-2 Spectral Compass & Async Precompute
    def _sample_doc_continuations(
        self,
        prefix: list[int],
        n_samples: int,
        lookahead_tokens: int,
        seed: int,
    ) -> list[list[int]]:
        """Ancestral burst sampling using `DOC_STOPS` (preserves multi-stanza `\\n\\n`) and reusing `self.pkv`."""
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        m = max(2, int(n_samples))
        max_new = max(2, int(lookahead_tokens))
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

        # Primary cached ancestral sampling path (0 prefix re-prefills!)
        if (
            self.pkv is not None
            and _cache_seq_len(self.pkv) == len(prefix)
            and self.last_logits is not None
        ):
            pkv_batch = _cache_clone_batch(self.pkv, m)
            if pkv_batch is not None:
                gen_rng = torch.Generator(device="cpu").manual_seed(int(seed))
                p0 = self.last_logits.float().softmax(-1)
                first_toks = torch.multinomial(p0, num_samples=m, replacement=True, generator=gen_rng).tolist()
                seqs: list[list[int]] = []
                alive: list[bool] = []
                for t0 in first_toks:
                    ti = int(t0)
                    if ti in eos_ids:
                        seqs.append([])
                        alive.append(False)
                    else:
                        seqs.append([ti])
                        alive.append(not _has_doc_stop(tok.decode([ti])))

                self._apply_global_steering(start_pos=self.start - 1)
                try:
                    with self._pause_fork_scorer(), torch.no_grad():
                        cur_pos = len(prefix)
                        step_toks = [int(t) for t in first_toks]
                        for step_i in range(max_new - 1):
                            if not any(alive):
                                break
                            x_step = torch.tensor([[t] for t in step_toks], dtype=torch.long, device=dev)
                            self._sync_hooks_pos(cur_pos)
                            out = arch.forward_hidden(model, input_ids=x_step, past_key_values=pkv_batch, use_cache=True)
                            pkv_batch = out.past_key_values
                            cur_pos += 1
                            lg = arch.logits(model, out.last_hidden_state[:, -1, :]).float().cpu()
                            probs = lg.softmax(-1)
                            sampled = torch.multinomial(probs, num_samples=1, generator=gen_rng).squeeze(-1).tolist()
                            step_toks = [int(t) for t in sampled]
                            for i, tid in enumerate(step_toks):
                                if not alive[i]:
                                    continue
                                if tid in eos_ids:
                                    alive[i] = False
                                    continue
                                seqs[i].append(tid)
                                if _has_doc_stop(tok.decode(seqs[i])):
                                    alive[i] = False
                finally:
                    self._clear_steering()
                return [cut_doc(tok, s) for s in seqs]

        # Uncached fallback using DOC_STOPS (NEVER legacy STOPS with '\n\n')
        from transformers import GenerationConfig

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        cfg = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
            stop_strings=list(DOC_STOPS),
        )
        self._apply_global_steering(start_pos=self.start - 1)
        out_seqs: list[list[int]] = []
        try:
            with self._pause_fork_scorer(), torch.no_grad():
                x = torch.tensor([prefix], dtype=torch.long, device=dev)
                batch_sz = min(16, m)
                for c in range(0, m, batch_sz):
                    k = min(batch_sz, m - c)
                    self._sync_hooks_pos(0)
                    y = model.generate(x.expand(k, -1), generation_config=cfg, tokenizer=tok)
                    for row in y[:, len(prefix) :].tolist():
                        out_seqs.append(cut_doc(tok, row))
        finally:
            self._clear_steering()
        return out_seqs

    def _build_fork_head_compass(
        self,
        model: Any,
        frontier: dict[str, Any],
        fh_dirs: torch.Tensor,
        prefix: list[int],
        b: int,
        layer: int,
        seed: int,
        t0: float,
    ) -> dict[str, Any]:
        """Zero-forward, zero-backward <10 ms compass using ForkHead(n_dirs>=2) axes + branch projection."""
        raw_axes = fh_dirs[:2].detach().float().cpu().numpy().astype(np.float64)
        u1 = raw_axes[0] / max(float(np.linalg.norm(raw_axes[0])), 1e-12)
        u2 = raw_axes[1] - float(np.dot(raw_axes[1], u1)) * u1
        u2_norm = float(np.linalg.norm(u2))
        if u2_norm < 1e-12:
            e = np.zeros_like(u1)
            e[int(np.argmin(np.abs(u1)))] = 1.0
            u2 = e - float(np.dot(e, u1)) * u1
            u2_norm = max(float(np.linalg.norm(u2)), 1e-12)
        u2 = u2 / u2_norm
        axes = np.vstack([u1, u2])

        # Alpha scale with ZERO forward passes: checkpoint metadata -> residual-norm-scaled default
        alpha_unit = 1.0
        fh_head = getattr(getattr(self.engine, "fork_scorer", None), "head", None)
        fh_meta = getattr(fh_head, "meta", None) if fh_head is not None else None
        rec_alpha = fh_meta.get("recommended_alpha") if isinstance(fh_meta, dict) else None
        if rec_alpha is not None and math.isfinite(float(rec_alpha)) and float(rec_alpha) > 0.0:
            alpha_unit = float(rec_alpha)
        elif self.last_hidden_norm > 0.0:
            alpha_unit = float(min(25.0, max(0.15, 0.08 * self.last_hidden_norm)))

        # Project candidate branch token-embedding / lm_head vectors onto axes[:2] (0 forwards!)
        branches = frontier["branches"][:5]
        n_b = len(branches)
        W = getattr(getattr(model, "lm_head", None), "weight", None)
        if W is None or int(W.shape[-1]) != axes.shape[1]:
            W = getattr(getattr(arch.text_model(model), "embed_tokens", None), "weight", None)

        anchors: np.ndarray | None = None
        var_expl = 1.0
        if W is not None and int(W.shape[-1]) == axes.shape[1] and n_b >= 2:
            n_vocab = int(W.shape[0])
            # Skip shared leading function-word token if every branch shares it
            first_ids = [br["preview_ids"][0] for br in branches if br.get("preview_ids")]
            skip_first = len(first_ids) == n_b and len(set(first_ids)) == 1
            vecs: list[np.ndarray] = []
            with torch.no_grad():
                for br in branches:
                    pids = list(br.get("preview_ids") or [])
                    if skip_first and len(pids) > 1:
                        pids = pids[1:]
                    use_ids = [int(i) for i in pids[:4] if 0 <= int(i) < n_vocab]
                    if not use_ids:
                        vecs.append(np.zeros(axes.shape[1], dtype=np.float64))
                    else:
                        idx_t = torch.tensor(use_ids, dtype=torch.long, device=W.device)
                        v = W.index_select(0, idx_t).detach().float().mean(dim=0).cpu().numpy().astype(np.float64)
                        vecs.append(v)
            R_mat = np.vstack(vecs)
            R_c = R_mat - R_mat.mean(axis=0, keepdims=True)
            raw_anc = R_c @ axes.T  # (n_b, 2)
            norms = np.linalg.norm(raw_anc, axis=1)
            max_norm = float(norms.max())
            if max_norm >= 1e-6:
                cand_anc = raw_anc / max_norm
                min_pdist = min(
                    float(np.linalg.norm(cand_anc[i] - cand_anc[j]))
                    for i in range(n_b)
                    for j in range(i + 1, n_b)
                )
                if min_pdist >= 1e-3:
                    anchors = cand_anc
                    tot_r = float((R_c ** 2).sum())
                    var_expl = float((raw_anc ** 2).sum() / max(tot_r, 1e-30))

        # Degenerate projection fallback only
        if anchors is None:
            angles = np.linspace(0.0, 2.0 * math.pi, n_b, endpoint=False)
            anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            var_expl = 1.0

        reading_meta = [
            {
                "id": idx,
                "label": br["distinctive"],
                "preview": br["preview_text"],
                "share": br["share"],
                "color": br["color"],
                "anchor": [round(float(anchors[idx, 0]), 4), round(float(anchors[idx, 1]), 4)],
            }
            for idx, br in enumerate(branches)
        ]
        probe_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        resp = {
            "source": "fork_head",
            "cursor_len": len(self.committed_ids),
            "variance_explained": round(float(min(1.0, max(0.0, var_expl))), 4),
            "alpha_unit": round(float(alpha_unit), 6),
            "probe_ms": probe_ms,
            "readings": reading_meta,
            "dots": [],
        }
        self.active_compass = {
            "cursor_len": len(self.committed_ids),
            "cursor_version": self._cursor_version,
            "seed": int(seed),
            "b": b,
            "layer": layer,
            "axes": axes,
            "anchors": anchors,
            "alpha_unit": float(alpha_unit),
            "readings": reading_meta,
            "source": "fork_head",
            "response": copy.deepcopy(resp),
        }
        return resp

    @staticmethod
    def _cluster_gradients_to_plane(
        G: np.ndarray,
        cont_texts: list[str],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray, float, list[dict[str, Any]], list[dict[str, Any]]]:
        """Pure-CPU NumPy gradient centering, clustering, 2D spectral plane, and reading metadata."""
        n_samples = len(cont_texts)
        G_c = G - G.mean(axis=0)
        norms = np.linalg.norm(G_c, axis=1, keepdims=True)
        if float(norms.max()) < 1e-12:
            rng = np.random.default_rng(int(seed))
            G_c = rng.normal(scale=1e-4, size=G.shape)
            norms = np.linalg.norm(G_c, axis=1, keepdims=True)
        X = G_c / norms.clip(1e-12)

        kmax = min(5, max(2, n_samples // 2))
        cl = R.cluster(X, min_share=0.05, kmax=kmax, sil_min=0.08)
        labels = np.asarray(cl["labels"], dtype=int)
        rd = [int(r) for r in cl["readings"]]
        if len(rd) < 2:
            _, _, Vt_g = np.linalg.svd(G_c, full_matrices=False)
            proj = G_c @ Vt_g[0]
            labels = (proj >= float(np.median(proj))).astype(int)
            if len(set(labels.tolist())) < 2:
                labels = (np.arange(n_samples) % 2).astype(int)
            rd = [int(r) for r in np.argsort(-np.bincount(labels, minlength=2))]

        M = np.array([G[labels == r].mean(axis=0) for r in rd], dtype=np.float64)
        c_mu = M.mean(axis=0)
        d = G.shape[1]

        def _orth(base: np.ndarray, mat: np.ndarray) -> np.ndarray:
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
            _, sv, Vt = np.linalg.svd(M - c_mu, full_matrices=False)
            u1 = Vt[0] / max(float(np.linalg.norm(Vt[0])), 1e-12)
            u2 = (Vt[1] - float(Vt[1] @ u1) * u1) if (Vt.shape[0] >= 2 and sv[1] > 1e-12) else _orth(u1, G_c)
            u2 /= max(float(np.linalg.norm(u2)), 1e-12)
            axes = np.vstack([u1, u2])
            var_expl = float((sv[:2] ** 2).sum() / max(float((sv ** 2).sum()), 1e-30))
        else:
            diff = M[0] - M[1]
            u1 = (diff / float(np.linalg.norm(diff))) if float(np.linalg.norm(diff)) > 1e-12 else _orth(np.zeros(d), G_c)
            u2 = _orth(u1, G_c)
            axes = np.vstack([u1, u2])
            var_expl = float(((G_c @ axes.T) ** 2).sum() / max(float((G_c ** 2).sum()), 1e-30))

        raw_anchors = (M - c_mu) @ axes.T
        scale = max(float(np.linalg.norm(raw_anchors, axis=1).max()), 1e-8)
        anchors = raw_anchors / scale
        dot_coords = ((G - c_mu) @ axes.T) / scale

        cluster_texts = {r: [cont_texts[i] for i in range(n_samples) if labels[i] == r] for r in rd}
        rep_texts = {r: (cluster_texts[r][0] if cluster_texts[r] else "") for r in rd}
        reading_meta: list[dict[str, Any]] = []
        color_map: dict[int, str] = {}
        for idx, r in enumerate(rd):
            col = READING_PALETTE[idx % len(READING_PALETTE)]
            color_map[r] = col
            others = [rep_texts[o] for o in rd if o != r]
            label = _distinctive_word(rep_texts[r], others) or f"Variant {chr(65 + (idx % 26))}"
            reading_meta.append({
                "id": int(r),
                "label": label,
                "preview": rep_texts[r],
                "share": round(float(len(cluster_texts[r]) / n_samples), 4),
                "color": col,
                "anchor": [round(float(anchors[idx, 0]), 4), round(float(anchors[idx, 1]), 4)],
            })

        dots = [
            {
                "x": round(float(dot_coords[i, 0]), 4),
                "y": round(float(dot_coords[i, 1]), 4),
                "reading": int(labels[i]),
                "color": color_map.get(int(labels[i]), "#999"),
                "text": cont_texts[i],
            }
            for i in range(n_samples)
        ]
        return axes, anchors, dot_coords, rd, labels, var_expl, reading_meta, dots

    def probe_compass(
        self,
        n_samples: int = 10,
        lookahead_tokens: int = 12,
        seed: int = 0,
        force: bool = False,
        force_gradient: bool = False,
    ) -> dict[str, Any]:
        """Build a 2D spectral steering plane right at the current cursor.

        Uses amortised `ForkHead` local subspace directions (<10 ms, 0 forwards/backwards)
        when trained with `n_dirs >= 2`, otherwise runs a compact OOM-safe `DOC_STOPS`
        burst + gradient PCA at the active cursor.
        """
        if (
            not force
            and not force_gradient
            and self._is_compass_ready()
            and self.active_compass is not None
            and self.active_compass.get("seed", 0) == int(seed)
            and "response" in self.active_compass
        ):
            return copy.deepcopy(self.active_compass["response"])

        t0 = time.perf_counter()
        tok, model = self.engine.ensure_loaded()
        prefix = self.prompt_ids + self.committed_ids
        b = len(prefix) - 1
        layer = self.engine.layer
        # Fast path: amortised ForkHead subspace (0 forwards, 0 backwards, <10 ms!)
        fh_dirs = (self.last_fh.get("dirs") if self.last_fh is not None else None) if not force_gradient else None
        if fh_dirs is not None and fh_dirs.shape[0] >= 2:
            frontier = self.last_frontier
            if frontier is None or len(frontier.get("branches", [])) < 2:
                # Build lightweight candidate labels directly from cached last_logits (0 forward passes)
                branches_synth: list[dict[str, Any]] = []
                if self.last_logits is not None:
                    p_lg = self.last_logits.float().softmax(-1)
                    vals, tids = torch.topk(p_lg, k=min(8, int(p_lg.numel())))
                    seen_k: set[str] = set()
                    for tid_v, prob_v in zip(tids.tolist(), vals.tolist()):
                        s_tok = tok.decode([int(tid_v)]).strip() or f"tok_{tid_v}"
                        k_tok = s_tok.lower()
                        if k_tok in seen_k:
                            continue
                        seen_k.add(k_tok)
                        idx_b = len(branches_synth)
                        branches_synth.append({
                            "index": idx_b,
                            "preview_text": tok.decode([int(tid_v)]),
                            "preview_ids": [int(tid_v)],
                            "first_word_text": tok.decode([int(tid_v)]),
                            "first_word_ids": [int(tid_v)],
                            "prob": float(prob_v),
                            "share": float(prob_v),
                            "distinctive": s_tok[:16],
                            "color": READING_PALETTE[idx_b % len(READING_PALETTE)],
                        })
                        if len(branches_synth) >= 4:
                            break
                    tot_s = sum(b_item["share"] for b_item in branches_synth) or 1.0
                    for b_item in branches_synth:
                        b_item["share"] = round(b_item["share"] / tot_s, 4)
                if len(branches_synth) < 2:
                    branches_synth = [
                        {"index": 0, "preview_text": "Variant A", "preview_ids": [], "first_word_text": "Variant A", "first_word_ids": [], "prob": 0.5, "share": 0.5, "distinctive": "Variant A", "color": READING_PALETTE[0]},
                        {"index": 1, "preview_text": "Variant B", "preview_ids": [], "first_word_text": "Variant B", "first_word_ids": [], "prob": 0.5, "share": 0.5, "distinctive": "Variant B", "color": READING_PALETTE[1]},
                    ]
                frontier = {"branches": branches_synth}
            return self._build_fork_head_compass(model, frontier, fh_dirs, prefix, b, layer, seed, t0)

        frontier = self.last_frontier or self.inspect_frontier(_schedule_precompute=False)

        # Standard Tier-2 path: DOC_STOPS burst + OOM-safe gradients at cursor `b`
        n_samples = max(6, int(n_samples))
        lookahead_tokens = max(4, int(lookahead_tokens))
        conts = self._sample_doc_continuations(prefix, n_samples=n_samples, lookahead_tokens=lookahead_tokens, seed=seed)
        cont_texts = [_strip_doc_stop(tok.decode(c)).strip() for c in conts]

        with self._pause_fork_scorer():
            self._clear_steering()
            G = _grads_at_oom_safe(model, prefix, conts, layer, b, batch=min(4, n_samples))

        axes, anchors, _, _, _, var_expl, reading_meta, dots = self._cluster_gradients_to_plane(G, cont_texts, seed)

        far = int(np.argmax(np.linalg.norm(anchors, axis=1)))
        v_far = anchors[far] @ axes
        v_far_norm = max(float(np.linalg.norm(v_far)), 1e-12)
        v_unit = v_far / v_far_norm
        a0 = alpha_for(G, v_unit, 0.25)
        if not math.isfinite(a0) or a0 <= 0.0 or a0 > 1e4:
            a0 = 1.0
        ref_cont = next((ct for ct in conts if ct), [])
        with self._pause_fork_scorer():
            a_far = (
                calibrate(model, self.engine.hook, prefix + ref_cont, b, v_unit, target=0.25, mode="cursor", a0=a0, iters=3)
                if ref_cont
                else a0
            )
        if not math.isfinite(a_far) or a_far <= 0.0:
            a_far = a0
        alpha_unit = float(a_far / v_far_norm)

        probe_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        resp = {
            "source": "gradients",
            "cursor_len": len(self.committed_ids),
            "variance_explained": round(var_expl, 4),
            "alpha_unit": round(float(alpha_unit), 6),
            "probe_ms": probe_ms,
            "readings": reading_meta,
            "dots": dots,
        }
        self.active_compass = {
            "cursor_len": len(self.committed_ids),
            "cursor_version": self._cursor_version,
            "seed": int(seed),
            "b": b,
            "layer": layer,
            "axes": axes,
            "anchors": anchors,
            "alpha_unit": alpha_unit,
            "readings": reading_meta,
            "source": "gradients",
            "response": copy.deepcopy(resp),
        }
        return resp

    def _schedule_compass_precompute(self) -> None:
        """Launch a cancellable daemon thread to precompute the Tier-2 compass for the current cursor."""
        if not self.auto_precompute_compass or self._is_compass_ready():
            return
        version = self._cursor_version
        cursor_len = len(self.committed_ids)
        with self._precompute_meta_lock:
            if (
                self._precompute_target_version == version
                and self._precompute_thread is not None
                and self._precompute_thread.is_alive()
            ):
                return
            self._precompute_cancel.clear()
            self._precompute_target_version = version
            th = threading.Thread(
                target=self._bg_precompute_compass,
                args=(version, cursor_len),
                daemon=True,
                name=f"compass-precompute-{self.session_id}",
            )
            self._precompute_thread = th
            th.start()

    def _bg_precompute_compass(self, version: int, cursor_len: int) -> None:
        """Chunked background Tier-2 compass worker; never holds _precompute_meta_lock across engine._lock."""
        try:
            delay = float(self._precompute_delay_s)
            if delay > 0.0 and self._precompute_cancel.wait(delay):
                return
            if self._cursor_version != version or self._precompute_cancel.is_set():
                return

            t0 = time.perf_counter()
            n_samples = 8
            lookahead_tokens = 10
            seed = 0

            # Chunk 1: verify state; if ForkHead fast-path is available, finish immediately under one short lock!
            with self.engine._lock:
                if (
                    self._cursor_version != version
                    or self._precompute_cancel.is_set()
                    or self.engine.hook is None
                    or self.engine.model is None
                    or len(self.committed_ids) != cursor_len
                ):
                    return
                if self._is_compass_ready():
                    return
                tok, model = self.engine.tok, self.engine.model
                prefix = list(self.prompt_ids + self.committed_ids)
                b = len(prefix) - 1
                layer = self.engine.layer
                frontier = self.last_frontier
                fh_dirs = self.last_fh.get("dirs") if self.last_fh is not None else None
                if (
                    fh_dirs is not None
                    and fh_dirs.shape[0] >= 2
                    and frontier is not None
                    and len(frontier.get("branches", [])) >= 2
                ):
                    self._build_fork_head_compass(model, frontier, fh_dirs, prefix, b, layer, seed, t0)
                    return
                conts = self._sample_doc_continuations(
                    prefix,
                    n_samples=n_samples,
                    lookahead_tokens=lookahead_tokens,
                    seed=seed,
                )
                cont_texts = [_strip_doc_stop(tok.decode(c)).strip() for c in conts]

            # Chunk 2: per-mini-batch gradients, releasing engine._lock between mini-batches
            grad_parts: list[np.ndarray] = []
            mini = 2
            for start_i in range(0, len(conts), mini):
                if self._cursor_version != version or self._precompute_cancel.is_set():
                    return
                part_conts = conts[start_i : start_i + mini]
                with self.engine._lock:
                    if (
                        self._cursor_version != version
                        or self._precompute_cancel.is_set()
                        or self.engine.hook is None
                        or self.engine.model is None
                    ):
                        return
                    with self._pause_fork_scorer():
                        self._clear_steering()
                        g_part = _grads_at_oom_safe(self.engine.model, prefix, part_conts, layer, b, batch=mini)
                    grad_parts.append(g_part)

            if self._cursor_version != version or self._precompute_cancel.is_set():
                return

            # Chunk 3: pure-CPU clustering & 2D plane (NO engine._lock held)
            G = np.concatenate(grad_parts, axis=0).astype(np.float64)
            axes, anchors, _, _, _, var_expl, reading_meta, dots = self._cluster_gradients_to_plane(G, cont_texts, seed)
            far = int(np.argmax(np.linalg.norm(anchors, axis=1)))
            v_far = anchors[far] @ axes
            v_far_norm = max(float(np.linalg.norm(v_far)), 1e-12)
            v_unit = v_far / v_far_norm
            a0 = alpha_for(G, v_unit, 0.25)
            if not math.isfinite(a0) or a0 <= 0.0 or a0 > 1e4:
                a0 = 1.0
            ref_cont = next((ct for ct in conts if ct), [])

            if self._cursor_version != version or self._precompute_cancel.is_set():
                return

            # Chunk 4: calibration & atomic commit under engine._lock
            with self.engine._lock:
                if (
                    self._cursor_version != version
                    or self._precompute_cancel.is_set()
                    or self.engine.hook is None
                    or self.engine.model is None
                    or len(self.committed_ids) != cursor_len
                ):
                    return
                with self._pause_fork_scorer():
                    a_far = (
                        calibrate(
                            self.engine.model,
                            self.engine.hook,
                            prefix + ref_cont,
                            b,
                            v_unit,
                            target=0.25,
                            mode="cursor",
                            a0=a0,
                            iters=3,
                        )
                        if ref_cont
                        else a0
                    )
                if self._cursor_version != version or self._precompute_cancel.is_set():
                    return
                if not math.isfinite(a_far) or a_far <= 0.0:
                    a_far = a0
                alpha_unit = float(a_far / v_far_norm)
                probe_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                resp = {
                    "source": "gradients",
                    "cursor_len": cursor_len,
                    "variance_explained": round(var_expl, 4),
                    "alpha_unit": round(float(alpha_unit), 6),
                    "probe_ms": probe_ms,
                    "readings": reading_meta,
                    "dots": dots,
                }
                self.active_compass = {
                    "cursor_len": cursor_len,
                    "cursor_version": version,
                    "seed": int(seed),
                    "b": b,
                    "layer": layer,
                    "axes": axes,
                    "anchors": anchors,
                    "alpha_unit": alpha_unit,
                    "readings": reading_meta,
                    "source": "gradients",
                    "response": copy.deepcopy(resp),
                }
        except Exception:
            # Never leak background thread tracebacks on shutdown or race cancellation
            return

    def steer_compass(self, x: float, y: float, max_new: int = 16) -> dict[str, Any]:
        """Generate a live steered preview at `(x, y)` on the active cursor's 2D compass."""
        if not self._is_compass_ready():
            self.probe_compass()
        tok, model = self.engine.ensure_loaded()
        comp = self.active_compass
        assert comp is not None
        p = np.array([float(x), float(y)], dtype=np.float64)
        local_vec = torch.tensor(comp["alpha_unit"] * (p @ comp["axes"]), dtype=torch.float32)

        prefix = self.prompt_ids + self.committed_ids
        b = len(prefix) - 1
        # Unified steering: global_hook & ReFT apply on [start-1 .. end], local hook applies on [b .. end].
        # This guarantees exact parity between the cropped-cache path and the full-prefill fallback path!
        self._apply_global_steering(
            start_pos=self.start - 1,
            local_vec=local_vec,
            local_b=b,
            local_only_at_b=False,
        )

        dev = self.engine._dev()
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        gen_ids: list[int] = []
        try:
            with self._pause_fork_scorer(), torch.no_grad():
                pkv_s = None
                if b >= 1 and _cache_can_crop(self.pkv, b):
                    pkv_try = _cache_clone_batch(self.pkv, 1)
                    if pkv_try is not None:
                        pkv_try, ok = _cache_crop(pkv_try, b)
                        if ok:
                            pkv_s = pkv_try
                if pkv_s is not None:
                    x_in = torch.tensor([[int(prefix[-1])]], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(b)
                    out = arch.forward_hidden(model, input_ids=x_in, past_key_values=pkv_s, use_cache=True)
                    pkv_s = out.past_key_values
                    cur_pos = b + 1
                else:
                    x_in = torch.tensor([prefix], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(0)
                    out = arch.forward_hidden(model, input_ids=x_in, use_cache=True)
                    pkv_s = out.past_key_values
                    cur_pos = len(prefix)

                nxt = int(arch.logits(model, out.last_hidden_state[0, -1][None])[0].argmax().item())
                for step_i in range(max(2, int(max_new))):
                    if nxt in eos_ids:
                        break
                    gen_ids.append(nxt)
                    if _has_doc_stop(tok.decode(gen_ids)):
                        break
                    if step_i == int(max_new) - 1:
                        break
                    x_step = torch.tensor([[nxt]], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(cur_pos)
                    out = arch.forward_hidden(model, input_ids=x_step, past_key_values=pkv_s, use_cache=True)
                    pkv_s = out.past_key_values
                    cur_pos += 1
                    nxt = int(arch.logits(model, out.last_hidden_state[0, -1][None])[0].argmax().item())
        finally:
            self._clear_steering()

        clean_ids = cut_doc(tok, gen_ids)
        preview_text = _strip_doc_stop(tok.decode(clean_ids))
        fw_ids = _extract_first_word_ids(tok, clean_ids)

        # Soft Gaussian weights over compass reading anchors (log-sum-exp stable for extreme coords)
        anchors = comp["anchors"]
        d2 = np.sum((anchors - p[None, :]) ** 2, axis=1)
        logits_w = -d2 / (2.0 * 0.55 ** 2)
        logits_w = logits_w - float(np.max(logits_w))
        w = np.exp(logits_w)
        w = w / max(float(w.sum()), 1e-12)
        shares = {str(rd["id"]): round(float(wi), 4) for rd, wi in zip(comp["readings"], w)}

        return {
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "preview_text": preview_text,
            "preview_ids": clean_ids,
            "first_word_text": tok.decode(fw_ids),
            "first_word_ids": fw_ids,
            "shares": shares,
        }

    def constellation(
        self,
        branch_indices: list[int],
        action: str = "chord",
    ) -> dict[str, Any]:
        """Commit a multi-word prismatic chord or blended phrase from 2+ competing branch poles."""
        frontier = self.last_frontier or self.inspect_frontier(_schedule_precompute=False)
        branches = frontier.get("branches") or []
        picked: list[dict[str, Any]] = []
        seen_idx: set[int] = set()
        for raw_i in branch_indices:
            i = int(raw_i)
            if i < 0 or i >= len(branches):
                raise IndexError(f"branch_index {i} out of range (0..{len(branches) - 1})")
            if i not in seen_idx:
                seen_idx.add(i)
                picked.append(branches[i])
        if not picked:
            raise ValueError("Select at least one branch to form a constellation.")

        words = [str(b.get("first_word_text", "")).strip() for b in picked]
        words = [w for w in words if w]
        if not words:
            raise ValueError("Selected branches have no words.")

        tok, _ = self.engine.ensure_loaded()
        cur_text = _strip_doc_stop(tok.decode(self.committed_ids))
        need_space = bool(cur_text) and not cur_text[-1].isspace()

        if action == "chord":
            body = f"[{' / '.join(words)}]"
        elif action == "blend":
            if len(words) == 1:
                body = words[0]
            elif len(words) == 2:
                body = f"{words[0]} ({words[1]})"
            else:
                body = f"{words[0]}—{', '.join(words[1:])}"
        else:
            raise ValueError(f"Invalid constellation action: {action!r} (expected 'chord' or 'blend')")

        commit_str = (" " if need_space else "") + body
        return self.step(custom_text=commit_str, source_tag="constellation")


# ------------------------------------------------------------------ Session Manager
class CoTranslationManager:
    """Thread-safe registry of active CoTranslationSessions backed by a LiveTranslationEngine."""

    def __init__(self, engine: Any, max_sessions: int = 8):
        self.engine = engine
        self.max_sessions = max(1, int(max_sessions))
        self._sessions: dict[str, CoTranslationSession] = {}

    def _get(self, session_id: str) -> CoTranslationSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id!r}")
        return self._sessions[session_id]

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def _cancel_session_precompute(self, session_id: str, wait: bool = False) -> None:
        sess = self._sessions.get(session_id)
        if sess is not None:
            sess.cancel_precompute(wait=wait)

    def cancel_all_precomputes(self, wait: bool = False) -> None:
        """Cancel all active background compass precomputation threads across sessions."""
        for sess in list(self._sessions.values()):
            sess.cancel_precompute(wait=wait)

    def start(
        self,
        source_text: str,
        src_lang: str = "Spanish",
        tgt_lang: str = "English",
        global_coords: dict[str, float] | None = None,
        lookahead_tokens: int = 9,
        max_branches: int = 6,
    ) -> dict[str, Any]:
        with self.engine._lock:
            # Evict oldest session to keep VRAM clean on 12B/27B GPUs
            while len(self._sessions) >= self.max_sessions:
                oldest = next(iter(self._sessions))
                evicted = self._sessions.pop(oldest, None)
                if evicted is not None:
                    evicted.cancel_precompute(wait=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            sess = CoTranslationSession(
                engine=self.engine,
                source_text=source_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                global_coords=global_coords,
                lookahead_tokens=lookahead_tokens,
                max_branches=max_branches,
            )
            self._sessions[sess.session_id] = sess
            return sess.state_dict(include_frontier=True)

    def get_state(self, session_id: str, include_frontier: bool = False) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).state_dict(include_frontier=include_frontier)

    def delete(self, session_id: str) -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=True)
        with self.engine._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id!r}")
            self._sessions.pop(session_id, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"deleted": True, "session_id": session_id}

    def step(self, session_id: str, **kw: Any) -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            return self._get(session_id).step(**kw)

    def redo(self, session_id: str, mode: str = "word") -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            return self._get(session_id).redo(mode=mode)

    def glide(self, session_id: str, max_words: int = 32, stop_on_fork: bool = True) -> Generator[dict[str, Any], None, None]:
        """Stepwise glide generator that acquires and releases `engine._lock` per step."""
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            _ = self._get(session_id)
        steps_taken = 0
        limit = max(1, int(max_words))
        while True:
            with self.engine._lock:
                if session_id not in self._sessions:
                    return
                sess = self._sessions[session_id]
                events, done = sess._glide_next(steps_taken, limit, stop_on_fork=stop_on_fork)
                for ev in events:
                    if ev.get("event") == "step":
                        steps_taken = int(ev["steps"])
            # Engine lock is RELEASED while yielding / flushing SSE bytes!
            for ev in events:
                yield ev
            if done:
                return

    def rewind(self, session_id: str, to_token_index: int) -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            return self._get(session_id).rewind(to_token_index)

    def checkout(self, session_id: str, node_id: str) -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            return self._get(session_id).checkout_node(node_id)

    def set_sliders(self, session_id: str, global_coords: dict[str, float]) -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            return self._get(session_id).set_global_coords(global_coords)

    def probe_compass(
        self,
        session_id: str,
        n_samples: int = 10,
        lookahead_tokens: int = 12,
        seed: int = 0,
        force: bool = False,
        force_gradient: bool = False,
    ) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).probe_compass(
                n_samples=n_samples,
                lookahead_tokens=lookahead_tokens,
                seed=seed,
                force=force,
                force_gradient=force_gradient,
            )

    def steer_compass(self, session_id: str, x: float, y: float, max_new: int = 16) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).steer_compass(x=x, y=y, max_new=max_new)

    def source_recipe(self, session_id: str, word: str | None = None) -> dict[str, Any]:
        with self.engine._lock:
            rec = self._get(session_id).get_source_recipe(word=word)
            return rec if rec is not None else {"word": word or "", "ingredients": [], "leftover": 1.0, "r2": 0.0, "unavailable": True}

    def constellation(self, session_id: str, branch_indices: list[int], action: str = "chord") -> dict[str, Any]:
        self._cancel_session_precompute(session_id, wait=False)
        with self.engine._lock:
            return self._get(session_id).constellation(branch_indices=branch_indices, action=action)

