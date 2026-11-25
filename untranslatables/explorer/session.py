"""Stateful KV-cached Live Co-Translation Engine (v2).

Designed for interactive token/word-by-word translation of long poems, multi-stanza
verse, and formatted paragraphs on quantized Gemma 3 12B / 27B (and smaller models).

Key capabilities:
  - Persistent mutable KV cache per session (prefill once; ~15 ms forward step;
    instant rewind via cache cropping with automatic fallback for sliding-window caches).
  - Tier-1 instant frontier on every step (zero backward passes):
      * Adaptive regime detection (`linear` | `subtle` | `fork`) combining Shannon
        entropy, post-dedup branch spread, and optional amortised `ForkHead`.
      * Surface-duplicate collapsing + 1-step function-word peek-ahead so forks
        hiding behind `"a"`/`"the"`/`"of"` surface immediately.
      * Batched KV-cached lookahead across up to 6 distinct starters, semantically
        deduplicated into genuinely distinct continuation ribbons.
  - Auto-glide SSE generator: streams whole words quickly through linear passages
    and automatically pauses the moment a complex interpretive fork opens.
  - Tier-2 on-demand 2D Spectral Compass at the active cursor (using amortised
    `ForkHead` directions when available, or OOM-safe gradient-space probing).
  - Lightweight explored-path history tree so rewinding and taking another branch
    never loses earlier drafts.
"""
from __future__ import annotations

import copy
import math
import re
import time
import uuid
from typing import Any, Generator

import numpy as np
import torch

from blind.fisher import seq_logp
from explorer import readings as R
from explorer.directions import alpha_for, calibrate
from explorer.frame import DOC_STOPS, build_doc_prompt, cut_doc
from explorer.live_engine import READING_PALETTE, _STOPWORDS
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


def _cache_crop(pkv: Any, length: int) -> tuple[Any, bool]:
    """Attempt in-place crop of KV cache to `length`. Returns `(pkv, ok)`."""
    if pkv is None:
        return None, False
    cur = _cache_seq_len(pkv)
    if length > cur:
        return pkv, False
    if length == cur:
        return pkv, True
    if hasattr(pkv, "crop"):
        try:
            pkv.crop(int(length))
            if _cache_seq_len(pkv) == int(length):
                return pkv, True
            return pkv, False
        except Exception:
            return pkv, False
    if isinstance(pkv, tuple):
        try:
            cropped = tuple(
                (layer[0][:, :, :length, :].contiguous(), layer[1][:, :, :length, :].contiguous())
                for layer in pkv
            )
            return cropped, True
        except Exception:
            return pkv, False
    return pkv, False


def _cache_clone_batch(pkv: Any, batch_size: int) -> Any:
    """Return an independent KV cache expanded to `batch_size` without mutating `pkv`.

    Returns None if cloning is unsupported for this cache type (caller falls back to prefill).
    """
    if pkv is None or batch_size <= 0:
        return None
    try:
        if isinstance(pkv, tuple):
            return tuple(
                (
                    layer[0].repeat(batch_size, 1, 1, 1).contiguous(),
                    layer[1].repeat(batch_size, 1, 1, 1).contiguous(),
                )
                for layer in pkv
            )
        cloned = copy.deepcopy(pkv)
        if batch_size == 1:
            return cloned
        if hasattr(cloned, "batch_repeat_interleave"):
            cloned.batch_repeat_interleave(int(batch_size))
            return cloned
        if hasattr(cloned, "key_cache") and hasattr(cloned, "value_cache"):
            cloned.key_cache = [k.repeat(batch_size, 1, 1, 1).contiguous() for k in cloned.key_cache]
            cloned.value_cache = [v.repeat(batch_size, 1, 1, 1).contiguous() for v in cloned.value_cache]
            return cloned
        if hasattr(cloned, "layers"):
            for layer in cloned.layers:
                if getattr(layer, "keys", None) is not None:
                    layer.keys = layer.keys.repeat(batch_size, 1, 1, 1).contiguous()
                if getattr(layer, "values", None) is not None:
                    layer.values = layer.values.repeat(batch_size, 1, 1, 1).contiguous()
            return cloned
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


def _content_words(text: str) -> list[str]:
    words = [w.lower().strip("'") for w in _WORD_RE.findall(text)]
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def _ribbon_similarity(a: str, b: str) -> float:
    """Fast lexical/semantic similarity for collapsing near-duplicate continuation ribbons."""
    na = re.sub(r"\s+", " ", a.strip().lower())
    nb = re.sub(r"\s+", " ", b.strip().lower())
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    if na == nb:
        return 1.0
    ca = _content_words(na)
    cb = _content_words(nb)
    if not ca or not cb:
        # Compare raw words if purely function words/punctuation
        wa = set(_WORD_RE.findall(na))
        wb = set(_WORD_RE.findall(nb))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)
    # Identical opening content words strongly signal same reading
    if len(ca) >= 2 and len(cb) >= 2 and ca[:2] == cb[:2]:
        return 0.85
    if ca[0] == cb[0] and (len(ca) == 1 or len(cb) == 1):
        return 0.85
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


def _has_doc_stop(text: str) -> bool:
    return any(st in text for st in DOC_STOPS)


def _strip_doc_stop(text: str) -> str:
    best = len(text)
    for st in DOC_STOPS:
        idx = text.find(st)
        if idx != -1 and idx < best:
            best = idx
    return text[:best]


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

        tok, _ = engine.ensure_loaded()
        self.prompt_ids, self.start = build_doc_prompt(tok, self.source_text, src_lang=src_lang, tgt_lang=tgt_lang)
        n_src_toks = len(tok(self.source_text, add_special_tokens=False)["input_ids"])
        self.max_total_tokens = max(256, 4 * n_src_toks + 128)

        self.committed_ids: list[int] = []
        self.committed_meta: list[dict[str, Any]] = []
        self.pkv: Any = None
        self.last_logits: torch.Tensor | None = None
        self.last_fh: dict[str, Any] | None = None
        self.last_frontier: dict[str, Any] | None = None
        self.active_compass: dict[str, Any] | None = None

        # Lightweight history tree of explored paths
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

        # Initial prefill of the source prompt
        self._prefill_full()

    # ---------------------------------------------------------- Steering hook management
    def _apply_global_steering(self, start_pos: int | None = None) -> None:
        gvec = self.engine._global_vector(self.global_coords)
        b_global = (self.start - 1) if start_pos is None else int(start_pos)
        if gvec is not None:
            self.engine.hook.set(torch.tensor(gvec, dtype=torch.float32), b=b_global, only_at_b=False)
        else:
            self.engine.hook.set(None)
        if self.engine.reft_controller is not None:
            rc = self.engine._resolve_coords(self.global_coords, set(self.engine.reft_controller.axis_names))
            self.engine.reft_controller.set_coords(rc or None, start_pos=b_global)

    def _sync_hooks_pos(self, pos: int) -> None:
        self.engine.hook.sync_pos(pos)
        if self.engine.reft_controller is not None:
            self.engine.reft_controller.sync_pos(pos)

    def _clear_steering(self) -> None:
        self.engine.hook.set(None)
        if self.engine.reft_controller is not None:
            self.engine.reft_controller.set_coords(None)

    def _prefill_full(self) -> None:
        """Full prefill of `prompt_ids + committed_ids` with current global steering."""
        _, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        seq = self.prompt_ids + self.committed_ids
        x = torch.tensor([seq], dtype=torch.long, device=dev)

        self._apply_global_steering()
        self._sync_hooks_pos(0)
        if self.engine.fork_scorer is not None:
            self.engine.fork_scorer.reset()
        try:
            with torch.no_grad():
                out = arch.forward_hidden(model, input_ids=x, use_cache=True)
                self.pkv = out.past_key_values
                h_last = out.last_hidden_state[0, -1]
                self.last_logits = arch.logits(model, h_last[None])[0].detach().cpu()
                self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
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
                self.last_logits = arch.logits(model, h_last[None])[0].detach().cpu()
                self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
        except Exception:
            # Safe fallback if sliding-window cache or hook encountered an edge case
            self._prefill_full()
        finally:
            self._clear_steering()

    # ---------------------------------------------------------- Batched KV-cached lookahead
    def _batched_lookahead(
        self,
        starters: list[list[int]],
        lookahead_steps: int,
        extra_vec: torch.Tensor | None = None,
        extra_b: int | None = None,
    ) -> list[list[int]]:
        """Run parallel greedy rollouts from `starters` sharing a cloned KV cache of the current prefix."""
        if not starters:
            return []
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        B = len(starters)
        prefix_len = len(self.prompt_ids) + len(self.committed_ids)

        # Group by starter length (usually all len 1, or some len 2 from function-word peek)
        max_st = max(len(s) for s in starters)
        if max_st == 1:
            return self._batched_lookahead_uniform(starters, lookahead_steps, extra_vec=extra_vec, extra_b=extra_b)

        # Split into len-1 and len-2+ groups so we keep unpadded batched execution clean
        by_len: dict[int, list[tuple[int, list[int]]]] = {}
        for idx, s in enumerate(starters):
            by_len.setdefault(len(s), []).append((idx, s))
        results: list[list[int]] = [[] for _ in range(B)]
        for _, group in by_len.items():
            sub_starters = [s for _, s in group]
            sub_out = self._batched_lookahead_uniform(
                sub_starters,
                max(2, lookahead_steps - (len(sub_starters[0]) - 1)),
                extra_vec=extra_vec,
                extra_b=extra_b,
            )
            for (orig_idx, _), seq in zip(group, sub_out):
                results[orig_idx] = seq
        return results

    def _batched_lookahead_uniform(
        self,
        starters: list[list[int]],
        lookahead_steps: int,
        extra_vec: torch.Tensor | None = None,
        extra_b: int | None = None,
    ) -> list[list[int]]:
        tok, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        B = len(starters)
        st_len = len(starters[0])
        prefix = self.prompt_ids + self.committed_ids
        prefix_len = len(prefix)

        # Set up steering (global + optional local compass vector)
        gvec = self.engine._global_vector(self.global_coords)
        combined: torch.Tensor | None = None
        if gvec is not None:
            combined = torch.tensor(gvec, dtype=torch.float32)
        if extra_vec is not None:
            ev = extra_vec.detach().float().cpu()
            combined = ev if combined is None else (combined + ev)
        steer_b = extra_b if extra_b is not None else (self.start - 1)

        if combined is not None:
            self.engine.hook.set(combined, b=steer_b, only_at_b=False)
        else:
            self.engine.hook.set(None)
        if self.engine.reft_controller is not None:
            rc = self.engine._resolve_coords(self.global_coords, set(self.engine.reft_controller.axis_names))
            self.engine.reft_controller.set_coords(rc or None, start_pos=self.start - 1)

        # Avoid mutating the session's main KV cache
        # If extra_vec acts at `extra_b == prefix_len - 1`, we only need to recompute from the last prefix token
        # when cloning; otherwise we clone `self.pkv` directly.
        use_cloned_cache = (extra_vec is None)
        pkv_batch = _cache_clone_batch(self.pkv, B) if use_cloned_cache else None

        seqs = [list(s) for s in starters]
        alive = [True] * B
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

        try:
            with torch.no_grad():
                if pkv_batch is not None:
                    x = torch.tensor(starters, dtype=torch.long, device=dev)
                    self._sync_hooks_pos(prefix_len)
                    out = arch.forward_hidden(model, input_ids=x, past_key_values=pkv_batch, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos = prefix_len + st_len
                else:
                    full_in = [prefix + s for s in starters]
                    x = torch.tensor(full_in, dtype=torch.long, device=dev)
                    self._sync_hooks_pos(0)
                    out = arch.forward_hidden(model, input_ids=x, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos = prefix_len + st_len

                h_last = out.last_hidden_state[:, -1, :]
                lg = arch.logits(model, h_last)
                nxt = lg.argmax(dim=-1).tolist()

                for step_i in range(int(lookahead_steps)):
                    for i in range(B):
                        if not alive[i]:
                            continue
                        tid = int(nxt[i])
                        if tid in eos_ids:
                            alive[i] = False
                            continue
                        seqs[i].append(tid)
                        if _has_doc_stop(tok.decode(seqs[i])):
                            alive[i] = False
                    if not any(alive) or step_i == int(lookahead_steps) - 1:
                        break
                    x_step = torch.tensor([[int(nxt[i])] for i in range(B)], dtype=torch.long, device=dev)
                    self._sync_hooks_pos(cur_pos)
                    out = arch.forward_hidden(model, input_ids=x_step, past_key_values=pkv_batch, use_cache=True)
                    pkv_batch = out.past_key_values
                    cur_pos += 1
                    lg = arch.logits(model, out.last_hidden_state[:, -1, :])
                    nxt = lg.argmax(dim=-1).tolist()
        finally:
            self._clear_steering()

        return [cut_doc(tok, s) for s in seqs]

    # ---------------------------------------------------------- Tier-1 Instant Frontier
    def inspect_frontier(self) -> dict[str, Any]:
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
        )
        top_vals, top_ids = torch.topk(p, k=min(28, int(p.numel())))
        top_pairs = [(int(tid), float(val)) for tid, val in zip(top_ids.tolist(), top_vals.tolist())]

        # Detect if top token is EOS or closing tag `</translation>`
        top1_id, top1_prob = top_pairs[0]
        top1_str = tok.decode([top1_id])
        if top1_id in eos_ids and top1_prob >= 0.45:
            at_end = True
        if top1_str.lstrip().startswith("</") and top1_prob >= 0.45:
            at_end = True

        fh_prob = float(self.last_fh["fork_prob"]) if self.last_fh is not None else None
        fh_ent = float(self.last_fh["pred_entropy"]) if self.last_fh is not None else None
        fh_fis = float(self.last_fh["pred_fisher_top"]) if self.last_fh is not None else None

        if at_end:
            frontier = {
                "at_end": True,
                "regime": "linear",
                "fork_score": 0.0,
                "entropy": round(entropy, 4),
                "top1_prob": round(top1_prob, 4),
                "fork_head_prob": round(fh_prob, 4) if fh_prob is not None else None,
                "fork_head_entropy": round(fh_ent, 4) if fh_ent is not None else None,
                "fork_head_fisher": round(fh_fis, 4) if fh_fis is not None else None,
                "ghost": {"text": "", "next_word_text": "", "next_word_ids": [], "full_ids": []},
                "branches": [],
            }
            self.last_frontier = frontier
            return frontier

        # 1. Surface-deduplicated next-token starters (collapse case / leading space duplicates)
        starters_map: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []
        for tid, prob in top_pairs:
            if tid in eos_ids:
                continue
            if prob < 0.012 and len(ordered_keys) >= 2:
                continue
            s = tok.decode([tid])
            if s.lstrip().startswith("</") or "<source" in s:
                continue
            # Keep newline/stanza breaks distinct from normal words
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
            starters_list = [{"ids": [top1_id], "prob": top1_prob, "raw_text": top1_str}]
        else:
            starters_list = sorted(starters_map.values(), key=lambda x: -x["prob"])

        # 2. Function-word 1-step peek-ahead: if the dominant starter is a short English function
        #    word (like "a", "the", "of") that may mask an immediate fork right behind it, peek 1 step!
        lead = starters_list[0]
        lead_norm = lead["raw_text"].strip().lower()
        peeked_entropy = entropy
        if (
            lead_norm in _PEEK_FUNCTION_WORDS
            and lead["prob"] >= 0.42
            and len(lead["ids"]) == 1
        ):
            peek_starters = self._peek_through_function_word(lead["ids"][0], lead["prob"])
            if peek_starters is not None:
                p_ent, expanded = peek_starters
                peeked_entropy = max(entropy, p_ent)
                if len(expanded) >= 2:
                    # Replace lead with the 2-token semantic starters + keep strong direct alternatives
                    direct_alts = [st for st in starters_list[1:] if st["prob"] >= 0.06]
                    starters_list = sorted(expanded + direct_alts, key=lambda x: -x["prob"])

        # Cap candidates before batched lookahead
        is_trivially_linear = (
            len(starters_list) == 1
            or (
                starters_list[0]["prob"] >= 0.84
                and peeked_entropy < 0.65
                and (fh_prob is None or fh_prob < 0.45)
            )
        )
        n_roll = 1 if is_trivially_linear else min(self.max_branches, len(starters_list))
        selected = starters_list[:n_roll]

        # 3. Batched KV-cached greedy lookahead across selected starters
        ribbons = self._batched_lookahead([s["ids"] for s in selected], lookahead_steps=self.lookahead_tokens)

        # 4. Semantic deduplication of lookahead ribbons (pool probability of near-identical meanings)
        merged: list[dict[str, Any]] = []
        for st, r_ids in zip(selected, ribbons):
            if not r_ids:
                continue
            r_text = _strip_doc_stop(tok.decode(r_ids))
            if not r_text:
                continue
            placed = False
            for m in merged:
                if _ribbon_similarity(r_text, m["preview_text"]) >= 0.72:
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
                })

        merged.sort(key=lambda x: -x["prob"])
        total_mass = sum(m["prob"] for m in merged) or 1.0

        # Normalise display shares while preserving raw_prob
        all_texts = [m["preview_text"] for m in merged]
        branches: list[dict[str, Any]] = []
        for idx, m in enumerate(merged[: self.max_branches]):
            others = [t for j, t in enumerate(all_texts) if j != idx]
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
            })

        # 5. Classify uncertainty regime honestly (linear vs subtle vs fork)
        eff_ent = max(entropy, peeked_entropy)
        top_share = branches[0]["share"] if branches else 1.0
        second_prob = branches[1]["prob"] if len(branches) >= 2 else 0.0
        n_distinct = len(branches)

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

        # Ghost continuation = top branch
        if branches:
            b0 = branches[0]
            ghost = {
                "text": b0["preview_text"],
                "next_word_text": b0["first_word_text"],
                "next_word_ids": b0["first_word_ids"],
                "full_ids": b0["preview_ids"],
            }
        else:
            ghost = {
                "text": top1_str,
                "next_word_text": top1_str,
                "next_word_ids": [top1_id],
                "full_ids": [top1_id],
            }

        frontier = {
            "at_end": False,
            "regime": regime,
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
        return frontier

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
            with torch.no_grad():
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
            if s.lstrip().startswith("</"):
                continue
            key = s.strip().lower() or f"__W_{tid}"
            joint_p = float(fn_prob * v)
            if key in seen:
                seen[key]["prob"] += joint_p
            else:
                seen[key] = {"ids": [int(fn_id), int(tid)], "prob": joint_p, "raw_text": s}
        expanded = sorted(seen.values(), key=lambda x: -x["prob"])[:4]
        return ent2, expanded

    # ---------------------------------------------------------- State serialization
    def _record_active_node(self) -> None:
        tok, _ = self.engine.ensure_loaded()
        node = self.history[self.active_node_id]
        node["token_ids"] = list(self.committed_ids)
        node["meta"] = [dict(m) for m in self.committed_meta]
        txt = _strip_doc_stop(tok.decode(self.committed_ids))
        node["text"] = txt
        tail = txt.strip().splitlines()[-1] if txt.strip() else ""
        node["snippet"] = tail[-44:] if len(tail) > 44 else tail

    def _split_history_node(self, fork_at: int, label_hint: str) -> None:
        """Create a sibling history node when diverging after a rewind."""
        nid = f"n{self._node_seq}"
        self._node_seq += 1
        self.history[nid] = {
            "id": nid,
            "parent_id": self.active_node_id,
            "fork_token_idx": int(fork_at),
            "token_ids": list(self.committed_ids),
            "meta": [dict(m) for m in self.committed_meta],
            "label": label_hint.strip()[:28] or f"Branch {self._node_seq - 1}",
            "created_at": time.time(),
        }
        self.active_node_id = nid
        self._record_active_node()

    def state_dict(self, include_frontier: bool = True) -> dict[str, Any]:
        tok, _ = self.engine.ensure_loaded()
        frontier = self.inspect_frontier() if include_frontier else self.last_frontier
        text = _strip_doc_stop(tok.decode(self.committed_ids))

        # Group committed tokens into clickable word spans for a clean UI
        tokens_out = [
            {
                "idx": i,
                "token_id": int(tid),
                "token": tok.decode([int(tid)]),
                **self.committed_meta[i],
            }
            for i, tid in enumerate(self.committed_ids)
        ]

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
            "has_compass": self.active_compass is not None and self.active_compass.get("cursor_len") == len(self.committed_ids),
            "history": history_summary,
            "active_node_id": self.active_node_id,
        }

    # ---------------------------------------------------------- Step / Glide / Rewind
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

        # If we rewound earlier and are now diverging from the active node's stored suffix, fork a sibling node!
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

        # Split history node if we rewound and picked different tokens than before
        if diverging:
            old_next = cur_node_ids[len(self.committed_ids) : len(self.committed_ids) + len(chosen_ids)]
            if old_next != chosen_ids:
                self._split_history_node(len(self.committed_ids), tok.decode(chosen_ids))

        ent = float(frontier.get("entropy", 0.0))
        fscore = float(frontier.get("fork_score", 0.0))
        is_fk = frontier.get("regime") == "fork"

        for idx, tid in enumerate(chosen_ids):
            self.committed_ids.append(int(tid))
            self.committed_meta.append({
                "entropy": round(ent, 3) if idx == 0 else 0.0,
                "fork_score": round(fscore, 3) if idx == 0 else 0.0,
                "is_fork": bool(is_fk) if idx == 0 else False,
                "source": tag,
            })

        # Local compass belonged to the previous cursor
        self.active_compass = None
        self._advance_cache(chosen_ids)
        self._record_active_node()
        return self.state_dict(include_frontier=True)

    def glide(
        self,
        max_words: int = 32,
        stop_on_fork: bool = True,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream words off the persistent KV cache, auto-pausing when a complex fork opens."""
        steps_taken = 0
        for _ in range(max(1, int(max_words))):
            frontier = self.last_frontier or self.inspect_frontier()
            if frontier.get("at_end"):
                yield {"event": "paused", "reason": "end", "steps": steps_taken, "state": self.state_dict(include_frontier=False)}
                return
            if stop_on_fork and steps_taken > 0 and frontier.get("regime") == "fork":
                yield {"event": "paused", "reason": "fork", "steps": steps_taken, "state": self.state_dict(include_frontier=False)}
                return

            # Fast step using top ghost word
            st = self.step(mode="word")
            steps_taken += 1
            next_f = st["frontier"]
            yield {
                "event": "step",
                "steps": steps_taken,
                "committed_text": st["committed_text"],
                "n_tokens": st["n_tokens"],
                "regime": next_f["regime"],
                "fork_score": next_f["fork_score"],
                "state": st,
            }
            if next_f.get("at_end"):
                yield {"event": "paused", "reason": "end", "steps": steps_taken, "state": st}
                return
            if stop_on_fork and next_f.get("regime") == "fork":
                yield {"event": "paused", "reason": "fork", "steps": steps_taken, "state": st}
                return

        yield {"event": "paused", "reason": "limit", "steps": steps_taken, "state": self.state_dict(include_frontier=False)}

    def rewind(self, to_token_index: int) -> dict[str, Any]:
        """Rewind the cursor to `to_token_index`, preserving the explored path in history."""
        target = max(0, min(int(to_token_index), len(self.committed_ids)))
        if target == len(self.committed_ids):
            return self.state_dict(include_frontier=True)

        # Ensure current path is saved cleanly before rewinding
        self._record_active_node()

        self.committed_ids = self.committed_ids[:target]
        self.committed_meta = self.committed_meta[:target]
        self.active_compass = None

        # Crop KV cache to `len(prompt_ids) + target` and recompute last_logits cleanly
        cache_target = len(self.prompt_ids) + target
        # To get exact `last_logits` at position `cache_target - 1`, crop to `cache_target - 1`
        # and run a single 1-token cached forward pass on the last token!
        if cache_target >= 1:
            cropped, ok = _cache_crop(self.pkv, cache_target - 1)
            if ok:
                self.pkv = cropped
                last_tok = self.committed_ids[-1] if target > 0 else self.prompt_ids[-1]
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
                        self.last_logits = arch.logits(model, out.last_hidden_state[0, -1][None])[0].detach().cpu()
                        self.last_fh = self.engine.fork_scorer.latest() if self.engine.fork_scorer is not None else None
                except Exception:
                    self._prefill_full()
                finally:
                    self._clear_steering()
            else:
                self._prefill_full()
        else:
            self._prefill_full()

        return self.state_dict(include_frontier=True)

    def checkout_node(self, node_id: str) -> dict[str, Any]:
        """Switch to any previously explored branch node in `self.history`."""
        if node_id not in self.history:
            raise KeyError(f"Unknown history node: {node_id!r}")
        self._record_active_node()
        node = self.history[node_id]
        self.active_node_id = node_id
        self.committed_ids = list(node["token_ids"])
        self.committed_meta = [dict(m) for m in node["meta"]]
        self.active_compass = None
        self._prefill_full()
        return self.state_dict(include_frontier=True)

    def set_global_coords(self, global_coords: dict[str, float]) -> dict[str, Any]:
        """Update global practice/tone sliders and refresh the active frontier."""
        self.global_coords = {str(k): float(v) for k, v in (global_coords or {}).items()}
        self.active_compass = None
        self._prefill_full()
        return self.state_dict(include_frontier=True)

    # ---------------------------------------------------------- Tier-2 On-Demand Spectral Compass
    def probe_compass(self, n_samples: int = 10, lookahead_tokens: int = 12, seed: int = 0) -> dict[str, Any]:
        """Build a 2D spectral steering plane right at the current cursor.

        Uses amortised `ForkHead` local subspace directions when trained with `n_dirs >= 2`,
        otherwise runs a compact OOM-safe burst + gradient PCA at the active cursor.
        """
        tok, model = self.engine.ensure_loaded()
        prefix = self.prompt_ids + self.committed_ids
        b = len(prefix) - 1
        layer = self.engine.layer
        frontier = self.last_frontier or self.inspect_frontier()

        # Fast path: amortised ForkHead subspace (zero backward passes!)
        fh_dirs = self.last_fh.get("dirs") if self.last_fh is not None else None
        if fh_dirs is not None and fh_dirs.shape[0] >= 2 and len(frontier.get("branches", [])) >= 2:
            axes = fh_dirs[:2].detach().float().cpu().numpy().astype(np.float64)
            # Build simple symmetric anchors around unit circle aligned to branch previews
            branches = frontier["branches"][:5]
            n_b = len(branches)
            angles = np.linspace(0.0, 2.0 * math.pi, n_b, endpoint=False)
            anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            v_unit = axes[0] / max(float(np.linalg.norm(axes[0])), 1e-12)
            ref_cont = branches[0]["preview_ids"]
            a_far = calibrate(model, self.engine.hook, prefix + ref_cont, b, v_unit, target=0.25, mode="cursor", a0=1.0, iters=3)
            if not math.isfinite(a_far) or a_far <= 0.0:
                a_far = 1.0
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
            self.active_compass = {
                "cursor_len": len(self.committed_ids),
                "b": b,
                "layer": layer,
                "axes": axes,
                "anchors": anchors,
                "alpha_unit": float(a_far),
                "readings": reading_meta,
                "source": "fork_head",
            }
            return {
                "source": "fork_head",
                "cursor_len": len(self.committed_ids),
                "variance_explained": 1.0,
                "readings": reading_meta,
                "dots": [],
            }

        # Standard Tier-2 path: compact burst + OOM-safe gradients at cursor `b`
        from explorer.sample import generate

        n_samples = max(6, int(n_samples))
        lookahead_tokens = max(4, int(lookahead_tokens))
        self._clear_steering()
        raw_conts = generate(
            model,
            tok,
            prefix,
            n=n_samples,
            max_new=lookahead_tokens,
            temperature=1.0,
            seed=int(seed),
            batch=min(16, n_samples),
        )
        conts = [cut_doc(tok, c) for c in raw_conts]
        cont_texts = [_strip_doc_stop(tok.decode(c)).strip() for c in conts]

        G = _grads_at_oom_safe(model, prefix, conts, layer, b, batch=min(4, n_samples))
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

        far = int(np.argmax(np.linalg.norm(anchors, axis=1)))
        v_far = anchors[far] @ axes
        v_far_norm = max(float(np.linalg.norm(v_far)), 1e-12)
        v_unit = v_far / v_far_norm
        a0 = alpha_for(G, v_unit, 0.25)
        if not math.isfinite(a0) or a0 <= 0.0 or a0 > 1e4:
            a0 = 1.0
        ref_cont = next((ct for ct in conts if ct), [])
        a_far = (
            calibrate(model, self.engine.hook, prefix + ref_cont, b, v_unit, target=0.25, mode="cursor", a0=a0, iters=3)
            if ref_cont
            else a0
        )
        if not math.isfinite(a_far) or a_far <= 0.0:
            a_far = a0
        alpha_unit = float(a_far / v_far_norm)

        # Human-readable labels from distinctive words
        cluster_texts = {r: [cont_texts[i] for i in range(n_samples) if labels[i] == r] for r in rd}
        rep_texts = {r: (cluster_texts[r][0] if cluster_texts[r] else "") for r in rd}
        reading_meta: list[dict[str, Any]] = []
        color_map: dict[int, str] = {}
        for idx, r in enumerate(rd):
            col = READING_PALETTE[idx % len(READING_PALETTE)]
            color_map[r] = col
            others = [rep_texts[o] for o in rd if o != r]
            label = _distinctive_word(rep_texts[r], others) or f"reading {idx + 1}"
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

        self.active_compass = {
            "cursor_len": len(self.committed_ids),
            "b": b,
            "layer": layer,
            "axes": axes,
            "anchors": anchors,
            "alpha_unit": alpha_unit,
            "readings": reading_meta,
            "source": "gradients",
        }
        return {
            "source": "gradients",
            "cursor_len": len(self.committed_ids),
            "variance_explained": round(var_expl, 4),
            "readings": reading_meta,
            "dots": dots,
        }

    def steer_compass(self, x: float, y: float, max_new: int = 16) -> dict[str, Any]:
        """Generate a live steered preview at `(x, y)` on the active cursor's 2D compass."""
        if self.active_compass is None or self.active_compass["cursor_len"] != len(self.committed_ids):
            self.probe_compass()
        tok, _ = self.engine.ensure_loaded()
        comp = self.active_compass
        assert comp is not None
        p = np.array([float(x), float(y)], dtype=np.float64)
        vec = torch.tensor(comp["alpha_unit"] * (p @ comp["axes"]), dtype=torch.float32)

        # Get first steered token + greedy lookahead via cloned KV cache (zero mutation to session!)
        prefix = self.prompt_ids + self.committed_ids
        b = len(prefix) - 1
        gvec = self.engine._global_vector(self.global_coords)
        combined = vec if gvec is None else (vec + torch.tensor(gvec, dtype=torch.float32))
        self.engine.hook.set(combined, b=b, only_at_b=False)
        if self.engine.reft_controller is not None:
            rc = self.engine._resolve_coords(self.global_coords, set(self.engine.reft_controller.axis_names))
            self.engine.reft_controller.set_coords(rc or None, start_pos=self.start - 1)

        _, model = self.engine.ensure_loaded()
        dev = self.engine._dev()
        eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        gen_ids: list[int] = []
        try:
            with torch.no_grad():
                # Prefill from scratch or from cropped cache at b (so steering applies cleanly AT position b!)
                pkv_s = None
                if b >= 1:
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

        # Soft Gaussian weights over compass reading anchors
        anchors = comp["anchors"]
        d2 = np.sum((anchors - p[None, :]) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * 0.55 ** 2))
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
                self._sessions.pop(oldest, None)
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

    def step(self, session_id: str, **kw: Any) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).step(**kw)

    def glide(self, session_id: str, max_words: int = 32, stop_on_fork: bool = True) -> Generator[dict[str, Any], None, None]:
        with self.engine._lock:
            sess = self._get(session_id)
            yield from sess.glide(max_words=max_words, stop_on_fork=stop_on_fork)

    def rewind(self, session_id: str, to_token_index: int) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).rewind(to_token_index)

    def checkout(self, session_id: str, node_id: str) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).checkout_node(node_id)

    def set_sliders(self, session_id: str, global_coords: dict[str, float]) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).set_global_coords(global_coords)

    def probe_compass(self, session_id: str, n_samples: int = 10, lookahead_tokens: int = 12, seed: int = 0) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).probe_compass(n_samples=n_samples, lookahead_tokens=lookahead_tokens, seed=seed)

    def steer_compass(self, session_id: str, x: float, y: float, max_new: int = 16) -> dict[str, Any]:
        with self.engine._lock:
            return self._get(session_id).steer_compass(x=x, y=y, max_new=max_new)
