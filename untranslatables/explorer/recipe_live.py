"""Live English-basis 'ingredients + leftover' decomposition & branch dispersion (Track A).

Zero-extra-forward-pass live diagnostic separating:
  - stylistic forks (e.g. 'will return' / 'back will come' / 'once more shall'):
    high token entropy, low semantic dispersion across lookahead ribbons.
  - conceptual gaps (e.g. 'duende', 'sprezzatura', 'saudade'):
    high token entropy, high semantic dispersion and/or high English-basis leftover.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
from scipy.optimize import nnls
import torch

from ut import arch
from ut.common import DATA, ROOT


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower()) if unicodedata.category(c) != "Mn")


def _unit(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    nrm = float(np.linalg.norm(arr))
    return arr / max(nrm, 1e-8)


def _ortho_sim(a: str, b: str) -> float:
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return 0.0
    return float(difflib.SequenceMatcher(None, fa, fb).ratio())


def _nnomp(D: np.ndarray, x: np.ndarray, k: int = 5) -> tuple[list[int], list[float], list[float]]:
    """Non-negative orthogonal matching pursuit over unit-norm atoms D (N, d) for unit target x (d,)."""
    r = x.astype(np.float64).copy()
    x64 = x.astype(np.float64)
    D64 = D.astype(np.float64)
    S: list[int] = []
    R2: list[float] = []
    w = np.zeros(0, dtype=np.float64)
    denom = float(x64 @ x64) or 1.0
    for _ in range(k):
        c = D64 @ r
        if S:
            c[S] = -np.inf
        j = int(np.argmax(c))
        if c[j] <= 1e-9:
            break
        S.append(j)
        w, _ = nnls(D64[S].T, x64)
        r = x64 - D64[S].T @ w
        R2.append(float(max(0.0, min(1.0, 1.0 - (r @ r) / denom))))
    return S, w.tolist(), R2


class LiveEnglishRecipeDecomposer:
    """Fast CPU NNOMP decomposer over a compact English concept pool."""

    def __init__(self, words: list[str], vectors: np.ndarray, centroid: np.ndarray, layer: int):
        self.words = list(words)
        self.vectors = np.asarray(vectors, dtype=np.float32)  # (N, d) unit-norm centred
        self.centroid = np.asarray(centroid, dtype=np.float32)
        self.layer = int(layer)

    @classmethod
    def from_npz(cls, path: str | Path) -> LiveEnglishRecipeDecomposer:
        p = Path(path)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes().decode("utf-8"))
            words = meta["words"]
            layer = int(meta["layer"])
            vecs = np.asarray(z["vectors"], dtype=np.float32)
            cent = np.asarray(z["centroid"], dtype=np.float32)
        # Re-normalise rows defensively
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        return cls(words=words, vectors=vecs, centroid=cent, layer=layer)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta_bytes = np.frombuffer(
            json.dumps({"words": self.words, "layer": self.layer}, ensure_ascii=False).encode("utf-8"),
            dtype=np.uint8,
        )
        with open(p, "wb") as f:
            np.savez(
                f,
                meta=meta_bytes,
                vectors=self.vectors.astype(np.float16),
                centroid=self.centroid.astype(np.float16),
            )
        return p

    def decompose(self, word: str, raw_vec: np.ndarray, k: int = 5) -> dict[str, Any]:
        """Decompose a source-token hidden state into top-k English concept ingredients + leftover."""
        x = _unit(np.asarray(raw_vec, dtype=np.float32) - self.centroid)
        fw = _fold(word.strip(" .,;:!?¡¿\"'()[]{}—–-"))
        keep = np.ones(len(self.words), dtype=bool)
        if fw:
            for i, pw in enumerate(self.words):
                fpw = _fold(pw)
                if fpw == fw or (len(fw) >= 5 and len(fpw) >= 5 and fpw[:5] == fw[:5]):
                    keep[i] = False
        idx = np.where(keep)[0]
        if len(idx) == 0:
            return {"word": word, "ingredients": [], "leftover": 1.0, "r2": 0.0}

        S, _w, R2 = _nnomp(self.vectors[idx], x, k=k)
        gains = np.diff([0.0] + R2[:k]) if R2 else np.zeros(0)
        ingredients = []
        for s_pos, g in zip(S[:k], gains):
            eng_word = self.words[int(idx[s_pos])]
            ingredients.append({
                "word": eng_word,
                "share": round(float(max(0.0, g)), 4),
                "sound_alike": bool(fw and _ortho_sim(fw, eng_word) >= 0.58),
            })
        r2_final = float(R2[min(k, len(R2)) - 1]) if R2 else 0.0
        return {
            "word": word,
            "ingredients": ingredients,
            "leftover": round(float(max(0.0, min(1.0, 1.0 - r2_final))), 4),
            "r2": round(r2_final, 4),
        }


def default_en_pool_path(model_key: str) -> Path:
    return ROOT / "results" / "explorer" / model_key / "en_pool.npz"


def load_en_pool_for_model(model_key: str | None) -> LiveEnglishRecipeDecomposer | None:
    if not model_key:
        return None
    p = default_en_pool_path(model_key)
    if p.exists():
        return LiveEnglishRecipeDecomposer.from_npz(p)
    return None


def extract_en_pool_from_model(
    model: Any,
    tok: Any,
    layer: int | None = None,
    max_words: int = 2000,
    batch: int = 64,
) -> LiveEnglishRecipeDecomposer:
    """Extract a compact centred English concept pool from `data/pools/en.json` using `model`."""
    pool_path = DATA / "pools" / "en.json"
    words_raw = json.loads(pool_path.read_text(encoding="utf-8"))
    if isinstance(words_raw, dict):
        words = list(words_raw.get("words") or words_raw.get("items") or [])[:max_words]
    else:
        words = [str(w) for w in words_raw[:max_words]]

    if layer is None:
        layer = arch.n_layers(model) // 2

    prompts = [f"The English word \"{w}\" means" for w in words]
    dev = arch.text_model(model).embed_tokens.weight.device
    vecs: list[np.ndarray] = []

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        for i in range(0, len(prompts), batch):
            part = prompts[i : i + batch]
            enc = tok(part, return_tensors="pt", padding=True).to(dev)
            attn = enc["attention_mask"]
            last_pos = attn.sum(dim=1) - 1
            with torch.no_grad():
                out = arch.forward_hidden(model, **enc, output_hidden_states=True)
                hs = out.hidden_states[layer + 1]
                rows = hs[torch.arange(len(part), device=dev), last_pos].float().cpu().numpy()
                vecs.append(rows)
    finally:
        tok.padding_side = old_side

    X = np.concatenate(vecs, axis=0).astype(np.float32)
    centroid = X.mean(axis=0)
    centred = X - centroid
    norms = np.linalg.norm(centred, axis=-1, keepdims=True)
    centred = centred / np.maximum(norms, 1e-8)
    return LiveEnglishRecipeDecomposer(words=words, vectors=centred, centroid=centroid, layer=int(layer))


def branch_semantic_dispersion(merged: list[dict[str, Any]], regime: str) -> tuple[float, str]:
    """Compute probability-weighted pairwise semantic dispersion across deduplicated branches.

    Returns (dispersion in [0, 1], fork_kind in {'linear', 'stylistic', 'conceptual'}).
    """
    if regime == "linear" or len(merged) <= 1:
        return 0.0, "linear"

    embs: list[np.ndarray] = []
    probs: list[float] = []
    for m in merged:
        e = m.get("emb")
        if e is not None:
            embs.append(np.asarray(e, dtype=np.float32))
            probs.append(float(max(1e-6, m.get("prob", 0.0))))

    if len(embs) < 2:
        # Fallback when ribbon embeddings aren't available: spread of first-word surfaces
        uniq_fw = {str(m.get("first_word_text", "")).strip().lower() for m in merged}
        disp = min(1.0, 0.25 * max(0, len(uniq_fw) - 1))
    else:
        W = sum(probs) or 1.0
        norm_p = [p / W for p in probs]
        num = 0.0
        den = 0.0
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                cos = float(np.clip(np.dot(embs[i], embs[j]), -1.0, 1.0))
                dist = 0.5 * (1.0 - cos)  # in [0, 1]
                w = norm_p[i] * norm_p[j]
                num += w * dist
                den += w
        disp = float(num / max(den, 1e-8))

    # Calibrated threshold: paraphrases ('will return' / 'back will come') sit below ~0.075 cosine-half-dist,
    # whereas conceptual forks ('magic' / 'spirit' / 'duende') disperse above ~0.075.
    fork_kind = "conceptual" if (regime == "fork" and disp >= 0.075) else "stylistic"
    return round(disp, 4), fork_kind


def main() -> None:
    import argparse
    from ut.common import load_model

    ap = argparse.ArgumentParser(description="Extract compact English concept pool for live recipe decomposition.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--max-words", type=int, default=2000)
    args = ap.parse_args()

    tok, model = load_model(args.model)
    dec = extract_en_pool_from_model(model, tok, layer=args.layer, max_words=args.max_words)
    out = dec.save(default_en_pool_path(args.model))
    print(f"Saved {len(dec.words)} English pool vectors (layer={dec.layer}) -> {out}")


if __name__ == "__main__":
    main()
