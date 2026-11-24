"""Roots, config hashing, resumable stages, model facts (HANDOFF_internals §11).

Every stage is `run(cfg: dict) -> list[Path]`.  `stage(name)` wraps it so that it is skipped when its outputs exist
and the config hash matches (Colab sessions get cut off), unless cfg["force"] is true.  Roots come from env vars:
  UT_RESULTS  results root     (default <repo>/results/internals)
  UT_CACHE    activations etc.  (default <repo>/cache/internals)
  UT_CORPUS   corpora + alignments (default $UT_CACHE/corpora)
  UT_DATA     test data        (default <repo>/data)
"""
import functools
import hashlib
import json
import os
import time
from pathlib import Path

from ut.common import ROOT, save_json

RESULTS = Path(os.environ.get("UT_RESULTS", ROOT / "results" / "internals"))
CACHE = Path(os.environ.get("UT_CACHE", ROOT / "cache" / "internals"))
CORPUS = Path(os.environ.get("UT_CORPUS", CACHE / "corpora"))
DATA = Path(os.environ.get("UT_DATA", ROOT / "data"))

PAIRS = {"es-en": ("es", "en"), "hi-en": ("hi", "en")}
MAIN = __import__("os").environ.get("UT_MAIN", "gemma12b")
SAE_LAYERS = {"gemma12b": [12, 24, 31, 41]}        # Gemma Scope 2 12B-IT resid_post (block outputs, 0-based)


def res_dir(model, *sub):
    d = RESULTS.joinpath(model, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(model, *sub):
    d = CACHE.joinpath(model, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cfg_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("force", "dry")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def stage(name):
    """Decorator: resumable stage. The marker <res_dir>/.done/<name>[_<tag>].json stores the config hash."""
    def deco(fn):
        @functools.wraps(fn)
        def run(cfg):
            model = cfg.get("model", "_global")
            tag = "_".join(str(cfg[k]) for k in ("pair", "bt", "layer") if k in cfg)
            mark = res_dir(model, ".done") / f"{name}{'_' + tag if tag else ''}.json"
            h = cfg_hash(cfg)
            if mark.exists() and not cfg.get("force"):
                m = json.loads(mark.read_text())
                if m.get("hash") == h and all(Path(p).exists() for p in m.get("outputs", [])):
                    print(f"[skip] {name} {tag} ({model}): done {m['when']}", flush=True)
                    return [Path(p) for p in m["outputs"]]
            t0 = time.time()
            print(f"[stage] {name} {tag} ({model})", flush=True)
            outs = fn(cfg) or []
            save_json(mark, dict(hash=h, cfg=cfg, outputs=[str(p) for p in outs], when=time.strftime("%a %d %b %H:%M"),
                                 minutes=round((time.time() - t0) / 60, 1)))
            print(f"[done] {name} {tag} ({model}) in {(time.time() - t0) / 60:.1f} min", flush=True)
            return outs
        run.stage_name = name
        return run
    return deco


def model_layers(model_key):
    """(n_layers, hidden size) from the config alone (no weights)."""
    from transformers import AutoConfig
    from ut.common import MODELS
    c = AutoConfig.from_pretrained(MODELS[model_key]["hf_id"])
    c = getattr(c, "text_config", None) or c
    return c.num_hidden_layers, c.hidden_size
