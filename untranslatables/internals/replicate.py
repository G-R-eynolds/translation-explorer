"""Part A replications (HANDOFF_internals §2).

R1  English detour: latent_lang.py (Wendler et al. logit lens, fr->es, de->it, pt->fr) on the new models; the result is
    copied to results/internals/<model>/replication_R1.json.
R2  Sentence round trip with the loophole fixed: sent/ in R2 mode (cognate masking default, B2_borrow, greedy-only B7,
    intended-sense recovery, no concept-layer score, no targeting, no dialect), written to
    results/internals/<model>/replication_R2/.  For the 8B bridge the 19 Nov candidates are re-scored rather than
    regenerated, so the comparison with the earlier run is like for like.
"""
import json
import os
import shutil
import subprocess
import sys

from ut.common import ROOT
from internals.common import res_dir, stage

R2_ENV = {"SENT_DIR_TEMPLATE": "results/internals/{model}/replication_R2",
          "SENT_FIG_TEMPLATE": "results/internals/{model}/replication_R2/figures"}
# generation batch per model (12 GB card: Gemma 3's KV cache is ~0.4 MB per token)
GEN_BATCH = {"gemma12b": 8, "qwen14b": 12, "nemo12b": 12}


def _py(*args, env=None):
    e = dict(os.environ, **(env or {}))
    print("$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, env=e)


@stage("R1")
def r1(cfg):
    m = cfg["model"]
    _py("latent_lang.py", "--model", m, "--n", str(cfg.get("n", 120)))
    src = ROOT / "results" / m / "latent_lang.json"
    dst = res_dir(m) / "replication_R1.json"
    shutil.copy(src, dst)
    return [dst]


@stage("R2")
def r2(cfg):
    m = cfg["model"]
    d = res_dir(m, "replication_R2")
    b = str(GEN_BATCH.get(m, 24))
    if cfg.get("reuse_from"):                              # the 8B bridge: re-score the 19 Nov candidates
        rows = [json.loads(l) for l in open(ROOT / "results" / "sent" / cfg["reuse_from"] / "candidates.jsonl")]
        with open(d / "candidates.jsonl", "w") as f:
            for c in rows:
                if c["category"] == "dialect":
                    continue
                if c["rung"] == "B2":
                    c["rung"] = "B2_borrow"; c["cid"] = c["cid"].replace("|B2|", "|B2_borrow|")
                if c["rung"] == "B7" and c["sample"] >= 0:
                    continue
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    else:
        _py("-m", "sent.ladder", "--model", m, "--greedy-only", "B7", "--batch", b, env=R2_ENV)
    _py("-m", "sent.score", "--model", m, "--rt-only", "--mask-rule", "cognate", "--batch", "16", env=R2_ENV)
    _py("-m", "sent.sense", "--model", m, "--batch", "16", env=R2_ENV)
    if not cfg.get("reuse_from"):
        _py("-m", "sent.spread", "--model", m, "--batch", b, env=R2_ENV)
    _py("-m", "sent.analyze_rt", "--model", m, env=R2_ENV)
    return [d / "validation.json", d / "curves.json", d / "sense.json"]


@stage("R2_bt")
def r2_bt(cfg):
    """Second back-translator on a model's R2 candidates (cfg: model, bt)."""
    m = cfg["model"]
    _py("-m", "sent.score", "--model", m, "--bt", cfg["bt"], "--rt-only", "--mask-rule", "cognate", "--batch", "16",
        env=R2_ENV)
    _py("-m", "sent.analyze_rt", "--model", m, env=R2_ENV)
    return [res_dir(m, "replication_R2") / "raw" / f"rt_{cfg['bt']}.json"]
