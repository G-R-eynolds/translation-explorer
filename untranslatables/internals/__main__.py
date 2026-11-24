"""CLI:  python -m internals <stage> --model gemma12b [--pair es-en] [--set key=value ...] [--force]

Each stage is a resumable run(cfg) function (internals.common.stage); this wrapper only builds cfg.
"""
import argparse
import importlib
import json

STAGES = {   # name -> module:function
    "R1": "internals.replicate:r1", "R2": "internals.replicate:r2", "R2_bt": "internals.replicate:r2_bt",
    "corpus": "internals.data:build_corpus", "flores": "internals.data:build_flores",
    "align": "internals.align:run", "quick": "internals.collect:quick", "collect": "internals.collect:collect",
    "evalstates": "internals.collect:eval_states", "cca": "internals.cca:run", "spans": "internals.spans:run",
    "hub": "internals.cca:hub", "probes": "internals.probes:run", "mlp": "internals.probes:mlp",
    "sae": "internals.sae:run", "mediate": "internals.mediate:run", "interchange": "internals.mediate:interchange",
    "figures": "internals.figures:run", "report": "internals.report:run",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--model", default=None)
    ap.add_argument("--pair", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="extra cfg entries key=value (JSON values allowed)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = {k: v for k, v in (("model", a.model), ("pair", a.pair)) if v is not None}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    if a.force:
        cfg["force"] = True
    mod, fn = STAGES[a.stage].split(":")
    outs = getattr(importlib.import_module(mod), fn)(cfg)
    for p in outs or []:
        print("output:", p)


if __name__ == "__main__":
    main()
