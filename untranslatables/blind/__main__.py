"""CLI:  python -m blind <stage> [--model g3_4b] [--set e1] [--kv key=value ...] [--force]"""
import argparse, importlib, json

STAGES = {"material": "blind.material:check", "atlas_corpus": "blind.material:build_atlas",
          **{n: f"blind.stages:{f}" for n, f in [("e0", "e0"), ("collect", "collect"), ("spectra", "spectra"),
             ("heatmaps", "heatmaps"), ("fdcheck", "fdcheck"), ("steer", "steer"), ("sae", "sae_stage"),
             ("practices", "practices"), ("validate", "validate"), ("atlas_dirs", "atlas_dirs"),
             ("atlas_langs", "atlas_langs"), ("atlas_cluster", "atlas_cluster")]},
          "figures": "blind.figures:run", "report": "blind.report:run"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--model"); ap.add_argument("--set")
    ap.add_argument("--kv", nargs="*", default=[]); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = {k: v for k, v in (("model", a.model), ("set", a.set)) if v}
    for kv in a.kv:
        k, v = kv.split("=", 1)
        try: cfg[k] = json.loads(v)
        except json.JSONDecodeError: cfg[k] = v
    if a.force: cfg["force"] = True
    mod, fn = STAGES[a.stage].split(":")
    getattr(importlib.import_module(mod), fn)(cfg)


if __name__ == "__main__":
    main()
