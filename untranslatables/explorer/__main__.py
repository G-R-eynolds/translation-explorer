"""CLI entry point for the explorer package across Phases 1–4.

Examples:
    python -m explorer run --model g3_4b_pt [--n 64] [--texts a,b] [--layer 17]
    python -m explorer report --model g3_4b_pt
    python -m explorer serve --mode replay|live [--model g3_4b_pt|tiny] [--port 8765]
    python -m explorer offline --out ../figures_for_dad/duende_workbench_offline.html
    python -m explorer axes --model g3_4b_pt --save results/explorer/g3_4b_pt/phase3/global_axes.npz
    python -m explorer train-lora --model g3_4b_pt --steps 120 --save results/explorer/g3_4b_pt/phase3/multiref_lora.pt
    python -m explorer train-reft --model g3_4b_pt --steps 160 --save results/explorer/g3_4b_pt/phase3/spectral_reft.pt
    python -m explorer train-fork-head --model g3_4b_pt --epochs 80
"""
import argparse
import inspect
from pathlib import Path


def _parse_model_dirs(raw) -> dict[str, str] | None:
    """Parse --model-dirs supporting k1=p1,k2=p2, p1,p2, p1 p2, and repeated flags."""
    if not raw:
        return None
    flat: list[str] = []
    if isinstance(raw, str):
        flat.append(raw)
    else:
        for item in raw:
            if isinstance(item, (list, tuple)):
                flat.extend(str(x) for x in item)
            else:
                flat.append(str(item))
    out: dict[str, str] = {}
    for entry in flat:
        for part in entry.split(","):
            tok = part.strip()
            if not tok:
                continue
            if "=" in tok:
                k, p = tok.split("=", 1)
                k, p = k.strip(), p.strip()
                if k and p:
                    out[k] = p
            else:
                key = Path(tok).name or tok
                out[key] = tok
    return out or None


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="python -m explorer")
    ap.add_argument(
        "stage",
        choices=["run", "report", "serve", "offline", "axes", "train-lora", "train-reft", "train-fork-head"],
    )
    ap.add_argument("--model", default="g3_4b_pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--texts", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--prompt-style", choices=["legacy", "doc"], default="legacy")
    ap.add_argument("--clustering", choices=["semantic", "hybrid", "grad"], default="semantic")
    ap.add_argument("--alpha-grad", type=float, default=0.5)
    ap.add_argument("--disp-min", type=float, default=0.035)
    # Phase 2 server / offline args
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--model-dirs", action="append", nargs="+", default=None)
    ap.add_argument("--sessions-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--global-axes", default=None)
    ap.add_argument("--fork-head", default=None)
    ap.add_argument("--reft", default=None)
    ap.add_argument("--lora", default=None)
    # Phase 3 / 4 training args
    ap.add_argument("--save", default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--kl-beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--from-texts", action="store_true")
    ap.add_argument("--n-dirs", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--entropy-thresh", type=float, default=None)
    a = ap.parse_args(argv)

    if a.stage in ("run", "report"):
        cfg = dict(
            model=a.model,
            n=a.n,
            layer=a.layer,
            texts=a.texts.split(",") if a.texts else None,
            prompt_style=a.prompt_style,
            clustering=a.clustering,
            alpha_grad=a.alpha_grad,
            disp_min=a.disp_min,
            results_dir=a.results_dir,
        )
        if a.stage == "run":
            from explorer.run import run
            run(cfg)
        from explorer.report import run as report
        print("report:", report(cfg))
        return

    if a.stage == "offline":
        from explorer.server import build_offline_html
        out = a.out or "../figures_for_dad/duende_workbench_offline.html"
        kwargs = {"results_dir": a.results_dir}
        sig = inspect.signature(build_offline_html)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        model_dirs = _parse_model_dirs(a.model_dirs)
        if model_dirs is not None and ("model_dirs" in sig.parameters or has_varkw):
            kwargs["model_dirs"] = model_dirs
        if a.sessions_dir is not None and ("sessions_dir" in sig.parameters or has_varkw):
            kwargs["sessions_dir"] = a.sessions_dir
        print("offline workbench:", build_offline_html(out, **kwargs))
        return

    if a.stage == "serve":
        from explorer.live_engine import LiveTranslationEngine, ReplayEngine
        from explorer.server import create_server
        replay = ReplayEngine(a.results_dir)
        live = None
        if a.mode == "live":
            live = LiveTranslationEngine(
                model_key=a.model,
                layer=a.layer,
                global_axes_path=a.global_axes,
                fork_head_path=a.fork_head,
                reft_path=a.reft,
                lora_path=a.lora,
            )
        srv = create_server(host=a.host, port=a.port, mode=a.mode, replay_engine=replay, live_engine=live)
        print(f"Serving Spectral Translation Workbench ({a.mode}) at http://{a.host}:{srv.server_address[1]}/")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if live is not None:
                live.close()
            srv.server_close()
        return

    if a.stage in ("axes", "train-lora", "train-reft"):
        from explorer.train_adapters import main as train_main
        mode_map = {"axes": "axes", "train-lora": "lora", "train-reft": "reft"}
        sub_argv = [
            "--mode", mode_map[a.stage],
            "--model", a.model,
            "--steps", str(a.steps),
            "--rank", str(a.rank),
            "--prompt-style", a.prompt_style,
        ]
        if a.stage == "train-lora":
            sub_argv += ["--kl-beta", str(a.kl_beta)]
        if a.layers:
            sub_argv += ["--layers", a.layers]
        if a.dataset:
            sub_argv += ["--dataset", a.dataset]
        out_path = a.out or a.save
        if out_path:
            sub_argv += ["--out", out_path]
        train_main(sub_argv)
        return

    if a.stage == "train-fork-head":
        from explorer.fork_head import main as fh_main
        sub_argv = ["--model", a.model, "--epochs", str(a.epochs), "--prompt-style", a.prompt_style]
        if a.layer is not None:
            sub_argv += ["--layer", str(a.layer)]
        if a.from_texts:
            sub_argv += ["--from-texts"]
        if a.n_dirs is not None:
            sub_argv += ["--n-dirs", str(a.n_dirs)]
        if a.dataset:
            sub_argv += ["--dataset", a.dataset]
        if a.entropy_thresh is not None:
            sub_argv += ["--entropy-thresh", str(a.entropy_thresh)]
        out_path = a.out or a.save
        if out_path:
            sub_argv += ["--out", out_path]
        fh_main(sub_argv)
        return


if __name__ == "__main__":
    main()
