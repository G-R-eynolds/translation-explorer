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


def main():
    ap = argparse.ArgumentParser(prog="python -m explorer")
    ap.add_argument(
        "stage",
        choices=["run", "report", "serve", "offline", "axes", "train-lora", "train-reft", "train-fork-head"],
    )
    ap.add_argument("--model", default="g3_4b_pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--texts", default=None)
    ap.add_argument("--layer", type=int, default=None)
    # Phase 2 server / offline args
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results-dir", default=None)
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
    a = ap.parse_args()

    if a.stage in ("run", "report"):
        cfg = dict(model=a.model, n=a.n, layer=a.layer, texts=a.texts.split(",") if a.texts else None)
        if a.stage == "run":
            from explorer.run import run
            run(cfg)
        from explorer.report import run as report
        print("report:", report(cfg))
        return

    if a.stage == "offline":
        from explorer.server import build_offline_html
        out = a.out or "../figures_for_dad/duende_workbench_offline.html"
        print("offline workbench:", build_offline_html(out, results_dir=a.results_dir))
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
        argv = ["--mode", mode_map[a.stage], "--model", a.model, "--steps", str(a.steps), "--rank", str(a.rank)]
        if a.stage == "train-lora":
            argv += ["--kl-beta", str(a.kl_beta)]
        out_path = a.out or a.save
        if out_path:
            argv += ["--out", out_path]
        train_main(argv)
        return

    if a.stage == "train-fork-head":
        from explorer.fork_head import main as fh_main
        argv = ["--model", a.model, "--epochs", str(a.epochs)]
        if a.layer is not None:
            argv += ["--layer", str(a.layer)]
        out_path = a.out or a.save
        if out_path:
            argv += ["--out", out_path]
        fh_main(argv)
        return


if __name__ == "__main__":
    main()
