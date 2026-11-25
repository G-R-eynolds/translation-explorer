"""Zero-dependency HTTP + SSE server and offline single-file HTML exporter for the Workbench.

Usage:
    python -m explorer.server --mode replay --port 8765
    python -m explorer.server --mode live --model tiny --port 8765
    python -m explorer.server --export-offline ../figures_for_dad/duende_workbench_offline.html
"""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from explorer.live_engine import DEFAULT_GLOBAL_AXES, LiveTranslationEngine, ReplayEngine

HERE = Path(__file__).resolve().parent
WORKBENCH_HTML = HERE / "static" / "workbench.html"
OFFLINE_HOOK = "<!-- OFFLINE_DATA_HOOK -->"


def build_offline_html(output_path: str | Path, results_dir: str | Path | None = None) -> Path:
    """Inject all saved ReplayEngine bundles as `window.__OFFLINE_DATA__` into a standalone HTML file."""
    engine = ReplayEngine(results_dir)
    texts = engine.list_texts()
    bundles = {t["id"]: engine.get_text_bundle(t["id"]) for t in texts}
    payload = {
        "mode": "offline",
        "texts": texts,
        "bundles": bundles,
    }
    raw_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    script_tag = f"<script>window.__OFFLINE_DATA__ = {raw_json};</script>"

    template = WORKBENCH_HTML.read_text(encoding="utf-8")
    if OFFLINE_HOOK in template:
        html = template.replace(OFFLINE_HOOK, script_tag, 1)
    else:
        html = template.replace("</body>", f"{script_tag}\n</body>", 1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


class WorkbenchServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        mode: str = "replay",
        replay_engine: ReplayEngine | None = None,
        live_engine: LiveTranslationEngine | None = None,
        quiet: bool = False,
    ):
        self.mode = mode
        self.replay = replay_engine if replay_engine is not None else ReplayEngine()
        self.live = live_engine
        self.quiet = quiet
        super().__init__(server_address, WorkbenchRequestHandler)


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        sys.stderr.write(f"[workbench] {self.address_string()} - {fmt % args}\n")

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        srv: WorkbenchServer = self.server  # type: ignore[assignment]

        try:
            if path in ("/", "/index.html", "/workbench.html"):
                body = WORKBENCH_HTML.read_bytes()
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", body)
                return

            if path == "/api/status":
                live_info = srv.live.status() if srv.live is not None else {
                    "model_key": None,
                    "loaded": False,
                    "device": "cpu",
                    "layer": None,
                    "active_forks": 0,
                    "global_axes_loaded": [],
                    "global_axes_schema": DEFAULT_GLOBAL_AXES,
                }
                self._send_json(HTTPStatus.OK, {
                    "mode": srv.mode,
                    "replay_texts": len(srv.replay.list_texts()),
                    **live_info,
                })
                return

            if path == "/api/replay/texts":
                self._send_json(HTTPStatus.OK, srv.replay.list_texts())
                return

            if path.startswith("/api/replay/"):
                text_id = unquote(path[len("/api/replay/"):]).strip("/")
                try:
                    bundle = srv.replay.get_text_bundle(text_id)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, bundle)
                return

            if path == "/api/stream":
                if srv.live is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Server is running in replay mode; live streaming requires --mode live."})
                    return
                source_text = (qs.get("text") or [""])[0].strip()
                if not source_text:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required query parameter 'text'."})
                    return
                max_new = int((qs.get("max_new") or ["48"])[0])
                entropy_threshold = float((qs.get("entropy_threshold") or ["1.5"])[0])
                compute_fisher = (qs.get("compute_fisher") or ["0"])[0].lower() in ("1", "true", "yes")
                fisher_iters = int((qs.get("fisher_iters") or ["3"])[0])

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                n_toks = 0
                try:
                    for item in srv.live.stream_translate(
                        source_text=source_text,
                        max_new=max_new,
                        entropy_threshold=entropy_threshold,
                        compute_fisher=compute_fisher,
                        fisher_iters=fisher_iters,
                    ):
                        n_toks += 1
                        chunk = f"event: token\ndata: {json.dumps(item, ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    done = f"event: done\ndata: {json.dumps({'n_tokens': n_toks})}\n\n".encode("utf-8")
                    self.wfile.write(done)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return

            if path == "/api/session/glide":
                if srv.live is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Live session gliding requires --mode live."})
                    return
                session_id = (qs.get("session_id") or [""])[0].strip()
                if not session_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required query parameter 'session_id'."})
                    return
                max_words = int((qs.get("max_words") or ["32"])[0])
                stop_on_fork = (qs.get("stop_on_fork") or ["1"])[0].lower() not in ("0", "false", "no")

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    for item in srv.live.sessions.glide(session_id, max_words=max_words, stop_on_fork=stop_on_fork):
                        ev = item.get("event", "step")
                        chunk = f"event: {ev}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except KeyError as exc:
                    err = f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")
                    self.wfile.write(err)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        srv: WorkbenchServer = self.server  # type: ignore[assignment]

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON body: {exc}"})
            return

        try:
            if path == "/api/replay/steer":
                text_id = body.get("text_id")
                if not text_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'text_id'."})
                    return
                x = float(body.get("x", 0.0))
                y = float(body.get("y", 0.0))
                global_coords = body.get("global_coords")
                try:
                    res = srv.replay.steer_replay(text_id, x, y, global_coords=global_coords)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, res)
                return

            if path == "/api/probe_fork":
                if srv.live is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Server is running in replay mode; /api/probe_fork requires --mode live."})
                    return
                source_text = str(body.get("source_text", "")).strip()
                if not source_text:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'source_text'."})
                    return
                res = srv.live.probe_fork(
                    source_text=source_text,
                    prefix_token_ids=body.get("prefix_token_ids") or [],
                    n_samples=int(body.get("n_samples", 16)),
                    lookahead_tokens=int(body.get("lookahead_tokens", 16)),
                    temperature=float(body.get("temperature", 1.0)),
                    seed=int(body.get("seed", 0)),
                )
                self._send_json(HTTPStatus.OK, res)
                return

            if path == "/api/steer":
                if srv.live is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Server is running in replay mode; /api/steer requires --mode live."})
                    return
                fork_id = body.get("fork_id")
                if not fork_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'fork_id'."})
                    return
                try:
                    res = srv.live.steer_at(
                        fork_id=str(fork_id),
                        x=float(body.get("x", 0.0)),
                        y=float(body.get("y", 0.0)),
                        global_coords=body.get("global_coords"),
                        max_new=int(body.get("max_new", 32)),
                        temperature=float(body.get("temperature", 0.0)),
                        only_at_b=bool(body.get("only_at_b", False)),
                    )
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, res)
                return

            # ------------------------------------------------ v2 Live Co-Translation Session API
            if path.startswith("/api/session/"):
                if srv.live is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"{path} requires --mode live."})
                    return

                if path == "/api/session/start":
                    source_text = str(body.get("source_text", "")).strip("\r\n")
                    if not source_text.strip():
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'source_text'."})
                        return
                    res = srv.live.sessions.start(
                        source_text=source_text,
                        src_lang=str(body.get("src_lang", "Spanish")),
                        tgt_lang=str(body.get("tgt_lang", "English")),
                        global_coords=body.get("global_coords"),
                        lookahead_tokens=int(body.get("lookahead_tokens", 9)),
                        max_branches=int(body.get("max_branches", 6)),
                    )
                    self._send_json(HTTPStatus.OK, res)
                    return

                session_id = str(body.get("session_id", "")).strip()
                if not session_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'session_id'."})
                    return

                try:
                    if path == "/api/session/step":
                        res = srv.live.sessions.step(
                            session_id,
                            mode=str(body.get("mode", "word")),
                            branch_index=body.get("branch_index"),
                            commit_full_branch=bool(body.get("commit_full_branch", False)),
                            custom_text=body.get("custom_text"),
                            custom_token_ids=body.get("custom_token_ids"),
                            source_tag=body.get("source_tag"),
                        )
                    elif path == "/api/session/rewind":
                        res = srv.live.sessions.rewind(session_id, int(body.get("to_token_index", 0)))
                    elif path == "/api/session/checkout":
                        node_id = str(body.get("node_id", "")).strip()
                        if not node_id:
                            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'node_id'."})
                            return
                        res = srv.live.sessions.checkout(session_id, node_id)
                    elif path == "/api/session/sliders":
                        res = srv.live.sessions.set_sliders(session_id, body.get("global_coords") or {})
                    elif path == "/api/session/compass/probe":
                        res = srv.live.sessions.probe_compass(
                            session_id,
                            n_samples=int(body.get("n_samples", 10)),
                            lookahead_tokens=int(body.get("lookahead_tokens", 12)),
                            seed=int(body.get("seed", 0)),
                        )
                    elif path == "/api/session/compass/steer":
                        res = srv.live.sessions.steer_compass(
                            session_id,
                            x=float(body.get("x", 0.0)),
                            y=float(body.get("y", 0.0)),
                            max_new=int(body.get("max_new", 16)),
                        )
                    else:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})
                        return
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, res)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: str = "replay",
    model_key: str = "tiny",
    layer: int | None = None,
    results_dir: str | Path | None = None,
    global_axes_path: str | Path | None = None,
    fork_head_path: str | Path | None = None,
    reft_path: str | Path | None = None,
    lora_path: str | Path | None = None,
    replay_engine: ReplayEngine | None = None,
    live_engine: LiveTranslationEngine | None = None,
    quiet: bool = False,
) -> WorkbenchServer:
    """Create a configured ThreadingHTTPServer ready for `.serve_forever()`."""
    replay = replay_engine if replay_engine is not None else ReplayEngine(results_dir)
    live = live_engine
    if mode == "live" and live is None:
        live = LiveTranslationEngine(
            model_key=model_key,
            layer=layer,
            global_axes_path=global_axes_path,
            fork_head_path=fork_head_path,
            reft_path=reft_path,
            lora_path=lora_path,
        )
    return WorkbenchServer((host, int(port)), mode=mode, replay_engine=replay, live_engine=live, quiet=quiet)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Spectral Translation Workbench HTTP/SSE server & offline exporter")
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--model", default="tiny", help="Model key from ut.common.MODELS (for --mode live)")
    ap.add_argument("--layer", type=int, default=None, help="Residual steering layer (default: n_layers // 2)")
    ap.add_argument("--results-dir", default=None, help="Saved explorer results directory for replay mode")
    ap.add_argument("--global-axes", default=None, help="Optional global tone axes file (.npz/.pt/.json)")
    ap.add_argument("--fork-head", default=None, help="Optional trained ForkHead checkpoint (.pt)")
    ap.add_argument("--reft", default=None, help="Optional trained SpectralReFT checkpoint (.pt)")
    ap.add_argument("--lora", default=None, help="Optional trained multi-reference LoRA checkpoint (.pt)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--export-offline", default=None, help="Write standalone offline HTML bundle and exit")
    args = ap.parse_args(argv)

    if args.export_offline:
        out = build_offline_html(args.export_offline, results_dir=args.results_dir)
        print(f"[workbench] exported offline bundle -> {out} ({out.stat().st_size / 1024:.1f} KiB)", flush=True)
        return

    srv = create_server(
        host=args.host,
        port=args.port,
        mode=args.mode,
        model_key=args.model,
        layer=args.layer,
        results_dir=args.results_dir,
        global_axes_path=args.global_axes,
        fork_head_path=args.fork_head,
        reft_path=args.reft,
        lora_path=args.lora,
    )
    bound_host, bound_port = srv.server_address
    print(f"[workbench] mode={args.mode} listening on http://{bound_host}:{bound_port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if srv.live is not None:
            srv.live.close()


if __name__ == "__main__":
    main()
