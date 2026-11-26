"""Shared pytest fixtures, HTTP/SSE client, and markers for Translation Explorer E2E tests."""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from .browser_driver import (
    ensure_playwright_importable,
    launch_headless_browser,
    new_isolated_page,
)


def pytest_configure(config: pytest.Config) -> None:
    """Enforce offline Hugging Face Hub mode and register E2E markers."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config.addinivalue_line("markers", "tier1: Tier 1 isolated happy-path feature E2E tests")
    config.addinivalue_line("markers", "tier2: Tier 2 boundary, negative, and error-handling E2E tests")
    config.addinivalue_line("markers", "tier3: Tier 3 cross-feature pairwise combination E2E tests")
    config.addinivalue_line("markers", "tier4: Tier 4 full real-world application workflow E2E tests")
    config.addinivalue_line("markers", "browser: Headless Chrome / Playwright browser UI E2E tests")


class HttpClient:
    """Pure-stdlib HTTP & SSE client that never raises on 4xx/5xx responses."""

    def request(
        self,
        method: str,
        url: str,
        json_body: Any = None,
        raw_body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, dict[str, str], Any]:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"

        req_headers = {"Connection": "close"}
        if headers:
            req_headers.update(headers)

        body_bytes: bytes | None = None
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif raw_body is not None:
            body_bytes = raw_body.encode("utf-8") if isinstance(raw_body, str) else bytes(raw_body)

        if body_bytes is not None:
            req_headers.setdefault("Content-Length", str(len(body_bytes)))

        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request(method.upper(), target, body=body_bytes, headers=req_headers)
            resp = conn.getresponse()
            status = resp.status
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            raw = resp.read()
        finally:
            conn.close()

        text = raw.decode("utf-8", errors="replace")
        ctype = resp_headers.get("content-type", "").lower()
        if "application/json" in ctype:
            try:
                return status, resp_headers, json.loads(text)
            except json.JSONDecodeError:
                return status, resp_headers, text
        stripped = text.strip()
        if stripped and stripped[0] in ("{", "["):
            try:
                return status, resp_headers, json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return status, resp_headers, text

    def get(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30.0):
        return self.request("GET", url, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        json_body: Any = None,
        raw_body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        return self.request("POST", url, json_body=json_body, raw_body=raw_body, headers=headers, timeout=timeout)

    def delete(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30.0):
        return self.request("DELETE", url, headers=headers, timeout=timeout)

    def options(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30.0):
        return self.request("OPTIONS", url, headers=headers, timeout=timeout)

    def sse_stream(
        self,
        url: str,
        timeout: float = 30.0,
        max_events: int | None = None,
    ) -> tuple[int, dict[str, str], Any]:
        """Connect to an SSE endpoint, capturing HTTP status & headers BEFORE streaming frames.

        If the server responds with a non-200 status or non-SSE content type (e.g.
        pre-header 400/404 JSON error on ``/api/session/glide``), returns
        ``(status, headers, parsed_json_or_text)``. Otherwise returns
        ``(status, headers, list[{"event": str, "data": Any}])``.
        """
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"

        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request(
                "GET",
                target,
                headers={"Accept": "text/event-stream", "Connection": "close"},
            )
            resp = conn.getresponse()
            status = resp.status
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            ctype = resp_headers.get("content-type", "").lower()

            if status != 200 or "text/event-stream" not in ctype:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    return status, resp_headers, json.loads(raw)
                except json.JSONDecodeError:
                    return status, resp_headers, raw

            events: list[dict[str, Any]] = []
            cur_event: str | None = None
            cur_data_lines: list[str] = []

            def _flush() -> None:
                nonlocal cur_event, cur_data_lines
                if cur_event is None and not cur_data_lines:
                    return
                raw_data = "\n".join(cur_data_lines)
                try:
                    parsed_data: Any = json.loads(raw_data)
                except json.JSONDecodeError:
                    parsed_data = raw_data
                events.append({"event": cur_event or "message", "data": parsed_data})
                cur_event = None
                cur_data_lines = []

            while True:
                raw_line = resp.readline()
                if not raw_line:
                    _flush()
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    _flush()
                    if max_events is not None and len(events) >= max_events:
                        break
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    cur_event = line[6:].strip()
                elif line.startswith("data:"):
                    cur_data_lines.append(line[5:].lstrip(" "))

            return status, resp_headers, events
        finally:
            conn.close()


@pytest.fixture(scope="function")
def http_client() -> HttpClient:
    return HttpClient()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # untranslatables/tests/e2e/conftest.py -> repo root is 3 parents up from untranslatables
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def g3_4b_results_dir(repo_root: Path) -> Path:
    p = repo_root / "untranslatables" / "results" / "explorer" / "g3_4b_pt"
    assert p.is_dir(), f"Missing real saved fixture directory: {p}"
    return p


@pytest.fixture(scope="session")
def tiny_model():
    """Load the locally cached offline ``tiny`` model once per test session."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from explorer.frame import load

    tok, model = load("tiny")
    return tok, model


@pytest.fixture(scope="session")
def replay_engine(g3_4b_results_dir: Path):
    from explorer.live_engine import ReplayEngine

    return ReplayEngine(g3_4b_results_dir)


@pytest.fixture(scope="session")
def synthetic_axes_path(tmp_path_factory: pytest.TempPathFactory, tiny_model) -> Path:
    """Create a deterministic 4-axis GlobalAxisBank .npz matching tiny's hidden_size=16."""
    from explorer.global_axes import GlobalAxisBank
    from ut import arch

    _, model = tiny_model
    d = arch.hidden_size(model)
    n_layers = arch.n_layers(model)
    layers = list(range(n_layers))

    rng = np.random.default_rng(42)
    canonical_axes = [
        ("literal_vs_idiomatic", {"neg": "Literal / word-for-word", "pos": "Idiomatic / natural"}),
        ("plain_vs_poetic", {"neg": "Plain / direct", "pos": "Poetic / elevated"}),
        ("foreignising_vs_domesticating", {"neg": "Foreignising", "pos": "Domesticating"}),
        ("concise_vs_glossed", {"neg": "Concise", "pos": "Glossed / explanatory"}),
    ]
    vectors: dict[str, dict[int, np.ndarray]] = {}
    scales: dict[str, float] = {}
    descriptions: dict[str, dict[str, str]] = {}
    for name, desc in canonical_axes:
        lmap: dict[int, np.ndarray] = {}
        for l_idx in layers:
            v = rng.standard_normal(d).astype(np.float64)
            v /= max(float(np.linalg.norm(v)), 1e-12)
            lmap[l_idx] = v
        vectors[name] = lmap
        scales[name] = 0.5
        descriptions[name] = desc

    bank = GlobalAxisBank(
        vectors=vectors,
        scales=scales,
        descriptions=descriptions,
        meta={"model": "tiny", "layers": layers, "synthetic_e2e": True},
    )
    out_dir = tmp_path_factory.mktemp("e2e_axes")
    return bank.save(out_dir / "tiny_global_axes.npz")


@pytest.fixture(scope="session")
def live_server(
    tiny_model,
    g3_4b_results_dir: Path,
    synthetic_axes_path: Path,
) -> Iterator[dict[str, Any]]:
    """Session-scoped live WorkbenchServer backed by ``tiny`` + real ``g3_4b_pt`` replay."""
    from explorer.live_engine import LiveTranslationEngine, ReplayEngine
    from explorer.server import WorkbenchServer

    tok, model = tiny_model
    live_eng = LiveTranslationEngine(
        model_key="tiny",
        global_axes_path=synthetic_axes_path,
        tok=tok,
        model=model,
    )
    rep_eng = ReplayEngine(g3_4b_results_dir)
    srv = WorkbenchServer(
        ("127.0.0.1", 0),
        mode="live",
        replay_engine=rep_eng,
        live_engine=live_eng,
        quiet=True,
    )
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"http://127.0.0.1:{port}",
            "port": port,
            "server": srv,
            "engine": live_eng,
            "replay": rep_eng,
        }
    finally:
        with contextlib.suppress(Exception):
            srv.shutdown()
        with contextlib.suppress(Exception):
            srv.server_close()
        with contextlib.suppress(Exception):
            live_eng.close()
        thread.join(timeout=5.0)


@pytest.fixture(scope="session")
def replay_server(replay_engine) -> Iterator[dict[str, Any]]:
    """Session-scoped replay-only WorkbenchServer (``live=None``)."""
    from explorer.server import WorkbenchServer

    srv = WorkbenchServer(
        ("127.0.0.1", 0),
        mode="replay",
        replay_engine=replay_engine,
        live_engine=None,
        quiet=True,
    )
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"http://127.0.0.1:{port}",
            "port": port,
            "server": srv,
            "replay": replay_engine,
        }
    finally:
        with contextlib.suppress(Exception):
            srv.shutdown()
        with contextlib.suppress(Exception):
            srv.server_close()
        thread.join(timeout=5.0)


@pytest.fixture(scope="session")
def headless_browser():
    """Single session-scoped headless Google Chrome instance via Playwright."""
    sync_api = ensure_playwright_importable()
    pw = sync_api.sync_playwright().start()
    browser = launch_headless_browser(pw)
    try:
        yield browser
    finally:
        with contextlib.suppress(Exception):
            browser.close()
        with contextlib.suppress(Exception):
            pw.stop()


@pytest.fixture(scope="function")
def browser_page(headless_browser):
    """Function-scoped isolated page yielding ``(page, page_errors, console_errors)``."""
    with new_isolated_page(headless_browser) as (_ctx, page, page_errors, console_errors, _ext):
        yield page, page_errors, console_errors


@pytest.fixture(scope="function")
def browser_page_with_requests(headless_browser):
    """Function-scoped isolated page also exposing blocked external request URLs."""
    with new_isolated_page(headless_browser) as (_ctx, page, page_errors, console_errors, ext_reqs):
        yield page, page_errors, console_errors, ext_reqs


@pytest.fixture(scope="session")
def single_model_offline_html(
    tmp_path_factory: pytest.TempPathFactory,
    g3_4b_results_dir: Path,
) -> Path:
    """Single-model standalone offline HTML built via ``build_offline_html``."""
    from explorer.server import build_offline_html

    out_dir = tmp_path_factory.mktemp("e2e_offline_single")
    out_file = out_dir / "single_offline.html"
    return Path(build_offline_html(out_file, results_dir=g3_4b_results_dir))
