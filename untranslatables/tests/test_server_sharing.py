"""Tests for auto-artifact discovery and optional --passcode sharing gate in explorer.server."""
from __future__ import annotations

import http.cookiejar
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from explorer.live_engine import LiveTranslationEngine
from explorer.server import create_server, discover_model_artifacts


def test_discover_model_artifacts_finds_trained_gemma3_checkpoints():
    found = discover_model_artifacts("g3_27b_pt")
    assert found["fork_head_path"] is not None and found["fork_head_path"].name == "fork_head.pt"
    assert found["reft_path"] is not None and found["reft_path"].name == "spectral_reft.pt"
    assert found["global_axes_path"] is not None and found["global_axes_path"].name == "global_axes.npz"


def test_server_passcode_gate_and_cookie_flow():
    eng = LiveTranslationEngine(model_key="tiny")
    srv = create_server(host="127.0.0.1", port=0, mode="live", live_engine=eng, quiet=True, passcode="duende42")
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"

        # 1. Unauthenticated / and /api/status return 401
        for path in ("/", "/api/status"):
            try:
                urllib.request.urlopen(f"{base}{path}", timeout=5)
                raise AssertionError(f"Expected 401 on {path}")
            except urllib.error.HTTPError as e:
                assert e.code == 401

        # 2. Wrong ?key= returns 401
        try:
            urllib.request.urlopen(f"{base}/?key=wrong", timeout=5)
            raise AssertionError("Expected 401 for wrong passcode")
        except urllib.error.HTTPError as e:
            assert e.code == 401

        # 3. Valid ?key= sets te_key cookie and allows subsequent /api/status without ?key=
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        with opener.open(f"{base}/?key=duende42", timeout=5) as r:
            assert r.status == 200
            assert "Translation" in r.read().decode("utf-8")

        with opener.open(f"{base}/api/status", timeout=5) as r:
            assert r.status == 200
            st = json.loads(r.read().decode("utf-8"))
            assert st["mode"] == "live"
    finally:
        srv.shutdown()
        srv.server_close()
        eng.close()
