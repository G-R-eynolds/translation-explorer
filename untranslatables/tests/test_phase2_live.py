"""CPU tests for Phase 2 live engine, HTTP/SSE server, and offline HTML bundle.

Run from untranslatables/:
    .venv-cpu/bin/python -m pytest -q tests/test_phase2_live.py
"""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from explorer.live_engine import DEFAULT_RESULTS_DIR, LiveTranslationEngine, ReplayEngine
from explorer.server import build_offline_html, create_server


@pytest.fixture(scope="module")
def tiny_model():
    from explorer.frame import load
    return load("tiny")


@pytest.fixture(scope="module")
def replay_engine():
    return ReplayEngine(DEFAULT_RESULTS_DIR)


# ------------------------------------------------------------------ ReplayEngine tests
def test_replay_loads_all_four_texts(replay_engine):
    texts = replay_engine.list_texts()
    ids = {t["id"] for t in texts}
    assert {"duende_flamenco", "duende_bare", "duende_cuento", "becquer_xxi"} <= ids
    required_keys = {"id", "text", "n_samples", "n_readings", "fork_t", "fork_token", "entropy_bits"}
    for t in texts:
        assert required_keys <= set(t.keys())
        assert t["n_samples"] == 64
        assert t["n_readings"] >= 2
        assert t["entropy_bits"] > 0


def test_replay_duende_flamenco_bundle_and_steering(replay_engine):
    bundle = replay_engine.get_text_bundle("duende_flamenco")
    assert bundle["n_samples"] == 64
    assert bundle["n_readings"] == 5
    assert len(bundle["readings"]) == 5
    assert len(bundle["dots"]) == 64
    assert bundle["fork_t"] == 7
    assert len(bundle["ref_tokens"]) == len(bundle["entropy"]) == len(bundle["fisher_top"])
    assert 0.0 < bundle["variance_explained"] <= 1.0

    # Moving to every reading anchor must shift dominant reading to that anchor's reading
    steered_hits = 0
    for rd in bundle["readings"]:
        rid = rd["id"]
        ax, ay = rd["anchor"]
        res = replay_engine.steer_replay("duende_flamenco", ax, ay)
        assert res["dominant_reading"] == rid
        assert isinstance(res["text"], str) and len(res["text"].strip()) > 0
        assert np.isclose(sum(res["shares"].values()), 1.0, atol=1e-2)
        assert res["shares"][str(rid)] == max(res["shares"].values())
        if res["source"] == "steered":
            steered_hits += 1
    # Top-3 steered readings (1, 3, 2) must pull real trajectories from steering/duende_flamenco.jsonl
    assert steered_hits >= 3

    # Exact origin returns valid baseline translation
    orig = replay_engine.steer_replay("duende_flamenco", 0.0, 0.0)
    assert isinstance(orig["text"], str) and len(orig["text"]) > 0


# ------------------------------------------------------------------ LiveTranslationEngine tests
def test_live_stream_probe_and_steer(tiny_model, tmp_path):
    tok, model = tiny_model
    d = model.config.hidden_size
    axes_path = tmp_path / "global_axes.npz"
    rng = np.random.default_rng(7)
    np.savez(
        axes_path,
        literal_idiomatic=(rng.normal(size=d) * 0.01).astype(np.float32),
        plain_poetic=(rng.normal(size=d) * 0.01).astype(np.float32),
    )

    eng = LiveTranslationEngine(model_key="tiny", global_axes_path=axes_path, tok=tok, model=model)
    try:
        # 1. Stream with entropy & optional Fisher power iteration
        streamed = list(
            eng.stream_translate(
                "Me gusta mucho el duende.",
                max_new=6,
                entropy_threshold=1.5,
                compute_fisher=True,
                fisher_iters=2,
            )
        )
        assert 1 <= len(streamed) <= 6
        first = streamed[0]
        for key in ("t", "token_id", "token", "text_so_far", "entropy", "fisher_top", "is_fork", "top_tokens"):
            assert key in first
        assert first["entropy"] >= 0.0
        assert first["fisher_top"] is not None and first["fisher_top"] >= 0.0
        assert len(first["top_tokens"]) >= 1

        # Also verify fast path with compute_fisher=False
        streamed_fast = list(eng.stream_translate("Me gusta mucho el duende.", max_new=4, compute_fisher=False))
        assert len(streamed_fast) >= 1
        assert streamed_fast[0]["fisher_top"] is None
        assert streamed_fast[0]["token_id"] == first["token_id"]

        # 2. Probe fork at prefix
        prefix_ids = [item["token_id"] for item in streamed[:2]]
        fork = eng.probe_fork(
            "Me gusta mucho el duende.",
            prefix_token_ids=prefix_ids,
            n_samples=8,
            lookahead_tokens=6,
            temperature=1.0,
            seed=0,
        )
        assert fork["fork_id"] in eng.forks
        assert len(fork["readings"]) >= 2
        assert len(fork["dots"]) == 8
        assert len(fork["samples"]) == 8
        assert 0.0 <= fork["variance_explained"] <= 1.0001
        # Plane axes must be orthonormal
        axes = eng.forks[fork["fork_id"]]["axes"]
        assert axes.shape == (2, d)
        assert np.allclose(axes @ axes.T, np.eye(2), atol=1e-5)

        # 3. Steer at (x, y) with global coords
        st = eng.steer_at(
            fork["fork_id"],
            x=0.75,
            y=-0.25,
            global_coords={"literal_idiomatic": 0.5, "plain_poetic": -0.25},
            max_new=6,
            temperature=0.0,
        )
        for key in ("text", "continuation", "x", "y", "shares", "tokens"):
            assert key in st
        assert st["global_coords_applied"] is True
        assert np.isclose(sum(st["shares"].values()), 1.0, atol=1e-2)
    finally:
        eng.close()


# ------------------------------------------------------------------ HTTP / SSE Server tests
def _http_json(url: str, payload: dict | None = None) -> tuple[int, dict | list]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def test_http_and_sse_server_ephemeral_port(tiny_model, replay_engine):
    tok, model = tiny_model
    live_eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    srv = create_server(
        host="127.0.0.1",
        port=0,
        mode="live",
        replay_engine=replay_engine,
        live_engine=live_eng,
        quiet=True,
    )
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # GET /
        with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Spectral Translation" in html

        # GET /api/status
        code, st = _http_json(f"{base}/api/status")
        assert code == 200
        assert st["mode"] == "live"
        assert st["model_key"] == "tiny"
        assert st["replay_texts"] >= 4

        # GET /api/replay/texts
        code, txts = _http_json(f"{base}/api/replay/texts")
        assert code == 200
        assert len(txts) >= 4

        # GET /api/replay/duende_flamenco
        code, bundle = _http_json(f"{base}/api/replay/duende_flamenco")
        assert code == 200
        assert bundle["fork_t"] == 7 and bundle["n_readings"] == 5

        # 404 on unknown text
        code, err = _http_json(f"{base}/api/replay/does_not_exist")
        assert code == 404 and "error" in err

        # POST /api/replay/steer
        rd_target = bundle["readings"][1]
        ax, ay = rd_target["anchor"]
        code, steered_rep = _http_json(f"{base}/api/replay/steer", {"text_id": "duende_flamenco", "x": ax, "y": ay})
        assert code == 200
        assert steered_rep["dominant_reading"] == rd_target["id"]

        # GET /api/stream (SSE)
        qs = urllib.parse.urlencode({"text": "Me gusta mucho el duende.", "max_new": "5"})
        with urllib.request.urlopen(f"{base}/api/stream?{qs}", timeout=15) as resp:
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8")
        events = [blk for blk in raw.strip().split("\n\n") if blk.strip()]
        tok_events = []
        saw_done = False
        for blk in events:
            ev_name, ev_data = None, None
            for line in blk.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    ev_data = json.loads(line.split(":", 1)[1].strip())
            if ev_name == "token":
                tok_events.append(ev_data)
            elif ev_name == "done":
                saw_done = True
        assert 1 <= len(tok_events) <= 5
        assert saw_done
        assert "entropy" in tok_events[0] and "token" in tok_events[0]

        # POST /api/probe_fork & POST /api/steer
        code, fork = _http_json(
            f"{base}/api/probe_fork",
            {
                "source_text": "Me gusta mucho el duende.",
                "prefix_token_ids": [tok_events[0]["token_id"]],
                "n_samples": 6,
                "lookahead_tokens": 5,
            },
        )
        assert code == 200
        assert "fork_id" in fork and len(fork["readings"]) >= 2

        code, live_st = _http_json(
            f"{base}/api/steer",
            {
                "fork_id": fork["fork_id"],
                "x": 0.5,
                "y": 0.25,
                "global_coords": {"literal_idiomatic": 0.2},
                "max_new": 5,
            },
        )
        assert code == 200
        assert "text" in live_st and "shares" in live_st
    finally:
        srv.shutdown()
        srv.server_close()
        live_eng.close()


# ------------------------------------------------------------------ Offline HTML bundle tests
def test_build_offline_html(tmp_path):
    out = build_offline_html(tmp_path / "workbench_offline.html", results_dir=DEFAULT_RESULTS_DIR)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "window.__OFFLINE_DATA__" in html
    m = re.search(r"window\.__OFFLINE_DATA__\s*=\s*(\{.*?\});</script>", html, flags=re.DOTALL)
    assert m is not None, "Embedded __OFFLINE_DATA__ JSON payload not found"
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["mode"] == "offline"
    assert {"duende_flamenco", "duende_bare", "duende_cuento", "becquer_xxi"} <= {t["id"] for t in data["texts"]}
    assert data["bundles"]["duende_flamenco"]["n_readings"] == 5
    assert data["bundles"]["duende_flamenco"]["fork_t"] == 7
