"""Tier 2 E2E Tests: Boundary, negative, error-code, concurrency, XSS & keyboard edge cases.

Covers FA1-FA5 + Headless Chrome UI edge cases strictly via public interfaces.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from .browser_driver import drag_svg_point

pytestmark = pytest.mark.tier2


def _start_session(http_client, base_url: str, source_text: str, **kwargs) -> dict[str, Any]:
    payload = {
        "source_text": source_text,
        "lookahead_tokens": kwargs.pop("lookahead_tokens", 5),
        "max_branches": kwargs.pop("max_branches", 4),
        **kwargs,
    }
    st, _, body = http_client.post(f"{base_url}/api/session/start", json_body=payload)
    assert st == 200, f"start failed ({st}): {body}"
    return body


def _cleanup_session(http_client, base_url: str, sid: str) -> None:
    http_client.delete(f"{base_url}/api/session/{sid}")


# ===========================================================================
# FA1 Boundaries — Validation, 400/404, Clamping, Eviction & Mode Guards
# ===========================================================================


def test_fa1_invalid_json_and_non_object_body_returns_400(live_server, http_client):
    """Malformed JSON, non-object JSON arrays, and missing required fields return HTTP 400 with {'error': ...}."""
    base = live_server["url"]
    cases = [
        ("POST", f"{base}/api/session/start", None, b"{bad json"),
        ("POST", f"{base}/api/session/start", [1, 2, 3], None),
        ("POST", f"{base}/api/session/start", {"source_text": "   \n\t  "}, None),
        ("POST", f"{base}/api/session/step", {"mode": "word"}, None),
        ("POST", f"{base}/api/session/checkout", {"session_id": "sess_abc"}, None),
        ("POST", f"{base}/api/session/redo", {"mode": "word"}, None),
    ]
    for method, url, jbody, rbody in cases:
        st, _, body = http_client.request(
            method,
            url,
            json_body=jbody,
            raw_body=rbody,
            headers={"Content-Type": "application/json"},
        )
        assert st == 400, f"{method} {url} with {jbody or rbody!r} expected 400, got {st}: {body}"
        assert isinstance(body, dict) and "error" in body


def test_fa1_unknown_session_and_node_ids_return_404(live_server, http_client):
    """Unknown session_id across all session routes and unknown node_id on checkout return HTTP 404."""
    base = live_server["url"]
    missing = "sess_does_not_exist_999"

    for method, path, payload in (
        ("GET", f"/api/session/{missing}", None),
        ("DELETE", f"/api/session/{missing}", None),
        ("POST", "/api/session/step", {"session_id": missing}),
        ("POST", "/api/session/rewind", {"session_id": missing, "to_token_index": 0}),
        ("POST", "/api/session/redo", {"session_id": missing, "mode": "word"}),
        ("POST", "/api/session/checkout", {"session_id": missing, "node_id": "n0"}),
        ("POST", "/api/session/sliders", {"session_id": missing, "global_coords": {}}),
        ("POST", "/api/session/compass/probe", {"session_id": missing}),
        ("POST", "/api/session/compass/steer", {"session_id": missing, "x": 0.0, "y": 0.0}),
    ):
        st, _, body = http_client.request(method, f"{base}{path}", json_body=payload)
        assert st == 404, f"{method} {path} expected 404, got {st}: {body}"
        assert isinstance(body, dict) and "error" in body

    # Unknown node_id on a valid session -> 404
    s = _start_session(http_client, base, "Hola mundo.")
    sid = s["session_id"]
    try:
        st, _, body = http_client.post(
            f"{base}/api/session/checkout",
            json_body={"session_id": sid, "node_id": "n_nonexistent"},
        )
        assert st == 404, f"checkout(unknown node) expected 404, got {st}: {body}"
        assert isinstance(body, dict) and "error" in body
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_out_of_range_branch_index_and_bad_types_return_400(live_server, http_client):
    """Out-of-range branch_index and wrong scalar types return HTTP 400 (NEVER 500)."""
    base = live_server["url"]
    s = _start_session(http_client, base, "La luna brilla sobre el mar.")
    sid = s["session_id"]
    try:
        bad_requests = [
            ("/api/session/step", {"session_id": sid, "branch_index": 999}),
            ("/api/session/step", {"session_id": sid, "branch_index": -1}),
            ("/api/session/step", {"session_id": sid, "branch_index": "not_an_int"}),
            ("/api/session/rewind", {"session_id": sid, "to_token_index": "not_an_int"}),
            ("/api/session/sliders", {"session_id": sid, "global_coords": {"plain_vs_poetic": "bad_float"}}),
            ("/api/session/compass/steer", {"session_id": sid, "x": "bad_float", "y": 0.0}),
        ]
        for path, payload in bad_requests:
            st, _, body = http_client.post(f"{base}{path}", json_body=payload)
            assert st == 400, f"POST {path} with {payload} expected 400 (not {st}): {body}"
            assert isinstance(body, dict) and "error" in body
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_rewind_clamping_and_redo_at_tip_noop(live_server, http_client):
    """Rewind clamps negative/excessive indices cleanly; redo at tip is a safe no-op returning 200."""
    base = live_server["url"]
    s = _start_session(http_client, base, "El río corre hacia el mar.")
    sid = s["session_id"]
    try:
        for _ in range(3):
            st, _, s = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "token"})
            assert st == 200
        assert s["n_tokens"] == 3

        # Rewind beyond tip clamps to current n_tokens and leaves can_redo False
        st_hi, _, s_hi = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": 9999},
        )
        assert st_hi == 200
        assert s_hi["n_tokens"] == 3
        assert s_hi["can_redo"] is False

        # Redo at tip is a safe 200 no-op
        st_rd, _, s_rd = http_client.post(
            f"{base}/api/session/redo",
            json_body={"session_id": sid, "mode": "word"},
        )
        assert st_rd == 200, f"redo at tip expected 200 no-op, got {st_rd}: {s_rd}"
        assert s_rd["n_tokens"] == 3
        assert s_rd["can_redo"] is False

        # Rewind to negative index clamps to 0 with can_redo True
        st_lo, _, s_lo = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": -5},
        )
        assert st_lo == 200
        assert s_lo["n_tokens"] == 0
        assert s_lo["can_redo"] is True
        assert s_lo["redo_remaining"] == 3
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_max_sessions_fifo_eviction(tiny_model, replay_engine, http_client):
    """Dedicated WorkbenchServer with max_sessions=2 evicts oldest session FIFO when a 3rd starts."""
    from explorer.live_engine import LiveTranslationEngine
    from explorer.server import WorkbenchServer
    from explorer.session import CoTranslationManager

    tok, model = tiny_model
    temp_eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    temp_eng.sessions = CoTranslationManager(temp_eng, max_sessions=2)
    srv = WorkbenchServer(
        ("127.0.0.1", 0),
        mode="live",
        replay_engine=replay_engine,
        live_engine=temp_eng,
        quiet=True,
    )
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        s1 = _start_session(http_client, base, "Primera frase.")["session_id"]
        s2 = _start_session(http_client, base, "Segunda frase.")["session_id"]
        s3 = _start_session(http_client, base, "Tercera frase.")["session_id"]

        st1, _, b1 = http_client.get(f"{base}/api/session/{s1}")
        st2, _, _ = http_client.get(f"{base}/api/session/{s2}")
        st3, _, _ = http_client.get(f"{base}/api/session/{s3}")
        assert st1 == 404, f"Oldest session {s1} should have been evicted (404), got {st1}: {b1}"
        assert st2 == 200, f"Session {s2} should remain alive (200), got {st2}"
        assert st3 == 200, f"Session {s3} should remain alive (200), got {st3}"
    finally:
        with contextlib.suppress(Exception):
            srv.shutdown()
        with contextlib.suppress(Exception):
            srv.server_close()
        with contextlib.suppress(Exception):
            temp_eng.close()
        thread.join(timeout=5.0)


def test_fa1_session_endpoints_rejected_in_replay_mode(replay_server, http_client):
    """Replay-only server rejects live session & SSE glide endpoints with HTTP 400 JSON."""
    base = replay_server["url"]
    for method, path, payload in (
        ("POST", "/api/session/start", {"source_text": "Hola"}),
        ("GET", "/api/session/any_id", None),
        ("DELETE", "/api/session/any_id", None),
        ("POST", "/api/session/redo", {"session_id": "any_id"}),
    ):
        st, _, body = http_client.request(method, f"{base}{path}", json_body=payload)
        assert st == 400, f"Replay server {method} {path} expected 400, got {st}: {body}"
        assert isinstance(body, dict) and "error" in body

    st_g, hdrs_g, body_g = http_client.sse_stream(f"{base}/api/session/glide?session_id=any_id")
    assert st_g == 400
    assert "application/json" in hdrs_g.get("content-type", "")
    assert isinstance(body_g, dict) and "error" in body_g


# ===========================================================================
# FA2 Boundaries — Funky Poems, Split Tags, O(N) Scaling & Oversized Bodies
# ===========================================================================


def test_fa2_long_funky_formatted_poem_roundtrip(live_server, http_client):
    """5-stanza poem with mixed indentation, tabs, triple blank lines, em-dashes & Unicode roundtrips cleanly."""
    base = live_server["url"]
    poem = (
        "  Primera línea con ñ y ácento\n"
        "    —¿Qué susurra el viento?\n"
        "\n\n\n"
        "\t«Segunda estrofa» en cœur y über,\n"
        "  coração de água e sal.\n"
        "\n"
        "Tercera estrofa breve.\n"
        "\n"
        "  Cuarta con ¡exclamación!\n"
        "\n"
        "    Quinta estrofa final."
    )
    s = _start_session(http_client, base, poem)
    sid = s["session_id"]
    try:
        assert s["source_text"] == poem
        aln = s["alignment"]
        stanzas = sorted({ln["stanza"] for ln in aln["source_lines"]})
        assert len(stanzas) >= 5, f"Expected >=5 stanzas parsed, got {stanzas}"

        st1, _, s1 = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "word"})
        assert st1 == 200
        st2, _, s2 = http_client.post(f"{base}/api/session/rewind", json_body={"session_id": sid, "to_token_index": 0})
        assert st2 == 200 and s2["can_redo"] is True
        st3, _, s3 = http_client.post(f"{base}/api/session/redo", json_body={"session_id": sid, "mode": "word"})
        assert st3 == 200 and s3["committed_token_ids"] == s1["committed_token_ids"]
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa2_empty_whitespace_and_single_char_source(live_server, http_client):
    """Whitespace-only source is rejected (400); single-character / single-word source starts cleanly."""
    base = live_server["url"]
    st_empty, _, body_empty = http_client.post(
        f"{base}/api/session/start",
        json_body={"source_text": "   \n\n\t  "},
    )
    assert st_empty == 400 and isinstance(body_empty, dict) and "error" in body_empty

    for tiny_src in ("¡", "Saudade"):
        s = _start_session(http_client, base, tiny_src)
        sid = s["session_id"]
        try:
            assert s["source_text"] == tiny_src
            assert len(s["alignment"]["source_lines"]) == 1
        finally:
            _cleanup_session(http_client, base, sid)


def test_fa2_cut_doc_split_closing_tag_across_tokens(tiny_model):
    """cut_doc strips closing </translation> tag when split across multiple BPE tokens."""
    from explorer.frame import cut_doc

    tok, _ = tiny_model
    tag_ids = tok.encode("</translation>", add_special_tokens=False)
    assert len(tag_ids) >= 2, f"Expected </translation> to span >=2 tokens, got {tag_ids}"
    full = "Dark swallows return\n\nsecond stanza</translation> trailing"
    y = tok.encode(full, add_special_tokens=False)
    cut = cut_doc(tok, y)
    decoded = tok.decode(cut, skip_special_tokens=True)
    assert "second stanza" in decoded
    assert "</" not in decoded
    assert "translation>" not in decoded
    assert "trailing" not in decoded


def test_fa2_cut_doc_long_sequence_linear_scaling(tiny_model):
    """cut_doc handles 500-token multi-stanza continuations in O(N) time (<50 ms)."""
    from explorer.frame import cut_doc

    tok, _ = tiny_model
    chunk = tok.encode("The dark swallows glide across the silent evening sky.\n\n", add_special_tokens=False)
    long_y = (chunk * ((500 // max(len(chunk), 1)) + 1))[:500]
    stop_y = long_y[:480] + tok.encode("</translation> extra words", add_special_tokens=False)

    t0 = time.perf_counter()
    res_no_stop = cut_doc(tok, long_y)
    res_stop = cut_doc(tok, stop_y)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(res_no_stop) == len(long_y)
    assert len(res_stop) <= 480
    assert "</translation>" not in tok.decode(res_stop, skip_special_tokens=True)
    assert elapsed_ms < 50.0, f"cut_doc took {elapsed_ms:.1f} ms for 2x500 tokens (possible O(N^2) regression)"


def test_fa2_cut_doc_empty_and_immediate_stop(tiny_model):
    """cut_doc handles empty token list and immediate </translation> or <source stops."""
    from explorer.frame import cut_doc

    tok, _ = tiny_model
    assert cut_doc(tok, []) == []
    for stop_text in ("</translation>", "<source lang=\"Spanish\">x"):
        ids = tok.encode(stop_text, add_special_tokens=False)
        out = cut_doc(tok, ids)
        assert tok.decode(out, skip_special_tokens=True).strip() == ""


def test_fa2_oversized_json_body_and_invalid_tone_axes_yaml(live_server, http_client, tmp_path: Path):
    """Server rejects oversized JSON payloads with 400/413; load_tone_axes_dataset rejects malformed YAML."""
    from explorer.global_axes import load_tone_axes_dataset

    base = live_server["url"]
    # 5 MiB JSON body should exceed any reasonable API body cap (e.g. 1-2 MiB).
    # Target /api/replay/steer with a valid text_id + oversized padding so pre-M3
    # servers fail fast (returning 200) without blocking live_server._lock on a 5M-char prefill.
    huge_payload = {
        "text_id": "duende_flamenco",
        "x": 0.0,
        "y": 0.0,
        "padding": "a" * (5 * 1024 * 1024),
    }
    st, _, body = http_client.post(f"{base}/api/replay/steer", json_body=huge_payload, timeout=15.0)
    assert st in (400, 413), f"Expected 400 or 413 for 5 MiB JSON body, got {st}: {str(body)[:200]}"
    assert isinstance(body, dict) and "error" in body

    bad_yaml = tmp_path / "bad_tone_axes.yaml"
    bad_yaml.write_text("just_a_key: 123\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tone_axes_dataset(bad_yaml)


# ===========================================================================
# FA3 Boundaries — Pre-Header SSE 400/404, Concurrency, Extreme Coords & KV
# ===========================================================================


def test_fa3_sse_glide_pre_header_validation_returns_400_and_404_json(live_server, http_client):
    """GET /api/session/glide validates session_id & int args BEFORE sending 200 text/event-stream."""
    base = live_server["url"]

    # Missing session_id -> 400 application/json
    st1, h1, b1 = http_client.sse_stream(f"{base}/api/session/glide")
    assert st1 == 400, f"Missing session_id expected 400, got {st1}: {b1}"
    assert "application/json" in h1.get("content-type", "")
    assert isinstance(b1, dict) and "error" in b1

    # Unknown session_id -> 404 application/json (NOT 200 text/event-stream!)
    st2, h2, b2 = http_client.sse_stream(f"{base}/api/session/glide?session_id=sess_missing_404")
    assert st2 == 404, f"Unknown session_id on glide expected pre-header 404, got {st2} ({h2}): {b2}"
    assert "application/json" in h2.get("content-type", "")
    assert isinstance(b2, dict) and "error" in b2

    # Invalid integer max_words -> 400 application/json
    s = _start_session(http_client, base, "El viento sopla.")
    sid = s["session_id"]
    try:
        st3, h3, b3 = http_client.sse_stream(f"{base}/api/session/glide?session_id={sid}&max_words=badint")
        assert st3 == 400, f"Invalid max_words expected 400, got {st3}: {b3}"
        assert "application/json" in h3.get("content-type", "")
        assert isinstance(b3, dict) and "error" in b3
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_concurrent_sliders_and_get_during_active_glide(live_server, http_client):
    """While SSE glide streams multiple words, concurrent GET /api/session/<id> and POST /sliders succeed 200."""
    base = live_server["url"]
    s = _start_session(http_client, base, "Cuando canta Camarón, tiene duende y sombra profunda.")
    sid = s["session_id"]
    try:
        glide_result: dict[str, Any] = {}
        first_step_seen = threading.Event()
        concurrent_done = threading.Event()
        concurrent_results: list[tuple[str, int, Any]] = []

        def _run_glide():
            import http.client as hc
            import urllib.parse

            parsed = urllib.parse.urlsplit(f"{base}/api/session/glide?session_id={sid}&max_words=10&stop_on_fork=0")
            conn = hc.HTTPConnection(parsed.hostname, parsed.port, timeout=30.0)
            try:
                conn.request("GET", f"{parsed.path}?{parsed.query}", headers={"Accept": "text/event-stream", "Connection": "close"})
                resp = conn.getresponse()
                glide_result["status"] = resp.status
                step_count = 0
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    txt = line.decode("utf-8", errors="replace").strip()
                    if txt == "event: step":
                        step_count += 1
                        if step_count == 1:
                            first_step_seen.set()
                            # Pause slightly reading so glide remains active while concurrent client hits server
                            concurrent_done.wait(timeout=10.0)
                glide_result["steps"] = step_count
            finally:
                conn.close()

        t = threading.Thread(target=_run_glide, daemon=True)
        t.start()
        assert first_step_seen.wait(timeout=10.0), "SSE glide never emitted first step event"

        t_start = time.perf_counter()
        st_get, _, b_get = http_client.get(f"{base}/api/session/{sid}", timeout=10.0)
        concurrent_results.append(("GET", st_get, b_get))
        st_sl, _, b_sl = http_client.post(
            f"{base}/api/session/sliders",
            json_body={"session_id": sid, "global_coords": {"plain_vs_poetic": 0.5}},
            timeout=10.0,
        )
        concurrent_results.append(("SLIDERS", st_sl, b_sl))
        concurrent_elapsed = time.perf_counter() - t_start
        concurrent_done.set()
        t.join(timeout=20.0)

        assert glide_result.get("status") == 200
        assert glide_result.get("steps", 0) >= 1
        for label, st_code, body in concurrent_results:
            assert st_code == 200, f"Concurrent {label} during glide failed ({st_code}): {body}"
        assert concurrent_elapsed < 10.0
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_compass_steer_auto_probes_and_handles_extreme_coords(live_server, http_client):
    """POST /api/session/compass/steer auto-probes if unprobed and returns finite shares at extreme coords."""
    base = live_server["url"]
    s = _start_session(http_client, base, "El eco resuena en la montaña.")
    sid = s["session_id"]
    try:
        for x, y in ((10.0, -10.0), (0.0, 0.0)):
            st, _, body = http_client.post(
                f"{base}/api/session/compass/steer",
                json_body={"session_id": sid, "x": x, "y": y, "max_new": 5},
            )
            assert st == 200, f"compass/steer({x},{y}) failed ({st}): {body}"
            shares = body["shares"]
            assert all(math.isfinite(float(v)) for v in shares.values())
            assert math.isclose(sum(float(v) for v in shares.values()), 1.0, rel_tol=1e-2, abs_tol=1e-2)
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_sliders_extreme_empty_alias_and_unknown_keys(live_server, http_client):
    """Sliders accept empty {}, alias keys, unknown axis keys, and extreme values without crashing."""
    base = live_server["url"]
    s = _start_session(http_client, base, "Silencio en la plaza.")
    sid = s["session_id"]
    try:
        for coords in (
            {},
            {"plain_poetic": 1.0},
            {"nonexistent_axis": 0.5, "plain_vs_poetic": 2.5},
            {"plain_vs_poetic": 0.0, "literal_vs_idiomatic": 0.0},
        ):
            st, _, body = http_client.post(
                f"{base}/api/session/sliders",
                json_body={"session_id": sid, "global_coords": coords},
            )
            assert st == 200, f"sliders({coords}) failed ({st}): {body}"
            assert isinstance(body["frontier"]["branches"], list)
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_kv_cache_rewind_parity_on_session(live_server, http_client):
    """Stepping 4 tokens then rewinding to 0 yields identical deterministic next-token choice as fresh session."""
    base = live_server["url"]
    src = "El cielo azul cubre la ciudad."
    s_fresh = _start_session(http_client, base, src)
    sid_fresh = s_fresh["session_id"]
    try:
        st_f1, _, sf1 = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid_fresh, "mode": "token"})
        assert st_f1 == 200
        expected_first_id = sf1["committed_token_ids"][0]

        for _ in range(3):
            st, _, _ = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid_fresh, "mode": "token"})
            assert st == 200

        st_rw, _, srw = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid_fresh, "to_token_index": 0},
        )
        assert st_rw == 200 and srw["n_tokens"] == 0
        # Ghost next-token prediction after KV rewind must match fresh prefill top-1 token
        ghost_ids = srw["frontier"]["ghost"]["full_ids"]
        assert len(ghost_ids) >= 1 and ghost_ids[0] == expected_first_id
    finally:
        _cleanup_session(http_client, base, sid_fresh)


def test_fa3_fork_head_n_dirs_zero_and_scorer_paused_exception_recovery(tiny_model):
    """extract_self_supervised_fork_dataset(n_dirs=0) disables dirs; LiveForkScorer.paused() recovers on exception."""
    from explorer.fork_head import ForkHead, LiveForkScorer, extract_self_supervised_fork_dataset
    from ut import arch

    tok, model = tiny_model
    d = arch.hidden_size(model)
    layer = arch.n_layers(model) // 2

    ds = extract_self_supervised_fork_dataset(
        model,
        tok,
        max_new_per_text=3,
        prompt_style="doc",
        n_dirs=0,
    )
    assert "y_dirs" not in ds or ds["y_dirs"] is None or ds["y_dirs"].shape[1] == 0

    head = ForkHead(d_model=d, hidden_dim=16, n_dirs=2)
    scorer = LiveForkScorer(head, model, layer=layer)
    try:
        inp = torch.tensor([[1, 2]], dtype=torch.long)
        with pytest.raises(RuntimeError, match="boom"):
            with scorer.paused():
                with scorer.paused():
                    with torch.no_grad():
                        model(input_ids=inp)
                raise RuntimeError("boom")
        assert len(scorer.history()) == 0
        # Hook must resume cleanly after exception
        with torch.no_grad():
            model(input_ids=inp)
        assert len(scorer.history()) == 1
    finally:
        scorer.remove()


# ===========================================================================
# FA4 Boundaries — Exact disp_min Boundary, Zero Variance, Labels Fallback & CLI
# ===========================================================================


def _make_two_cluster_cloud(raw_disp: float, n_per_cluster: int = 32, d: int = 16, seed: int = 0) -> np.ndarray:
    """Create unit-normalized two-cluster cloud with exact target raw_disp = 1 - ||mean(X)||."""
    rng = np.random.default_rng(seed)
    cos_t = 1.0 - float(raw_disp)
    sin_t = math.sqrt(max(1.0 - cos_t * cos_t, 0.0))
    X = np.zeros((2 * n_per_cluster, d), dtype=np.float64)
    X[:n_per_cluster, 0] = cos_t
    X[:n_per_cluster, 1] = sin_t
    X[n_per_cluster:, 0] = cos_t
    X[n_per_cluster:, 1] = -sin_t
    # Tiny orthogonal jitter so silhouette is strictly finite & near 1.0
    X[:, 2:] = rng.normal(0.0, 1e-4, size=(2 * n_per_cluster, d - 2))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def test_fa4_readings_disp_gate_boundary_0034_vs_0036():
    """Exact raw_disp=0.034 collapses to k=1 while raw_disp=0.036 splits to k=2; disp_min=0.0 disables gate."""
    from explorer import readings

    X_below = _make_two_cluster_cloud(0.034)
    X_above = _make_two_cluster_cloud(0.036)
    lpt = np.zeros(len(X_below), dtype=np.float64)

    res_below = readings.readings_from(X_below, lpt, drop_q=0.0, disp_min=0.035)
    assert res_below["k"] == 1
    assert res_below["readings"] == [0]
    assert float(res_below["silhouette"]) == 0.0

    res_above = readings.readings_from(X_above, lpt, drop_q=0.0, disp_min=0.035)
    assert res_above["k"] == 2
    assert len(res_above["readings"]) == 2

    res_disabled = readings.readings_from(X_below, lpt, drop_q=0.0, disp_min=0.0)
    assert res_disabled["k"] == 2


def test_fa4_readings_identical_embeddings_and_all_tail_ties():
    """Zero-variance identical embeddings and tied logp values produce valid k=1 without NaN or crash."""
    from explorer import readings

    E = np.zeros((20, 16), dtype=np.float64)
    E[:, 0] = 1.0
    lpt = np.full(20, -1.5, dtype=np.float64)
    res = readings.readings_from(E, lpt, disp_min=0.035)
    assert res["k"] == 1
    assert res["readings"] == [0]
    assert math.isfinite(float(res["silhouette"]))
    assert not any(math.isnan(v) for v in res["shares"].values())


def test_fa4_mapgrid_r2_collinear_and_zero_variance_fallback():
    """spectral_basis handles R==2 with zero within-reading variance and R==1 without LinAlgError/NaN."""
    from explorer import grad_readings

    d = 10
    # R==2 with identical samples inside each reading (zero within-reading residual)
    g0 = np.tile(np.eye(1, d, 0), (6, 1))
    g1 = np.tile(np.eye(1, d, 1), (6, 1))
    G2 = np.vstack([g0, g1])
    res2 = grad_readings.spectral_basis(G2, np.array([0] * 6 + [1] * 6), [0, 1], n_dims=2, whiten=True)
    assert np.all(np.isfinite(res2["axes"]))
    for r in (0, 1):
        assert np.all(np.isfinite(res2["anchors"][r]))
        assert np.all(np.isfinite(res2["dots"][r]))

    # R==1 single reading
    res1 = grad_readings.spectral_basis(g0, np.zeros(6, dtype=int), [0], n_dims=2, whiten=True)
    assert np.all(np.isfinite(res1["axes"]))


def test_fa4_labels_missing_model_and_missing_yaml_fallback(tmp_path: Path):
    """ReplayEngine on an unknown model/text ID still generates non-empty labels and NEVER 'reading 0'."""
    from explorer.live_engine import ReplayEngine

    sim_dir = tmp_path / "unknown_model_xyz"
    (sim_dir / "samples").mkdir(parents=True)
    (sim_dir / "branches").mkdir(parents=True)
    (sim_dir / "steering").mkdir(parents=True)

    tid = "unlisted_poem_999"
    sample_doc = {
        "id": tid,
        "text": "El sol cae sobre el puente.",
        "kind": "hard",
        "k": 2,
        "silhouette": 0.4,
        "entropy_bits": 1.0,
        "readings": [0, 1],
        "shares": {"0": 0.5, "1": 0.5},
        "labels": [0, 0, 1, 1],
        "logp": [-1.0, -1.1, -1.2, -1.3],
        "lens": [5, 5, 5, 5],
        "texts": [
            "The sun falls on the bridge.",
            "The sun drops on the bridge.",
            "The sun sets over the span.",
            "The sun sinks over the arch.",
        ],
        "ys": [[1, 2], [1, 3], [1, 4], [1, 5]],
    }
    (sim_dir / "samples" / f"{tid}.json").write_text(json.dumps(sample_doc), encoding="utf-8")
    (sim_dir / "samples" / f"{tid}_emb.npy").write_bytes(b"")  # unused if numpy load skipped, write valid array instead
    np.save(sim_dir / "samples" / f"{tid}_emb.npy", np.eye(4, 8, dtype=np.float32))
    branch_doc = {
        "id": tid,
        "t": 1,
        "why": "max_fisher",
        "ref": "The sun falls on the bridge.",
        "ref_tokens": ["The", " sun", " falls", " on", " the", " bridge."],
        "entropy": [0.1, 1.5, 0.2, 0.1, 0.1, 0.1],
        "fisher_top": [0.1, 2.0, 0.2, 0.1, 0.1, 0.1],
        "specificity": 0.8,
    }
    (sim_dir / "branches" / f"{tid}.json").write_text(json.dumps(branch_doc), encoding="utf-8")

    eng = ReplayEngine(sim_dir)
    bundle = eng.get_text_bundle(tid)
    bare_re = re.compile(r"^reading\s+\d+$", re.IGNORECASE)
    for rd in bundle["readings"]:
        lbl = str(rd.get("label", "")).strip()
        assert lbl and not bare_re.match(lbl), f"Unlisted text produced bare fallback label {lbl!r}"


def test_fa4_cli_invalid_stage_and_non_cpu_model_error(repo_root: Path):
    """CLI rejects unknown stage with exit code 2 and rejects non-CPU models on CPU cleanly."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "untranslatables")
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""

    p_bad = subprocess.run(
        [sys.executable, "-m", "explorer", "not-a-valid-stage"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p_bad.returncode == 2

    p_gpu = subprocess.run(
        [sys.executable, "-m", "explorer", "axes", "--model", "g3_27b_pt"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p_gpu.returncode != 0
    combined = (p_gpu.stdout + "\n" + p_gpu.stderr).lower()
    assert any(w in combined for w in ("cuda", "gpu", "cpu", "g3_27b_pt")), (
        f"Expected clear GPU/CPU error message, got: {combined[:300]}"
    )


def test_fa4_extract_fork_dataset_missing_branches_raises(tiny_model, tmp_path: Path):
    """extract_fork_dataset raises FileNotFoundError when branches/ directory does not exist."""
    from explorer.fork_head import extract_fork_dataset

    tok, model = tiny_model
    with pytest.raises(FileNotFoundError):
        extract_fork_dataset(model, tok, tmp_path / "does_not_exist")


# ===========================================================================
# FA5 & Browser UI Boundaries — XSS, Empty Dirs, Keyboard, Scroll, AbortCtrl
# ===========================================================================


@pytest.mark.browser
def test_fa5_build_offline_html_escapes_script_injection(
    g3_4b_results_dir: Path, tmp_path: Path, browser_page
):
    """Offline HTML escapes '</script>' payloads in text data so XSS never executes in Chrome."""
    from explorer.server import build_offline_html

    sim_dir = tmp_path / "xss_model"
    shutil.copytree(g3_4b_results_dir, sim_dir)
    sp = sim_dir / "samples" / "duende_flamenco.json"
    doc = json.loads(sp.read_text(encoding="utf-8"))
    payload = "</script><script>window.__XSS__=1;</script>"
    doc["texts"][0] = f"He has soul {payload}"
    sp.write_text(json.dumps(doc), encoding="utf-8")

    out_html = tmp_path / "xss_offline.html"
    build_offline_html(out_html, results_dir=sim_dir)
    raw_html = out_html.read_text(encoding="utf-8")
    assert payload not in raw_html, "Raw unescaped </script> payload leaked into offline HTML"

    page, page_errors, console_errors = browser_page
    page.goto(out_html.resolve().as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelectorAll('#replaySelect option').length === 9", timeout=10000)
    assert page.evaluate("() => window.__XSS__") is None
    assert page_errors == []
    assert console_errors == []


def test_fa5_build_offline_html_empty_or_missing_optional_dirs(g3_4b_results_dir: Path, tmp_path: Path):
    """build_offline_html works cleanly when sessions_dir is empty/None and map_*.json files are absent."""
    from explorer.server import build_offline_html

    nomap_dir = tmp_path / "nomap_model"
    shutil.copytree(g3_4b_results_dir, nomap_dir)
    for mp in nomap_dir.glob("map_*.json"):
        mp.unlink()
    empty_sessions = tmp_path / "empty_sessions"
    empty_sessions.mkdir()

    out_html = tmp_path / "nomap_offline.html"
    build_offline_html(
        out_html,
        model_dirs={"nomap": nomap_dir},
        sessions_dir=empty_sessions,
    )
    html_text = out_html.read_text(encoding="utf-8")
    m = re.search(r"window\.__OFFLINE_DATA__\s*=\s*(\{.*?\})\s*;\s*</script>", html_text, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["models"]["nomap"]["maps"] == {}
    assert data["models"]["nomap"]["sessions"] == []


@pytest.mark.browser
def test_fa5_ui_keyboard_shift_digit_commits_full_branch_via_ev_code(live_server, browser_page):
    """Shift+Digit1 (ev.key='!') MUST trigger /api/session/step with branch_index=0 and commit_full_branch=true."""
    page, page_errors, console_errors = browser_page
    captured_steps: list[dict[str, Any]] = []

    def _on_request(req):
        if req.method == "POST" and req.url.endswith("/api/session/step"):
            try:
                captured_steps.append(json.loads(req.post_data or "{}"))
            except Exception:
                pass

    page.on("request", _on_request)
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)
    # Blur any input so global keyboard handler receives shortcut
    page.evaluate("() => document.activeElement && document.activeElement.blur()")

    # Ensure at least one branch exists in current session frontier
    page.keyboard.down("Shift")
    page.keyboard.press("Digit1")
    page.keyboard.up("Shift")

    page.wait_for_function(
        "() => document.querySelectorAll('#translationCanvas .ctok').length >= 1",
        timeout=5000,
    )
    matching = [
        p for p in captured_steps
        if p.get("branch_index") == 0 and p.get("commit_full_branch") is True
    ]
    assert len(matching) >= 1, (
        f"Shift+Digit1 did not send branch_index=0 & commit_full_branch=true; captured={captured_steps}"
    )
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_fa5_ui_keyboard_space_backspace_arrows_escape(live_server, browser_page):
    """Space/ArrowRight step, Backspace/ArrowLeft rewind, c/s/h open drawers, and Escape closes them."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)
    page.evaluate("() => document.activeElement && document.activeElement.blur()")

    # Space advances
    page.keyboard.press("Space")
    page.wait_for_function("() => document.querySelectorAll('#translationCanvas .ctok').length >= 1", timeout=10000)
    c1 = page.locator("#translationCanvas .ctok").count()

    # ArrowRight advances
    page.keyboard.press("ArrowRight")
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length > {c1}", timeout=10000)
    c2 = page.locator("#translationCanvas .ctok").count()

    # Backspace rewinds
    page.keyboard.press("Backspace")
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length < {c2}", timeout=10000)
    c3 = page.locator("#translationCanvas .ctok").count()

    # If Backspace rewound a single multi-token word all the way to 0, step once so ArrowLeft has >=1 token to rewind
    if c3 == 0:
        page.keyboard.press("Space")
        page.wait_for_function("() => document.querySelectorAll('#translationCanvas .ctok').length >= 1", timeout=10000)
        c3 = page.locator("#translationCanvas .ctok").count()

    # ArrowLeft rewinds further
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length < {c3}", timeout=10000)

    # Open sliders drawer with 's', then close with Escape
    page.keyboard.press("s")
    page.locator("#slidersDrawer").wait_for(state="visible", timeout=5000)
    page.keyboard.press("Escape")
    page.wait_for_function(
        "() => !document.querySelector('#slidersDrawer').classList.contains('open')",
        timeout=5000,
    )
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_fa5_ui_keyboard_shortcuts_suppressed_in_inputs_and_tab_indents_source(live_server, browser_page):
    """Shortcuts do not fire inside #sourceInput/#customWord, and Tab inserts indentation in #sourceInput."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    # Switch to edit mode if source textarea is currently hidden by annotated view
    src_box = page.locator("#sourceInput")
    if not src_box.is_visible():
        edit_btn = page.get_by_role("button", name=re.compile(r"edit", re.I)).first
        if edit_btn.count() > 0:
            edit_btn.click()
            src_box.wait_for(state="visible", timeout=5000)

    src_box.click()
    before_val = src_box.input_value()
    page.keyboard.press("Tab")
    after_val = src_box.input_value()
    assert len(after_val) > len(before_val), "Pressing Tab inside #sourceInput must insert indentation"
    # Verify Tab did NOT trigger glide and Space inside textarea does not step
    page.keyboard.press("Space")
    assert page.locator("#translationCanvas .ctok").count() == 0

    # Custom word input suppresses global shortcuts
    page.locator("#customWord").click()
    page.keyboard.press("Space")
    page.keyboard.press("Backspace")
    assert page.locator("#translationCanvas .ctok").count() == 0
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_fa5_ui_long_poem_canvas_scrollable_and_replay_tokens_pre_wrap(live_server, browser_page):
    """#translationCanvas has bounded max-height + scrollable overflow-y; #replayTokens uses white-space: pre-wrap."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    canvas_style = page.evaluate(
        """() => {
            const cs = getComputedStyle(document.querySelector('#translationCanvas'));
            return { overflowY: cs.overflowY, maxHeight: cs.maxHeight };
        }"""
    )
    assert canvas_style["overflowY"] in ("auto", "scroll"), (
        f"#translationCanvas must scroll for long poems, got overflowY={canvas_style['overflowY']!r}"
    )
    assert canvas_style["maxHeight"] != "none", (
        f"#translationCanvas must have bounded maxHeight, got {canvas_style['maxHeight']!r}"
    )

    page.locator("#btnSwitchMode").click()
    page.locator("#replayView").wait_for(state="visible", timeout=5000)
    replay_ws = page.evaluate("() => getComputedStyle(document.querySelector('#replayTokens')).whiteSpace")
    assert replay_ws == "pre-wrap", f"#replayTokens must use white-space: pre-wrap, got {replay_ws!r}"
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_fa5_ui_compass_rapid_drag_abort_controller_no_stale_overwrite(live_server, browser_page):
    """Rapid compass drags abort/ignore stale in-flight responses so slower response #1 never overwrites #2."""
    page, page_errors, console_errors = browser_page
    steer_call = {"n": 0}

    def _intercept_steer(route):
        steer_call["n"] += 1
        idx = steer_call["n"]
        if idx == 1:
            time.sleep(0.35)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "x": 0.8,
                    "y": 0.8,
                    "preview_text": "STALE_FIRST_RESPONSE",
                    "preview_ids": [10],
                    "first_word_text": "STALE",
                    "first_word_ids": [10],
                    "shares": {"0": 1.0},
                }),
            )
        else:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "x": -0.8,
                    "y": -0.8,
                    "preview_text": "FRESH_LATEST_RESPONSE",
                    "preview_ids": [11],
                    "first_word_text": "FRESH",
                    "first_word_ids": [11],
                    "shares": {"0": 1.0},
                }),
            )

    page.route("**/api/session/compass/steer", _intercept_steer)
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    page.locator("#btnCompass").click()
    page.locator("#compassDrawer").wait_for(state="visible", timeout=10000)
    page.wait_for_function("() => document.querySelectorAll('#compassSvg circle').length >= 1", timeout=20000)

    # Trigger two rapid compass drags
    drag_svg_point(page, "#compassSvg", 0.8, 0.8, steps=1)
    drag_svg_point(page, "#compassSvg", -0.8, -0.8, steps=1)

    # Wait past the 350ms delayed first response
    page.wait_for_timeout(600)
    preview = (page.locator("#steeredPreview").text_content() or "").strip()
    assert "FRESH_LATEST_RESPONSE" in preview, (
        f"Stale compass steer response overwrote latest drag: preview={preview!r}"
    )
    assert "STALE_FIRST_RESPONSE" not in preview
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_fa5_ui_word_click_rewind_and_token_word_toggle_and_offline_extreme_coords(
    live_server, single_model_offline_html: Path, browser_page
):
    """Token/Word toggle changes step mode; clicking mid-word rewinds to word_start_idx; offline extreme coords have no NaN."""
    page, page_errors, console_errors = browser_page
    step_modes: list[str] = []
    rewind_indices: list[int] = []

    def _on_req(req):
        if req.method == "POST" and req.url.endswith("/api/session/step"):
            with contextlib.suppress(Exception):
                step_modes.append(json.loads(req.post_data or "{}").get("mode", ""))
        elif req.method == "POST" and req.url.endswith("/api/session/rewind"):
            with contextlib.suppress(Exception):
                rewind_indices.append(int(json.loads(req.post_data or "{}").get("to_token_index", -1)))

    page.on("request", _on_req)
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    # Toggle to Token mode and step once (avoid matching #btnStep 'Next word')
    toggle = page.locator("button:not(#btnStep):not(#btnCommitSteeredWord)").filter(
        has_text=re.compile(r"token|word\s*/\s*token|mode", re.I)
    ).first
    toggle.wait_for(state="visible", timeout=5000)
    toggle.click()
    page.locator("#btnStep").click()
    page.wait_for_function("() => document.querySelectorAll('#translationCanvas .ctok').length >= 1", timeout=10000)
    assert "token" in step_modes, f"Expected Token/Word toggle to send mode='token', saw modes={step_modes}"

    # Commit a multi-token word and click its trailing subword token -> rewind should snap to word_start_idx
    page.locator("#customWord").fill("untranslatableness")
    page.locator("#customWord").press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('untranslatableness')",
        timeout=10000,
    )
    tok_count = page.locator("#translationCanvas .ctok").count()
    assert tok_count >= 2
    page.locator("#translationCanvas .ctok").nth(tok_count - 1).click()
    page.wait_for_timeout(300)
    assert len(rewind_indices) >= 1
    assert rewind_indices[-1] < tok_count - 1, (
        f"Clicking mid-word token {tok_count - 1} should snap rewind to word_start_idx, got {rewind_indices[-1]}"
    )

    # Verify offline single-reading / control_museo extreme compass coordinates produce no NaN
    page.goto(single_model_offline_html.resolve().as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelectorAll('#replaySelect option').length === 9", timeout=10000)
    page.locator("#replaySelect").select_option("control_museo")
    drag_svg_point(page, "#replayCompass", -1.4, -1.4)
    drag_svg_point(page, "#replayCompass", 1.4, 1.4)
    readings_txt = page.locator("#replayReadings").text_content() or ""
    assert "NaN" not in readings_txt
    assert len((page.locator("#replayOut").text_content() or "").strip()) > 0
    assert page_errors == []
    assert console_errors == []
