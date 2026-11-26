"""Tier 3 E2E Tests: Cross-feature pairwise combinations across FA1-FA5 + Browser UI.

Every test exercises genuine multi-feature state transitions across at least two
feature areas or subsystems without mocking away internal state.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from .browser_driver import drag_svg_point

pytestmark = pytest.mark.tier3


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


def test_comb_fa1_fa3_rewind_sliders_then_diverge_branch(live_server, http_client):
    """FA1 x FA3: Step -> rewind -> adjust global sliders -> diverge into n1 -> checkout n0 preserves suffix."""
    base = live_server["url"]
    s = _start_session(http_client, base, "El viento murmura en la noche oscura.")
    sid = s["session_id"]
    try:
        for _ in range(3):
            st, _, s = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "token"})
            assert st == 200
        n0_ids = list(s["committed_token_ids"])

        st_rw, _, srw = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": 1},
        )
        assert st_rw == 200 and srw["can_redo"] is True

        st_sl, _, ssl = http_client.post(
            f"{base}/api/session/sliders",
            json_body={"session_id": sid, "global_coords": {"plain_vs_poetic": 0.9}},
        )
        assert st_sl == 200
        assert math.isclose(ssl["global_coords"]["plain_vs_poetic"], 0.9, abs_tol=1e-6)

        st_div, _, sdiv = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " poetic_breeze"},
        )
        assert st_div == 200
        assert sdiv["active_node_id"] == "n1"
        hist = {h["id"]: h for h in sdiv["history"]}
        assert hist["n0"]["n_tokens"] == len(n0_ids)
        assert hist["n1"]["parent_id"] == "n0"
        assert hist["n1"]["fork_token_idx"] == 1

        st_co, _, sco = http_client.post(
            f"{base}/api/session/checkout",
            json_body={"session_id": sid, "node_id": "n0"},
        )
        assert st_co == 200
        assert sco["committed_token_ids"] == n0_ids
    finally:
        _cleanup_session(http_client, base, sid)


def test_comb_fa1_fa3_glide_to_pause_then_compass_steer_and_commit(live_server, http_client):
    """FA1 x FA3: SSE glide -> compass probe -> compass steer -> commit steered word -> compass invalidates."""
    base = live_server["url"]
    s = _start_session(http_client, base, "Cuando canta Camarón, tiene duende.")
    sid = s["session_id"]
    try:
        st, hdrs, events = http_client.sse_stream(
            f"{base}/api/session/glide?session_id={sid}&max_words=3&stop_on_fork=0"
        )
        assert st == 200 and "text/event-stream" in hdrs.get("content-type", "")
        paused = [e for e in events if e["event"] == "paused"]
        assert len(paused) == 1
        n_after_glide = paused[0]["data"]["state"]["n_tokens"]
        assert n_after_glide >= 1

        st_p, _, probe = http_client.post(
            f"{base}/api/session/compass/probe",
            json_body={"session_id": sid, "n_samples": 4, "lookahead_tokens": 4, "seed": 0},
        )
        assert st_p == 200
        assert probe["cursor_len"] == n_after_glide

        st_s, _, steered = http_client.post(
            f"{base}/api/session/compass/steer",
            json_body={"session_id": sid, "x": 0.8, "y": 0.2, "max_new": 5},
        )
        assert st_s == 200 and len(steered["first_word_ids"]) >= 1

        st_c, _, committed = http_client.post(
            f"{base}/api/session/step",
            json_body={
                "session_id": sid,
                "custom_token_ids": steered["first_word_ids"],
                "source_tag": "steered",
            },
        )
        assert st_c == 200
        assert committed["n_tokens"] == n_after_glide + len(steered["first_word_ids"])
        assert committed["tokens"][-1]["source"] == "steered"
        assert committed["has_compass"] is False
    finally:
        _cleanup_session(http_client, base, sid)


def test_comb_fa1_fa2_multistanza_poem_rewind_redo_and_alignment_sync(live_server, http_client):
    """FA1 x FA2: Stepping across stanza break updates alignment.active_source_line; rewind/redo keeps it synced."""
    base = live_server["url"]
    poem = (
        "Primera línea del poema.\n"
        "\n"
        "Segunda estrofa en silencio.\n"
        "\n"
        "Tercera estrofa al alba."
    )
    s0 = _start_session(http_client, base, poem)
    sid = s0["session_id"]
    try:
        line_start = s0["alignment"]["active_source_line"]
        st1, _, s1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": "First line of the poem."},
        )
        assert st1 == 200
        idx_before_break = s1["n_tokens"]

        st2, _, s2 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": "\n\nSecond stanza in silence.\n\nThird stanza"},
        )
        assert st2 == 200
        line_after_stanzas = s2["alignment"]["active_source_line"]
        assert line_after_stanzas > line_start, (
            f"active_source_line should advance across stanzas ({line_start} -> {line_after_stanzas})"
        )

        # Rewind before the stanza break
        st_rw, _, srw = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": idx_before_break},
        )
        assert st_rw == 200
        assert srw["can_redo"] is True
        assert srw["alignment"]["active_source_line"] <= line_after_stanzas

        # Redo back across the stanza break
        cur = srw
        for _ in range(30):
            if not cur["can_redo"]:
                break
            st_r, _, cur = http_client.post(
                f"{base}/api/session/redo",
                json_body={"session_id": sid, "mode": "word"},
            )
            assert st_r == 200
        assert cur["committed_token_ids"] == s2["committed_token_ids"]
        assert cur["alignment"]["active_source_line"] == line_after_stanzas
    finally:
        _cleanup_session(http_client, base, sid)


def test_comb_fa1_fa3_forkhead_ndirs2_with_step_rewind_and_paused_scorer(tiny_model, tmp_path: Path):
    """FA1 x FA3: ForkHead(n_dirs=2) + CoTranslationSession step/rewind + zero-pass compass + unpolluted LiveForkScorer."""
    from explorer.fork_head import ForkHead
    from explorer.live_engine import LiveTranslationEngine
    from explorer.session import CoTranslationSession
    from ut import arch

    tok, model = tiny_model
    d = arch.hidden_size(model)
    layer = arch.n_layers(model) // 2

    head = ForkHead(d_model=d, hidden_dim=32, n_dirs=2)
    ckpt = tmp_path / "fh_comb.pt"
    head.save(ckpt, meta={"layer": layer, "recommended_alpha": 1.1})

    eng = LiveTranslationEngine(model_key="tiny", fork_head_path=ckpt, tok=tok, model=model)
    try:
        sess = CoTranslationSession(eng, "Cuando canta Camarón, tiene duende.", lookahead_tokens=4, max_branches=4)
        hist_after_start = len(eng.fork_scorer.history())
        # Lookahead branches must NOT have polluted LiveForkScorer with dozens of extra records
        assert hist_after_start <= 3, (
            f"LiveForkScorer appears unpaused during lookahead (history len={hist_after_start})"
        )

        # Probe compass at cursor 0 with 0 model forward passes
        fwd_calls = 0

        def _count(_m, _i, _o):
            nonlocal fwd_calls
            fwd_calls += 1

        h_cnt = arch.layers(model)[0].register_forward_hook(_count)
        try:
            p0 = sess.probe_compass()
        finally:
            h_cnt.remove()
        assert fwd_calls == 0
        assert p0["source"] == "fork_head"
        assert p0["cursor_len"] == 0
        assert float(p0["probe_ms"]) < 10.0

        # Step 2 tokens, probe again, rewind to 0
        sess.step(mode="token")
        sess.step(mode="token")
        p2 = sess.probe_compass()
        assert p2["source"] == "fork_head"
        assert p2["cursor_len"] == 2
        st_rw = sess.rewind(0)
        assert st_rw["n_tokens"] == 0 and st_rw["can_redo"] is True
    finally:
        eng.close()


def test_comb_fa1_fa2_custom_newline_indent_plus_nested_branch_tree(live_server, http_client):
    """FA1 x FA2: Indented em-dash branch n1 + stanza-break child branch n2 preserve exact whitespace on checkout."""
    base = live_server["url"]
    s = _start_session(http_client, base, "Primera línea\n  —segunda\n\nTercera")
    sid = s["session_id"]
    try:
        st, _, s = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "custom_text": "First line"})
        assert st == 200
        fork1 = s["n_tokens"]
        st, _, s_n0 = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "custom_text": " plain"})
        assert st == 200
        txt_n0 = s_n0["committed_text"]

        # Rewind -> branch n1 with indented em-dash
        http_client.post(f"{base}/api/session/rewind", json_body={"session_id": sid, "to_token_index": fork1})
        st, _, s_n1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": "\n  —only"},
        )
        assert st == 200 and s_n1["active_node_id"] == "n1"
        fork2 = s_n1["n_tokens"]
        st, _, s_n1 = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "custom_text": " wind"})
        assert st == 200
        txt_n1 = s_n1["committed_text"]

        # Rewind inside n1 -> branch n2 with stanza break
        http_client.post(f"{base}/api/session/rewind", json_body={"session_id": sid, "to_token_index": fork2})
        st, _, s_n2 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": "\n\nThird stanza"},
        )
        assert st == 200 and s_n2["active_node_id"] == "n2"
        txt_n2 = s_n2["committed_text"]

        for nid, expected in (("n0", txt_n0), ("n1", txt_n1), ("n2", txt_n2)):
            st_c, _, sc = http_client.post(
                f"{base}/api/session/checkout",
                json_body={"session_id": sid, "node_id": nid},
            )
            assert st_c == 200
            assert sc["committed_text"] == expected
        assert "\n  —only" in txt_n1
        assert "\n\nThird stanza" in txt_n2
    finally:
        _cleanup_session(http_client, base, sid)


def test_comb_fa3_fa1_glide_early_disconnect_then_delete_session(live_server, http_client):
    """FA3 x FA1: Early client disconnect during SSE glide leaves server healthy; DELETE cleans up -> 404."""
    base = live_server["url"]
    s = _start_session(http_client, base, "El mar canta bajo las estrellas.")
    sid = s["session_id"]

    # Stream only 1 event and disconnect abruptly
    st, hdrs, events = http_client.sse_stream(
        f"{base}/api/session/glide?session_id={sid}&max_words=10&stop_on_fork=0",
        max_events=1,
    )
    assert st == 200 and "text/event-stream" in hdrs.get("content-type", "")
    assert len(events) == 1

    # Server stays responsive and DELETE succeeds
    st_del, _, b_del = http_client.delete(f"{base}/api/session/{sid}")
    assert st_del == 200 and b_del.get("deleted") is True

    # Subsequent GET and SSE glide both return 404
    st_g, _, _ = http_client.get(f"{base}/api/session/{sid}")
    assert st_g == 404
    st_gl, h_gl, b_gl = http_client.sse_stream(f"{base}/api/session/glide?session_id={sid}")
    assert st_gl == 404
    assert "application/json" in h_gl.get("content-type", "")
    assert isinstance(b_gl, dict) and "error" in b_gl


def test_comb_fa2_fa3_multistanza_poem_framing_plus_session_step_and_compass(live_server, http_client):
    """FA2 x FA3: Portuguese 2-stanza poem steps and probes compass cleanly using DOC_STOPS + alignment."""
    base = live_server["url"]
    pt_poem = (
        "No meio do caminho tinha uma pedra\n"
        "  tinha uma pedra no meio do caminho\n"
        "\n"
        "Nunca me esquecerei desse acontecimento."
    )
    s = _start_session(http_client, base, pt_poem, src_lang="Portuguese", tgt_lang="English")
    sid = s["session_id"]
    try:
        assert s["src_lang"] == "Portuguese"
        assert len({ln["stanza"] for ln in s["alignment"]["source_lines"]}) >= 2
        st_step, _, s1 = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "word"})
        assert st_step == 200
        st_p, _, probe = http_client.post(
            f"{base}/api/session/compass/probe",
            json_body={"session_id": sid, "n_samples": 4, "lookahead_tokens": 5, "seed": 1},
        )
        assert st_p == 200
        assert probe["cursor_len"] == s1["n_tokens"]
        assert float(probe["alpha_unit"]) > 0.0
    finally:
        _cleanup_session(http_client, base, sid)


def test_comb_fa4_fa5_readings_disp_gate_plus_report_plus_replay_engine(
    g3_4b_results_dir: Path, tmp_path: Path, monkeypatch
):
    """FA4 x FA5: Re-clustering control_museo + duende_flamenco with disp_min=0.035 propagates to report.html & ReplayEngine."""
    import explorer.report as report_mod
    import explorer.run as run_mod
    from explorer import readings
    from explorer.live_engine import ReplayEngine

    sim_dir = tmp_path / "g3_4b_pt"
    shutil.copytree(g3_4b_results_dir, sim_dir)

    # Re-cluster control_museo and duende_flamenco with disp_min=0.035 and write updated sample JSONs
    for tid in ("control_museo", "duende_flamenco"):
        sp = sim_dir / "samples" / f"{tid}.json"
        S = json.loads(sp.read_text(encoding="utf-8"))
        E = np.load(sim_dir / "samples" / f"{tid}_emb.npy")
        lpt = np.array(S["logp"], dtype=np.float64) / np.maximum(np.array(S["lens"], dtype=np.float64), 1.0)
        res = readings.readings_from(E, lpt, disp_min=0.035)
        S.update(
            k=res["k"],
            silhouette=res["silhouette"],
            readings=res["readings"],
            shares={str(k): v for k, v in res["shares"].items()},
            labels=res["labels"],
            entropy_bits=readings.entropy(res["shares"]),
        )
        sp.write_text(json.dumps(S, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(report_mod, "OUT", tmp_path)
    monkeypatch.setattr(run_mod, "OUT", tmp_path)
    html_path = report_mod.run({"model": "g3_4b_pt"})
    html_text = Path(html_path).read_text(encoding="utf-8")
    assert not re.search(r"\breading\s+\d+\b", html_text, re.IGNORECASE)

    rep = ReplayEngine(sim_dir)
    ctrl_bundle = rep.get_text_bundle("control_museo")
    duende_bundle = rep.get_text_bundle("duende_flamenco")
    assert ctrl_bundle["n_readings"] == 1
    assert duende_bundle["n_readings"] >= 2
    steered = rep.steer_replay("duende_flamenco", 0.7, -0.3)
    assert steered["text"] and not re.match(r"^reading\s+\d+$", steered["dominant_label"], re.I)


@pytest.mark.browser
def test_comb_fa5_offline_steering_parity_with_server_replay_api(
    single_model_offline_html: Path, replay_engine, browser_page
):
    """FA5: Offline workbench.html steering updates from steering_points & RBF shares matching ReplayEngine."""
    page, page_errors, console_errors = browser_page
    page.goto(single_model_offline_html.resolve().as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelectorAll('#replaySelect option').length === 9", timeout=10000)
    page.locator("#replaySelect").select_option("duende_flamenco")

    # Drag to two distinct opposite compass coordinates and verify both text & live share bars update
    drag_svg_point(page, "#replayCompass", 0.9, 0.4, extent=1.6)
    page.wait_for_timeout(150)
    out_a = (page.locator("#replayOut").text_content() or "").strip()
    shares_a = (page.locator("#replayReadings").text_content() or "").strip()

    drag_svg_point(page, "#replayCompass", -0.9, -0.5, extent=1.6)
    page.wait_for_timeout(150)
    out_b = (page.locator("#replayOut").text_content() or "").strip()
    shares_b = (page.locator("#replayReadings").text_content() or "").strip()

    assert out_a and out_b
    assert shares_a != shares_b, (
        "Offline compass drag MUST dynamically update live RBF reading share bars (parity with server replay)"
    )

    # Verify offline steer uses steering_points (not only nearest raw dot) by comparing against bundle steering_points
    bundle = replay_engine.get_text_bundle("duende_flamenco")
    valid_steered_texts = {p["text"].strip() for p in bundle["steering_points"]}
    assert out_a in valid_steered_texts or out_b in valid_steered_texts, (
        f"Offline steered output did not come from steering_points: out_a={out_a!r}, out_b={out_b!r}"
    )
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_comb_fa4_fa5_cli_offline_plus_multi_model_browser_switcher(
    repo_root: Path, g3_4b_results_dir: Path, tmp_path: Path, browser_page
):
    """FA4 x FA5: CLI `python -m explorer offline` builds standalone HTML; multi-model switcher toggles models in Chrome."""
    from explorer.server import build_offline_html

    # 1) CLI single-model build into tmp_path
    cli_out = tmp_path / "cli_offline.html"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "untranslatables")
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "explorer",
            "offline",
            "--results-dir",
            str(g3_4b_results_dir),
            "--out",
            str(cli_out),
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"CLI offline failed: {proc.stderr}"
    assert cli_out.is_file() and cli_out.stat().st_size > 10_000

    # 2) Multi-model bundle with distinct text count in second model so switching is directly observable
    m2_dir = tmp_path / "g3_27b_sim"
    shutil.copytree(g3_4b_results_dir, m2_dir)
    # Remove one text in m2 so model switch changes #replaySelect option count (9 -> 8)
    for sub in ("samples/rosalia_dicen.json", "samples/rosalia_dicen_emb.npy", "branches/rosalia_dicen.json"):
        p = m2_dir / sub
        if p.exists():
            p.unlink()

    multi_out = tmp_path / "multi_switcher.html"
    build_offline_html(
        multi_out,
        model_dirs={"g3_4b_pt": g3_4b_results_dir, "g3_27b_sim": m2_dir},
    )

    page, page_errors, console_errors = browser_page
    page.goto(multi_out.resolve().as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelectorAll('#replaySelect option').length >= 8", timeout=10000)

    # Locate the model switcher <select> (distinct from #replaySelect and #srcLang)
    model_select = page.evaluate_handle(
        """() => {
            const selects = Array.from(document.querySelectorAll('select'));
            return selects.find(s => {
                if (s.id === 'replaySelect' || s.id === 'srcLang' || s.id === 'presetSelect') return false;
                const vals = Array.from(s.options).map(o => o.value + ' ' + o.textContent);
                return vals.some(v => v.includes('g3_27b_sim') || v.includes('27b') || v.includes('g3_4b'));
            }) || null;
        }"""
    ).as_element()
    assert model_select is not None, "Multi-model offline HTML must render a visible model switcher <select>"

    count_before = page.locator("#replaySelect option").count()
    model_select.select_option("g3_27b_sim")
    page.wait_for_function(
        f"() => document.querySelectorAll('#replaySelect option').length !== {count_before}",
        timeout=5000,
    )
    assert page.locator("#replaySelect option").count() == 8
    drag_svg_point(page, "#replayCompass", 0.5, 0.5, extent=1.6)
    assert page_errors == []
    assert console_errors == []


def test_comb_fa2_fa3_forkhead_dataset_plus_doc_prompt_style_plus_tone_axes(tiny_model):
    """FA2 x FA3: Self-supervised ForkHead(n_dirs=2) extraction with prompt_style='doc' ingests expanded tone_axes.yaml."""
    from explorer.fork_head import extract_self_supervised_fork_dataset
    from explorer.global_axes import load_tone_axes_dataset

    tok, model = tiny_model
    tone_ds = load_tone_axes_dataset()
    assert len(tone_ds["items"]) > 16
    assert any("\n\n" in str(it.get("source", "")) for it in tone_ds["items"])

    ds = extract_self_supervised_fork_dataset(
        model,
        tok,
        max_new_per_text=4,
        entropy_fork_thresh=0.0,
        prompt_style="doc",
        n_dirs=2,
    )
    assert ds["H"].shape[0] >= 10
    assert "y_dirs" in ds and ds["y_dirs"].shape[1] == 2
    assert "y_dirs_mask" in ds and ds["y_dirs_mask"].any()
    assert float(ds["meta"]["recommended_alpha"]) > 0.0


@pytest.mark.browser
def test_comb_fa1_fa5_ui_live_branch_tree_and_hierarchical_paths_drawer(live_server, browser_page):
    """FA1 x FA5: Building n0 -> n1 -> n2 in Chrome renders hierarchical Paths tree and restores n0 on click."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    # Commit two words on n0
    for word in ("origin", "path"):
        page.locator("#customWord").fill(word)
        page.locator("#customWord").press("Enter")
        page.wait_for_function(
            f"() => (document.querySelector('#translationCanvas')?.textContent || '').includes('{word}')",
            timeout=10000,
        )

    # Rewind 1 word and fork n1
    page.locator("#btnRewind").click()
    page.wait_for_timeout(300)
    page.locator("#customWord").fill("branchOne")
    page.locator("#customWord").press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('branchOne')",
        timeout=10000,
    )

    # Step another word on n1, rewind, and fork n2
    page.locator("#customWord").fill("leafA")
    page.locator("#customWord").press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('leafA')",
        timeout=10000,
    )
    page.locator("#btnRewind").click()
    page.wait_for_timeout(300)
    page.locator("#customWord").fill("branchTwo")
    page.locator("#customWord").press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('branchTwo')",
        timeout=10000,
    )

    # Open Paths drawer and verify 3 nodes with hierarchical indentation / depth nesting
    page.locator("#btnHistory").click()
    page.locator("#historyDrawer").wait_for(state="visible", timeout=5000)
    page.wait_for_function("() => document.querySelectorAll('#historyList .hist-item').length === 3", timeout=5000)

    tree_metrics = page.evaluate(
        """() => Array.from(document.querySelectorAll('#historyList .hist-item')).map(el => ({
            text: el.textContent || '',
            left: el.getBoundingClientRect().left,
            depth: el.dataset.depth || el.getAttribute('aria-level') || ''
        }))"""
    )
    # Child nodes (n1, n2) must be visually indented or carry increasing depth attributes relative to root n0
    lefts = [m["left"] for m in tree_metrics]
    depths = [m["depth"] for m in tree_metrics]
    assert max(lefts) > min(lefts) or len({d for d in depths if d}) >= 2, (
        f"Paths drawer must show hierarchical nesting for n0->n1->n2; got {tree_metrics}"
    )

    # Clicking root n0 restores 'path' in canvas
    page.locator("#historyList .hist-item").first.click()
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('path')",
        timeout=10000,
    )
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_comb_fa3_fa5_ui_live_glide_sliders_and_escape_cancel(live_server, browser_page):
    """FA3 x FA5: Trigger glide -> cancel/pause with Escape -> open sliders -> adjust -> close with Escape."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    page.locator("#btnGlide").click()
    page.wait_for_timeout(150)
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=10000)

    page.keyboard.press("s")
    page.locator("#slidersDrawer").wait_for(state="visible", timeout=5000)
    slider = page.locator("#sliderGrid input[type='range']").first
    slider.fill("-0.5")
    slider.dispatch_event("change")
    page.keyboard.press("Escape")
    page.wait_for_function(
        "() => !document.querySelector('#slidersDrawer').classList.contains('open')",
        timeout=5000,
    )
    assert page_errors == []
    assert console_errors == []


def test_comb_fa1_fa4_replay_labels_and_cors_options_with_live_session(live_server, http_client):
    """FA1 x FA4: Live server serves both CORS OPTIONS (including DELETE) and curated Replay titles alongside live sessions."""
    base = live_server["url"]
    st_opt, h_opt, _ = http_client.options(f"{base}/api/session/start")
    assert st_opt == 204
    allow_methods = h_opt.get("access-control-allow-methods", "")
    assert "DELETE" in allow_methods.upper(), f"CORS Allow-Methods should include DELETE, got {allow_methods!r}"

    st_r, _, replay_texts = http_client.get(f"{base}/api/replay/texts")
    assert st_r == 200 and len(replay_texts) == 9
    assert all(str(t.get("title", "")).strip() for t in replay_texts)

    # Live session operates cleanly on the same hybrid server
    s = _start_session(http_client, base, replay_texts[0]["text"])
    sid = s["session_id"]
    st_d, _, b_d = http_client.delete(f"{base}/api/session/{sid}")
    assert st_d == 200 and b_d.get("deleted") is True
