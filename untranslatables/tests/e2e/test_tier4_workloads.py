"""Tier 4 E2E Tests: Full real-world multi-step end-to-end application workflows (W1-W6).

Each test executes a complete user journey across the real system:
- W1: Multi-stanza formatted poem co-translation (ES/PT/FR via HTTP/SSE)
- W2: Comparative branching exploration & divergent poetic readings (n0 -> n1 -> n2)
- W3: Full live browser editorial walkthrough in headless Chrome
- W4: Standalone zero-server multi-model offline showcase browser walkthrough
- W5: End-to-end CLI pipeline (python -m explorer subprocesses -> LiveTranslationEngine)
- W6: ForkHead(n_dirs=2) <10 ms zero-backward-pass compass workflow
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
from pathlib import Path
from typing import Any

import pytest

from .browser_driver import drag_svg_point

pytestmark = pytest.mark.tier4


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


# ===========================================================================
# W1 — Multi-Stanza Formatted Poem Co-Translation (ES / PT / FR)
# ===========================================================================


def test_w1_multistanza_formatted_poem_cotranslation(live_server, http_client):
    """W1: Full HTTP/SSE co-translation lifecycle on a multi-stanza indented poem with em-dashes & accents."""
    base = live_server["url"]
    becquer = (
        "Volverán las oscuras golondrinas\n"
        "  en tu balcón sus nidos a colgar,\n"
        "  —y otra vez con el ala a sus cristales\n"
        "    jugando llamarán.\n"
        "\n"
        "Pero aquéllas que el vuelo refrenaban\n"
        "  tu hermosura y mi dicha a contemplar,\n"
        "  ésas… ¡no volverán!"
    )
    s = _start_session(http_client, base, becquer, src_lang="Spanish", tgt_lang="English")
    sid = s["session_id"]
    try:
        # Verify stanza parsing & initial alignment
        aln0 = s["alignment"]
        assert len({ln["stanza"] for ln in aln0["source_lines"]}) >= 2
        assert aln0["active_source_line"] == 0

        # 1) Token & word stepping
        st, _, s = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "token"})
        assert st == 200 and s["n_tokens"] == 1
        st, _, s = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "word"})
        assert st == 200 and s["n_tokens"] >= 2

        # 2) Custom indented em-dash line & stanza break insertion
        st, _, s = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": "\n  —playing at the glass\n\nYet those swallows"},
        )
        assert st == 200
        assert "\n  —playing at the glass\n\nYet those swallows" in s["committed_text"]
        assert s["alignment"]["active_source_line"] >= 1
        n_before_sliders = s["n_tokens"]

        # 3) Slider tone adjustment (poetic + idiomatic)
        st, _, s = http_client.post(
            f"{base}/api/session/sliders",
            json_body={"session_id": sid, "global_coords": {"plain_vs_poetic": 0.85, "literal_vs_idiomatic": 0.4}},
        )
        assert st == 200 and s["n_tokens"] == n_before_sliders

        # 4) SSE glide for 2 words
        st_g, hdrs_g, events = http_client.sse_stream(
            f"{base}/api/session/glide?session_id={sid}&max_words=2&stop_on_fork=0"
        )
        assert st_g == 200 and "text/event-stream" in hdrs_g.get("content-type", "")
        paused = [e for e in events if e["event"] == "paused"]
        assert len(paused) == 1
        after_glide_state = paused[0]["data"]["state"]
        full_ids_after_glide = list(after_glide_state["committed_token_ids"])
        assert len(full_ids_after_glide) > n_before_sliders

        # 5) Rewind + true Redo back to tip
        st_rw, _, srw = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": n_before_sliders},
        )
        assert st_rw == 200 and srw["can_redo"] is True
        cur = srw
        for _ in range(20):
            if not cur["can_redo"]:
                break
            st_rd, _, cur = http_client.post(f"{base}/api/session/redo", json_body={"session_id": sid, "mode": "word"})
            assert st_rd == 200
        assert cur["committed_token_ids"] == full_ids_after_glide

        # 6) Compass probe + steer + commit steered word
        st_p, _, probe = http_client.post(
            f"{base}/api/session/compass/probe",
            json_body={"session_id": sid, "n_samples": 4, "lookahead_tokens": 5, "seed": 0},
        )
        assert st_p == 200 and float(probe["alpha_unit"]) > 0.0
        st_st, _, steered = http_client.post(
            f"{base}/api/session/compass/steer",
            json_body={"session_id": sid, "x": 0.6, "y": -0.5, "max_new": 5},
        )
        assert st_st == 200 and len(steered["first_word_ids"]) >= 1
        st_cm, _, final_state = http_client.post(
            f"{base}/api/session/step",
            json_body={
                "session_id": sid,
                "custom_token_ids": steered["first_word_ids"],
                "source_tag": "steered",
            },
        )
        assert st_cm == 200 and final_state["tokens"][-1]["source"] == "steered"
        assert "\n\n" in final_state["committed_text"]
        assert "—" in final_state["committed_text"]

        # 7) Also verify quick PT and FR poem starts succeed in the same workflow
        for lang, poem in (
            ("Portuguese", "Saudade do mar\n\nE do vento antigo."),
            ("French", "Les feuilles d'automne\n\n  —tombent sans bruit."),
        ):
            sp = _start_session(http_client, base, poem, src_lang=lang, tgt_lang="English")
            assert sp["src_lang"] == lang
            assert len({ln["stanza"] for ln in sp["alignment"]["source_lines"]}) >= 2
            http_client.delete(f"{base}/api/session/{sp['session_id']}")
    finally:
        st_del, _, b_del = http_client.delete(f"{base}/api/session/{sid}")
        assert st_del == 200 and b_del.get("deleted") is True


# ===========================================================================
# W2 — Comparative Branching Exploration & Divergent Poetic Readings
# ===========================================================================


def test_w2_comparative_branching_exploration_tree(live_server, http_client):
    """W2: Build 3-level branch tree n0 (baseline) -> n1 (slider-steered) -> n2 (compass-steered) and verify lossless checkout."""
    base = live_server["url"]
    duende = "Cuando canta Camarón, tiene duende; no es técnica, es algo que te sube por la piel."
    s = _start_session(http_client, base, duende)
    sid = s["session_id"]
    try:
        # Path n0: literal baseline stepping
        for txt in ("When", " Camarón", " sings,", " he has soul"):
            st, _, s = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "custom_text": txt})
            assert st == 200
        n0_ids = list(s["committed_token_ids"])
        n0_text = s["committed_text"]
        fork1_idx = 2

        # Rewind to fork1_idx, enable poetic slider, diverge into n1
        st, _, srw1 = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": fork1_idx},
        )
        assert st == 200 and srw1["can_redo"] is True
        st, _, _ = http_client.post(
            f"{base}/api/session/sliders",
            json_body={"session_id": sid, "global_coords": {"plain_vs_poetic": 0.9}},
        )
        assert st == 200
        st, _, s_n1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " dark fire"},
        )
        assert st == 200 and s_n1["active_node_id"] == "n1"
        fork2_idx = s_n1["n_tokens"]
        st, _, s_n1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " rises through the skin"},
        )
        assert st == 200
        n1_ids = list(s_n1["committed_token_ids"])
        n1_text = s_n1["committed_text"]

        # Rewind inside n1, probe + steer compass, and commit into nested child n2
        st, _, _ = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": fork2_idx},
        )
        assert st == 200
        st_p, _, _ = http_client.post(
            f"{base}/api/session/compass/probe",
            json_body={"session_id": sid, "n_samples": 4, "lookahead_tokens": 4, "seed": 2},
        )
        assert st_p == 200
        st_s, _, steered = http_client.post(
            f"{base}/api/session/compass/steer",
            json_body={"session_id": sid, "x": -0.7, "y": 0.6, "max_new": 4},
        )
        assert st_s == 200
        # Ensure divergence from n1's suffix at fork2_idx
        steered_ids = steered["first_word_ids"]
        if steered_ids and n1_ids[fork2_idx : fork2_idx + len(steered_ids)] == steered_ids:
            st_c, _, s_n2 = http_client.post(
                f"{base}/api/session/step",
                json_body={"session_id": sid, "custom_text": " spirit_compass", "source_tag": "steered"},
            )
        else:
            st_c, _, s_n2 = http_client.post(
                f"{base}/api/session/step",
                json_body={"session_id": sid, "custom_token_ids": steered_ids, "source_tag": "steered"},
            )
        assert st_c == 200 and s_n2["active_node_id"] == "n2"
        n2_ids = list(s_n2["committed_token_ids"])
        n2_text = s_n2["committed_text"]

        # Verify 3-level tree topology
        hist = {h["id"]: h for h in s_n2["history"]}
        assert set(hist.keys()) == {"n0", "n1", "n2"}
        assert hist["n0"]["parent_id"] is None and hist["n0"]["n_tokens"] == len(n0_ids)
        assert hist["n1"]["parent_id"] == "n0" and hist["n1"]["fork_token_idx"] == fork1_idx
        assert hist["n1"]["n_tokens"] == len(n1_ids)
        assert hist["n2"]["parent_id"] == "n1" and hist["n2"]["fork_token_idx"] == fork2_idx
        assert hist["n2"]["n_tokens"] == len(n2_ids)

        # Lossless checkout across every branch and verify non-corrupted frontier
        for nid, expected_ids, expected_text in (
            ("n0", n0_ids, n0_text),
            ("n2", n2_ids, n2_text),
            ("n1", n1_ids, n1_text),
            ("n0", n0_ids, n0_text),
        ):
            st_co, _, sco = http_client.post(
                f"{base}/api/session/checkout",
                json_body={"session_id": sid, "node_id": nid},
            )
            assert st_co == 200
            assert sco["active_node_id"] == nid
            assert sco["committed_token_ids"] == expected_ids
            assert sco["committed_text"] == expected_text
            assert isinstance(sco["frontier"]["branches"], list)
    finally:
        http_client.delete(f"{base}/api/session/{sid}")


# ===========================================================================
# W3 — Full Live Browser Editorial Walkthrough (Headless Chrome)
# ===========================================================================


@pytest.mark.browser
def test_w3_live_browser_full_editorial_walkthrough(live_server, browser_page):
    """W3: End-to-end headless Chrome editorial walkthrough across live studio, compass, paths, theme & replay."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    # 1) Select Bécquer preset (index 1) and start session
    page.locator("#presetSelect").select_option(index=1)
    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)
    page.evaluate("() => document.activeElement && document.activeElement.blur()")

    # 2) Verify annotated source view renders poem lines
    page.wait_for_function(
        """() => {
            const els = Array.from(document.querySelectorAll('#liveStudio *')).filter(el => {
                if (el.tagName === 'TEXTAREA' || el.tagName === 'OPTION' || el.tagName === 'SCRIPT') return false;
                return el.children.length === 0 && (el.textContent || '').includes('Volverán');
            });
            return els.length >= 1;
        }""",
        timeout=5000,
    )

    # 3) Step via Space and ArrowRight
    page.keyboard.press("Space")
    page.wait_for_function("() => document.querySelectorAll('#translationCanvas .ctok').length >= 1", timeout=10000)
    c1 = page.locator("#translationCanvas .ctok").count()
    page.keyboard.press("ArrowRight")
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length > {c1}", timeout=10000)
    c2 = page.locator("#translationCanvas .ctok").count()

    # 4) Toggle Token/Word mode (avoid matching #btnStep 'Next word')
    tok_word_btn = page.locator("button:not(#btnStep):not(#btnCommitSteeredWord)").filter(
        has_text=re.compile(r"token|word\s*/\s*token|mode", re.I)
    ).first
    tok_word_btn.wait_for(state="visible", timeout=5000)
    tok_word_btn.click()

    # 5) Rewind via Backspace, then Redo via Redo button
    page.keyboard.press("Backspace")
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length < {c2}", timeout=10000)
    redo_btn = page.get_by_role("button", name=re.compile(r"redo|forward", re.I)).first
    redo_btn.wait_for(state="visible", timeout=5000)
    assert redo_btn.is_enabled()
    redo_btn.click()
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length === {c2}", timeout=10000)

    # 6) Rewind and diverge into n1 via Shift+Digit1 (or custom word if identical)
    page.keyboard.press("Backspace")
    page.wait_for_function(f"() => document.querySelectorAll('#translationCanvas .ctok').length < {c2}", timeout=10000)
    page.keyboard.down("Shift")
    page.keyboard.press("Digit1")
    page.keyboard.up("Shift")
    page.wait_for_timeout(400)
    # Also type a distinct custom word to guarantee a child branch exists for Paths check
    page.locator("#btnRewind").click()
    page.wait_for_timeout(250)
    page.locator("#customWord").fill("editorial_branch")
    page.locator("#customWord").press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('editorial_branch')",
        timeout=10000,
    )

    # 7) Inspect hierarchical Paths drawer and checkout Main path (n0)
    page.locator("#btnHistory").click()
    page.locator("#historyDrawer").wait_for(state="visible", timeout=5000)
    assert page.locator("#historyList .hist-item").count() >= 2
    page.locator("#historyList .hist-item").first.click()
    page.wait_for_timeout(300)

    # 8) Open Compass via 'c', drag SVG knob, commit steered word
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    page.keyboard.press("c")
    page.locator("#compassDrawer").wait_for(state="visible", timeout=10000)
    page.wait_for_function("() => document.querySelectorAll('#compassSvg circle').length >= 1", timeout=20000)
    drag_svg_point(page, "#compassSvg", 0.75, 0.35)
    page.wait_for_function("() => !document.querySelector('#btnCommitSteeredWord').disabled", timeout=15000)
    before_commit = page.locator("#translationCanvas .ctok").count()
    page.locator("#btnCommitSteeredWord").click()
    page.wait_for_function(
        f"() => document.querySelectorAll('#translationCanvas .ctok').length > {before_commit}",
        timeout=10000,
    )

    # 9) Toggle dark theme
    theme0 = page.evaluate("() => document.documentElement.dataset.theme || ''")
    page.locator("#btnTheme").click()
    assert page.evaluate("() => document.documentElement.dataset.theme || ''") != theme0

    # 10) Switch to Replay Archive and verify #replayTokens white-space: pre-wrap
    page.locator("#btnSwitchMode").click()
    page.locator("#replayView").wait_for(state="visible", timeout=5000)
    assert page.evaluate("() => getComputedStyle(document.querySelector('#replayTokens')).whiteSpace") == "pre-wrap"
    drag_svg_point(page, "#replayCompass", 0.5, -0.5, extent=1.6)

    assert page_errors == [], f"Uncaught page errors during W3: {page_errors}"
    assert console_errors == [], f"Console errors during W3: {console_errors}"


# ===========================================================================
# W4 — Standalone Zero-Server Multi-Model Offline Showcase Walkthrough
# ===========================================================================


@pytest.mark.browser
def test_w4_standalone_multimodel_offline_showcase_walkthrough(
    g3_4b_results_dir: Path, tmp_path: Path, browser_page_with_requests
):
    """W4: Multi-model offline HTML via file:// (0 servers, 0 network requests) with switcher, RBF shares & session tree."""
    from explorer.server import build_offline_html

    # Build second model copy with 8 texts so switching is unambiguously visible in DOM
    m2_dir = tmp_path / "g3_27b_sim"
    shutil.copytree(g3_4b_results_dir, m2_dir)
    for rel in ("samples/monte_catalejo.json", "samples/monte_catalejo_emb.npy", "branches/monte_catalejo.json"):
        p = m2_dir / rel
        if p.exists():
            p.unlink()

    sessions_dir = tmp_path / "recorded_sessions"
    sessions_dir.mkdir()
    recorded = {
        "session_id": "sess_becquer_w4",
        "title": "Bécquer Rima LIII (Recorded Co-Translation)",
        "source_text": "Volverán las oscuras golondrinas\n\npero aquéllas… ¡no volverán!",
        "nodes": {
            "n0": {
                "id": "n0",
                "parent_id": None,
                "fork_token_idx": 0,
                "label": "Literal path",
                "text": "The dark swallows will return\n\nbut those will not return!",
            },
            "n1": {
                "id": "n1",
                "parent_id": "n0",
                "fork_token_idx": 3,
                "label": "Poetic branch",
                "text": "The shadowed swallows shall return\n\nyet those will come no more!",
            },
        },
        "history": [
            {"id": "n0", "parent_id": None, "fork_token_idx": 0, "label": "Literal path", "n_tokens": 10, "snippet": "The dark swallows will return"},
            {"id": "n1", "parent_id": "n0", "fork_token_idx": 3, "label": "Poetic branch", "n_tokens": 11, "snippet": "The shadowed swallows shall return"},
        ],
    }
    (sessions_dir / "becquer_recorded.json").write_text(json.dumps(recorded), encoding="utf-8")

    out_html = tmp_path / "multimodel_showcase.html"
    build_offline_html(
        out_html,
        model_dirs={"g3_4b_pt": g3_4b_results_dir, "g3_27b_sim": m2_dir},
        sessions_dir=sessions_dir,
    )

    page, page_errors, console_errors, _ext_blocked = browser_page_with_requests
    local_http_reqs: list[str] = []
    page.on(
        "request",
        lambda req: local_http_reqs.append(req.url)
        if req.url.startswith(("http://", "https://"))
        and not any(h in req.url for h in ("fonts.googleapis.com", "fonts.gstatic.com"))
        else None,
    )

    page.goto(out_html.resolve().as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelectorAll('#replaySelect option').length >= 8", timeout=10000)

    # Verify human-readable labels (never bare "reading 0")
    page.locator("#replaySelect").select_option("duende_flamenco")
    readings_text = page.locator("#replayReadings").text_content() or ""
    assert readings_text.strip()
    assert not re.search(r"\breading\s+\d+\b", readings_text, re.IGNORECASE), (
        f"Offline showcase displayed bare reading label: {readings_text!r}"
    )

    # Drag compass to two opposite corners and assert live RBF shares + output update
    drag_svg_point(page, "#replayCompass", 0.9, 0.6, extent=1.6)
    page.wait_for_timeout(150)
    shares1 = (page.locator("#replayReadings").text_content() or "").strip()
    out1 = (page.locator("#replayOut").text_content() or "").strip()

    drag_svg_point(page, "#replayCompass", -0.9, -0.6, extent=1.6)
    page.wait_for_timeout(150)
    shares2 = (page.locator("#replayReadings").text_content() or "").strip()
    out2 = (page.locator("#replayOut").text_content() or "").strip()
    assert out1 and out2
    assert shares1 != shares2, "Offline compass drag must update live RBF share bars"

    # Switch between embedded models
    model_select = page.evaluate_handle(
        """() => {
            const selects = Array.from(document.querySelectorAll('select'));
            return selects.find(s => {
                if (s.id === 'replaySelect' || s.id === 'srcLang' || s.id === 'presetSelect') return false;
                const vals = Array.from(s.options).map(o => o.value);
                return vals.includes('g3_27b_sim') || vals.includes('g3_4b_pt');
            }) || null;
        }"""
    ).as_element()
    assert model_select is not None, "Multi-model offline showcase missing model switcher <select>"
    model_select.select_option("g3_27b_sim")
    page.wait_for_function("() => document.querySelectorAll('#replaySelect option').length === 8", timeout=5000)

    # Verify bundled recorded session is discoverable/playable in the offline UI
    has_session_ui = page.evaluate(
        """() => {
            const txt = document.body.innerText || '';
            const opts = Array.from(document.querySelectorAll('option, button')).map(el => el.textContent || '');
            return txt.includes('Bécquer') || opts.some(t => /session|recorded|co-translation|rima/i.test(t));
        }"""
    )
    assert has_session_ui, "Bundled recorded v2 session tree should be surfaced in offline showcase UI"

    assert local_http_reqs == [], f"Zero HTTP requests allowed in offline showcase, saw: {local_http_reqs}"
    assert page_errors == [], f"Uncaught page errors during W4: {page_errors}"
    assert console_errors == [], f"Console errors during W4: {console_errors}"


# ===========================================================================
# W5 — End-to-End CLI Pipeline (python -m explorer Subprocesses)
# ===========================================================================


def test_w5_cli_end_to_end_pipeline(repo_root: Path, g3_4b_results_dir: Path, tmp_path: Path):
    """W5: Execute real CLI stages (axes, train-lora, train-reft, train-fork-head, offline, report) into tmp_path."""
    import yaml
    from explorer.global_axes import load_tone_axes_dataset
    from explorer.live_engine import LiveTranslationEngine

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "untranslatables")
    env["HF_HUB_OFFLINE"] = "1"

    # Build a compact 2-item tone_axes YAML in tmp_path so CPU CLI steps run quickly
    full_ds = load_tone_axes_dataset()
    mini_ds = {"axes": full_ds["axes"], "items": full_ds["items"][:2]}
    mini_yaml = tmp_path / "mini_tone_axes.yaml"
    mini_yaml.write_text(yaml.safe_dump(mini_ds, allow_unicode=True), encoding="utf-8")

    axes_path = tmp_path / "global_axes.npz"
    lora_path = tmp_path / "multiref_lora.pt"
    reft_path = tmp_path / "spectral_reft.pt"
    fh_path = tmp_path / "fork_head.pt"
    offline_path = tmp_path / "offline.html"

    commands = [
        [sys.executable, "-m", "explorer", "axes", "--model", "tiny", "--prompt-style", "doc", "--dataset", str(mini_yaml), "--out", str(axes_path)],
        [sys.executable, "-m", "explorer", "train-lora", "--model", "tiny", "--prompt-style", "doc", "--dataset", str(mini_yaml), "--steps", "2", "--out", str(lora_path)],
        [sys.executable, "-m", "explorer", "train-reft", "--model", "tiny", "--prompt-style", "doc", "--dataset", str(mini_yaml), "--steps", "2", "--out", str(reft_path)],
        [sys.executable, "-m", "explorer", "train-fork-head", "--model", "tiny", "--prompt-style", "doc", "--from-texts", "--n-dirs", "2", "--epochs", "2", "--out", str(fh_path)],
        [sys.executable, "-m", "explorer", "offline", "--results-dir", str(g3_4b_results_dir), "--out", str(offline_path)],
    ]
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"CLI command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    for artifact in (axes_path, lora_path, reft_path, fh_path, offline_path):
        assert artifact.is_file() and artifact.stat().st_size > 0, f"Missing/empty artifact: {artifact}"

    # Run `report` stage in a real subprocess pointing OUT at an isolated tmp copy (NEVER mutates repo files)
    rep_root = tmp_path / "report_out"
    shutil.copytree(g3_4b_results_dir, rep_root / "g3_4b_pt")
    rep_script = (
        "import pathlib, sys; "
        "import explorer.run as r, explorer.report as rep; "
        "r.OUT = rep.OUT = pathlib.Path(sys.argv[1]); "
        "sys.argv = ['explorer', 'report', '--model', 'g3_4b_pt']; "
        "from explorer.__main__ import main; main()"
    )
    rep_proc = subprocess.run(
        [sys.executable, "-c", rep_script, str(rep_root)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert rep_proc.returncode == 0, f"CLI report failed:\n{rep_proc.stderr}"
    rep_html = rep_root / "g3_4b_pt" / "report.html"
    assert rep_html.is_file() and rep_html.stat().st_size > 1000
    assert not re.search(r"\breading\s+\d+\b", rep_html.read_text(encoding="utf-8"), re.IGNORECASE)

    # Load all 4 trained artifacts simultaneously into LiveTranslationEngine
    eng = LiveTranslationEngine(
        model_key="tiny",
        global_axes_path=axes_path,
        lora_path=lora_path,
        reft_path=reft_path,
        fork_head_path=fh_path,
    )
    try:
        eng.ensure_loaded()
        st = eng.status()
        assert bool(st["global_axes_loaded"])
        assert st["lora_loaded"] is True
        assert st["reft_loaded"] is True
        assert st["fork_head_loaded"] is True
    finally:
        eng.close()


# ===========================================================================
# W6 — ForkHead(n_dirs=2) <10 ms Zero-Backward-Pass Compass Workflow
# ===========================================================================


def test_w6_fork_head_ndirs2_sub10ms_compass_workflow(tiny_model, replay_engine, http_client, tmp_path: Path):
    """W6: Extract n_dirs=2 dataset -> train & save ForkHead -> live server <10 ms 0-pass compass -> steer & commit."""
    from explorer.fork_head import extract_self_supervised_fork_dataset, train_fork_head
    from explorer.live_engine import LiveTranslationEngine
    from explorer.server import WorkbenchServer
    from ut import arch

    tok, model = tiny_model

    # 1) Extract self-supervised dataset with n_dirs=2
    ds = extract_self_supervised_fork_dataset(
        model,
        tok,
        max_new_per_text=4,
        entropy_fork_thresh=0.0,
        prompt_style="doc",
        n_dirs=2,
    )
    assert "y_dirs" in ds and "y_dirs_mask" in ds
    assert ds["y_dirs"].shape[1] == 2
    assert ds["y_dirs_mask"].any()
    rec_alpha = float(ds["meta"]["recommended_alpha"])
    assert math.isfinite(rec_alpha) and rec_alpha > 0.0

    # 2) Train & save ForkHead(n_dirs=2) checkpoint
    ckpt = tmp_path / "w6_fork_head_ndirs2.pt"
    train_res = train_fork_head(
        ds,
        hidden_dim=32,
        n_dirs=2,
        epochs=4,
        seed=0,
        save_path=ckpt,
    )
    assert ckpt.is_file()
    assert float(train_res["head"].meta.get("recommended_alpha", 0.0)) > 0.0

    # 3) Boot dedicated live WorkbenchServer with fork_head_path loaded
    eng = LiveTranslationEngine(model_key="tiny", fork_head_path=ckpt, tok=tok, model=model)
    srv = WorkbenchServer(
        ("127.0.0.1", 0),
        mode="live",
        replay_engine=replay_engine,
        live_engine=eng,
        quiet=True,
    )
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    sid = None
    try:
        s0 = _start_session(http_client, base, "Cuando canta Camarón, tiene duende.")
        sid = s0["session_id"]

        for _ in range(2):
            st, _, s0 = http_client.post(f"{base}/api/session/step", json_body={"session_id": sid, "mode": "word"})
            assert st == 200
        cursor_before = s0["n_tokens"]

        # 4) Probe compass while counting forward passes -> MUST be 0 passes & <10 ms
        fwd_calls = 0

        def _count_fwd(_m, _i, _o):
            nonlocal fwd_calls
            fwd_calls += 1

        h_cnt = arch.layers(model)[0].register_forward_hook(_count_fwd)
        try:
            st_p, _, probe = http_client.post(
                f"{base}/api/session/compass/probe",
                json_body={"session_id": sid, "n_samples": 6, "lookahead_tokens": 5, "seed": 0},
            )
        finally:
            h_cnt.remove()

        assert st_p == 200, f"compass/probe failed: {st_p} {probe}"
        assert probe["source"] == "fork_head"
        assert fwd_calls == 0, f"Expected 0 model forward/backward passes during ForkHead compass probe, got {fwd_calls}"
        assert float(probe["probe_ms"]) < 10.0, f"Expected probe_ms < 10.0 ms, got {probe['probe_ms']}"
        assert float(probe["alpha_unit"]) > 0.0
        assert probe["cursor_len"] == cursor_before
        assert len(probe["readings"]) >= 1

        # 5) Steer compass & commit steered branch
        st_s, _, steered = http_client.post(
            f"{base}/api/session/compass/steer",
            json_body={"session_id": sid, "x": 0.8, "y": 0.4, "max_new": 5},
        )
        assert st_s == 200
        assert len(steered["first_word_ids"]) >= 1
        assert math.isclose(sum(float(v) for v in steered["shares"].values()), 1.0, rel_tol=1e-2, abs_tol=1e-2)

        st_c, _, committed = http_client.post(
            f"{base}/api/session/step",
            json_body={
                "session_id": sid,
                "custom_token_ids": steered["first_word_ids"],
                "source_tag": "steered",
            },
        )
        assert st_c == 200
        assert committed["n_tokens"] == cursor_before + len(steered["first_word_ids"])
        assert committed["tokens"][-1]["source"] == "steered"
    finally:
        if sid is not None:
            http_client.delete(f"{base}/api/session/{sid}")
        with contextlib.suppress(Exception):
            srv.shutdown()
        with contextlib.suppress(Exception):
            srv.server_close()
        with contextlib.suppress(Exception):
            eng.close()
        thread.join(timeout=5.0)
