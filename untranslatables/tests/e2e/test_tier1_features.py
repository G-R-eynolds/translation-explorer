"""Tier 1 E2E Tests: Isolated happy-path feature coverage across FA1-FA5 + Browser UI.

Every test exercises Translation Explorer through its public HTTP/SSE API,
public Python entry points, CLI subprocesses, or headless Chrome driving
``workbench.html``. All assertions enforce the exact contracts specified in
``PROJECT.md`` without mocking away the system under test.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from .browser_driver import drag_svg_point

pytestmark = pytest.mark.tier1


# ---------------------------------------------------------------------------
# Shared lightweight opaque-box helpers
# ---------------------------------------------------------------------------

def _extract_offline_data(html_text: str) -> dict[str, Any]:
    """Extract and parse ``window.__OFFLINE_DATA__ = ...;`` from offline HTML."""
    m = re.search(
        r"window\.__OFFLINE_DATA__\s*=\s*(\{.*?\})\s*;\s*</script>",
        html_text,
        flags=re.DOTALL,
    )
    assert m is not None, "Could not locate window.__OFFLINE_DATA__ script payload in offline HTML"
    raw = m.group(1).replace("<\\/", "</")
    return json.loads(raw)


def _start_session(http_client, base_url: str, source_text: str, **kwargs) -> dict[str, Any]:
    payload = {
        "source_text": source_text,
        "lookahead_tokens": kwargs.pop("lookahead_tokens", 5),
        "max_branches": kwargs.pop("max_branches", 4),
        **kwargs,
    }
    status, _hdrs, body = http_client.post(f"{base_url}/api/session/start", json_body=payload)
    assert status == 200, f"Expected 200 from /api/session/start, got {status}: {body}"
    return body


def _cleanup_session(http_client, base_url: str, session_id: str) -> None:
    # Best-effort cleanup (harmless even before DELETE endpoint lands in M3).
    http_client.delete(f"{base_url}/api/session/{session_id}")


# ===========================================================================
# FA1 — Live Co-Translation Session Lifecycle, Stepping, Rewind, Redo & Tree
# ===========================================================================


def test_fa1_session_start_get_delete_lifecycle(live_server, http_client):
    """POST /api/session/start -> GET /api/session/<id> -> DELETE -> 404 on subsequent GET."""
    base = live_server["url"]
    state = _start_session(http_client, base, "Cuando canta Camarón, tiene duende.")
    sid = state["session_id"]

    assert isinstance(sid, str) and sid.startswith("sess_")
    assert state["n_tokens"] == 0
    assert state["active_node_id"] == "n0"
    assert state.get("can_redo") is False, f"Expected can_redo=False at start, got {state.get('can_redo')!r}"
    assert state.get("redo_remaining") == 0, f"Expected redo_remaining=0 at start, got {state.get('redo_remaining')!r}"
    assert isinstance(state["frontier"], dict)
    assert state["frontier"]["regime"] in ("linear", "subtle", "fork")
    assert isinstance(state["frontier"]["branches"], list)
    assert isinstance(state["frontier"]["ghost"], dict)

    # GET /api/session/<id> returns matching state_dict
    status, _hdrs, fetched = http_client.get(f"{base}/api/session/{sid}")
    assert status == 200, f"GET /api/session/<id> expected 200, got {status}: {fetched}"
    assert fetched["session_id"] == sid
    assert fetched["active_node_id"] == "n0"
    assert fetched["n_tokens"] == 0

    # DELETE /api/session/<id> removes session cleanly
    d_status, _dhdrs, d_body = http_client.delete(f"{base}/api/session/{sid}")
    assert d_status == 200, f"DELETE /api/session/<id> expected 200, got {d_status}: {d_body}"
    assert d_body.get("deleted") is True
    assert d_body.get("session_id") == sid

    # Subsequent GET returns 404 JSON error
    g2_status, _g2hdrs, g2_body = http_client.get(f"{base}/api/session/{sid}")
    assert g2_status == 404, f"Expected 404 after DELETE, got {g2_status}: {g2_body}"
    assert isinstance(g2_body, dict) and "error" in g2_body


def test_fa1_step_word_and_token_modes(live_server, http_client):
    """POST /api/session/step advances by 1 in token mode and >=1 in word mode with word_start_idx."""
    base = live_server["url"]
    s0 = _start_session(http_client, base, "El camino se hace al andar.")
    sid = s0["session_id"]
    try:
        st_tok, _, s1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "mode": "token"},
        )
        assert st_tok == 200, f"step(token) failed: {st_tok} {s1}"
        assert s1["n_tokens"] == 1
        assert len(s1["tokens"]) == 1
        t0 = s1["tokens"][0]
        assert t0["source"] == "step"
        assert "word_start_idx" in t0, f"Missing word_start_idx in token metadata: {t0}"
        assert isinstance(t0["word_start_idx"], int) and 0 <= t0["word_start_idx"] <= t0["idx"]

        st_word, _, s2 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "mode": "word"},
        )
        assert st_word == 200, f"step(word) failed: {st_word} {s2}"
        assert s2["n_tokens"] >= 2
        assert len(s2["committed_token_ids"]) == s2["n_tokens"]
        assert s2["tokens"][-1]["source"] == "step"
        # word_start_idx must be non-decreasing across tokens
        w_starts = [t["word_start_idx"] for t in s2["tokens"]]
        assert all(isinstance(w, int) and 0 <= w <= t["idx"] for w, t in zip(w_starts, s2["tokens"]))
        assert w_starts == sorted(w_starts)
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_step_custom_text_and_custom_token_ids(live_server, http_client, tiny_model):
    """POST /api/session/step accepts custom_text (typed) and custom_token_ids with source_tag."""
    base = live_server["url"]
    tok, _ = tiny_model
    s0 = _start_session(http_client, base, "Volverán las oscuras golondrinas.")
    sid = s0["session_id"]
    try:
        st1, _, s1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " shadow"},
        )
        assert st1 == 200, f"custom_text step failed: {st1} {s1}"
        assert "shadow" in s1["committed_text"]
        assert s1["tokens"][-1]["source"] == "typed"

        custom_ids = tok.encode(" moonlight", add_special_tokens=False)
        assert len(custom_ids) >= 1
        st2, _, s2 = http_client.post(
            f"{base}/api/session/step",
            json_body={
                "session_id": sid,
                "custom_token_ids": custom_ids,
                "source_tag": "steered",
            },
        )
        assert st2 == 200, f"custom_token_ids step failed: {st2} {s2}"
        assert s2["n_tokens"] == s1["n_tokens"] + len(custom_ids)
        assert s2["tokens"][-1]["source"] == "steered"
        assert "moonlight" in s2["committed_text"]
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_step_branch_and_commit_full_branch(live_server, http_client):
    """Selecting branch_index commits first_word_ids or full preview_ids with source='branch'."""
    base = live_server["url"]
    s0 = _start_session(http_client, base, "Camarón tiene duende cuando canta.")
    sid = s0["session_id"]
    try:
        branches0 = s0["frontier"]["branches"]
        assert len(branches0) >= 1
        b0 = branches0[0]

        st1, _, s1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "branch_index": 0, "commit_full_branch": False},
        )
        assert st1 == 200
        assert s1["n_tokens"] == len(b0["first_word_ids"])
        assert s1["tokens"][-1]["source"] == "branch"

        branches1 = s1["frontier"]["branches"]
        assert len(branches1) >= 1
        b1 = branches1[0]
        st2, _, s2 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "branch_index": 0, "commit_full_branch": True},
        )
        assert st2 == 200
        assert s2["n_tokens"] == s1["n_tokens"] + len(b1["preview_ids"])
        assert len(b1["preview_ids"]) >= len(b1["first_word_ids"])
        assert s2["tokens"][-1]["source"] == "branch"
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_non_destructive_rewind_and_true_redo(live_server, http_client):
    """Rewind preserves full node suffix (can_redo=True) and POST /api/session/redo replays it without duplicating."""
    base = live_server["url"]
    s = _start_session(http_client, base, "El viento canta sobre el mar oscuro.")
    sid = s["session_id"]
    try:
        for _ in range(4):
            st, _, s = http_client.post(
                f"{base}/api/session/step",
                json_body={"session_id": sid, "mode": "token"},
            )
            assert st == 200
        full_ids = list(s["committed_token_ids"])
        assert len(full_ids) == 4
        assert len(s["history"]) == 1

        # Rewind to token index 1: suffix must stay preserved on n0
        st_rw, _, srw = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": 1},
        )
        assert st_rw == 200
        assert srw["n_tokens"] == 1
        assert srw["committed_token_ids"] == full_ids[:1]
        assert srw.get("can_redo") is True
        assert srw.get("redo_remaining") == 3
        n0_entry = next(h for h in srw["history"] if h["id"] == "n0")
        assert n0_entry["n_tokens"] == 4, "Rewind must be non-destructive to n0's saved suffix"

        # Redo 1 token along saved suffix without creating a duplicate branch
        st_rd1, _, srd1 = http_client.post(
            f"{base}/api/session/redo",
            json_body={"session_id": sid, "mode": "token"},
        )
        assert st_rd1 == 200, f"POST /api/session/redo expected 200, got {st_rd1}: {srd1}"
        assert srd1["n_tokens"] == 2
        assert srd1["committed_token_ids"] == full_ids[:2]
        assert len(srd1["history"]) == 1
        assert srd1["can_redo"] is True
        assert srd1["redo_remaining"] == 2

        # Redo remaining tokens until tip
        cur = srd1
        for _ in range(4):
            if not cur["can_redo"]:
                break
            st_r, _, cur = http_client.post(
                f"{base}/api/session/redo",
                json_body={"session_id": sid, "mode": "word"},
            )
            assert st_r == 200
        assert cur["committed_token_ids"] == full_ids
        assert cur["can_redo"] is False
        assert cur["redo_remaining"] == 0
        assert len(cur["history"]) == 1
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_nested_branch_on_branch_and_checkout(live_server, http_client):
    """Build 3-level nested branch tree n0 -> n1 -> n2 and verify lossless checkout across all nodes."""
    base = live_server["url"]
    s = _start_session(http_client, base, "La noche guarda silencios antiguos.")
    sid = s["session_id"]
    try:
        for w in (" Night", " keeps", " old"):
            st, _, s = http_client.post(
                f"{base}/api/session/step",
                json_body={"session_id": sid, "custom_text": w},
            )
            assert st == 200
        n0_ids = list(s["committed_token_ids"])
        fork1_idx = 1

        # Rewind n0 to fork1_idx and diverge -> n1
        st, _, _ = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": fork1_idx},
        )
        assert st == 200
        st, _, s_n1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " alpha"},
        )
        assert st == 200
        assert s_n1["active_node_id"] == "n1"
        st, _, s_n1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " river"},
        )
        assert st == 200
        n1_ids = list(s_n1["committed_token_ids"])
        fork2_idx = len(n1_ids) - 1

        # Rewind n1 and diverge -> n2 (parent_id='n1')
        st, _, _ = http_client.post(
            f"{base}/api/session/rewind",
            json_body={"session_id": sid, "to_token_index": fork2_idx},
        )
        assert st == 200
        st, _, s_n2 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": " beta"},
        )
        assert st == 200
        assert s_n2["active_node_id"] == "n2"
        n2_ids = list(s_n2["committed_token_ids"])

        hist = {h["id"]: h for h in s_n2["history"]}
        assert set(hist.keys()) == {"n0", "n1", "n2"}
        assert hist["n0"]["parent_id"] is None
        assert hist["n0"]["n_tokens"] == len(n0_ids)
        assert hist["n1"]["parent_id"] == "n0"
        assert hist["n1"]["fork_token_idx"] == fork1_idx
        assert hist["n1"]["n_tokens"] == len(n1_ids)
        assert hist["n2"]["parent_id"] == "n1"
        assert hist["n2"]["fork_token_idx"] == fork2_idx
        assert hist["n2"]["n_tokens"] == len(n2_ids)

        # Checkout each node and verify exact token recovery without truncating any sibling/parent
        for node_id, expected_ids in (("n0", n0_ids), ("n1", n1_ids), ("n2", n2_ids), ("n0", n0_ids)):
            st_c, _, sc = http_client.post(
                f"{base}/api/session/checkout",
                json_body={"session_id": sid, "node_id": node_id},
            )
            assert st_c == 200
            assert sc["active_node_id"] == node_id
            assert sc["committed_token_ids"] == expected_ids
            hist_after = {h["id"]: h for h in sc["history"]}
            assert hist_after["n0"]["n_tokens"] == len(n0_ids)
            assert hist_after["n1"]["n_tokens"] == len(n1_ids)
            assert hist_after["n2"]["n_tokens"] == len(n2_ids)
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa1_frontier_branch_schema_and_sub_branches(live_server, http_client):
    """Frontier branches conform to full schema including has_newline and depth-2 sub_branches."""
    base = live_server["url"]
    s = _start_session(http_client, base, "El poeta contempla la luna en el río.")
    sid = s["session_id"]
    try:
        frontier = s["frontier"]
        branches = frontier["branches"]
        assert len(branches) >= 1
        required_keys = {
            "index",
            "preview_text",
            "preview_ids",
            "first_word_text",
            "first_word_ids",
            "prob",
            "share",
            "distinctive",
            "color",
            "has_newline",
            "sub_branches",
        }
        share_sum = 0.0
        for br in branches:
            missing = required_keys - set(br.keys())
            assert not missing, f"Branch missing keys {missing}: {br}"
            assert isinstance(br["has_newline"], bool)
            assert isinstance(br["sub_branches"], list)
            share_sum += float(br["share"])
            if frontier["regime"] == "linear":
                assert br["sub_branches"] == []
            for sub in br["sub_branches"]:
                for sk in ("preview_text", "first_word_text", "prob", "share"):
                    assert sk in sub, f"sub_branch missing key {sk}: {sub}"
        assert math.isclose(share_sum, 1.0, rel_tol=1e-2, abs_tol=1e-2)
    finally:
        _cleanup_session(http_client, base, sid)


# ===========================================================================
# FA2 — Poem & Long-Document Formatting, Alignment & Multilingual Framing
# ===========================================================================


def test_fa2_doc_prompt_multistanza_formatting(tiny_model):
    """build_doc_prompt preserves multi-stanza blank lines, indentation, em-dashes, and XML tags."""
    from explorer.frame import build_doc_prompt, doc_prompt_text

    tok, _ = tiny_model
    poem = (
        "Volverán las oscuras golondrinas\n"
        "  en tu balcón sus nidos a colgar,\n"
        "\n"
        "  —¿mas aquéllas que el vuelo refrenaban?\n"
        "    ¡ésas… no volverán!"
    )
    prompt_txt = doc_prompt_text(poem, src_lang="Spanish", tgt_lang="English")
    assert '<source lang="Spanish">' in prompt_txt
    assert '<translation lang="English">' in prompt_txt
    assert poem in prompt_txt

    ids, start = build_doc_prompt(tok, poem, src_lang="Spanish", tgt_lang="English")
    assert isinstance(ids, list) and len(ids) > 10
    assert start == len(ids)


def test_fa2_doc_prompt_multilingual_shots():
    """doc_prompt_text supports ES/PT/FR/DE/IT with language-aware framing and non-Spanish shots."""
    from explorer.frame import DOC_SHOTS, doc_prompt_text

    # Verify DOC_SHOTS covers multiple languages (not Spanish-only)
    assert isinstance(DOC_SHOTS, (dict, list, tuple))
    for lang, sample in (
        ("Spanish", "Tiene duende cuando canta."),
        ("Portuguese", "Tenho uma saudade imensa do mar."),
        ("French", "L'esprit d'escalier nous rattrape."),
        ("German", "Waldeinsamkeit im stillen Tal."),
        ("Italian", "Il dolce far niente d'estate."),
    ):
        txt = doc_prompt_text(sample, src_lang=lang, tgt_lang="English")
        assert f'<source lang="{lang}">' in txt
        assert '<translation lang="English">' in txt
        assert sample in txt
        # Non-Spanish languages should have language-appropriate shots (not exclusively Spanish exemplars)
        if lang != "Spanish":
            assert f'<source lang="{lang}">' in txt.split(sample)[0], (
                f"Expected language-aware few-shot exemplar for {lang} before the query source"
            )

    # Also verify non-English target language tag
    fr_tgt = doc_prompt_text("Tiene duende.", src_lang="Spanish", tgt_lang="French")
    assert '<translation lang="French">' in fr_tgt


def test_fa2_cut_doc_preserves_stanzas_and_stops_at_close_tag(tiny_model):
    """cut_doc preserves blank stanza lines, indentation, and em-dashes while cutting at DOC_STOPS."""
    from explorer.frame import cut_doc

    tok, _ = tiny_model
    body = "The dark swallows will return\n  to hang their nests,\n\n  —yet never again."
    full = body + "</translation>\n<source lang=\"Spanish\">extra"
    ids = tok.encode(full, add_special_tokens=False)
    cut_ids = cut_doc(tok, ids)
    decoded = tok.decode(cut_ids, skip_special_tokens=True)
    assert "</translation>" not in decoded
    assert "<source" not in decoded
    assert "\n\n" in decoded
    assert "  " in decoded
    assert "—" in decoded


def test_fa2_live_session_alignment_schema(live_server, http_client):
    """Live session state_dict exposes stanza-aware source↔target alignment metadata."""
    base = live_server["url"]
    poem = (
        "Volverán las oscuras golondrinas\n"
        "en tu balcón sus nidos a colgar,\n"
        "\n"
        "pero aquéllas que el vuelo refrenaban\n"
        "ésas… ¡no volverán!"
    )
    s = _start_session(http_client, base, poem)
    sid = s["session_id"]
    try:
        for _ in range(3):
            st, _, s = http_client.post(
                f"{base}/api/session/step",
                json_body={"session_id": sid, "mode": "word"},
            )
            assert st == 200

        assert "alignment" in s, f"Missing alignment metadata in state_dict keys: {list(s.keys())}"
        aln = s["alignment"]
        assert isinstance(aln, dict)
        lines = aln.get("source_lines")
        assert isinstance(lines, list) and len(lines) >= 4
        stanzas = set()
        for entry in lines:
            assert isinstance(entry["index"], int)
            assert isinstance(entry["text"], str)
            assert isinstance(entry["stanza"], int)
            stanzas.add(entry["stanza"])
        assert len(stanzas) >= 2, f"Expected >=2 stanzas for 2-stanza poem, got {stanzas}"
        active_line = aln["active_source_line"]
        assert isinstance(active_line, int) and 0 <= active_line < len(lines)
        spans = aln["spans"]
        assert isinstance(spans, list)
        for sp in spans:
            for k in ("target_start", "target_end", "source_line", "confidence"):
                assert k in sp, f"Alignment span missing {k}: {sp}"
            assert 0 <= sp["target_start"] <= sp["target_end"] <= s["n_tokens"]
            assert 0 <= sp["source_line"] < len(lines)
            assert 0.0 <= float(sp["confidence"]) <= 1.0
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa2_live_session_preserves_custom_newline_and_indentation(live_server, http_client):
    """Committing indented em-dash lines and blank stanza breaks preserves exact whitespace without premature stop."""
    base = live_server["url"]
    s = _start_session(http_client, base, "Primera estrofa.\n\nSegunda estrofa.")
    sid = s["session_id"]
    try:
        custom = "\n  —only the wind\n\nSecond stanza"
        st, _, s1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "custom_text": custom},
        )
        assert st == 200
        assert "\n  —only the wind\n\nSecond stanza" in s1["committed_text"]
        assert s1["frontier"]["at_end"] is False
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa2_multilingual_session_start_and_branch_has_newline(live_server, http_client):
    """Sessions accept multilingual src_lang/tgt_lang pairs and report accurate branch has_newline."""
    base = live_server["url"]
    for src_lang, tgt_lang, text in (
        ("Portuguese", "English", "Saudade é a presença da ausência."),
        ("French", "English", "La nuit tombe doucement sur Paris."),
        ("German", "French", "Über allen Gipfeln ist Ruh."),
        ("Italian", "English", "Nel mezzo del cammin di nostra vita."),
    ):
        s = _start_session(http_client, base, text, src_lang=src_lang, tgt_lang=tgt_lang)
        sid = s["session_id"]
        try:
            assert s["src_lang"] == src_lang
            assert s["tgt_lang"] == tgt_lang
            for br in s["frontier"]["branches"]:
                assert "has_newline" in br and isinstance(br["has_newline"], bool)
                assert br["has_newline"] == ("\n" in br["preview_text"])
        finally:
            _cleanup_session(http_client, base, sid)


def test_fa2_tone_axes_yaml_expanded_schema_and_models_registry():
    """tone_axes.yaml is expanded with multi-stanza poems & multilingual items; MODELS registry has target scale keys."""
    from explorer.frame import load_texts
    from explorer.global_axes import load_tone_axes_dataset
    from ut.common import MODELS

    ds = load_tone_axes_dataset()
    axes = set(ds["axes"].keys())
    expected_axes = {
        "literal_vs_idiomatic",
        "plain_vs_poetic",
        "foreignising_vs_domesticating",
        "concise_vs_glossed",
    }
    assert expected_axes.issubset(axes), f"Missing canonical axes: {expected_axes - axes}"

    items = ds["items"]
    assert len(items) > 16, f"Expected tone_axes.yaml expanded beyond 16 items, got {len(items)}"

    has_multistanza = any(
        "\n\n" in str(it.get("source", ""))
        and any("\n\n" in str(tr.get("text", "")) for tr in it.get("translations", []))
        for it in items
    )
    assert has_multistanza, "Expected at least one multi-stanza ('\\n\\n') poem item in tone_axes.yaml"

    non_spanish = [
        it for it in items
        if str(it.get("lang", it.get("src_lang", "Spanish"))).lower() not in ("spanish", "es", "")
    ]
    assert len(non_spanish) >= 2, (
        f"Expected non-Spanish items (PT/FR/DE/IT) in tone_axes.yaml, found {len(non_spanish)}"
    )

    for it in items:
        assert it.get("id") and it.get("source")
        assert len(it.get("translations", [])) >= 2
        for tr in it["translations"]:
            assert tr.get("text")
            for ax, val in tr.get("coords", {}).items():
                assert -1.0 <= float(val) <= 1.0, f"Coord {ax}={val} out of [-1, 1] in item {it['id']}"

    texts = load_texts()
    assert len(texts) >= 9
    assert any(t["id"] == "control_museo" for t in texts)
    assert any(t["id"] == "duende_flamenco" for t in texts)

    for key in ("g3_4b_pt", "g3_12b_pt", "g3_27b_pt", "qwen72b_base", "tiny"):
        assert key in MODELS, f"Model key {key!r} missing from ut.common.MODELS"
    assert MODELS["g3_12b_pt"].get("quant") == "nf4"
    assert MODELS["g3_27b_pt"].get("quant") == "nf4"
    assert MODELS["qwen72b_base"].get("quant") == "nf4"
    assert MODELS["tiny"].get("cpu_ok") is True


# ===========================================================================
# FA3 — Steering, Sliders, Compass, ForkHead(n_dirs=2) & SSE Glide
# ===========================================================================


def test_fa3_session_sliders_update_coords_and_frontier(live_server, http_client):
    """POST /api/session/sliders persists global_coords and refreshes frontier without losing committed tokens."""
    base = live_server["url"]
    s0 = _start_session(http_client, base, "El viento murmura entre los pinos.")
    sid = s0["session_id"]
    try:
        st_step, _, s1 = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "mode": "token"},
        )
        assert st_step == 200
        coords = {"plain_vs_poetic": 0.8, "literal_vs_idiomatic": -0.5}
        st, _, s2 = http_client.post(
            f"{base}/api/session/sliders",
            json_body={"session_id": sid, "global_coords": coords},
        )
        assert st == 200
        assert s2["n_tokens"] == s1["n_tokens"]
        assert math.isclose(s2["global_coords"]["plain_vs_poetic"], 0.8, abs_tol=1e-6)
        assert math.isclose(s2["global_coords"]["literal_vs_idiomatic"], -0.5, abs_tol=1e-6)
        assert isinstance(s2["frontier"]["branches"], list)
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_compass_probe_and_steer_gradients_path(live_server, http_client):
    """Compass probe (gradient path) returns alpha_unit & probe_ms; steer does not mutate committed tokens."""
    base = live_server["url"]
    s0 = _start_session(http_client, base, "Cuando canta Camarón, tiene duende.")
    sid = s0["session_id"]
    try:
        st_p, _, probe = http_client.post(
            f"{base}/api/session/compass/probe",
            json_body={"session_id": sid, "n_samples": 6, "lookahead_tokens": 5, "seed": 0},
        )
        assert st_p == 200, f"compass/probe failed: {st_p} {probe}"
        assert probe["source"] == "gradients"
        assert probe["cursor_len"] == 0
        assert "alpha_unit" in probe and float(probe["alpha_unit"]) > 0.0
        assert "probe_ms" in probe and float(probe["probe_ms"]) >= 0.0
        assert isinstance(probe["readings"], list) and len(probe["readings"]) >= 2
        assert isinstance(probe["dots"], list) and len(probe["dots"]) == 6

        st_s, _, steered = http_client.post(
            f"{base}/api/session/compass/steer",
            json_body={"session_id": sid, "x": 0.7, "y": -0.4, "max_new": 6},
        )
        assert st_s == 200, f"compass/steer failed: {st_s} {steered}"
        assert isinstance(steered["preview_text"], str)
        assert isinstance(steered["preview_ids"], list)
        assert isinstance(steered["first_word_text"], str)
        assert isinstance(steered["first_word_ids"], list)
        shares = steered["shares"]
        assert isinstance(shares, dict) and len(shares) >= 1
        assert math.isclose(sum(float(v) for v in shares.values()), 1.0, rel_tol=1e-2, abs_tol=1e-2)

        # Steer must remain purely speculative (cursor unchanged)
        st_g, _, after = http_client.get(f"{base}/api/session/{sid}")
        assert st_g == 200
        assert after["n_tokens"] == 0
        assert after["has_compass"] is True
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_fork_head_ndirs2_fast_compass_and_scorer_paused(tiny_model, tmp_path):
    """ForkHead(n_dirs=2) enables zero-forward-pass <10 ms compass probe and LiveForkScorer.paused()."""
    from explorer.fork_head import ForkHead, LiveForkScorer
    from explorer.live_engine import LiveTranslationEngine
    from explorer.session import CoTranslationSession
    from ut import arch

    tok, model = tiny_model
    d = arch.hidden_size(model)
    layer = arch.n_layers(model) // 2

    head = ForkHead(d_model=d, hidden_dim=32, n_dirs=2)
    ckpt = tmp_path / "tiny_fork_head_ndirs2.pt"
    head.save(ckpt, meta={"layer": layer, "recommended_alpha": 1.25})
    reloaded = ForkHead.load(ckpt)
    assert reloaded.n_dirs == 2
    assert math.isclose(float(reloaded.meta["recommended_alpha"]), 1.25, rel_tol=1e-5)

    # Verify LiveForkScorer.paused() suppresses hook recording
    scorer = LiveForkScorer(reloaded, model, layer=layer)
    try:
        assert hasattr(scorer, "paused"), "LiveForkScorer must provide paused() context manager"
        inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            model(input_ids=inp)
        assert len(scorer.history()) == 1
        with scorer.paused():
            with torch.no_grad():
                model(input_ids=inp)
                model(input_ids=inp)
        assert len(scorer.history()) == 1, "Forward passes inside scorer.paused() must not record history"
        with torch.no_grad():
            model(input_ids=inp)
        assert len(scorer.history()) == 2
    finally:
        scorer.remove()

    # Verify LiveTranslationEngine + CoTranslationSession zero-forward-pass ForkHead compass
    eng = LiveTranslationEngine(model_key="tiny", fork_head_path=ckpt, tok=tok, model=model)
    try:
        sess = CoTranslationSession(
            eng,
            "Cuando canta Camarón, tiene duende.",
            lookahead_tokens=5,
            max_branches=4,
        )
        # Count model forward passes during probe_compass()
        fwd_calls = 0

        def _count_fwd(_mod, _inp, _out):
            nonlocal fwd_calls
            fwd_calls += 1

        h_cnt = arch.layers(model)[0].register_forward_hook(_count_fwd)
        try:
            t0 = time.perf_counter()
            res = sess.probe_compass(n_samples=6, lookahead_tokens=5, seed=0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            h_cnt.remove()

        assert res["source"] == "fork_head"
        assert fwd_calls == 0, f"ForkHead(n_dirs=2) compass must run 0 forward passes, ran {fwd_calls}"
        assert "alpha_unit" in res and float(res["alpha_unit"]) > 0.0
        assert "probe_ms" in res and float(res["probe_ms"]) < 10.0, (
            f"Expected probe_ms < 10.0 ms, got {res.get('probe_ms')} (wall {elapsed_ms:.2f} ms)"
        )
        assert elapsed_ms < 25.0, f"Wall time {elapsed_ms:.2f} ms exceeded fast budget"
        for r in res["readings"]:
            ax, ay = r["anchor"]
            assert math.isfinite(ax) and math.isfinite(ay)
    finally:
        eng.close()


def test_fa3_sse_glide_emits_step_and_paused_events(live_server, http_client):
    """GET /api/session/glide streams text/event-stream with event: step and terminal event: paused."""
    base = live_server["url"]
    s0 = _start_session(http_client, base, "El viento corre sobre las olas.")
    sid = s0["session_id"]
    try:
        status, hdrs, events = http_client.sse_stream(
            f"{base}/api/session/glide?session_id={sid}&max_words=3&stop_on_fork=0"
        )
        assert status == 200
        assert "text/event-stream" in hdrs.get("content-type", "")
        assert isinstance(events, list) and len(events) >= 2
        step_events = [e for e in events if e["event"] == "step"]
        paused_events = [e for e in events if e["event"] == "paused"]
        assert len(step_events) >= 1
        assert len(paused_events) == 1
        for se in step_events:
            d = se["data"]
            for k in ("steps", "committed_text", "n_tokens", "regime", "fork_score", "state"):
                assert k in d, f"SSE step payload missing {k}"
        pd = paused_events[0]["data"]
        assert pd["reason"] in ("fork", "end", "limit")
        assert isinstance(pd["state"], dict)
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_compass_ready_flag_in_state_dict(live_server, http_client):
    """state_dict['compass_ready'] is a bool, True after compass probe, and tracks cursor validity."""
    base = live_server["url"]
    s0 = _start_session(http_client, base, "El silencio habla más alto.")
    sid = s0["session_id"]
    try:
        assert "compass_ready" in s0 and isinstance(s0["compass_ready"], bool)
        st_p, _, _ = http_client.post(
            f"{base}/api/session/compass/probe",
            json_body={"session_id": sid, "n_samples": 4, "lookahead_tokens": 4, "seed": 0},
        )
        assert st_p == 200
        st_g, _, s_after = http_client.get(f"{base}/api/session/{sid}")
        assert st_g == 200
        assert s_after["compass_ready"] is True

        # Moving cursor invalidates previous compass cache
        st_st, _, s_step = http_client.post(
            f"{base}/api/session/step",
            json_body={"session_id": sid, "mode": "token"},
        )
        assert st_st == 200
        assert isinstance(s_step["compass_ready"], bool)
        assert s_step["has_compass"] is False
    finally:
        _cleanup_session(http_client, base, sid)


def test_fa3_api_status_and_extract_fork_dataset_ndirs2(live_server, http_client, tiny_model):
    """GET /api/status reports live capabilities; extract_self_supervised_fork_dataset populates y_dirs."""
    from explorer.fork_head import extract_self_supervised_fork_dataset
    from ut import arch

    base = live_server["url"]
    st, _, status_body = http_client.get(f"{base}/api/status")
    assert st == 200
    assert status_body["mode"] == "live"
    assert status_body["model_key"] == "tiny"
    assert bool(status_body["global_axes_loaded"]) and len(status_body["global_axes_loaded"]) == 4
    assert len(status_body["global_axes_schema"]) == 4
    for key in ("fork_head_loaded", "reft_loaded", "lora_loaded", "active_forks"):
        assert key in status_body

    tok, model = tiny_model
    d = arch.hidden_size(model)
    ds = extract_self_supervised_fork_dataset(
        model,
        tok,
        max_new_per_text=4,
        entropy_fork_thresh=0.0,  # guarantee fork tokens on tiny random logits
        prompt_style="doc",
        n_dirs=2,
    )
    assert "y_dirs" in ds and "y_dirs_mask" in ds, f"Dataset missing y_dirs/y_dirs_mask: {list(ds.keys())}"
    N = int(ds["H"].shape[0])
    assert N >= 2
    assert ds["y_dirs"].shape == (N, 2, d)
    assert ds["y_dirs"].dtype == torch.float32
    assert ds["y_dirs_mask"].shape == (N,)
    assert ds["y_dirs_mask"].dtype == torch.bool
    assert ds["y_dirs_mask"].any(), "Expected at least one masked fork token when entropy_fork_thresh=0.0"

    # Masked fork rows must have orthonormal 2D frames
    for idx in torch.where(ds["y_dirs_mask"])[0].tolist():
        D = ds["y_dirs"][idx].double()
        gram = D @ D.T
        assert torch.allclose(gram, torch.eye(2, dtype=torch.float64), atol=1e-3), (
            f"y_dirs[{idx}] is not orthonormal: {gram}"
        )
    # Non-masked rows (if any) must be zero
    for idx in torch.where(~ds["y_dirs_mask"])[0].tolist():
        assert torch.allclose(ds["y_dirs"][idx], torch.zeros(2, d))

    rec_alpha = ds.get("meta", {}).get("recommended_alpha")
    assert rec_alpha is not None and math.isfinite(float(rec_alpha)) and float(rec_alpha) > 0.0


# ===========================================================================
# FA4 — Reading Discovery, mapgrid, Human-Readable Labels & CLI Pipelines
# ===========================================================================


def test_fa4_readings_from_disp_gate_separates_control_and_hard(g3_4b_results_dir: Path):
    """Uncentered raw_disp gate (disp_min=0.035) collapses real control_museo to k=1 while keeping all 8 hard texts >=2."""
    from explorer import readings

    samples_dir = g3_4b_results_dir / "samples"
    hard_checked = 0
    for jp in sorted(samples_dir.glob("*.json")):
        S = json.loads(jp.read_text(encoding="utf-8"))
        tid = S["id"]
        emb_path = samples_dir / f"{tid}_emb.npy"
        E = np.load(emb_path)
        lpt = np.array(S["logp"], dtype=np.float64) / np.maximum(np.array(S["lens"], dtype=np.float64), 1.0)

        res = readings.readings_from(E, lpt, disp_min=0.035)
        if S["kind"] == "control":
            assert res["k"] == 1, f"{tid} should collapse to k=1 under disp_min=0.035, got {res['k']}"
            assert res["readings"] == [0]
            assert float(res["silhouette"]) == 0.0
        else:
            hard_checked += 1
            assert res["k"] >= 2 and len(res["readings"]) >= 2, (
                f"Hard text {tid} unexpectedly collapsed: k={res['k']}, readings={res['readings']}"
            )
    assert hard_checked == 8


def test_fa4_readings_orthographic_normalization(monkeypatch):
    """readings.py normalizes curly/straight quotes and repeated whitespace before semantic embedding."""
    from explorer import readings

    variants = [
        "“The museum”   opens  at  9:00.",
        '"The museum" opens at 9:00.',
        "‘It’s’  open   daily.",
        "'It's' open daily.",
    ]

    # Check if readings exposes a direct text normalization helper; otherwise observe embed_texts input
    norm_fn = None
    for attr in ("normalize_text", "normalize_orthography", "normalize_surface", "_normalize_text"):
        fn = getattr(readings, attr, None)
        if callable(fn):
            norm_fn = fn
            break

    if norm_fn is not None:
        assert norm_fn(variants[0]) == norm_fn(variants[1])
        assert norm_fn(variants[2]) == norm_fn(variants[3])
    else:
        captured: list[str] = []

        class _CaptureTok:
            def __call__(self, batch_texts, **kwargs):
                captured.extend(batch_texts)
                n = len(batch_texts)
                return {
                    "input_ids": torch.ones((n, 2), dtype=torch.long),
                    "attention_mask": torch.ones((n, 2), dtype=torch.long),
                }

        class _DummyEnc:
            device = torch.device("cpu")

            def __call__(self, **batch):
                n = batch["input_ids"].shape[0]
                return type("Out", (), {"last_hidden_state": torch.ones((n, 2, 4))})()

        monkeypatch.setitem(readings._EMB, "t", _CaptureTok())
        monkeypatch.setitem(readings._EMB, "m", _DummyEnc())
        readings.embed_texts(variants)
        assert len(captured) == 4
        assert captured[0] == captured[1], f"Curly quotes/whitespace not normalized: {captured[0]!r} != {captured[1]!r}"
        assert captured[2] == captured[3], f"Curly apostrophes/whitespace not normalized: {captured[2]!r} != {captured[3]!r}"


def test_fa4_mapgrid_spectral_basis_r2_and_generic_phrases(g3_4b_results_dir: Path):
    """spectral_basis handles R==2 without rank-1 collapse and mapgrid.phrases_by_reading works beyond duende_flamenco."""
    from explorer import grad_readings, mapgrid

    rng = np.random.default_rng(7)
    d = 12
    g0 = rng.normal(loc=0.0, scale=0.3, size=(12, d))
    g0[:, 0] += 2.0
    g0[:, 1] += np.linspace(-1.0, 1.0, 12)
    g1 = rng.normal(loc=0.0, scale=0.3, size=(12, d))
    g1[:, 0] -= 2.0
    g1[:, 1] += np.linspace(1.0, -1.0, 12)
    G = np.vstack([g0, g1])
    labels = np.array([0] * 12 + [1] * 12)

    basis = grad_readings.spectral_basis(G, labels, [0, 1], n_dims=2, whiten=True)
    axes = basis["axes"]
    assert axes.shape == (2, d)
    assert np.allclose(axes @ axes.T, np.eye(2), atol=1e-5)
    all_dots = np.vstack([np.array(basis["dots"][0]), np.array(basis["dots"][1])])
    assert np.all(np.isfinite(all_dots))
    assert float(np.std(all_dots[:, 0])) > 1e-3
    assert float(np.std(all_dots[:, 1])) > 1e-3

    # Verify mapgrid.py itself uses spectral_basis (generic non-duende phrases)
    for tid in ("machado_caminante", "becquer_xxi", "monte_catalejo"):
        S = json.loads((g3_4b_results_dir / "samples" / f"{tid}.json").read_text(encoding="utf-8"))
        cands = mapgrid.phrases_by_reading(S, S["readings"])
        for r in S["readings"]:
            assert r in cands and len(cands[r]) >= 1, (
                f"mapgrid.phrases_by_reading returned empty candidates for {tid} reading {r}: {cands}"
            )
            assert all(isinstance(p, str) and p.strip() for p in cands[r])


def test_fa4_readings_labels_yaml_and_replay_engine(repo_root: Path, g3_4b_results_dir: Path):
    """readings_labels.yaml exists with required schema and ReplayEngine never emits bare 'reading <n>' labels."""
    from explorer.frame import load_texts
    from explorer.live_engine import ReplayEngine

    labels_path = repo_root / "untranslatables" / "explorer" / "data" / "readings_labels.yaml"
    assert labels_path.is_file(), f"Missing curated labels file: {labels_path}"
    data = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data.get("texts"), dict)
    assert isinstance(data.get("models"), dict)

    for item in load_texts():
        tid = item["id"]
        assert tid in data["texts"], f"Missing text metadata for {tid} in readings_labels.yaml"
        assert str(data["texts"][tid].get("title", "")).strip()
        assert str(data["texts"][tid].get("description", "")).strip()
    assert "g3_4b_pt" in data["models"] and len(data["models"]["g3_4b_pt"]) >= 1

    eng = ReplayEngine(g3_4b_results_dir)
    listed = eng.list_texts()
    assert len(listed) == 9
    bare_re = re.compile(r"^reading\s+\d+$", re.IGNORECASE)
    for meta in listed:
        assert str(meta.get("title", "")).strip(), f"Replay text {meta['id']} missing human-readable title"
        bundle = eng.get_text_bundle(meta["id"])
        for rd in bundle["readings"]:
            lbl = str(rd.get("label", "")).strip()
            assert lbl and not bare_re.match(lbl), (
                f"Bare fallback label {lbl!r} found for {meta['id']} reading {rd['id']}"
            )


def test_fa4_report_html_human_readable_labels(g3_4b_results_dir: Path, tmp_path: Path, monkeypatch):
    """report.run writes curated human-readable titles/labels and zero bare 'reading <n>' strings."""
    import explorer.report as report_mod
    import explorer.run as run_mod

    tmp_model_dir = tmp_path / "g3_4b_pt"
    shutil.copytree(g3_4b_results_dir, tmp_model_dir)
    monkeypatch.setattr(report_mod, "OUT", tmp_path)
    monkeypatch.setattr(run_mod, "OUT", tmp_path)

    out_html = report_mod.run({"model": "g3_4b_pt"})
    assert Path(out_html).resolve().parent == tmp_model_dir.resolve()
    html_text = Path(out_html).read_text(encoding="utf-8")
    assert "<!doctype html>" in html_text.lower()
    assert not re.search(r"\breading\s+\d+\b", html_text, re.IGNORECASE), (
        "report.html must never contain bare 'reading <n>' fallback labels"
    )
    # Curated title from readings_labels.yaml should appear (not just raw 'duende_flamenco')
    assert "duende" in html_text.lower()


def test_fa4_cli_help_and_subcommands(repo_root: Path):
    """python -m explorer --help succeeds and advertises all stages and newly forwarded flags."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "untranslatables")
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "explorer", "--help"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr}"
    out = proc.stdout
    for stage in ("run", "report", "serve", "offline", "axes", "train-lora", "train-reft", "train-fork-head"):
        assert stage in out, f"CLI stage {stage!r} missing from --help"
    for flag in ("--prompt-style", "--n-dirs", "--from-texts", "--model-dirs"):
        assert flag in out, f"CLI flag {flag!r} missing from explorer/__main__.py --help"


# ===========================================================================
# FA5 — Replay Mode, Multi-Model Offline Showcase, Browser UI & Repo Hygiene
# ===========================================================================


def test_fa5_replay_http_endpoints_and_steer(replay_server, http_client):
    """Replay HTTP endpoints serve all 9 texts, bundle schemas, and RBF-interpolated steering."""
    base = replay_server["url"]
    st, _, status_body = http_client.get(f"{base}/api/status")
    assert st == 200
    assert status_body["mode"] == "replay"
    assert status_body["replay_texts"] == 9

    st_list, _, texts = http_client.get(f"{base}/api/replay/texts")
    assert st_list == 200 and isinstance(texts, list) and len(texts) == 9

    st_b, _, bundle = http_client.get(f"{base}/api/replay/duende_flamenco")
    assert st_b == 200
    assert bundle["id"] == "duende_flamenco"
    assert len(bundle["readings"]) >= 2
    assert len(bundle["steering_points"]) >= 1

    st_s, _, steered = http_client.post(
        f"{base}/api/replay/steer",
        json_body={"text_id": "duende_flamenco", "x": 0.6, "y": -0.3},
    )
    assert st_s == 200
    assert steered["text_id"] == "duende_flamenco"
    assert isinstance(steered["text"], str) and len(steered["text"]) > 0
    assert math.isclose(sum(float(v) for v in steered["shares"].values()), 1.0, rel_tol=1e-2, abs_tol=1e-2)


def test_fa5_build_offline_html_single_and_multi_model(g3_4b_results_dir: Path, tmp_path: Path):
    """build_offline_html supports backwards-compatible single-model AND multi-model + sessions_dir bundles."""
    from explorer.server import build_offline_html

    # 1) Backwards-compatible single-model call
    single_out = tmp_path / "single.html"
    build_offline_html(single_out, results_dir=g3_4b_results_dir)
    single_data = _extract_offline_data(single_out.read_text(encoding="utf-8"))
    assert single_data["mode"] == "offline"
    assert len(single_data["texts"]) == 9
    assert len(single_data["bundles"]) == 9
    assert "default_model" in single_data and "models" in single_data
    assert single_data["default_model"] in single_data["models"]

    # 2) Multi-model + sessions_dir bundle
    m2_dir = tmp_path / "g3_12b_sim"
    shutil.copytree(g3_4b_results_dir, m2_dir)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    sample_session = {
        "session_id": "sess_demo01",
        "title": "Bécquer Rima LIII (Demo)",
        "source_text": "Volverán las oscuras golondrinas\n\npero aquéllas… ¡no volverán!",
        "history": [
            {"id": "n0", "parent_id": None, "fork_token_idx": 0, "label": "Main path", "n_tokens": 5, "snippet": "Dark swallows return"},
            {"id": "n1", "parent_id": "n0", "fork_token_idx": 2, "label": "Poetic", "n_tokens": 6, "snippet": "Shadowed swallows glide"},
        ],
    }
    (sessions_dir / "demo_session.json").write_text(json.dumps(sample_session), encoding="utf-8")

    multi_out = tmp_path / "multi.html"
    build_offline_html(
        multi_out,
        model_dirs={"g3_4b_pt": g3_4b_results_dir, "g3_12b_sim": m2_dir},
        sessions_dir=sessions_dir,
    )
    multi_data = _extract_offline_data(multi_out.read_text(encoding="utf-8"))
    assert multi_data["mode"] == "offline"
    assert set(multi_data["models"].keys()) == {"g3_4b_pt", "g3_12b_sim"}
    for mkey in ("g3_4b_pt", "g3_12b_sim"):
        entry = multi_data["models"][mkey]
        for k in ("label", "summary", "texts", "bundles", "maps", "sessions"):
            assert k in entry, f"Offline model {mkey} missing key {k}"
        assert len(entry["texts"]) == 9
        assert "duende_flamenco" in entry["maps"]
    assert any(len(multi_data["models"][k]["sessions"]) >= 1 for k in multi_data["models"])


def test_fa5_gitignore_hygiene(repo_root: Path):
    """Repository .gitignore ignores .agents/ and *.token and zero matching files are tracked."""
    r1 = subprocess.run(
        ["git", "check-ignore", ".agents/"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0, ".agents/ MUST be ignored by .gitignore (F1)"

    r2 = subprocess.run(
        ["git", "check-ignore", "secret.token"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, "*.token MUST be ignored by .gitignore (F1)"

    r3 = subprocess.run(
        ["git", "ls-files", ".agents", "*.token"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert r3.stdout.strip() == "", f"Tracked secret/teamwork files found: {r3.stdout}"


@pytest.mark.browser
def test_fa5_ui_live_bootstrap_and_start_step_rewind_redo(live_server, browser_page):
    """Headless Chrome loads live studio, starts session, steps, rewinds, and exercises Redo button."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")

    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)
    assert "tiny" in (page.locator("#modelPill").text_content() or "").lower()
    assert page.locator("#liveStudio").is_visible()
    assert not page.locator("#replayView").is_visible()
    assert len(page.locator("#sourceInput").input_value().strip()) > 0

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    # Step twice
    page.locator("#btnStep").click()
    page.wait_for_function("() => document.querySelectorAll('#translationCanvas .ctok').length >= 1", timeout=10000)
    count1 = page.locator("#translationCanvas .ctok").count()

    page.locator("#btnStep").click()
    page.wait_for_function(
        f"() => document.querySelectorAll('#translationCanvas .ctok').length > {count1}",
        timeout=10000,
    )
    count2 = page.locator("#translationCanvas .ctok").count()
    assert count2 > count1

    # Rewind one word
    page.locator("#btnRewind").click()
    page.wait_for_function(
        f"() => document.querySelectorAll('#translationCanvas .ctok').length < {count2}",
        timeout=10000,
    )
    count_rw = page.locator("#translationCanvas .ctok").count()
    assert count_rw < count2

    # True Redo button must exist, become enabled after rewind, and restore tokens
    redo_btn = page.get_by_role("button", name=re.compile(r"redo|forward", re.I)).first
    redo_btn.wait_for(state="visible", timeout=5000)
    assert redo_btn.is_enabled(), "Redo/Forward button must be enabled when can_redo is true"
    redo_btn.click()
    page.wait_for_function(
        f"() => document.querySelectorAll('#translationCanvas .ctok').length > {count_rw}",
        timeout=10000,
    )
    assert page.locator("#translationCanvas .ctok").count() == count2
    assert page_errors == [], f"Uncaught page errors: {page_errors}"
    assert console_errors == [], f"Console errors: {console_errors}"


@pytest.mark.browser
def test_fa5_ui_live_custom_word_newline_sliders_paths_theme(live_server, browser_page):
    """Live UI supports custom word, newline/stanza controls, sliders drawer, paths drawer, and dark theme."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    # Custom word via Enter
    page.locator("#customWord").fill("luna")
    page.locator("#customWord").press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#translationCanvas')?.textContent || '').includes('luna')",
        timeout=10000,
    )

    # Explicit newline / stanza insertion control (↵ or ¶)
    nl_btn = page.get_by_role("button", name=re.compile(r"↵|¶|newline|line\s*break|stanza", re.I)).first
    nl_btn.wait_for(state="visible", timeout=5000)
    before_ctoks = page.locator("#translationCanvas .ctok").count()
    nl_btn.click()
    page.wait_for_function(
        f"() => document.querySelectorAll('#translationCanvas .ctok').length > {before_ctoks}",
        timeout=10000,
    )
    canvas_text = page.locator("#translationCanvas").inner_text()
    assert "\n" in canvas_text, f"Expected newline in translation canvas after newline control, got {canvas_text!r}"

    # Sliders drawer & range adjustment
    page.locator("#btnSliders").click()
    page.locator("#slidersDrawer").wait_for(state="visible", timeout=5000)
    slider = page.locator("#sliderGrid input[type='range']").first
    slider.wait_for(state="visible", timeout=5000)
    slider.fill("0.6")
    slider.dispatch_event("change")

    # History / Paths drawer
    page.locator("#btnHistory").click()
    page.locator("#historyDrawer").wait_for(state="visible", timeout=5000)
    assert page.locator("#historyList .hist-item").count() >= 1

    # Theme toggle
    theme_before = page.evaluate("() => document.documentElement.dataset.theme || ''")
    page.locator("#btnTheme").click()
    theme_after = page.evaluate("() => document.documentElement.dataset.theme || ''")
    assert theme_after != theme_before
    assert theme_after in ("dark", "light")

    assert page_errors == [], f"Uncaught page errors: {page_errors}"
    assert console_errors == [], f"Console errors: {console_errors}"


@pytest.mark.browser
def test_fa5_ui_live_compass_drawer_and_source_alignment(live_server, browser_page):
    """Live UI opens Compass drawer, drags SVG knob, shows source↔target alignment, and toggles Replay view."""
    page, page_errors, console_errors = browser_page
    page.goto(live_server["url"], wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#modeLabel')?.textContent?.trim() === 'Live'", timeout=10000)

    page.locator("#btnStart").click()
    page.wait_for_function("() => !document.querySelector('#btnStep').disabled", timeout=15000)

    # Annotated source view should render poem lines (not only hidden raw textarea)
    page.wait_for_function(
        """() => {
            const firstLine = (document.querySelector('#sourceInput')?.value || '').split('\\n')[0].trim();
            if (!firstLine) return false;
            const els = Array.from(document.querySelectorAll('#liveStudio *')).filter(el => {
                if (el.tagName === 'TEXTAREA' || el.tagName === 'OPTION' || el.tagName === 'SCRIPT') return false;
                return el.children.length === 0 && (el.textContent || '').includes(firstLine.slice(0, 12));
            });
            return els.length >= 1;
        }""",
        timeout=5000,
    )

    # Open compass and drag knob
    page.locator("#btnCompass").click()
    page.locator("#compassDrawer").wait_for(state="visible", timeout=10000)
    page.wait_for_function(
        "() => document.querySelectorAll('#compassSvg circle').length >= 1",
        timeout=20000,
    )
    drag_svg_point(page, "#compassSvg", 0.7, -0.5)
    page.wait_for_function(
        "() => !document.querySelector('#btnCommitSteeredWord').disabled",
        timeout=15000,
    )
    assert len((page.locator("#steeredPreview").text_content() or "").strip()) > 0

    # Switch to Replay Archive and back
    page.locator("#btnSwitchMode").click()
    page.locator("#replayView").wait_for(state="visible", timeout=5000)
    assert page.locator("#replaySelect option").count() == 9

    assert page_errors == [], f"Uncaught page errors: {page_errors}"
    assert console_errors == [], f"Console errors: {console_errors}"


@pytest.mark.browser
def test_fa5_ui_offline_html_loads_in_headless_browser(single_model_offline_html: Path, browser_page_with_requests):
    """Standalone file:// offline HTML loads with zero HTTP servers, zero JS errors, and 9 embedded texts."""
    page, page_errors, console_errors, ext_reqs = browser_page_with_requests
    http_api_requests: list[str] = []
    page.on(
        "request",
        lambda req: http_api_requests.append(req.url)
        if req.url.startswith(("http://127.0.0.1", "http://localhost"))
        else None,
    )

    page.goto(single_model_offline_html.resolve().as_uri(), wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelectorAll('#replaySelect option').length === 9",
        timeout=10000,
    )
    assert "offline" in (page.locator("#modeLabel").text_content() or "").lower()
    assert page.locator("#replayView").is_visible()
    assert len((page.locator("#replayOut").text_content() or "").strip()) > 0
    assert http_api_requests == [], f"Offline HTML must not make local server API requests: {http_api_requests}"
    assert page_errors == [], f"Uncaught page errors: {page_errors}"
    assert console_errors == [], f"Console errors: {console_errors}"
