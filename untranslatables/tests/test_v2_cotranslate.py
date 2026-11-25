"""CPU tests for v2 Live Co-Translation Engine, doc/poem framing, quantized registry, and /api/session/* API."""
from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request

import numpy as np
import pytest
import torch

from explorer.adapters import SpectralReFTController
from explorer.frame import build_doc_prompt, cut_doc, doc_prompt_text, load
from explorer.live_engine import LiveTranslationEngine
from explorer.server import create_server
from explorer.session import _extract_first_word_ids, _ribbon_similarity
from explorer.steer import CursorHook
from ut.common import MODELS


@pytest.fixture(scope="module")
def tiny_model():
    return load("tiny")


def test_quantized_gemma3_registry():
    for key, hf_id, quant in [
        ("g3_12b_pt", "google/gemma-3-12b-pt", "nf4"),
        ("g3_27b_pt", "google/gemma-3-27b-pt", "nf4"),
        ("g3_12b_pt_bf16", "google/gemma-3-12b-pt", None),
        ("g3_27b_pt_8bit", "google/gemma-3-27b-pt", "int8"),
    ]:
        assert key in MODELS
        assert MODELS[key]["hf_id"] == hf_id
        assert MODELS[key]["quant"] == quant
        assert MODELS[key]["base"] is True


def test_doc_prompt_preserves_stanzas_and_stops_cleanly(tiny_model):
    tok, _ = tiny_model
    poem = "Linea uno\n  con sangria.\n\nSegunda estrofa\n  —y pausa."
    prompt = doc_prompt_text(poem, src_lang="Spanish", tgt_lang="English")
    assert "<source lang=\"Spanish\">" in prompt
    assert poem in prompt
    assert prompt.endswith("<translation lang=\"English\">\n")

    ids, start = build_doc_prompt(tok, poem)
    assert start == len(ids)

    # Blank lines inside a poem must NOT trigger stop; </translation> must stop cleanly
    out_with_stanzas = "Line one\n  with indent.\n\nSecond stanza.\n</translation>\n<source>ignored"
    y = tok(out_with_stanzas, add_special_tokens=False)["input_ids"]
    trimmed = tok.decode(cut_doc(tok, y))
    assert "\n\nSecond stanza." in trimmed
    assert "</translation>" not in trimmed
    assert "ignored" not in trimmed


def test_ribbon_similarity_and_word_slicing(tiny_model):
    tok, _ = tiny_model
    assert _ribbon_similarity("the dark spirit of flamenco", "The dark spirit of flamenco") >= 0.85
    assert _ribbon_similarity("a touch of magic in his voice", "a deep sadness in the night") < 0.5
    ids = tok(" hello world", add_special_tokens=False)["input_ids"]
    fw = _extract_first_word_ids(tok, ids)
    assert 1 <= len(fw) <= len(ids)


def test_hooks_sync_pos_with_kv_cache(tiny_model):
    _, model = tiny_model
    d = model.config.hidden_size
    hk = CursorHook(model, 0)
    reft = SpectralReFTController(model, layers=[0], axis_names=["a", "b"], rank=2)
    try:
        v = torch.ones(d) * 0.25
        hk.set(v, b=4, end_b=5)
        reft.set_coords({"a": 0.5, "b": -0.2}, start_pos=4)

        x0 = torch.tensor([[1, 2, 3, 4]])
        hk.sync_pos(0)
        reft.sync_pos(0)
        with torch.no_grad():
            out0 = model(input_ids=x0, use_cache=True)
            assert hk.pos == 4
            assert reft._pos[0] == 4

            # Multi-token cached append at position 4 (length 2 -> positions 4, 5)
            x1 = torch.tensor([[5, 6]])
            hk.sync_pos(4)
            reft.sync_pos(4)
            _ = model(input_ids=x1, past_key_values=out0.past_key_values, use_cache=True)
            assert hk.pos == 6
            assert reft._pos[0] == 6
    finally:
        hk.remove()
        reft.remove()


def test_cotranslation_session_lifecycle(tiny_model, tmp_path):
    tok, model = tiny_model
    d = model.config.hidden_size
    axes_path = tmp_path / "axes.npz"
    rng = np.random.default_rng(11)
    np.savez(
        axes_path,
        literal_vs_idiomatic=(rng.normal(size=d) * 0.01).astype(np.float32),
        plain_vs_poetic=(rng.normal(size=d) * 0.01).astype(np.float32),
    )

    eng = LiveTranslationEngine(model_key="tiny", global_axes_path=axes_path, tok=tok, model=model)
    try:
        poem = "Volverán las oscuras golondrinas\n\nPero aquellas no volverán."
        s0 = eng.sessions.start(poem, src_lang="Spanish", tgt_lang="English", lookahead_tokens=5, max_branches=4)
        sid = s0["session_id"]
        assert s0["n_tokens"] == 0
        assert s0["frontier"]["regime"] in ("linear", "subtle", "fork")
        assert 0.0 <= s0["frontier"]["fork_score"] <= 1.0

        # Step forward twice
        s1 = eng.sessions.step(sid, mode="word")
        assert s1["n_tokens"] >= 1
        s2 = eng.sessions.step(sid, mode="word")
        assert s2["n_tokens"] > s1["n_tokens"]
        orig_node_id = s2["active_node_id"]
        orig_tokens = list(s2["committed_token_ids"])

        # Rewind to token 1 and diverge via custom typed word -> creates sibling history node!
        sr = eng.sessions.rewind(sid, 1)
        assert sr["n_tokens"] == 1
        sd = eng.sessions.step(sid, custom_text=" customword", source_tag="typed")
        assert len(sd["history"]) >= 2
        assert sd["active_node_id"] != orig_node_id

        # Checkout original explored path
        sc = eng.sessions.checkout(sid, orig_node_id)
        assert sc["committed_token_ids"] == orig_tokens

        # Update sliders
        ss = eng.sessions.set_sliders(sid, {"plain_vs_poetic": 0.7})
        assert ss["global_coords"]["plain_vs_poetic"] == pytest.approx(0.7)

        # Tier-2 compass probe + steer must not mutate committed tokens
        comp = eng.sessions.probe_compass(sid, n_samples=6, lookahead_tokens=4, seed=0)
        assert len(comp["readings"]) >= 2
        st = eng.sessions.steer_compass(sid, x=0.6, y=-0.3, max_new=6)
        assert "preview_text" in st and "preview_ids" in st and "shares" in st
        assert eng.sessions._get(sid).committed_ids == orig_tokens

        # Glide generator yields step & paused events
        events = list(eng.sessions.glide(sid, max_words=3, stop_on_fork=True))
        assert events[-1]["event"] == "paused"
    finally:
        eng.close()


def _http_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_v2_session_http_and_sse_endpoints(tiny_model):
    tok, model = tiny_model
    live_eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    srv = create_server(host="127.0.0.1", port=0, mode="live", live_engine=live_eng, quiet=True)
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        code, s0 = _http_json(f"{base}/api/session/start", {"source_text": "Cuando canta Camarón,\ntiene duende."})
        assert code == 200 and "session_id" in s0
        sid = s0["session_id"]

        code, s1 = _http_json(f"{base}/api/session/step", {"session_id": sid, "mode": "word"})
        assert code == 200 and s1["n_tokens"] >= 1

        code, sr = _http_json(f"{base}/api/session/rewind", {"session_id": sid, "to_token_index": 0})
        assert code == 200 and sr["n_tokens"] == 0

        # SSE glide
        qs = urllib.parse.urlencode({"session_id": sid, "max_words": "2", "stop_on_fork": "1"})
        with urllib.request.urlopen(f"{base}/api/session/glide?{qs}", timeout=20) as resp:
            assert resp.status == 200
            raw = resp.read().decode("utf-8")
        assert "event: paused" in raw

        code, comp = _http_json(f"{base}/api/session/compass/probe", {"session_id": sid, "n_samples": 6, "lookahead_tokens": 4})
        assert code == 200 and len(comp["readings"]) >= 2

        code, st = _http_json(f"{base}/api/session/compass/steer", {"session_id": sid, "x": 0.4, "y": 0.2, "max_new": 5})
        assert code == 200 and "preview_text" in st
    finally:
        srv.shutdown()
        srv.server_close()
        live_eng.close()
