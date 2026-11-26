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
    for key, hf_id, quant, family in [
        ("g3_12b_pt", "google/gemma-3-12b-pt", "nf4", "gemma3"),
        ("g3_27b_pt", "google/gemma-3-27b-pt", "nf4", "gemma3"),
        ("g3_12b_pt_bf16", "google/gemma-3-12b-pt", None, "gemma3"),
        ("g3_27b_pt_8bit", "google/gemma-3-27b-pt", "int8", "gemma3"),
        ("qwen72b_base", "Qwen/Qwen2.5-72B", "nf4", "qwen"),
    ]:
        assert key in MODELS
        assert MODELS[key]["hf_id"] == hf_id
        assert MODELS[key]["quant"] == quant
        assert MODELS[key]["family"] == family
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


# ------------------------------------------------------------------ M1 Core Engine Unit Tests
def test_kv_cache_crop_and_exact_logit_parity(tiny_model, tmp_path, monkeypatch):
    from ut import arch

    tok, model = tiny_model
    d = model.config.hidden_size
    axes_path = tmp_path / "axes.npz"
    rng = np.random.default_rng(21)
    np.savez(axes_path, plain_vs_poetic=(rng.normal(size=d) * 0.02).astype(np.float32))

    eng = LiveTranslationEngine(model_key="tiny", global_axes_path=axes_path, tok=tok, model=model)
    try:
        s0 = eng.sessions.start("La noche cae sobre el río.", lookahead_tokens=4, max_branches=3)
        sid = s0["session_id"]
        sess = eng.sessions._get(sid)
        sess.auto_precompute_compass = False

        prefill_calls = 0
        orig_prefill = sess._prefill_full

        def _spy_prefill():
            nonlocal prefill_calls
            prefill_calls += 1
            return orig_prefill()

        monkeypatch.setattr(sess, "_prefill_full", _spy_prefill)

        def _uncached_logits():
            seq = sess.prompt_ids + sess.committed_ids
            x = torch.tensor([seq], dtype=torch.long, device=eng._dev())
            sess._apply_global_steering()
            sess._sync_hooks_pos(0)
            try:
                with torch.no_grad():
                    out = arch.forward_hidden(model, input_ids=x, use_cache=False)
                    return arch.logits(model, out.last_hidden_state[0, -1][None])[0].detach().cpu()
            finally:
                sess._clear_steering()

        # Step 3 tokens incrementally without any full prefill
        for _ in range(3):
            sess.step(mode="token")
        assert prefill_calls == 0
        assert torch.allclose(sess.last_logits, _uncached_logits(), atol=1e-4)
        n0_id = sess.active_node_id

        # Rewind to token 1: transformers 5.x negative-delta crop must succeed with 0 prefills!
        sess.rewind(1)
        assert prefill_calls == 0
        assert torch.allclose(sess.last_logits, _uncached_logits(), atol=1e-4)

        # Diverge to create n1, then checkout n0 using LCP crop with 0 prefills!
        sess.step(custom_text=" moon", source_tag="typed")
        assert sess.active_node_id != n0_id
        sess.checkout_node(n0_id)
        assert prefill_calls == 0
        assert torch.allclose(sess.last_logits, _uncached_logits(), atol=1e-4)

        # Update sliders: must crop to start - 1 instead of full prefill and match exact uncached logits!
        sess.set_global_coords({"plain_vs_poetic": 0.6})
        assert prefill_calls == 0
        assert torch.allclose(sess.last_logits, _uncached_logits(), atol=1e-4)
    finally:
        eng.close()


def test_gemma3_sliding_window_cache_crop_safety():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from explorer.session import _cache_can_crop, _cache_crop, _cache_seq_len

    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    torch.manual_seed(0)
    g3 = Gemma3ForCausalLM(cfg).eval()

    with torch.no_grad():
        # Case 1: Below sliding_window (cur=6 < 8) -> crop to 4 succeeds with exact logit parity
        x6 = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        pkv6 = g3(input_ids=x6, use_cache=True).past_key_values
        assert _cache_can_crop(pkv6, 4) is True
        pkv4, ok = _cache_crop(pkv6, 4)
        assert ok is True and _cache_seq_len(pkv4) == 4
        nxt_tok = torch.tensor([[5]], dtype=torch.long)
        lg_crop = g3(input_ids=nxt_tok, past_key_values=pkv4, use_cache=True).logits[0, -1]
        lg_ref = g3(input_ids=x6[:, :5], use_cache=False).logits[0, -1]
        assert torch.allclose(lg_crop, lg_ref, atol=1e-4)

        # Case 2: Past sliding_window (cur=12 >= 8) with record_past=False -> rejected cleanly with ZERO mutation
        x12 = torch.arange(1, 13, dtype=torch.long).unsqueeze(0)
        pkv12 = g3(input_ids=x12, use_cache=True).past_key_values
        assert _cache_can_crop(pkv12, 11) is False
        pre_lens = [int(l.keys.shape[-2]) for l in pkv12.layers]
        _, ok_past = _cache_crop(pkv12, 11)
        assert ok_past is False
        assert [int(l.keys.shape[-2]) for l in pkv12.layers] == pre_lens
        # Subsequent forward on untouched pkv12 succeeds without SDPA crash
        out_next = g3(input_ids=torch.tensor([[13]], dtype=torch.long), past_key_values=pkv12, use_cache=True)
        assert out_next.logits.shape[-1] == 64

        # Case 3: Past sliding_window with record_past=True -> 1st crop (12->9) ok, 2nd crop (9->7) safely rejected
        from transformers.cache_utils import DynamicCache

        pkv_rec = DynamicCache(config=cfg)
        for l in pkv_rec.layers:
            if hasattr(l, "record_past"):
                l.record_past = True
        pkv_rec = g3(input_ids=x12, past_key_values=pkv_rec, use_cache=True).past_key_values
        assert _cache_can_crop(pkv_rec, 9) is True
        pkv_rec, ok1 = _cache_crop(pkv_rec, 9)
        assert ok1 is True and _cache_seq_len(pkv_rec) == 9
        # 2nd crop past window would underfill sliding keys -> must be rejected cleanly with no SDPA crash!
        assert _cache_can_crop(pkv_rec, 7) is False
        _, ok2 = _cache_crop(pkv_rec, 7)
        assert ok2 is False
        # Forward after rejected 2nd crop still matches reference at position 9 -> 10
        lg_rec = g3(input_ids=torch.tensor([[10]], dtype=torch.long), past_key_values=pkv_rec, use_cache=True).logits[0, -1]
        lg_ref10 = g3(input_ids=x12[:, :10], use_cache=False).logits[0, -1]
        assert torch.allclose(lg_rec, lg_ref10, atol=1e-4)


def test_cache_clone_batch_immutability(tiny_model, monkeypatch):
    import copy as copy_mod

    from explorer.session import _cache_clone_batch, _cache_seq_len

    _, model = tiny_model
    x = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        pkv = model(input_ids=x, use_cache=True).past_key_values

    # Snapshot every source KV tensor
    snap = [(l.keys.clone(), l.values.clone()) for l in pkv.layers]

    # Verify single-allocation path does NOT call copy.deepcopy on DynamicCache
    monkeypatch.setattr(copy_mod, "deepcopy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("deepcopy called")))
    cloned = _cache_clone_batch(pkv, 4)
    assert cloned is not None
    assert _cache_seq_len(cloned) == 6

    with torch.no_grad():
        nxt = torch.tensor([[7], [8], [9], [10]], dtype=torch.long)
        out = model(input_ids=nxt, past_key_values=cloned, use_cache=True)
        assert _cache_seq_len(out.past_key_values) == 7
        for l in out.past_key_values.layers:
            l.keys.zero_()
            l.values.zero_()

    # Source cache must remain bit-identical
    assert _cache_seq_len(pkv) == 6
    for l, (k0, v0) in zip(pkv.layers, snap):
        assert torch.equal(l.keys, k0)
        assert torch.equal(l.values, v0)


def test_non_destructive_history_redo_and_nested_branching(tiny_model):
    tok, model = tiny_model
    eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    try:
        s0 = eng.sessions.start("El viento canta entre los pinos.", lookahead_tokens=4, max_branches=3)
        sid = s0["session_id"]
        eng.sessions._get(sid).auto_precompute_compass = False

        # Step 4 words on n0
        st = s0
        for _ in range(4):
            st = eng.sessions.step(sid, mode="word")
        orig_ids = list(st["committed_token_ids"])
        assert len(orig_ids) >= 4
        assert st["can_redo"] is False
        assert st["redo_remaining"] == 0
        assert all(isinstance(t["word_start_idx"], int) for t in st["tokens"])
        assert st["tokens"][0]["word_start_idx"] == 0

        # Rewind to token 2 -> can_redo is True
        sr = eng.sessions.rewind(sid, 2)
        assert sr["n_tokens"] == 2
        assert sr["can_redo"] is True
        assert sr["redo_remaining"] == len(orig_ids) - 2

        # Bug 3 regression: double rewind to token 1 must NOT truncate saved suffix of n0!
        sr2 = eng.sessions.rewind(sid, 1)
        assert sr2["n_tokens"] == 1
        assert sr2["can_redo"] is True
        assert sr2["redo_remaining"] == len(orig_ids) - 1

        # Bug 1 regression: stepping the SAME token [orig_ids[1]] must NOT truncate n0 or create duplicate node!
        ss_same = eng.sessions.step(sid, custom_token_ids=[orig_ids[1]], source_tag="step")
        assert len(ss_same["history"]) == 1
        assert ss_same["can_redo"] is True
        assert ss_same["redo_remaining"] == len(orig_ids) - 2

        # True redo(mode="token") and redo(mode="word") advance cleanly along orig_ids with 0 duplicate nodes
        rd1 = eng.sessions.redo(sid, mode="token")
        assert rd1["n_tokens"] == 3
        assert rd1["committed_token_ids"] == orig_ids[:3]
        assert len(rd1["history"]) == 1

        while rd1["can_redo"]:
            rd1 = eng.sessions.redo(sid, mode="word")
        assert rd1["committed_token_ids"] == orig_ids
        assert rd1["redo_remaining"] == 0
        assert len(rd1["history"]) == 1

        with pytest.raises(ValueError):
            eng.sessions.redo(sid, mode="invalid_mode")

        # Rewind n0 to 1 and diverge -> creates child n1 under n0
        eng.sessions.rewind(sid, 1)
        sn1 = eng.sessions.step(sid, custom_text=" alpha beta gamma", source_tag="typed")
        n1_id = sn1["active_node_id"]
        n1_full_ids = list(sn1["committed_token_ids"])
        assert n1_id != "n0"
        n1_meta = next(h for h in sn1["history"] if h["id"] == n1_id)
        assert n1_meta["parent_id"] == "n0"

        # Bug 2 regression: rewind n1 then checkout n0 must NOT truncate n1's saved suffix!
        eng.sessions.rewind(sid, 2)
        sc0 = eng.sessions.checkout(sid, "n0")
        assert sc0["committed_token_ids"] == orig_ids
        sc1 = eng.sessions.checkout(sid, n1_id)
        assert sc1["committed_token_ids"] == n1_full_ids

        # Multi-level nested branching: rewind n1 to 2 and diverge -> creates n2 with parent_id == n1!
        eng.sessions.rewind(sid, 2)
        sn2 = eng.sessions.step(sid, custom_text=" nestedleaf", source_tag="typed")
        n2_id = sn2["active_node_id"]
        n2_ids = list(sn2["committed_token_ids"])
        n2_meta = next(h for h in sn2["history"] if h["id"] == n2_id)
        assert n2_meta["parent_id"] == n1_id
        assert len(sn2["history"]) == 3

        # Checkout all 3 nodes cleanly
        assert eng.sessions.checkout(sid, "n0")["committed_token_ids"] == orig_ids
        assert eng.sessions.checkout(sid, n1_id)["committed_token_ids"] == n1_full_ids
        assert eng.sessions.checkout(sid, n2_id)["committed_token_ids"] == n2_ids
    finally:
        eng.close()


def test_multistanza_poem_framing_cut_doc_and_compass_stops(tiny_model):
    from explorer.frame import get_doc_shots

    tok, model = tiny_model

    # 1. Language-aware multi-stanza DOC_SHOTS across ES, PT, FR, DE, IT and non-English tgt_lang
    for src_l, tgt_l, expected_src_word, expected_tgt_word in [
        ("Spanish", "English", "plaza", "square"),
        ("Portuguese", "English", "praça", "square"),
        ("French", "English", "rivière", "river"),
        ("German", "English", "Fluss", "river"),
        ("Italian", "English", "fiume", "river"),
        ("Spanish", "French", "plaza", "silencieuse"),
    ]:
        s_src, s_tgt, shots = get_doc_shots(src_l, tgt_l)
        assert s_src == src_l and s_tgt == tgt_l and len(shots) == 2
        assert "\n\n" in shots[1][0] and "  " in shots[1][0] and "—" in shots[1][0]
        assert "\n\n" in shots[1][1] and "  " in shots[1][1] and "—" in shots[1][1]
        p = doc_prompt_text("Hola", src_lang=src_l, tgt_lang=tgt_l)
        assert expected_src_word in p and expected_tgt_word in p

    # 2. O(N) cut_doc decode count & zero partial closing-tag leak ('</translation' must NEVER leak!)
    decode_calls = 0
    orig_decode = tok.decode

    class _CountingTok:
        def decode(self, ids, *a, **k):
            nonlocal decode_calls
            decode_calls += 1
            return orig_decode(ids, *a, **k)

    ctok = _CountingTok()
    long_clean = tok("Stanza one line.\n  Indented — line.\n\nStanza two.\n" * 15, add_special_tokens=False)["input_ids"]
    decode_calls = 0
    kept = cut_doc(ctok, long_clean)
    assert kept == long_clean
    assert decode_calls == 1  # O(N) single-decode fast path!

    with_stop = long_clean + tok("\n</translation>\n<source>leaked", add_special_tokens=False)["input_ids"]
    decode_calls = 0
    trimmed_ids = cut_doc(ctok, with_stop)
    trimmed = orig_decode(trimmed_ids)
    assert decode_calls <= 18  # O(log N) binary search, NOT O(N)
    assert "\n\nStanza two." in trimmed
    assert "</translation" not in trimmed
    assert "</" not in trimmed
    assert "leaked" not in trimmed

    # Trailing partial tag (truncated mid-tag at lookahead limit) also stripped cleanly
    partial = tok("Line one.\n\nLine two.\n</trans", add_special_tokens=False)["input_ids"]
    assert "</trans" not in orig_decode(cut_doc(tok, partial))

    # 3. Split closing-tag protection in inspect_frontier & DOC_STOPS preservation in probe_compass
    eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    try:
        s0 = eng.sessions.start("Linea uno\n\nLinea dos", lookahead_tokens=4, max_branches=3)
        sess = eng.sessions._get(s0["session_id"])
        sess.auto_precompute_compass = False

        # Force top token to be '<' followed by '/translation>' (simulated via _rollout_frontier returning empty cut_doc)
        lt_ids = tok("<", add_special_tokens=False)["input_ids"]
        if lt_ids:
            fake_lg = torch.full_like(sess.last_logits, -20.0)
            fake_lg[lt_ids[0]] = 20.0
            sess.last_logits = fake_lg
            orig_roll = sess._rollout_frontier
            sess._rollout_frontier = lambda sl, *a, **k: (sl[:1], [[]], [None], [None], [0], 0.0)
            fr = sess.inspect_frontier(_schedule_precompute=False)
            sess._rollout_frontier = orig_roll
            assert fr["at_end"] is True
            assert fr["branches"] == []

        # Confirm _sample_doc_continuations preserves '\n\n' instead of cutting at legacy STOPS
        sess._prefill_full()
        poem_cont = tok("First stanza\n\nSecond stanza\n</translation>", add_special_tokens=False)["input_ids"]
        cleaned = orig_decode(cut_doc(tok, poem_cont))
        assert "\n\nSecond stanza" in cleaned and "</" not in cleaned
    finally:
        eng.close()


def test_semantic_ribbon_similarity_and_depth2_sub_branches(tiny_model):
    tok, model = tiny_model

    # Fix ca[:2] == cb[:2] false-merge bug!
    assert _ribbon_similarity("Camarón sings with duende", "Camarón sings with magic") < 0.72

    # Orthographic, contraction, and numeral/time normalization must merge (>= 0.72)
    assert _ribbon_similarity("at ten o'clock in the morning", "at 10:00 a.m.") >= 0.72
    assert _ribbon_similarity("at ten o’clock in the morning", "at ten in the morning") >= 0.72
    assert _ribbon_similarity("he doesn't know", "he does not know") >= 0.72
    assert _ribbon_similarity("she won’t stay", "she will not stay") >= 0.72

    eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    try:
        s0 = eng.sessions.start("Cuando canta Camarón, tiene duende.", lookahead_tokens=6, max_branches=4)
        sess = eng.sessions._get(s0["session_id"])
        sess.auto_precompute_compass = False

        # Verify every branch has has_newline: bool and sub_branches: list
        fr = s0["frontier"]
        assert fr["branches"]
        for br in fr["branches"]:
            assert isinstance(br["has_newline"], bool)
            assert isinstance(br["sub_branches"], list)
            if fr["regime"] in ("fork", "subtle"):
                for sb in br["sub_branches"]:
                    assert {"preview_text", "first_word_text", "prob", "share"} <= set(sb.keys())
            else:
                assert br["sub_branches"] == []

        # Force trivially linear logits -> sub_branches must be []
        lin_lg = torch.full_like(sess.last_logits, -30.0)
        word_id = tok(" Hello", add_special_tokens=False)["input_ids"][0]
        lin_lg[word_id] = 30.0
        sess.last_logits = lin_lg
        fr_lin = sess.inspect_frontier(_schedule_precompute=False)
        assert fr_lin["regime"] == "linear"
        assert all(br["sub_branches"] == [] for br in fr_lin["branches"])
    finally:
        eng.close()


def test_fork_head_ndirs2_fast_compass_and_async_precompute(tiny_model, tmp_path, monkeypatch):
    import time
    from contextlib import contextmanager

    from explorer.fork_head import ForkHead
    from ut import arch

    tok, model = tiny_model
    d = arch.hidden_size(model)

    # 1. ForkHead(n_dirs=2) fast-path: 0 forwards, 0 backwards, <10 ms, meta recommended_alpha
    fh_path = tmp_path / "fork_head_nd2.pt"
    head = ForkHead(d_model=d, hidden_dim=16, n_dirs=2)
    head.save(fh_path, meta={"layer": 0, "recommended_alpha": 0.42})

    eng = LiveTranslationEngine(model_key="tiny", fork_head_path=fh_path, tok=tok, model=model)
    try:
        # Install spy on LiveForkScorer.paused if available (or attach a lightweight paused() wrapper)
        pause_entries = 0
        orig_paused = getattr(eng.fork_scorer, "paused", None)

        @contextmanager
        def _spy_paused():
            nonlocal pause_entries
            pause_entries += 1
            if callable(orig_paused):
                with orig_paused():
                    yield
            else:
                yield

        eng.fork_scorer.paused = _spy_paused

        s0 = eng.sessions.start("Cuando canta Camarón, tiene duende.", lookahead_tokens=5, max_branches=4)
        sid = s0["session_id"]
        sess = eng.sessions._get(sid)
        sess.auto_precompute_compass = False
        assert pause_entries >= 1  # paused during frontier rollout!

        # Ensure >=2 branches exist at cursor for ForkHead fast-path
        if len(sess.last_frontier["branches"]) < 2:
            # Synthesize balanced 2-branch frontier from real token IDs
            t1 = tok(" magic", add_special_tokens=False)["input_ids"][0]
            t2 = tok(" spirit", add_special_tokens=False)["input_ids"][0]
            lg = torch.full_like(sess.last_logits, -20.0)
            lg[t1] = 1.0
            lg[t2] = 0.9
            sess.last_logits = lg
            sess.inspect_frontier(_schedule_precompute=False)

        fwd_calls = 0
        orig_fwd = arch.forward_hidden

        def _counting_fwd(*a, **k):
            nonlocal fwd_calls
            fwd_calls += 1
            return orig_fwd(*a, **k)

        monkeypatch.setattr(arch, "forward_hidden", _counting_fwd)
        comp = eng.sessions.probe_compass(sid)
        assert comp["source"] == "fork_head"
        assert fwd_calls == 0  # ZERO forward or backward passes!
        assert comp["probe_ms"] < 10.0
        assert comp["alpha_unit"] == pytest.approx(0.42)
        assert len(comp["readings"]) >= 2

        # Residual-norm-scaled fallback when recommended_alpha is absent
        del eng.fork_scorer.head.meta["recommended_alpha"]
        comp2 = sess.probe_compass(force=True)
        assert comp2["source"] == "fork_head"
        assert fwd_calls == 0
        assert comp2["alpha_unit"] > 0.0

        # Steer compass also pauses fork_scorer
        monkeypatch.setattr(arch, "forward_hidden", orig_fwd)
        before_pauses = pause_entries
        eng.sessions.steer_compass(sid, x=0.3, y=-0.2, max_new=4)
        assert pause_entries > before_pauses
    finally:
        eng.close()

    # 2. Async background Tier-2 compass precompute + stale cursor cancellation + clean close()
    eng2 = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    try:
        s0 = eng2.sessions.start("Cuando canta Camarón, tiene duende.", lookahead_tokens=4, max_branches=3)
        sid = s0["session_id"]
        sess = eng2.sessions._get(sid)
        sess._precompute_delay_s = 0.0
        sess._schedule_compass_precompute()

        # Poll briefly for background thread completion (tiny CPU takes <1s)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if eng2.sessions.get_state(sid, include_frontier=False)["compass_ready"]:
                break
            time.sleep(0.05)
        st_ready = eng2.sessions.get_state(sid, include_frontier=False)
        assert st_ready["compass_ready"] is True

        # Moving cursor invalidates stale compass cleanly
        st_moved = eng2.sessions.step(sid, mode="token")
        assert st_moved["n_tokens"] == 1
        if st_moved["frontier"]["regime"] != "fork":
            assert st_moved["compass_ready"] is False
    finally:
        eng2.close()


def test_directions_calibrate_caches_p0_and_steer_compass_parity(tiny_model, tmp_path, monkeypatch):
    from explorer.directions import calibrate, measured_kl
    from explorer.frame import build_prompt
    import explorer.session as sess_mod
    from ut import arch

    tok, model = tiny_model
    d = arch.hidden_size(model)
    ids, _ = build_prompt(tok, "Hola.")
    seq = ids + [5, 6, 7]
    b = len(ids) - 1
    rng = np.random.default_rng(5)
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)

    hk = CursorHook(model, 0)
    fwd_calls = 0
    orig_fwd = arch.forward_hidden

    def _counting_fwd(*a, **k):
        nonlocal fwd_calls
        fwd_calls += 1
        return orig_fwd(*a, **k)

    monkeypatch.setattr(arch, "forward_hidden", _counting_fwd)
    try:
        a = calibrate(model, hk, seq, b, v, target=0.002, mode="cursor", a0=1.0, iters=5)
        # 1 unsteered p0 + 5 bisection iterations == 6 forwards (NOT 2 * 5 = 10!)
        assert fwd_calls == 6
        monkeypatch.setattr(arch, "forward_hidden", orig_fwd)
        kl = measured_kl(model, hk, seq, b, torch.tensor(a * v, dtype=torch.float32), "cursor")
        assert abs(kl - 0.002) < 0.001
    finally:
        hk.remove()

    # steer_compass cached path vs forced full-prefill fallback parity with active global_coords
    axes_path = tmp_path / "axes.npz"
    np.savez(axes_path, plain_vs_poetic=(rng.normal(size=d) * 0.03).astype(np.float32))
    eng = LiveTranslationEngine(model_key="tiny", global_axes_path=axes_path, tok=tok, model=model)
    try:
        s0 = eng.sessions.start(
            "Cuando canta Camarón, tiene duende.",
            global_coords={"plain_vs_poetic": 0.8},
            lookahead_tokens=4,
            max_branches=3,
        )
        sid = s0["session_id"]
        sess = eng.sessions._get(sid)
        sess.auto_precompute_compass = False
        sess.step(mode="token")
        sess.step(mode="token")
        assert len(sess.committed_ids) == 2

        sess.probe_compass(n_samples=6, lookahead_tokens=4, seed=0)
        res_cached = sess.steer_compass(x=0.5, y=-0.4, max_new=6)

        monkeypatch.setattr(sess_mod, "_cache_can_crop", lambda *a, **k: False)
        res_fallback = sess.steer_compass(x=0.5, y=-0.4, max_new=6)
        assert res_cached["preview_ids"] == res_fallback["preview_ids"]
        assert res_cached["preview_text"] == res_fallback["preview_text"]
    finally:
        eng.close()


def test_fast_glide_and_lock_release_between_steps(tiny_model, monkeypatch):
    tok, model = tiny_model
    eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    try:
        s0 = eng.sessions.start("La casa está cerca del río.", lookahead_tokens=4, max_branches=3)
        sid = s0["session_id"]
        sess = eng.sessions._get(sid)
        sess.auto_precompute_compass = False

        rollout_calls = 0
        orig_roll = sess._rollout_frontier

        def _spy_roll(*a, **k):
            nonlocal rollout_calls
            rollout_calls += 1
            return orig_roll(*a, **k)

        monkeypatch.setattr(sess, "_rollout_frontier", _spy_roll)

        gen = eng.sessions.glide(sid, max_words=3, stop_on_fork=False)
        first_ev = next(gen)
        assert first_ev["event"] == "step"

        # Another thread MUST be able to acquire engine._lock between yielded glide steps!
        lock_acquired = {"ok": False}

        def _probe_lock():
            got = eng._lock.acquire(blocking=False)
            lock_acquired["ok"] = got
            if got:
                eng._lock.release()

        th = threading.Thread(target=_probe_lock)
        th.start()
        th.join(timeout=5.0)
        assert lock_acquired["ok"] is True

        rest = list(gen)
        assert rest[-1]["event"] == "paused"
        assert rest[-1]["reason"] in ("limit", "end")
        # Full rollout only runs once on final pause (NOT on every intermediate glided step!)
        assert rollout_calls <= 1
    finally:
        eng.close()


def test_global_axes_exact_layer_match_and_reft_primary(tiny_model, tmp_path):
    from explorer.global_axes import GlobalAxisBank
    from ut import arch

    tok, model = tiny_model
    d = arch.hidden_size(model)
    rng = np.random.default_rng(77)

    # Create a GlobalAxisBank strictly on layer [0] while engine.layer == 1!
    v0 = rng.normal(size=d).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    bank = GlobalAxisBank(
        vectors={"plain_vs_poetic": {0: v0}, "literal_vs_idiomatic": {0: v0}},
        scales={"plain_vs_poetic": 0.4, "literal_vs_idiomatic": 0.4},
        meta={"cursor_calibrated": True},
    )
    bank_path = tmp_path / "global_axes.npz"
    bank.save(bank_path)

    # (A) Without ReFT: global_hook attaches to exact bank layer [0], shifting layer-0 activations
    eng_bank = LiveTranslationEngine(model_key="tiny", layer=1, global_axes_path=bank_path, tok=tok, model=model)
    try:
        assert eng_bank.layer == 1
        assert eng_bank.global_hook is not None
        assert eng_bank.global_hook.layers == [0]
        assert eng_bank.status()["global_steering_mode"] == "global_axes_multilayer"

        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        rec_base = []
        rec_steered = []
        h = arch.layers(model)[0].register_forward_hook(lambda m, i, o: rec_base.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
        with torch.no_grad():
            model(input_ids=x, use_cache=False)
        h.remove()

        applied = eng_bank.apply_steering(global_coords={"plain_vs_poetic": 1.0}, global_start=2)
        assert applied is True
        h2 = arch.layers(model)[0].register_forward_hook(lambda m, i, o: rec_steered.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
        try:
            with torch.no_grad():
                model(input_ids=x, use_cache=False)
        finally:
            h2.remove()
            eng_bank.clear_steering()

        # Layer 0 shifted at positions >= 2 and untouched before 2!
        assert (rec_steered[0][0, :2] - rec_base[0][0, :2]).abs().max().item() < 1e-6
        assert torch.allclose(rec_steered[0][0, 2] - rec_base[0][0, 2], torch.tensor(0.4 * v0), atol=1e-5)
    finally:
        eng_bank.close()

    # (B) With BOTH GlobalAxisBank AND SpectralReFT loaded: ReFT-primary (zero double-steering!)
    reft_path = tmp_path / "spectral_reft.pt"
    reft = SpectralReFTController(
        model,
        layers=[0],
        axis_names=["literal_vs_idiomatic", "plain_vs_poetic"],
        rank=2,
    )
    reft.save(reft_path)
    reft.remove()

    eng_both = LiveTranslationEngine(
        model_key="tiny",
        layer=1,
        global_axes_path=bank_path,
        reft_path=reft_path,
        tok=tok,
        model=model,
    )
    eng_reft_only = LiveTranslationEngine(
        model_key="tiny",
        layer=1,
        reft_path=reft_path,
        tok=tok,
        model=model,
    )
    try:
        st = eng_both.status()
        assert st["global_steering_mode"] == "reft"
        assert set(st["global_axes_loaded"]) == {"plain_vs_poetic", "literal_vs_idiomatic"}

        s_both = eng_both.sessions.start("Hola mundo", global_coords={"plain_vs_poetic": 0.9}, lookahead_tokens=4)
        s_only = eng_reft_only.sessions.start("Hola mundo", global_coords={"plain_vs_poetic": 0.9}, lookahead_tokens=4)
        lg_both = eng_both.sessions._get(s_both["session_id"]).last_logits
        lg_only = eng_reft_only.sessions._get(s_only["session_id"]).last_logits
        assert torch.allclose(lg_both, lg_only, atol=1e-5)
    finally:
        eng_both.close()
        eng_reft_only.close()


def test_lora_merge_rejects_quantized_layers():
    import torch.nn as nn

    from explorer.adapters import LoRALinear

    # Mock Linear4bit (packed uint8 weight + quant_state)
    class Linear4bit(nn.Linear):
        def __init__(self, in_f, out_f):
            super().__init__(in_f, out_f, bias=False)
            self.weight = nn.Parameter(torch.randint(0, 255, (out_f * in_f // 2, 1), dtype=torch.uint8), requires_grad=False)
            self.weight.quant_state = object()

    # Mock Linear8bitLt (int8 weight + CB/SCB)
    class Linear8bitLt(nn.Linear):
        def __init__(self, in_f, out_f):
            super().__init__(in_f, out_f, bias=False)
            self.weight = nn.Parameter(torch.randint(-127, 127, (out_f, in_f), dtype=torch.int8), requires_grad=False)
            self.weight.CB = torch.ones(out_f)
            self.weight.SCB = torch.ones(out_f)

    for cls in (Linear4bit, Linear8bitLt):
        base = cls(16, 8)
        lora = LoRALinear(base, rank=2, alpha=4.0)
        with torch.no_grad():
            lora.lora_B.fill_(0.5)
        w_before = base.weight.clone()
        b_before = lora.lora_B.clone()
        for inplace in (False, True):
            with pytest.raises(RuntimeError, match="quantized"):
                lora.merge(inplace=inplace)
        # Ensure zero mutation occurred!
        assert torch.equal(base.weight, w_before)
        assert torch.equal(lora.lora_B, b_before)


def test_replay_engine_labels_yaml_and_map_loading(tmp_path):
    import re

    from explorer.live_engine import DEFAULT_RESULTS_DIR, ReplayEngine

    # (A) Real DEFAULT_RESULTS_DIR: map_duende_flamenco.json loads; zero bare 'reading N' labels; titles present
    rep = ReplayEngine(DEFAULT_RESULTS_DIR)
    texts = rep.list_texts()
    assert len(texts) >= 4
    for t in texts:
        assert isinstance(t["title"], str) and len(t["title"]) > 0
        assert "has_map" in t

    duende = rep.get_text_bundle("duende_flamenco")
    assert duende["has_map"] is True
    assert isinstance(duende["map"], dict) and len(duende["map"]["grid"]) == 225
    assert rep.get_text_map("duende_flamenco") is duende["map"]

    bare_re = re.compile(r"^reading\s+\d+$", re.IGNORECASE)
    for t in texts:
        b = rep.get_text_bundle(t["id"])
        for r in b["readings"]:
            assert not bare_re.match(r["label"]), f"Bare fallback label in {t['id']}: {r['label']}"
            assert r["label_source"] in ("curated", "auto")

    # (B) Custom readings_labels.yaml overrides title, description, and reading label
    custom_yaml = tmp_path / "custom_labels.yaml"
    custom_yaml.write_text(
        "texts:\n"
        "  duende_flamenco:\n"
        "    title: \"Camarón's Duende\"\n"
        "    description: \"Flamenco spirit test\"\n"
        "models:\n"
        "  g3_4b_pt:\n"
        "    duende_flamenco:\n"
        "      0: \"Curated Soul Reading\"\n",
        encoding="utf-8",
    )
    rep_cur = ReplayEngine(DEFAULT_RESULTS_DIR, labels_path=custom_yaml, model_key="g3_4b_pt")
    b_cur = rep_cur.get_text_bundle("duende_flamenco")
    assert b_cur["title"] == "Camarón's Duende"
    assert b_cur["description"] == "Flamenco spirit test"
    r0 = next(r for r in b_cur["readings"] if r["id"] == 0)
    assert r0["label"] == "Curated Soul Reading"
    assert r0["label_source"] == "curated"

    # (C) Missing YAML path degrades gracefully
    rep_missing = ReplayEngine(DEFAULT_RESULTS_DIR, labels_path=tmp_path / "does_not_exist.yaml")
    b_miss = rep_missing.get_text_bundle("duende_flamenco")
    assert b_miss["title"] == "Duende Flamenco"
    assert all(r["label_source"] == "auto" for r in b_miss["readings"])


def test_state_dict_alignment_metadata(tiny_model):
    tok, model = tiny_model
    eng = LiveTranslationEngine(model_key="tiny", tok=tok, model=model)
    try:
        poem = "En la plaza callada\n  cae la lluvia lenta.\n\nNadie cruza el umbral\n  —solo el viento."
        s0 = eng.sessions.start(poem, lookahead_tokens=4, max_branches=3)
        sid = s0["session_id"]
        eng.sessions._get(sid).auto_precompute_compass = False

        al0 = s0["alignment"]
        assert len(al0["source_lines"]) == 4
        assert [sl["stanza"] for sl in al0["source_lines"]] == [0, 0, 1, 1]
        assert al0["active_source_line"] == 0
        assert al0["spans"] == []

        # Commit two lines + stanza break + third line
        s1 = eng.sessions.step(sid, custom_text="In the quiet square\n  the slow rain falls.\n\nNo one crosses", source_tag="typed")
        al1 = s1["alignment"]
        n_toks = s1["n_tokens"]
        assert len(al1["spans"]) >= 3
        for sp in al1["spans"]:
            assert 0 <= sp["target_start"] < sp["target_end"] <= n_toks
            assert 0 <= sp["source_line"] < len(al1["source_lines"])
            assert 0.0 <= sp["confidence"] <= 1.0
        assert al1["active_source_line"] == 2

        # Multi-sentence prose alignment
        s_prose = eng.sessions.start("El sol brilla. La luna espera.", lookahead_tokens=4, max_branches=3)
        sid_p = s_prose["session_id"]
        eng.sessions._get(sid_p).auto_precompute_compass = False
        assert len(s_prose["alignment"]["source_lines"]) == 2
        s_p1 = eng.sessions.step(sid_p, custom_text="The sun shines. The moon waits.", source_tag="typed")
        assert len(s_p1["alignment"]["spans"]) == 2
        assert s_p1["alignment"]["spans"][0]["source_line"] == 0
        assert s_p1["alignment"]["spans"][1]["source_line"] == 1
    finally:
        eng.close()

