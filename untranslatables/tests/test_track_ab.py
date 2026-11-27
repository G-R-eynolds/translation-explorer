"""Tests for Track A (stylistic vs conceptual forks, live recipe strip, constellation) and Track B diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

import numpy as np
import pytest

from explorer.live_engine import LiveTranslationEngine
from explorer.recipe_live import (
    LiveEnglishRecipeDecomposer,
    branch_semantic_dispersion,
    extract_en_pool_from_model,
)
from explorer.server import create_server


def test_live_recipe_decomposer_and_sound_alike(tmp_path: Path):
    rng = np.random.default_rng(7)
    d = 32
    vecs = rng.normal(size=(20, d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    words = ["spirit", "magic", "duet", "charm", "soul"] + [f"w{i}" for i in range(15)]
    cent = np.zeros(d, dtype=np.float32)

    dec = LiveEnglishRecipeDecomposer(words=words, vectors=vecs, centroid=cent, layer=2)
    p = tmp_path / "en_pool.npz"
    dec.save(p)
    dec2 = LiveEnglishRecipeDecomposer.from_npz(p)

    # Target close to 'duet' + 'spirit'
    target = 0.65 * vecs[2] + 0.35 * vecs[0]
    res = dec2.decompose("duende", target, k=5)
    assert res["word"] == "duende"
    assert len(res["ingredients"]) > 0
    assert 0.0 <= res["leftover"] <= 1.0
    # 'duet' should be tagged sound_alike for 'duende' if selected
    duet_hits = [ing for ing in res["ingredients"] if ing["word"] == "duet"]
    if duet_hits:
        assert duet_hits[0]["sound_alike"] is True


def test_branch_semantic_dispersion_stylistic_vs_conceptual():
    # Tight paraphrases -> stylistic
    e_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_near = np.array([0.98, 0.20, 0.0, 0.0], dtype=np.float32)
    e_near /= np.linalg.norm(e_near)
    disp_st, kind_st = branch_semantic_dispersion(
        [{"emb": e_base, "prob": 0.5}, {"emb": e_near, "prob": 0.5}],
        regime="fork",
    )
    assert kind_st == "stylistic"
    assert disp_st < 0.075

    # Orthogonal conceptual poles -> conceptual
    e_orth = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    disp_co, kind_co = branch_semantic_dispersion(
        [{"emb": e_base, "prob": 0.5}, {"emb": e_orth, "prob": 0.5}],
        regime="fork",
    )
    assert kind_co == "conceptual"
    assert disp_co >= 0.075


def test_live_session_recipe_and_constellation_endpoints():
    eng = LiveTranslationEngine(model_key="tiny")
    tok, model = eng.ensure_loaded()
    eng.en_decomposer = extract_en_pool_from_model(model, tok, layer=1, max_words=24, batch=8)

    srv = create_server(host="127.0.0.1", port=0, mode="live", live_engine=eng, quiet=True)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        def post(path: str, payload: dict) -> dict:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))

        st = post("/api/session/start", {"source_text": "Cuando canta Camarón, tiene duende."})
        sid = st["session_id"]
        assert "fork_kind" in st["frontier"]
        assert "semantic_dispersion" in st["frontier"]

        rec = post("/api/session/recipe", {"session_id": sid, "word": "duende"})
        assert rec["word"] == "duende"
        assert len(rec["ingredients"]) > 0
        assert 0.0 <= rec["leftover"] <= 1.0

        # Ensure there are at least 2 branches or test constellation chord/blend on available branches
        n_br = len(st["frontier"]["branches"])
        if n_br >= 2:
            res_chord = post("/api/session/constellation", {"session_id": sid, "branch_indices": [0, 1], "action": "chord"})
            assert "[" in res_chord["committed_text"] and "/" in res_chord["committed_text"]
    finally:
        srv.shutdown()
        srv.server_close()
        eng.close()


def test_diagnose_unalignability_cpu_smoke(tmp_path: Path, monkeypatch):
    import diagnose_unalignability as du

    monkeypatch.setattr(du, "ROOT", tmp_path)
    monkeypatch.setattr(du, "default_en_pool_path", lambda m: tmp_path / f"{m}_en_pool.npz")
    out = du.run_diagnostic("tiny", limit=2)
    assert len(out["rows"]) == 2
    for r in out["rows"]:
        assert "source_sharpness" in r
        assert "en_leftover" in r
        assert "cycle_consistency_cc" in r
        assert "rig_proxy" in r
