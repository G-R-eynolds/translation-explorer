"""Cross-phase integration test on CPU (`tiny` model): Phases 1 + 2 + 3 + 4 wired together."""
from pathlib import Path
import numpy as np
import pytest
import torch

from explorer.frame import load


@pytest.fixture(scope="module")
def tiny():
    return load("tiny")


def test_full_stack_live_engine_with_axes_reft_and_fork_head(tiny, tmp_path: Path):
    from explorer.adapters import SpectralReFTController
    from explorer.fork_head import ForkHead
    from explorer.global_axes import extract_global_axes
    from explorer.live_engine import LiveTranslationEngine
    from explorer.train_adapters import train_spectral_reft
    from ut import arch

    tok, model = tiny
    d = arch.hidden_size(model)
    layer = arch.n_layers(model) // 2

    # Phase 3a: extract GlobalAxisBank on 2 items and save to .npz
    bank = extract_global_axes(model, tok, layers=[layer], max_items=2, batch=4)
    axes_path = tmp_path / "global_axes.npz"
    bank.save(axes_path)

    # Phase 3b: train tiny SpectralReFT for 2 steps and save checkpoint
    reft_path = tmp_path / "spectral_reft.pt"
    res = train_spectral_reft(model, tok, layers=[layer], rank=4, steps=2, save_path=reft_path)
    res["controller"].remove()
    assert reft_path.exists()

    # Phase 4: save a ForkHead checkpoint
    fh_path = tmp_path / "fork_head.pt"
    head = ForkHead(d_model=d, hidden_dim=32, n_dirs=0)
    head.save(fh_path, meta={"layer": layer})

    # Phase 2: boot LiveTranslationEngine wired to ALL artifacts simultaneously
    eng = LiveTranslationEngine(
        model_key="tiny",
        layer=layer,
        global_axes_path=axes_path,
        fork_head_path=fh_path,
        reft_path=reft_path,
        tok=tok,
        model=model,
    )
    try:
        st = eng.status()
        assert st["loaded"] is True
        assert st["fork_head_loaded"] is True
        assert st["reft_loaded"] is True
        assert len(st["global_axes_loaded"]) == 4

        # Stream with global slider alias AND canonical axis name + live ForkHead scores
        events = list(
            eng.stream_translate(
                "Cuando canta Camarón, tiene duende.",
                max_new=5,
                global_coords={"plain_poetic": 0.6, "foreignising_vs_domesticating": -0.4},
            )
        )
        assert len(events) >= 1
        assert "fork_head_prob" in events[0]
        assert 0.0 <= events[0]["fork_head_prob"] <= 1.0
        assert events[0]["fork_head_entropy"] >= 0.0

        # Probe fork + steer combining local 2D compass + global axes + ReFT
        fork = eng.probe_fork("Cuando canta Camarón, tiene duende.", prefix_token_ids=[], n_samples=6, lookahead_tokens=4)
        steered = eng.steer_at(
            fork["fork_id"],
            x=0.7,
            y=-0.3,
            global_coords={"plain_poetic": 0.8},
            max_new=6,
        )
        assert steered["global_coords_applied"] is True
        assert isinstance(steered["text"], str)
        assert abs(sum(steered["shares"].values()) - 1.0) < 1e-2
    finally:
        eng.close()
