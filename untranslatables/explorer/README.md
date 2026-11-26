# `explorer/`: Live Spectral Co-Translation Studio (v2)

Real-time token/word-by-word co-translation engine, interpretive fork discovery, continuous multi-layer `SpectralReFT` practice steering, and `<10 ms` amortised `ForkHead(n_dirs=2)` 2D steering compass.

## Quick Commands

```bash
# 1. Live v2 Studio (Gemma 3 12B NF4 on 12 GB RTX 4070, or g3_27b_pt on 22.5 GB L4 / A100)
python -m explorer serve --mode live --model g3_12b_pt \
  --global-axes results/explorer/g3_12b_pt/phase3/global_axes.npz \
  --lora results/explorer/g3_12b_pt/phase3/multiref_lora.pt \
  --reft results/explorer/g3_12b_pt/phase3/spectral_reft.pt \
  --fork-head results/explorer/g3_12b_pt/fork_head.pt \
  --port 8765

# 2. Train v2 artifacts for any model scale (g3_12b_pt | g3_27b_pt | qwen72b_base)
MODEL=g3_27b_pt bash explorer/queues/train_g3_scale.sh

# 3. Run real-GPU live v2 latency + formatting benchmark & record offline sessions
python -m explorer.bench_live --model g3_27b_pt

# 4. Build static HTML Phase-1 reports & multi-model offline showcase
python -m explorer report --model g3_27b_pt
python -m explorer report --model g3_4b_pt

# 5. Run full CPU + headless Chrome test suite (158 tests)
pytest -q tests/test_v2_cotranslate.py tests/test_explorer.py tests/test_phase1_multisteer.py \
  tests/test_phase2_live.py tests/test_phase3_adapters.py tests/test_phase4_fork_head.py \
  tests/test_integration_all_phases.py tests/e2e/
```

## Module Map

| File | Responsibility |
|---|---|
| `session.py` | Stateful KV-cached `CoTranslationSession` & `CoTranslationManager`: word/token stepping, exact rewind/redo, non-destructive history tree with LCP cache reuse, fast `glide()` to forks, depth-2 `sub_branches`, `<10 ms` `ForkHead` compass + Tier-2 gradient compass, source↔target line alignment, Gemma 3 `DynamicSlidingWindowLayer` crop safety. |
| `live_engine.py` | `LiveTranslationEngine` (unified ReFT-primary global steering + local `CursorHook` + `LiveForkScorer`) and `ReplayEngine` (Phase-1 archive + curated reading labels). |
| `server.py` | Threading HTTP/SSE server (`/api/session/*`, `/api/replay/*`, 2 MiB body cap, pre-header SSE validation) and multi-model standalone `build_offline_html()`. |
| `static/workbench.html` | Clean two-pane editorial web UI: live source editor & aligned line view, bounded scrollable translation canvas, keyboard-first stepping (`Space`, `Tab`, `Backspace`, `1–6`, `Shift+1–6`, `C`, `S`, `H`, `Esc`), 2D SVG compass, hierarchical Paths tree, and multi-model offline viewer. |
| `frame.py` | Doc/poem prompt framing (`build_doc_prompt`, `DOC_STOPS`) preserving `\n`, `\n\n`, and indentation, plus legacy few-shot framing and model loader. |
| `readings.py` | Semantic sentence-embedder clustering with uncentered raw-dispersion gate (`disp_min=0.035`) + orthographic normalization eliminating factual control false positives (`control_museo: 5 → 1`). |
| `directions.py` | OOM-safe per-sample gradients (`grads_at`), Fisher-whitened natural contrast directions, and KL-bisection calibration. |
| `multisteer.py` | Multi-layer `MultiLayerCursorHook` (with `sync_pos` KV-cache tracking) and fast cached-$p_0$ bisection. |
| `adapters.py` & `train_adapters.py` | Pure-PyTorch multi-reference `LoRALinear` (quantized merge guard) and continuous `SpectralReFTController` with automatic residual-norm-calibrated `max_delta_norm` (`10%` of median `||h||`). |
| `fork_head.py` | `ForkHead(n_dirs=2)` side-head predicting fork probability, entropy, Fisher $\lambda_1$, and orthonormal local steering directions in `<10 ms`, plus re-entrant `LiveForkScorer.paused()`. |
| `mapgrid.py` & `report.py` | 15×15 2D steering-plane sweep (with `spectral_basis` handling $R=2$ without rank collapse) and static HTML report generator. |
| `bench_live.py` | End-to-end real-GPU live v2 latency, formatting, slider, and session-recording benchmark. |
| `data/` | `texts.yaml` (9 benchmark texts), `tone_axes.yaml` (expanded multi-stanza poems & prose across ES/PT/FR/DE/IT), and `readings_labels.yaml` (curated reading labels for `g3_4b_pt` & `g3_27b_pt`). |

See [`results/explorer/SCALE_COMPARISON.md`](../results/explorer/SCALE_COMPARISON.md) for complete 4B vs 12B vs 27B empirical results.
