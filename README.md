# Translation Explorer

Interactive **v2 Live Co-Translation Studio** and empirical interpretability toolkit for exploring how multilingual language models represent *untranslatables*, interpretive forks, translation blind spots (Fisher-information spectra), and continuous real-time spectral steering across whole poems and literary paragraphs.

Built around Spanish→English (and Romance/Germanic) literary case studies: Camarón's *duende*, Bécquer's *Rimas*, Machado's *Proverbios y cantares*, Rosalía de Castro, *sobremesa*, and *saudade*.

## Highlights (v2 Live Co-Translation Studio)
- **True real-time token/word-by-word co-translation:** Persistent KV-cached stepping, instant non-destructive rewind (with Gemma 3 `DynamicSlidingWindowLayer` safety), exact **Redo**, **Auto-Glide to Fork**, and a visual **branch tree** (`Paths`) with LCP cache reuse.
- **Adaptive uncertainty & depth-2 lookahead:** Linear/quiet on predictable words; rich semantic branches + depth-2 *"new token of the new token"* sub-forks on complex interpretive forks.
- **<10 ms zero-backward 2D Spectral Compass:** `ForkHead(n_dirs=2)` predicts fork probability, entropy, Fisher $\lambda_1$, and local orthonormal steering directions in **3.3–5.7 ms** (a **650×–4,100× speedup** over gradient backprop), with on-demand Tier-2 gradient compass fallback.
- **Auto-scaled continuous `SpectralReFT` tone/practice sliders:** Live continuous control over *Literal ↔ Idiomatic*, *Plain ↔ Poetic*, *Foreignising ↔ Domesticating*, and *Concise ↔ Glossed* calibrated automatically to Gemma 3's residual-stream scale.
- **Long-form & formatted poetry support:** Preserves line breaks, stanza breaks (`\n\n`), and indentation (`Tab`), with interactive source↔target line alignment.
- **Trained multi-scale checkpoints included:** Complete v2 artifacts (`global_axes.npz`, `multiref_lora.pt`, `spectral_reft.pt`, `fork_head.pt`) for **`g3_4b_pt`**, **`g3_12b_pt` (NF4)**, and **`g3_27b_pt` (NF4)**, plus ready-to-run support for **`qwen72b_base` (Qwen2.5-72B NF4)**.

---

## Quick Start (No GPU Needed)

### 1. Open the multi-model offline showcase
Double-click or open in any browser:
- [`figures_for_dad/duende_workbench_offline.html`](figures_for_dad/duende_workbench_offline.html) — standalone offline workbench with model switcher across **`g3_27b_pt`**, **`g3_4b_pt`**, and **`g3_12b_pt`**, interactive 2D spectral direction compass, token-level fork underlines, and recorded live v2 co-translation sessions.
- See [`untranslatables/results/explorer/SCALE_COMPARISON.md`](untranslatables/results/explorer/SCALE_COMPARISON.md) for the complete 4B vs 12B vs 27B quantitative benchmark, live GPU latencies, literary excerpts, and Colab CU cost breakdown.

### 2. Run the local replay server
```bash
cd untranslatables
python -m explorer serve --mode replay --results-dir results/explorer/g3_27b_pt --port 8765
```
Then open `http://127.0.0.1:8765/`.

### 3. Run the full CPU + headless Chrome test suite (158 tests)
```bash
cd untranslatables
pytest -q tests/test_v2_cotranslate.py tests/test_explorer.py tests/test_phase1_multisteer.py \
  tests/test_phase2_live.py tests/test_phase3_adapters.py tests/test_phase4_fork_head.py \
  tests/test_integration_all_phases.py tests/e2e/
```

---

## Live GPU Workflows

### Desktop RTX 4070 (12 GB VRAM) — Gemma 3 12B NF4 (`g3_12b_pt`)
```bash
cd untranslatables
python -m explorer serve --mode live --model g3_12b_pt \
  --global-axes results/explorer/g3_12b_pt/phase3/global_axes.npz \
  --lora results/explorer/g3_12b_pt/phase3/multiref_lora.pt \
  --reft results/explorer/g3_12b_pt/phase3/spectral_reft.pt \
  --fork-head results/explorer/g3_12b_pt/fork_head.pt \
  --port 8765
```

### Colab L4 (22.5 GB) / A100 — Gemma 3 27B NF4 (`g3_27b_pt`) or Qwen2.5-72B (`qwen72b_base`)
```bash
cd untranslatables
python -m explorer serve --mode live --model g3_27b_pt \
  --global-axes results/explorer/g3_27b_pt/phase3/global_axes.npz \
  --lora results/explorer/g3_27b_pt/phase3/multiref_lora.pt \
  --reft results/explorer/g3_27b_pt/phase3/spectral_reft.pt \
  --fork-head results/explorer/g3_27b_pt/fork_head.pt \
  --port 8765
```
Or train/refresh any scale (`g3_12b_pt`, `g3_27b_pt`, `qwen72b_base`) end-to-end:
```bash
cd untranslatables
MODEL=g3_27b_pt bash explorer/queues/train_g3_scale.sh
python -m explorer.bench_live --model g3_27b_pt
```

---

## Repository Layout

| Path | Contents |
|---|---|
| `untranslatables/explorer/` | v2 Live Co-Translation Studio: KV-cached session engine (`session.py`), live & replay engines (`live_engine.py`), HTTP/SSE server & multi-model offline builder (`server.py`), editorial UI (`static/workbench.html`), doc/poem framing (`frame.py`), reading discovery (`readings.py`), multi-layer steering (`multisteer.py`), global axes & expanded multi-stanza `tone_axes.yaml`, auto-scaled `SpectralReFT` & multi-ref LoRA (`adapters.py`, `train_adapters.py`), `ForkHead(n_dirs=2)` (`fork_head.py`), and live GPU benchmark (`bench_live.py`). |
| `untranslatables/results/explorer/` | Saved checkpoints, Phase-1 outputs, live GPU benchmarks, recorded sessions, static reports, and [`SCALE_COMPARISON.md`](untranslatables/results/explorer/SCALE_COMPARISON.md) for `g3_4b_pt`, `g3_12b_pt`, and `g3_27b_pt`. |
| `untranslatables/tests/` | 73 unit/integration tests + 85-test 4-tier E2E suite (`tests/e2e/`) including headless Chrome browser tests. |
| `figures_for_dad/` | Editorial visualisations and multi-model standalone offline workbench (`duende_workbench_offline.html`). |
| `untranslatables/blind/`, `sent/`, `internals/` | Translation blind spots (generalised Fisher eigenproblem), sentence-level untranslatability, and internal probes. |
| `colab/` | Colab training & live-tunnel launch helpers (`colab/explorer/colab_live.sh`). |
