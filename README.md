# Translation Explorer

Code, experiments, and visualisations exploring how multilingual language models represent *untranslatables*, interpretive forks, translation blind spots (Fisher-information spectra), and real-time spectral steering during translation.

Built around a Spanish→English case study (*“Cuando canta Camarón, tiene duende; me gusta mucho el duende.”*, Lorca's *duende*, Bécquer, Machado, Rosalía de Castro, and related experiments).

## Quick Start (No GPU Needed)

### 1. Open the interactive offline workbench
Double-click or open in any browser:
- [`figures_for_dad/duende_workbench_offline.html`](figures_for_dad/duende_workbench_offline.html) — interactive 2D spectral direction compass, token-level entropy/Fisher fork underlines, and steering explorer across all 4 saved `g3_4b_pt` texts (`duende_flamenco`, `duende_bare`, `duende_cuento`, `becquer_xxi`).

### 2. Run the local replay server
```bash
cd untranslatables
.venv-cpu/bin/python -m explorer serve --mode replay --port 8765
```
Then open `http://127.0.0.1:8765/`.

### 3. Run the CPU test suite (39 tests)
```bash
cd untranslatables
.venv-cpu/bin/python -m pytest -q tests/test_explorer.py tests/test_phase1_multisteer.py tests/test_phase2_live.py tests/test_phase3_adapters.py tests/test_phase4_fork_head.py tests/test_integration_all_phases.py
```

## Repository Layout

| Path | Contents |
|---|---|
| `untranslatables/explorer/` | Real-time spectral translation workbench (Phases 1–4): streaming + fork-burst engine (`live_engine.py`, `server.py`, `static/workbench.html`), multi-layer steering & fast calibration (`multisteer.py`), gradient-space reading discovery (`grad_readings.py`), global tone/practice axes (`global_axes.py`, `data/tone_axes.yaml`), pure-PyTorch LoRA & continuous `SpectralReFT` (`adapters.py`, `train_adapters.py`), and amortised `ForkHead` side-head (`fork_head.py`). |
| `untranslatables/blind/` | Translation blind spots as a generalised Fisher-information eigenproblem (`G_src v = λ (G_L + εI) v`). |
| `untranslatables/sent/` & `untranslatables/internals/` | Sentence-level untranslatability, concept-space recipes, and internal representation probes. |
| `untranslatables/results/explorer/g3_4b_pt/` | Saved Gemma 3 4B base (`g3_4b_pt`) samples, embeddings, branch timelines, and steering sweeps for offline replay and plotting. |
| `figures_for_dad/` | Editorial visualisations (`duende_five_ways.html`, `duende_directions_numbers.png`/`.svg`, `duende_workbench_offline.html`, and earlier recipe/borrowing figures). |
| `translation-lattice/` | Earlier translation-lattice geometry experiments. |
| `colab/` & `results_colab/` | Colab notebooks, scripts, and 70B/0.5B experiment summaries. |
| `literature/` & `untranslatables-literature/` | Background reading notes and references. |

## GPU Workflows (When GPU Is Online)

Ready-to-run queue scripts live in `untranslatables/explorer/queues/`:
- `phase1_finish_and_map.sh` — finish remaining texts in `texts.yaml`, build `map_duende_flamenco.json`, refresh report.
- `phase3_axes_and_train.sh` — extract universal Fisher-whitened global axes + train diversity-preserving multi-reference LoRA and continuous `SpectralReFT`.
- `phase4_train_fork_head.sh` — extract hidden-state fork dataset and train `ForkHead`.

Launch the full live workbench wired to trained artifacts:
```bash
cd untranslatables
python -m explorer serve --mode live --model g3_4b_pt \
  --global-axes results/explorer/g3_4b_pt/phase3/global_axes.npz \
  --reft results/explorer/g3_4b_pt/phase3/spectral_reft.pt \
  --fork-head results/explorer/g3_4b_pt/fork_head.pt \
  --port 8765
```
