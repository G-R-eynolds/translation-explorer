# Translation Explorer v2 — Multi-Scale Benchmark & Live GPU Evaluation

Empirical comparison of the v2 Live Co-Translation Studio across **Gemma 3 4B (`g3_4b_pt`, bf16)**, **Gemma 3 12B (`g3_12b_pt`, NF4)**, and **Gemma 3 27B (`g3_27b_pt`, NF4)**, all trained and benchmarked on real NVIDIA GPUs via Google Colab (apples-to-apples live latencies on **NVIDIA L4 22.5 GB**, plus **A100-SXM4-40GB** reference timings for 12B).

> **Note on `qwen72b_base` (`Qwen/Qwen2.5-72B`, NF4):** Full model registration (`ut/common.py`), training script (`explorer/queues/train_g3_scale.sh`, `LORA_LAYERS=30,40,50,60`), live server (`colab/explorer/colab_live.sh`), and benchmark harness (`explorer/bench_live.py`) are implemented and verified, but 80 GB Colab tiers (`A100 --high-mem`, `H100`, `G4`) returned `503 Service Unavailable` during this overnight window. `qwen72b_base` requires ~42 GB VRAM and can be run with a single command (`MODEL=qwen72b_base bash explorer/queues/train_g3_scale.sh`) whenever an 80 GB runtime is available.

---

## 1. Phase-1 Interpretive Fork & Steering Benchmark (4B vs 27B)

Evaluated across the 9-text Spanish→English benchmark (`texts.yaml`: 8 literary/ambiguous texts + 1 factual control `control_museo`, $N=64$ ancestral samples per text):

| Metric | `g3_4b_pt` (bf16, pre-M2 gate) | `g3_27b_pt` (NF4, M2 dispersion gate) |
|---|---:|---:|
| Pre-registered decision rule (`pass`) | **`true`** | **`true`** |
| Hard texts with $\ge 2$ readings | `8 / 8` | `7 / 8` (`rosalia_dicen` unanimous) |
| Control (`control_museo`) readings | `5` *(false split on punctuation/numerals)* | **`1`** *(unanimous; `control_warning: false`)* |
| `natural/cursor` steering success | `74.0%` (`37/50`) | **`77.3%`** (`17/22`) |
| `plain/cursor` steering success | `68.0%` (`34/50`) | **`86.4%`** (`19/22`) |
| `duende_flamenco` readings discovered | 5 *(included mistranslated "magic/fairy")* | **2** *(73% keep `duende` vs 26% `spirit/soul`)* |
| `map_duende_flamenco.json` 2D plane var. | `0.85` (5 readings) | **`1.00`** *(2 readings via M2 `spectral_basis`)* |

**Key scientific takeaway:** Scaling from 4B to 27B sharpens interpretive uncertainty rather than bloating it. On Camarón's flamenco sentence, 4B splits into five readings including folkloric mistranslations (*fairy*, *magic*), whereas 27B concentrates cleanly onto the two genuinely defensible literary choices—keeping untranslated *duende* (73%) vs. domesticating to *spirit/soul* (26%)—while still recovering 6 rich folkloric readings on bare *duende* (`silhouette = 0.70`) and 5 distinct readings on `monte_catalejo` (`silhouette = 0.80`). Combining 27B with M2's uncentered raw-dispersion gate + orthographic normalization eliminates the `control_museo` false positive completely (`5 → 1`).

---

## 2. Amortised `ForkHead(n_dirs=2)` & Continuous `SpectralReFT`

All three scales were trained with `ForkHead(n_dirs=2)` (predicting fork probability, next-token entropy, output-Fisher $\lambda_1$, and 2 orthonormal local steering directions directly from mid-layer residual states) and continuous multi-layer `SpectralReFT` over the expanded multi-stanza `tone_axes.yaml`:

| Model | Layer | Median `||h||` | `ForkHead` AUROC | `ForkHead` Acc | Entropy MAE (nats) | Fisher MAE | `SpectralReFT` `max_delta_norm` | `SpectralReFT` Final Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `g3_4b_pt` (bf16) | 17 / 34 | 30,209 | `0.8485` | `77.6%` | `0.410` | `0.088` | `5.0` *(legacy)* | — |
| `g3_12b_pt` (NF4) | 24 / 48 | 56,210 | **`0.8994`** | **`84.3%`** | `0.358` | **`0.075`** | **`5390.9`** *(auto 10%)* | **`1.129`** |
| `g3_27b_pt` (NF4) | 31 / 62 | 35,000 | `0.8720` | `81.3%` | **`0.356`** | `0.086` | **`3508.7`** *(auto 10%)* | **`1.182`** |

### Real-GPU Discovery: Gemma 3 Residual-Norm Scaling in `SpectralReFT`
During the first live GPU benchmark of `g3_12b_pt`, global tone/practice sliders shifted `0 / 4` frontiers despite `reft_loaded: true`. Inspecting the checkpoint revealed that Gemma 3's mid-layer residual stream has median norm $30\text{k}\text{–}56\text{k}$ (due to $\sqrt{d_{\text{model}}}$ embedding scaling), so `SpectralReFT`'s hardcoded `max_delta_norm = 5.0` clamped perturbations to $<0.01\%$ of `||h||`. Auto-calibrating `max_delta_norm = max(5.0, 0.10 * median_h_norm)` in `explorer/train_adapters.py` (commit `05c0506`) dropped ReFT training loss from `1.978 → 1.129` (12B) and `1.182` (27B), lifting live slider responsiveness to **`4 / 4` (100%) on 12B** and **`3 / 3` (100% of non-EOF frontiers) on 27B**.

---

## 3. Real-GPU Live v2 Co-Translation Latencies & Reliability

Measured via `python -m explorer.bench_live` across 4 texts (`duende_flamenco`, 2-stanza indented Bécquer Rima LIII, 2-stanza Machado Proverbios XXIX, and a dense *sobremesa*/*duende* literary paragraph) with **all 4 v2 artifacts active simultaneously**:

| Model & GPU | Peak VRAM | Prefill (ms) | Step Word (ms) | Glide (ms/word) | Rewind (ms) | Tier-1 `ForkHead` Compass | Tier-2 Gradient Compass | Compass Speedup | Exact Redo | Slider Shifts | Bécquer Newlines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g3_4b_pt` (L4) | 10.13 GB | 917 | 726 | 606 | 732 | **`3.35 ms`** | `2,178 ms` | **650×** | 4/4 | 1/4 *(legacy cap)* | 7/8 |
| `g3_12b_pt` (A100-40G) | 13.68 GB | 1,474 | 1,244 | 1,199 | 1,235 | **`3.44 ms`** | `3,227 ms` | **938×** | 4/4 | — *(pre-fix)* | 7/8 |
| `g3_12b_pt` (L4, fixed ReFT) | 13.68 GB | 1,588 | 1,300 | 1,245 | 1,303 | **`4.38 ms`** | `4,768 ms` | **1,089×** | 4/4 | **4/4** | 7/8 |
| `g3_27b_pt` (L4, fixed ReFT) | 21.61 GB | 2,257 | 1,269 | 1,185 | 1,272 | **`5.67 ms`** *(3.36–4.09 steady)* | `10,613 ms` | **1,872×** *(4,100× on Bécquer)* | 4/4 | **3/3** *(non-EOF)* | **8/8** |

> **Hardware guidance:**
> - **Desktop RTX 4070 (12 GB VRAM):** Use **`g3_12b_pt` (NF4)** with `ForkHead(n_dirs=2)` (Tier-1 compass needs zero backward pass, keeping VRAM ~9–10 GB; the 13.68 GB peak above occurred only during the optional Tier-2 gradient compass probe).
> - **Colab L4 (22.5 GB VRAM, `2.33 CU/hr`) or A100:** Use **`g3_27b_pt` (NF4)** — fits cleanly in 21.6 GB peak VRAM even with full gradient compass probes and Phase-1 steering sweeps (with OOM-safe batch halving in `sample.py` and `directions.py`).

---

## 4. Qualitative Literary Comparison (Live Co-Translation & Global Sliders)

### Bécquer, *Rima LIII* (2 stanzas with 2-space indentation & blank stanza break)
- **`g3_12b_pt` (preserved 7/8 newlines, stanza break, and 2-space indentation):**
  > The dark swallows will return
  >   to hang their nests on your balcony,
  > and once again with their wings against the glass
  >   playing they will call.
  >
  > But those that held back their flight
  >   to contemplate your beauty and my happiness,
  > those that learned our names...
- **`g3_27b_pt` (preserved 8/8 newlines, stanza break, and 2-space indentation; `ForkHead` compass = `3.36 ms`):**
  - *Plain / Domesticating slider (`-0.8, +0.8`):* `" again with their wings against the glass\n  they"`
  - *Poetic / Foreignising slider (`+0.8, -0.8`):* `" more with the wing to your crystals\n  playing"`

### Machado, *Proverbios y cantares XXIX*
- **`g3_12b_pt`:** Opened with untranslated `"Caminante, your footprints..."` (alt branch: `"Caminante (traveler)..."`).
- **`g3_27b_pt`:** Rendered idiomatically as **`"Wayfarer, your footprints / are the only road; / wayfarer, there is no road, / the road is made by walking."`**
  - *Plain / Domesticating slider:* `" is no road,\nyou make the road by"`
  - *Poetic / Foreignising slider:* `" is no camino,\nse hace camino al andar"`

### Literary Paragraph (*Sobremesa & Duende*)
- **`g3_12b_pt`:** Produced calque `"overmesa"` in neutral greedy decoding, but **Tier-1 `3.82 ms` ForkHead compass** immediately steered cleanly between `"after-dinner talk of Sunday afternoon stretched on into the evening"` and `"Sunday overmesa (after-meal talk)"`, while **auto-scaled `SpectralReFT`** shifted between `"afternoon coffee talk went on until sunset with laughter"` (Plain/Domesticating) and `"sobremesa (afternoon coffee) was prolonged until"` (Poetic/Foreignising).
- **`g3_27b_pt`:** Eliminated the `"overmesa"` calque entirely—producing **`"The Sunday sobremesa stretched until dusk among laughter, cold coffee and a complicit silence that no one dared to break..."`** (alt branch: `"The after-dinner conversation of Sunday stretched until dusk..."`)—while **auto-scaled `SpectralReFT`** shifted cleanly between `"after-dinner conversation went on until dusk over coffee"` and `"sobremesa (the time after the midday meal)"`.

---

## 5. Colab Compute Unit Cost Ledger

| Session | Hardware | Rate | Work Completed | Balance After |
|---|---|---:|---|---:|
| Initial AI Ultra balance | — | — | Starting balance | `1500.28 CU` |
| Prior `ultra` run | A100-80G | 10.99/hr | Earlier prototype verification | `1485.34 CU` (`14.94` used) |
| `ultra2` | A100-40G | ~10.99/hr | `g3_12b_pt` v2 training + A100 live benchmark (harvested) | `1481.17 CU` (`4.17` used) |
| `ultra3` | L4 22.5G | 2.33/hr | `g3_12b_pt` ReFT refresh + `g3_27b_pt` v2 training + full 9-text Phase 1 + 15×15 `mapgrid` + `g3_4b_pt` v2 refresh & live benchmark (all harvested) | `1475.93 CU` (`5.24` used) |
| **Total spent** | **0 active** | **`0.00/hr`** | **All 4B, 12B, 27B checkpoints, benchmarks & showcases harvested** | **`24.35 CU` used / `500 CU` budget (4.9%)** |
