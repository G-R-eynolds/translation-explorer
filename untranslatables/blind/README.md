# blind/: translation blind spots as a Fisher-information spectrum

This implements `../../HANDOFF_blind_spots.md`. The predictions are in `PREDICTIONS.md`, written before any run.

```
python -m blind <stage> --model g3_4b [--set e1|e2|e3|atlas] [--kv key=value ...] [--force]
remote/run.sh blind blind/queues/main.sh      # E0-E3 on g3_4b, E1+E2 on 3b/aya8b/g3_12b, figures, draft report
remote/run.sh atlas blind/queues/atlas.sh     # E4, only after the E1 validation passes (stop point 2)
.venv-cpu/bin/python -m pytest -q tests/test_blind.py
```

| module | what |
|---|---|
| `common.py` | paths, stored layers (Gemma Scope 2 detailed depths), channel prompts, items, resumable stages |
| `material.py` | YAML checks; builds the atlas corpus from FLORES-101 |
| `fisher.py` | GPU: shared-source prompts, additive gradient leaves, sampling, per-sample gradients, KL, steering |
| `spectrum.py` | CPU: Gram-matrix generalised eigenproblem, matched split-half null, U, spans, directions |
| `stages.py` | e0, collect, spectra, heatmaps, fdcheck, steer, sae, practices, validate, atlas_dirs/langs/cluster |
| `figures.py`, `report.py` | §8 figures (PNG + SVG, `captions.json`) and the draft `results/blind/report.md` |

**Outputs:** `results/blind/<model>/`. The gradients go in `cache/blind/<model>/grads/<set>/<item>/<channel>.npz`, as scaled float16, so that `remote/pull.sh` never copies gigabytes back to the laptop.

## Decisions and deviations from the handoff

- **Models**
  - `ayax8b` (Aya Expanse) is still gated for the desktop account, so `aya8b` (Aya 23 8B) replaces it, as in the internals run.
- **Data**
  - **Atlas corpus:** FLORES-101 devtest Spanish replaces FLORES+, which is gated. The sample is 200 sentences of 8–25 words, taken round-robin by domain, with near-duplicates removed.
- **Prompts and sampling**
  - **Tokenisation:** the prompt is built from separately tokenised pieces (chat prefix, text, instruction plus chat tail). The boundary between the text and "\n\n" is therefore never merged into one token, and this is the same in every channel.
  - **Identical-h check:** E0 asserts that the source ids are identical across channels. It also checks the source states numerically. On Gemma 3 4B they are exactly equal when two channel prompts have the same length; otherwise they differ by about 0.4% in per-position relative norm, which is bf16 kernel-shape noise (prefix-only vs full prompt gives the same). First run: 108 of 120 channel comparisons exactly equal, the rest at most 1.3%. The pass limit is 3%. Max |Δ|/RMS is not usable here because a few of Gemma's residual dimensions reach about 8e4.
  - **E3 output length:** the gloss, footnote and thick practices get 6|s|+48 new tokens instead of 2|s|+16, which would cut a gloss off mid-sentence. Length-normalised numbers are reported next to the raw ones.
- **Gradients**
  - A zero leaf is added at the source positions of each stored layer, rather than a detached copy being spliced in. The gradient is then the total derivative, it still flows to earlier stored layers, and each batch row gets its own per-sample gradient.
- **Spectra and statistics**
  - **Extra U variant:** besides raw U, every row also has a trace-normalised U (only the shape of each channel's sensitivity counts, not its overall size) and a length-normalised one.
  - **Null and directions:** U uses the index-wise median null spectrum over 20 splits; the U_null interval is that formula applied to each split's own null. Directions for steering, SAE and the finite-difference check come from all K samples; how many of them count as above the null comes from the split analysis.
  - **Headline layer:** the 50% layer is the headline, fixed in advance in PREDICTIONS.md; robustness is reported at every layer × ε.
- **Checks**
  - **Finite-difference step for "seen" directions:** the step α is set so that the larger of the two predicted KLs is 0.5 nats. For a blind direction that is the source channel, as the handoff says; for a "seen" direction the source channel is insensitive, so the target channel sets the step.
  - **Test 4 (steering flips):** pre-counted automatically from the `flip` markers in `data/builtin.yaml`. The final count comes from reading `steering_e1.jsonl`.
