# Untranslatables inside multilingual models

Design and rationale: `../HANDOFF_untranslatables.md` (v2) and `../literature/NOTES_*.md`. Predictions,
written before any run: `PREDICTIONS.md`. **Nothing has run on the GPU yet.** Every stage has run end to end on the
laptop CPU with a tiny random Llama, and thread 2 also with Qwen2.5-0.5B. Unit tests: `.venv-cpu/bin/python -m pytest -q`.

## Layout

```
ut/              shared core
  common.py        model registry (3b, 8b, qwen7b, aya8b; qwen05/tiny for CPU tests), loading, chat, sampling, ETAs
  represent.py     span-mean residual states at every layer (disk cache), steering + patching hooks
  space.py         concept space at a layer: pools and target words centred per language and context kind; cognates
  lens.py          logit lens + prefix-token sets (method of Wendler et al. [2402.10588], reimplemented)
  lm.py            token log-probs;  nli.py  meaning clustering, H, KLE;  stats.py  AUROC/bootstrap/matching
  words.py         word data loader + validator (python -m ut.words)
data/            words/<lang>.yaml (84 words), distinctions.yaml (6 x 40 items), human_renderings.yaml (to fill),
                 pools/ (8,000 frequency-matched words per language, MUSE pairs and dictionaries)
pools.py         §5.2 pools: --build (done), --casing (GPU, once), --extract
familiarity.py   §4 gate + tick-box page       layers.py   §5.3 layer choice
recipe.py        §5.1/5.4/6.5 recipes, usage clouds, nulls, pivots, layer dependence, loanwords
baselines.py     §5.5 leftover vs CSLS / frequency / tokens      patching.py  §5.6 what the model would say
latent_lang.py   §6.1 English share by layer   coverage.py  §6.2   distinctions.py  §6.3   steering.py  §6.4
spread.py        §7.1-7.2 sampling under bare / gloss / gloss+sense / gloss+context, + iterative re-asks
ignorance.py     §7.2-7.5 diagnostics, other ignorance signals, 2x2 confusion, human reference
probes.py        §7.6 stretch               figures.py  13 figures (§9)        report.py  results/report.html
run_stage.sh     stages 0-6 in the handoff's order, with timings and stop points
```

## Running (when the desktop is free)

```bash
remote/run.sh s0 ./run_stage.sh 0 3b && remote/pull.sh      # once: pool casing; pull brings data/pools back
remote/run.sh s1 ./run_stage.sh 1 3b                        # STOP: tick results/3b/familiarity_check.html
remote/run.sh s2 ./run_stage.sh 2 3b                        # STOP: saudade/tristeza card + baselines
remote/run.sh s3 ./run_stage.sh 3 3b                        # then the same for 8b
remote/run.sh s4 ./run_stage.sh 4 8b                        # thread 2 on the 8B (and 3b, qwen7b, aya8b for signals)
remote/run.sh s5 ./run_stage.sh 5 8b --compare qwen7b aya8b --small 3b   # STOP: uncertainty map
remote/run.sh s6 ./run_stage.sh 6 8b && remote/pull.sh
```
Roughly 1.5-2.5 GPU-hours per model for stages 1-5; the 8B, Qwen and Aya (nf4) are 2-3x slower than the 3B.
Each script prints an ETA and warns when a run will exceed 15 minutes.

## Decisions (user, 18 Nov 2026)
Glosses and sense clarifications in the source language; distinction controls are distinctions English keeps, with
*tempo* scored as its own 'English splits' type; patching follows Dumas et al. (single-layer patch, layer chosen on
ordinary words) plus word-span and averaged variants; contexts written by Claude Code.

## Things found while building
* transformers 5 records `hidden_states` before user hooks, so at a hooked layer itself the recorded state is the
  pre-hook value; everything downstream is modified (tests/test_core.py). No experiment reads the hooked layer.
* With a source-language gloss, a small model sometimes answers in the source language ('durata'); prompts now
  ask for English explicitly, and ignorance.py reports the off-language rate per condition.
* Pretrained tuned lenses exist only for Llama-2 / Llama-3.0-8B, so §6.1 uses the logit lens alone.
* The llm-latent-language repo has no licence: its method is reimplemented with attribution, not copied.
* `python -m ut.words` leak warnings were reviewed: all cognates or names, not translations.

## Needs a human
Check the word lists, contexts, glosses and distinction items (stop point 1). Add Belli's footnotes as glosses and
four more Belli contexts per Romanesco item (`TODO_USER`). Fill `data/human_renderings.yaml` (§7.4). Accept the HF
licences for Llama-3.1-8B and Aya-23-8B. Set a git identity and commit `PREDICTIONS.md` before the first run.

## Sentence-level experiments (`sent/`, 19 Nov 2026)

Design: `../HANDOFF_sentence_untranslatability.md`. Test set: `data/sentences.yaml` (40 sentences + 2 optional Belli
lines; **unchecked, stop point 1**; validate with `python -m sent.data`). Outputs: `results/sent/<model>/`,
figures in `results/sent/figures/<model>/`, report `results/sent/report.md`.

```
sent/data.py       sentences, hot-span offsets (computed from substrings), words, centring texts
sent/prompts.py    every prompt: ladder rungs B0-B7, back-translation, copy oracle, span extraction, glosses
sent/ladder.py     step 1 (GPU): candidates.jsonl, 1 greedy + 5 samples at T=0.8 per rung
sent/score.py      step 2 (GPU): raw round-trip log-probs and per-layer OT statistics for every candidate,
                   plus masked / mismatched / spelling-control rows
sent/measure.py    the maths (leftovers, closed-form semi-relaxed OT, envelope, U, A); no model
sent/analyze.py    step 2b (CPU): layer + tau + epsilon on controls only, curves, spans, validation §6.1-6.5
sent/spread.py     step 3a (GPU): H per span over 20 standard translations (NLI clusters)
sent/targeting.py  step 3b (GPU): glosses on heat-map words vs random words, scored in both spaces
sent/figures.py    the 6 figures + captions.json, titles chosen from the statistics
sent/report.py     report.md from the JSON only
sent/run.sh        steps 1, 2, 3, all, bt, 6, smoke (see its header for the evening schedule)
```
Run: `remote/run.sh sent1-3b sent/run.sh 1 3b`, then `2`, then `3`, then `all 8b`, `bt 8b aya8b`, `6 8b 3b 8b`.
CPU check of the whole chain (Qwen2.5-0.5B, 5 sentences, ~6 min): `sent/run.sh smoke x` with `.venv-cpu` python.

Decisions taken while building (worth a look):
* **OT target marginal is free** (`reg_m = (tau, 0)`), as the handoff's text says; its parenthetical `(tau, inf)`
  would make the target binding and penalise commentary. With POT's default KL reference this has a closed form
  (checked against POT in `tests/test_sent.py`), so tau and the layer are recalibrated on the CPU.
* Target tokens carry mass 1 each, so a longer English text can never *raise* a token's leftover.
* **Layer choice ties** (AUROC 1.0 at many layers) are broken by the standardised mean difference; a warning is
  written if the chosen layer falls inside the English-detour band.
* |t| is counted with one fixed tokenizer (Llama-3.2-3B's) for every model.
* Candidates have **verbatim copies of the source sentence removed** (small models often restate it), and a second
  "masked" score removes source words rare in English (Zipf < 3), for the copying check.
* Spread H uses only renderings that really occur in their translation (or explicit "(none)"); H is n/a below 10.
* Targeting: the model writes each gloss separately (<= 8 words, hard-capped) and the code appends them
  "[g1; g2]" identically for both arms; a small model asked to rewrite the sentence with glosses added none.
* Centring texts per language = the natural contexts of `data/words/<lang>.yaml` (never the test sentences).
