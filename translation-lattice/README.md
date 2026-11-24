# Translation lattices with a local Llama

Sample many line-for-line translations of a poem, measure per line whether renderings **differ by
meaning** (not wording, dialect or rhyme) using quantities taken from the model's own probabilities and
hidden states, then draw the lattice. Background, design and rationale: `../HANDOFF.md`. Findings: `results.md`.

Test poem: G. G. Belli, *Er giorno der giudizzio* (1831), `data/source.json`. **Lines 7–14 still
need checking against a print edition.** The scripts never alter the text.

## Layout

```
data/            source.json, the 12 demo samples + hand clusters (weak reference),
                 meaning_edits_<model>.json (hand-written controls), controls_spotcheck_<model>.json
runs/<model>/    everything a model produces:
  samples_llama.json (+ _logprobs, _attempts)   Phase 1
  paraphrase_it.json                            Romanesco -> Italian sanity check
  restyles.json, controls.json                  Phase 3 controls
  spine.json                                    central route used as M2's fixed prefix
  clusters_{m1,m1r,m2,b0,b1,b2}.json            per-metric meaning clusters (prototype schema);
                                                b2 = the handoff's "nli", m1r = M1 as a per-token rate
  out/<metric>/index.html                       lattices
  results/                                      back-translations, score matrices, hidden-state
                                                distances, validation.json, layer sweep, logit lens, figures
common.py        model registry/loading, pinned chat template, sampling, prefix-cached scoring
sample.py        Phase 1 sampling (+ --paraphrase sanity check)
backtranslate.py M1: sample m Italian back-translations per rendering (prompt never contains the source)
score.py         M1: teacher-force every pooled back-translation under every rendering's BT prompt
controls.py      style-only (restyle) and meaning-edit control pairs
layers.py        hidden states after a fixed common prefix, LEACE, layer sweep, logit lens
metrics.py       distances, KLE, spectral clustering, validation (AUROC, tau, ARI), clusters_*.json
epistemic.py     §8: pool two models' samples, MI between model and reading vs a permutation null
figures.py       layer-sweep figure + tables.md for results.md
build.py         samples + clusters -> lattice page (now takes --samples/--clusters/--out, and
                 --display-top N to merge small clusters into one "other readings" node, for display only)
remote/          sync/run/pull/status helpers for the WSL2 GPU desktop (same as the dissertation code)
```

## Running

The GPU desktop is reached as `ssh desktop-gpu`. Its env is `~/translation-lattice/.venv`, built with
`uv sync --extra quant` and pinned like the dissertation code (torch 2.14.1+cu130, transformers 5.18).
Jobs run in tmux: `remote/run.sh <job> python ...`, `remote/status.sh <job>`, `remote/pull.sh`.

```bash
M=3b   # 3b = Llama-3.2-3B-Instruct bf16 | 8b = Llama-3.1-8B-Instruct nf4 (needs licence) | qwen7b
remote/run.sh p1 sh -c "python sample.py --model $M --paraphrase && python sample.py --model $M --per-stance 12"
remote/run.sh m1 sh -c "for s in lattice demo; do python backtranslate.py --model $M --set \$s --m 6 --temperature 0.7 && python score.py --model $M --set \$s; done"
remote/run.sh sp sh -c "python metrics.py --model $M --stage spine && python controls.py --model $M --restyle"
#   spot-check runs/$M/restyles.json -> data/controls_spotcheck_$M.json;
#   write data/meaning_edits_$M.json (minimal interpretive edits of this run's samples)
remote/run.sh c  sh -c "python controls.py --model $M --assemble && python backtranslate.py --model $M --set controls --m 6 --temperature 0.7 && python score.py --model $M --set controls"
remote/run.sh l  sh -c "python layers.py --model $M --extract --lens && python metrics.py --model $M --stage all && python layers.py --model $M --analyze && python metrics.py --model $M --stage all"
remote/run.sh f  sh -c "python figures.py --model $M"
# §8, second model (about 35 min for the pooled scoring at m=4):
remote/run.sh q  sh -c "python sample.py --model qwen7b --paraphrase && python sample.py --model qwen7b --per-stance 12 && python backtranslate.py --model $M --set pool --pool-with qwen7b --m 4 --temperature 0.7 && python score.py --model $M --set pool --pool-with qwen7b && python epistemic.py --model $M --other qwen7b"
remote/pull.sh
# lattices (laptop): uv run --no-project --with numpy --with scikit-learn python build.py \
#   --samples runs/$M/samples_llama.json --clusters runs/$M/clusters_m1r.json --out runs/$M/out/m1r --display-top 3
```

Timings on the RTX 4070 (3B, bf16): sampling ~1 min; M1 back-translation + scoring for the lattice ~15 min;
hidden-state extraction 3–12 min (slower when the Windows side holds VRAM); `layers.py --analyze` is CPU-bound,
about 1–3 min per layer, ~45 min in all.

`metrics.py --stage all` runs twice: the first pass writes `clusters_m1r.json`, which the layer sweep uses
as labels, and the second picks up the M2/B1 distances. The hand-written controls are tied to one run's samples.
For a new model or poem, rewrite `meaning_edits_<model>.json` and the spot-check file.

## Deviations from the handoff (all deliberate, all recorded in outputs)

* **Model**: Llama-3.2-3B-Instruct in bf16, not 3.1-8B in nf4. The 8B licence hadn't been accepted on the
  HF account. The `8b` key is ready, so re-run with `M=8b` once access is granted.
* **Back-translation temperature 0.7** (no top-p/top-k). At T=1.0 the 3B's samples had junk tails. Scoring uses
  the same tempered distribution softmax(logits/0.7), so the Monte Carlo JSD estimator stays exact.
* **M1r, a per-token rate.** The string-level JSD (M1 as specified) saturates at ln 2 for almost every pair.
  The added M1r is the symmetric KL *rate* between back-translation distributions, in nats per token.
  See results.md.
* **Lattice clustering is average-linkage cut at τ**, not spectral/eigengap. Eigengap chose k=1 on every
  line, and its results are kept in `validation.json`.
* The BT prompt shows the neighbouring English lines labelled "do not translate". With unlabelled context
  the 3B translated the neighbours too.
