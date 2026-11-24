#!/usr/bin/env bash
# Run one stage of the pipeline for one model, on the desktop (via remote/run.sh) or locally.
#   remote/run.sh s1-3b ./run_stage.sh 1 3b
# Stages follow the handoff's order of work (§10); each ends at a stop point for a human check.
# Rough timings on the RTX 4070 are for the 3B in bf16; the nf4 8B / Qwen / Aya are about 2-3x slower on GPU stages.
set -euo pipefail
stage="$1"; M="$2"; shift 2
case "$stage" in
  0)  # once, model-independent decisions taken with the 3B: German casing + proper-noun/acronym removal (~5 min)
      python pools.py --casing --model 3b ;;
  1)  # familiarity gate (~5 min).  STOP: tick results/$M/familiarity_check.html -> data/familiarity_ticks_$M.json
      python familiarity.py --model "$M" ;;
  2)  # thread 1 core (~40 min: pools 5, extraction 1, recipes+nulls+layer dependence 20 CPU, patching 10)
      python pools.py --extract --model "$M"
      python layers.py --model "$M"
      python recipe.py --model "$M" --extract --analyze
      python baselines.py --model "$M"
      python patching.py --model "$M" ;;
      # STOP: show the saudade/tristeza card and the baseline comparison (figures.py --model $M)
  3)  # English bias (~40 min: latent language 3, distinctions 5, coverage 10 CPU, steering 20); quick GPU ones first
      python latent_lang.py --model "$M"
      python distinctions.py --model "$M"
      python coverage.py --model "$M"
      python steering.py --model "$M" ;;
  4)  # thread 2 sampling (~10 min for the 3B, ~30 min for an 8B in nf4)
      python spread.py --model "$M" ;;
  5)  # thread 2 analysis, after stage 4 for the comparison models (~15 min).  STOP: uncertainty map
      python ignorance.py --model "$M" "$@" ;;          # e.g. --compare qwen7b aya8b --small 3b
  6)  # stretch probes (~15 min CPU), figures, report
      python probes.py --model "$M"
      python figures.py --model "$M" --models 3b 8b qwen7b aya8b
      python report.py --model "$M" ;;
  *)  echo "unknown stage $stage"; exit 1 ;;
esac
