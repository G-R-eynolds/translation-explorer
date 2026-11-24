#!/usr/bin/env bash
# Queue B (21 Nov): everything after the Part A replications, in the handoff's §12 order, adapted to one run:
# alignment; N1 on the main model (gemma12b) with its analysis; N2 and N3 on gemma12b (N3 is gated on N1 inside the
# stage); R1/R2 on qwen14b; N1 on the comparison models; sarvam (hi only); second back-translator for R2.
# A failed step is logged and skipped; every stage is resumable, so re-running the queue continues where it stopped.
set -uo pipefail
run() { echo "===== $(date '+%a %H:%M') START $*"; python -m internals "$@" && echo "===== $(date '+%a %H:%M') ok $*" || echo "===== $(date '+%a %H:%M') FAILED $*"; }
n1() {   # n1 <model> [pairs...]
  local M=$1; shift; local P=("${@:-es-en hi-en}"); [[ $# -eq 0 ]] && P=(es-en hi-en)
  for p in "${P[@]}"; do run quick --model "$M" --pair "$p"; done
  for p in "${P[@]}"; do run collect --model "$M" --pair "$p"; done
  if [[ " ${P[*]} " == *" es-en "* ]]; then run evalstates --model "$M"; else run evalstates --model "$M" --set langs='["hi"]' flores_langs='["hi","en","bho"]'; fi
  for p in "${P[@]}"; do run cca --model "$M" --pair "$p"; done
  run spans --model "$M"
  [[ " ${P[*]} " == *" es-en "* ]] && run hub --model "$M"
  run probes --model "$M"
  run mlp --model "$M"
}
run align --set what='["es-en","hi-en","flores"]'
n1 gemma12b
run sae --model gemma12b
run mediate --model gemma12b
run interchange --model gemma12b
run R1 --model qwen14b
run R2 --model qwen14b
n1 qwen14b
run R1 --model nemo12b
n1 nemo12b
n1 aya8b
n1 8b
n1 sarvam2b hi-en
run R2_bt --model gemma12b --set bt=aya8b
run R2_bt --model qwen14b --set bt=aya8b
run R2 --model nemo12b
echo "===== queue B finished $(date)"
