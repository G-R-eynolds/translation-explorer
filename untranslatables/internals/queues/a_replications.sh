#!/usr/bin/env bash
# Queue A (21 Nov): Part A replications on the cached models first, then Gemma 3 12B (waits for its download).
set -uo pipefail
run() { echo "===== $(date '+%a %H:%M') START $*"; python -m internals "$@" && echo "===== $(date '+%a %H:%M') ok $*" || echo "===== $(date '+%a %H:%M') FAILED $*"; }
run R1 --model aya8b
run R1 --model 8b
run R2 --model 8b --set reuse_from=8b
run R1 --model gemma12b
run R2 --model gemma12b
echo "===== queue A finished $(date)"
