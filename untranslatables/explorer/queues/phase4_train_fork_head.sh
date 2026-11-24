#!/usr/bin/env bash
# Phase 4 GPU queue: extract hidden-state fork dataset & train amortised ForkHead side-head.
# Run later on desktop-gpu via:
#   remote/run.sh phase4 bash explorer/queues/phase4_train_fork_head.sh
set -euo pipefail

MODEL="${MODEL:-g3_4b_pt}"

echo "=== Extracting fork dataset & training ForkHead for $MODEL ==="
python -m explorer.fork_head --model "$MODEL" --epochs 80

echo "===== Phase 4 finished $(date) ====="
