#!/usr/bin/env bash
# Phase 1 GPU queue (DO NOT RUN while GPU is off).
# Run later on desktop-gpu via:
#   remote/run.sh phase1 bash explorer/queues/phase1_finish_and_map.sh
set -euo pipefail

MODEL="${MODEL:-g3_4b_pt}"

echo "=== [1/3] Finishing remaining explorer texts for $MODEL ==="
EXPLORER_FAST=1 python -m explorer run --model "$MODEL" --n 64

echo "=== [2/3] Generating interactive 2D direction map for duende_flamenco ==="
python -m explorer.mapgrid --model "$MODEL" --text duende_flamenco --n 15

echo "=== [3/3] Refreshing static HTML report ==="
python -m explorer report --model "$MODEL"

echo "===== Phase 1 finished $(date) ====="
