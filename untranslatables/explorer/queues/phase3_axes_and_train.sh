#!/usr/bin/env bash
# Phase 3 GPU queue: extract universal global tone/practice axes + train non-collapsing LoRA & SpectralReFT.
# Run later on desktop-gpu via:
#   remote/run.sh phase3 bash explorer/queues/phase3_axes_and_train.sh
set -euo pipefail

MODEL="${MODEL:-g3_4b_pt}"
OUT="results/explorer/${MODEL}/phase3"
mkdir -p "$OUT"

echo "=== [1/3] Extracting universal Fisher-whitened global axes (no training) ==="
python -m explorer.train_adapters --mode axes --model "$MODEL" --out "$OUT/global_axes.npz"

echo "=== [2/3] Training diversity-preserving multi-reference LoRA ==="
python -m explorer.train_adapters --mode lora --model "$MODEL" --steps 120 --kl-beta 0.1 --out "$OUT/multiref_lora.pt"

echo "=== [3/3] Training continuous SpectralReFT slider controller ==="
python -m explorer.train_adapters --mode reft --model "$MODEL" --steps 160 --out "$OUT/spectral_reft.pt"

echo "===== Phase 3 finished $(date) ====="
