#!/usr/bin/env bash
# Combined ~1h desktop GPU runner for Phases 1, 3, and 4 + offline workbench refresh.
# Run from untranslatables/:
#   bash explorer/queues/run_all_desktop.sh
set -euo pipefail

MODEL="${MODEL:-g3_4b_pt}"
OUT="results/explorer/${MODEL}"
mkdir -p "$OUT/phase3"

# Prefer local .venv if present and not already activated
if [[ -z "${VIRTUAL_ENV:-}" && -x ".venv/bin/python" ]]; then
  export PATH="$(pwd)/.venv/bin:$PATH"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "===== Desktop GPU run starting $(date) (model=$MODEL, python=$(which python)) ====="

echo "=== [1/6] Phase 1: finish remaining explorer texts (EXPLORER_FAST=1) ==="
EXPLORER_FAST=1 python -m explorer run --model "$MODEL" --n 64

echo "=== [2/6] Phase 1: 2D direction map for duende_flamenco ==="
python -m explorer.mapgrid --model "$MODEL" --text duende_flamenco --n 15

echo "=== [3/6] Phase 1: refresh HTML report & offline workbench ==="
python -m explorer report --model "$MODEL"
python -m explorer offline --results-dir "$OUT" --out ../figures_for_dad/duende_workbench_offline.html

echo "=== [4/6] Phase 3: extract global tone/practice axes ==="
python -m explorer.train_adapters --mode axes --model "$MODEL" --out "$OUT/phase3/global_axes.npz"

echo "=== [5/6] Phase 3: train multi-reference LoRA & continuous SpectralReFT ==="
python -m explorer.train_adapters --mode lora --model "$MODEL" --layers "${LORA_LAYERS:-12,17,22,27}" --steps 120 --kl-beta 0.1 --out "$OUT/phase3/multiref_lora.pt"
python -m explorer.train_adapters --mode reft --model "$MODEL" --steps 160 --out "$OUT/phase3/spectral_reft.pt"

echo "=== [6/6] Phase 4: extract fork dataset (all texts) & train ForkHead ==="
python -m explorer.fork_head --model "$MODEL" --epochs 80

echo "===== All phases finished $(date) ====="
