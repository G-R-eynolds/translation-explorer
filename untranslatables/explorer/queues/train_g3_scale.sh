#!/usr/bin/env bash
# Train v2 Live Co-Translation artifacts (doc-prompt global axes, multi-ref LoRA, SpectralReFT,
# and self-supervised ForkHead) for quantized Gemma 3 12B (`g3_12b_pt`) or 27B (`g3_27b_pt`).
#
# Examples:
#   MODEL=g3_12b_pt bash explorer/queues/train_g3_scale.sh    # ~8 GB nf4 (RTX 4070 12 GB / Colab T4/L4)
#   MODEL=g3_27b_pt bash explorer/queues/train_g3_scale.sh    # ~16.5 GB nf4 (Colab L4 22.5 GB / A100)
set -euo pipefail

MODEL="${MODEL:-g3_12b_pt}"
OUT="results/explorer/${MODEL}"
mkdir -p "$OUT/phase3"

if [[ -z "${VIRTUAL_ENV:-}" && -x ".venv/bin/python" ]]; then
  export PATH="$(pwd)/.venv/bin:$PATH"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Default mid-upper LoRA layers scaled to model depth (48L=12B, 62L=27B, 80L=72B)
if [[ -z "${LORA_LAYERS:-}" ]]; then
  case "$MODEL" in
    qwen72b*) LORA_LAYERS="30,40,50,60" ;;
    g3_27b*)  LORA_LAYERS="22,31,40,48" ;;
    g3_12b*)  LORA_LAYERS="16,24,32,38" ;;
    *)        LORA_LAYERS="12,17,22,27" ;;
  esac
fi

echo "===== Training v2 live artifacts for $MODEL (LORA_LAYERS=$LORA_LAYERS) ====="

echo "=== [1/4] Extracting doc-prompt global tone/practice axes ==="
if [[ ! -f "$OUT/phase3/global_axes.npz" ]]; then
  python -m explorer.train_adapters \
    --mode axes --model "$MODEL" --prompt-style doc \
    --out "$OUT/phase3/global_axes.npz"
else
  echo "  (already exists: $OUT/phase3/global_axes.npz)"
fi

echo "=== [2/4] Training diversity-preserving multi-reference LoRA (doc prompt) ==="
if [[ ! -f "$OUT/phase3/multiref_lora.pt" ]]; then
  python -m explorer.train_adapters \
    --mode lora --model "$MODEL" --prompt-style doc \
    --layers "$LORA_LAYERS" --steps "${LORA_STEPS:-100}" --kl-beta 0.1 \
    --out "$OUT/phase3/multiref_lora.pt"
else
  echo "  (already exists: $OUT/phase3/multiref_lora.pt)"
fi

echo "=== [3/4] Training continuous SpectralReFT slider controller (doc prompt) ==="
if [[ ! -f "$OUT/phase3/spectral_reft.pt" ]]; then
  python -m explorer.train_adapters \
    --mode reft --model "$MODEL" --prompt-style doc \
    --steps "${REFT_STEPS:-140}" \
    --out "$OUT/phase3/spectral_reft.pt"
else
  echo "  (already exists: $OUT/phase3/spectral_reft.pt)"
fi

echo "=== [4/4] Training self-supervised ForkHead (doc prompt) ==="
if [[ ! -f "$OUT/fork_head.pt" ]]; then
  python -m explorer.fork_head \
    --model "$MODEL" --prompt-style doc --from-texts \
    --epochs "${FORK_EPOCHS:-80}" \
    --out "$OUT/fork_head.pt"
else
  echo "  (already exists: $OUT/fork_head.pt)"
fi

if [[ "${RUN_PHASE1:-0}" == "1" ]]; then
  echo "=== [5/5] Phase 1: 9-text reading & steering benchmark (EXPLORER_FAST=1) + mapgrid + report ==="
  EXPLORER_FAST=1 python -m explorer run --model "$MODEL" --n 64
  if [[ ! -f "$OUT/map_duende_flamenco.json" ]]; then
    python -m explorer.mapgrid --model "$MODEL" --text duende_flamenco --n 15
  fi
  python -m explorer report --model "$MODEL"
fi

echo "===== Done! Launch live server with: ====="
echo "python -m explorer serve --mode live --model $MODEL \\"
echo "  --global-axes $OUT/phase3/global_axes.npz \\"
echo "  --lora $OUT/phase3/multiref_lora.pt \\"
echo "  --reft $OUT/phase3/spectral_reft.pt \\"
echo "  --fork-head $OUT/fork_head.pt --port 8765"
