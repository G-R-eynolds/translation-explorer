#!/usr/bin/env bash
# Sync, then run a job in tmux on the desktop.  Usage: remote/run.sh <job> <command...>
# OMP/MKL threads capped: the last project's CPU analysis starved the GPU job of a core.
source "$(dirname "$0")/_common.sh"
name="$1"; shift
"$(dirname "$0")/sync.sh"
cmd=$(printf '%q ' "$@")
rsh "mkdir -p $REMOTE_DIR/logs && cd $REMOTE_DIR && tmux new-session -d -s '$name' \
  \"TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/.local/bin/uv run --extra quant $cmd 2>&1 | tee logs/$name.log; \
echo __EXIT_\\\${PIPESTATUS[0]} >> logs/$name.log\""
echo "started '$name'. remote/status.sh $name"
