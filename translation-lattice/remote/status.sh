#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
rsh "tmux ls 2>/dev/null || echo 'no tmux sessions'; /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true"
[[ $# -ge 1 ]] && rsh "tail -n ${2:-20} $REMOTE_DIR/logs/$1.log"
true
