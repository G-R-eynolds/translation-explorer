#!/usr/bin/env bash
# Step 1 on the desktop:  remote/run.sh explorer bash explorer/queue.sh      (progress: remote/status.sh explorer 30)
set -uo pipefail
EXPLORER_FAST=${EXPLORER_FAST:-1} python -m explorer run --model g3_4b_pt --n 64 && echo "===== explorer finished $(date)" || echo "===== explorer FAILED"
