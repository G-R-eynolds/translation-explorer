#!/usr/bin/env bash
# The whole internals run, strictly one job at a time (21 Nov: running the corpus download, the weight prefetch and a
# GPU queue side by side coincided with WSL becoming unreachable).  Every stage is resumable: re-running this script
# skips what is done (config hash + outputs present) and continues where it stopped.
#   remote/run.sh qall internals/queues/all.sh       # progress: remote/status.sh qall 40
set -uo pipefail
run() { echo "===== $(date '+%a %H:%M') START $*"; python -m internals "$@" && echo "===== $(date '+%a %H:%M') ok $*" || echo "===== $(date '+%a %H:%M') FAILED $*"; }
run corpus --pair es-en
run corpus --pair hi-en
run flores
bash internals/queues/a_replications.sh
bash internals/queues/b_main.sh
run figures
run report
echo "===== all finished $(date)"
