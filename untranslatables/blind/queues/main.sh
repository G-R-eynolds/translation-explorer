#!/usr/bin/env bash
# Blind spots, steps 1-4 of HANDOFF_blind_spots §7 (E0, E1, E2, E3 on g3_4b; E1+E2 on 3b, aya8b, g3_12b), then figures and
# the draft report.  Sequential and resumable: rerunning skips finished stages and finished (item, channel) files.
#   remote/run.sh blind blind/queues/main.sh        # progress: remote/status.sh blind 40
# The atlas (E4) is NOT here: it waits for stop point 2 (E1 validation) -> blind/queues/atlas.sh.
set -uo pipefail
run() { echo "===== $(date '+%a %H:%M') START $*"; python -m blind "$@" && echo "===== $(date '+%a %H:%M') ok $*" || echo "===== $(date '+%a %H:%M') FAILED $*"; }
M=g3_4b
python -m blind material || exit 1
# E0 first: the identical-h assert and the gradient check must pass before anything else is worth running
python -m blind e0 --model $M || { echo "===== E0 FAILED: stopping"; exit 1; }
# E1
run collect --model $M --set e1
run spectra --model $M --set e1
run fdcheck --model $M
run steer --model $M --set e1
run sae --model $M --set e1
run validate --model $M
run heatmaps --model $M --set e1 --kv head_only=true
# E2
run collect --model $M --set e2
run spectra --model $M --set e2
run heatmaps --model $M --set e2
run steer --model $M --set e2
run sae --model $M --set e2
# E3
run collect --model $M --set e3
run spectra --model $M --set e3
run practices --model $M
run figures --model $M
run report
# E5: other models (E1 + E2)
for M2 in 3b aya8b g3_12b; do
  run collect --model $M2 --set e1
  run spectra --model $M2 --set e1
  run steer --model $M2 --set e1
  run validate --model $M2
  run collect --model $M2 --set e2
  run spectra --model $M2 --set e2
done
run report
echo "===== all finished $(date)"
