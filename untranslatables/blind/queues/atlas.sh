#!/usr/bin/env bash
# E4 atlas (HANDOFF_blind_spots §5 E4), only after E1 passed (stop point 2).  Then stop point 5: label the clusters
# from the steering pairs in results/blind/g3_4b/atlas.md before writing anything up.
set -uo pipefail
run() { echo "===== $(date '+%a %H:%M') START $*"; python -m blind "$@" && echo "===== $(date '+%a %H:%M') ok $*" || echo "===== $(date '+%a %H:%M') FAILED $*"; }
M=g3_4b; N=${N:-200}
run collect --model $M --set atlas --kv n=$N
run atlas_dirs --model $M --set atlas --kv n=$N
run atlas_cluster --model $M --kv n=$N
# rerun 3 example sentences per cluster into it/fr/pt, then re-cluster to fill in "kept by"
ITEMS=$(python -c "import json;d=json.load(open('results/blind/$M/atlas.json'));print(json.dumps(sorted({e['item'] for c in d['clusters'][:12] for e in c['examples']})))")
run collect --model $M --set atlas --kv "langs=[\"en\",\"it\",\"fr\",\"pt\"]" "items=$ITEMS" n=$N
run atlas_langs --model $M
run atlas_cluster --model $M --kv n=$N --force
run figures --model $M --force
