#!/usr/bin/env bash
# Drive the untranslatability experiments on a Colab GPU through the official Google Colab CLI.
#   ./run_all.sh                        # full run, 70B model, stages A B C1 C2 D Z
#   DRY=1 ./run_all.sh                  # 10-minute rehearsal: Qwen 0.5B on a T4
#   STAGES="C1 C2 D Z" KEEP=1 ./run_all.sh   # subset; KEEP=1 leaves the session running afterwards
# Every flag used here is from the CLI's README; check `colab --help` / `colab <cmd> --help` if one is rejected.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SESSION="${SESSION:-ut}"
if [ "${DRY:-0}" = 1 ]; then
  GPU="${GPU:-T4}"; MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
else
  GPU="${GPU:-A100}"; MODEL_ID="${MODEL_ID:-unsloth/Llama-3.3-70B-Instruct-bnb-4bit}"
fi
STAGES="${STAGES:-A B C1 C2 D Z}"
SLUG="$(echo "$MODEL_ID" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+|_+$//g')"
OUTDIR="${OUTDIR:-$HERE/../../results_colab}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_${SLUG}_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

cleanup() {
  if [ "${KEEP:-0}" = 1 ]; then echo "KEEP=1: session '$SESSION' left running — stop it with: colab stop -s $SESSION"
  else echo "stopping session $SESSION"; colab stop -s "$SESSION" || true; fi
}
trap cleanup EXIT

fetch() {   # pull the packed results and unpack them into $OUTDIR/<model slug>/
  colab download -s "$SESSION" "/content/ut_results_${SLUG}.tar.gz" "$OUTDIR/ut_results_${SLUG}.tar.gz" \
    && tar -xzf "$OUTDIR/ut_results_${SLUG}.tar.gz" -C "$OUTDIR" && echo "results -> $OUTDIR/$SLUG"
}

if ! colab sessions </dev/null 2>/dev/null | grep -qw "$SESSION"; then
  echo "== creating session $SESSION ($GPU, high-RAM)"
  colab new -s "$SESSION" --gpu "$GPU" --high-mem
fi
echo "== GPU check"
echo 'import subprocess; print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],capture_output=True,text=True).stdout)' \
  | colab exec -s "$SESSION"

echo "== installing packages"
colab install -s "$SESSION" -r "$HERE/requirements.txt"

echo "== setup ($MODEL_ID)"
printf 'MODEL_ID = %s\n' "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$MODEL_ID")" | colab exec -s "$SESSION"
colab exec -s "$SESSION" --timeout 0 -f "$HERE/00_setup.py"

for st in $STAGES; do
  f="$(ls "$HERE"/${st}_*.py | head -1)"
  echo "== stage $st ($(basename "$f")) — $(date +%H:%M)"
  colab exec -s "$SESSION" --timeout 0 -f "$f"
  fetch
done
colab log -s "$SESSION" -o "$OUTDIR/session_${SLUG}.ipynb" || true
echo "== done — $(date +%H:%M)"
