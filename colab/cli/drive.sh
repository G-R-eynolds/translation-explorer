#!/usr/bin/env bash
# Run stages one at a time in an existing session; halt (leaving the session up) on any Python traceback, because
# `colab exec` exits 0 even when the cell raises.  Usage: ./drive.sh 00_setup A B C1 ...
set -uo pipefail
S="${SESSION:-ut}"; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/../../results_colab"
SLUG="${SLUG:-unsloth_Llama_3_3_70B_Instruct_bnb_4bit}"
mkdir -p "$OUT/logs"
for st in "$@"; do
  f="$(ls "$HERE"/${st}*.py | head -1)"; log="$OUT/logs/${st}.log"
  echo "== $(date +%H:%M:%S) START $st ($(basename "$f"))" | tee -a "$OUT/logs/timeline.txt"
  colab exec -s "$S" --timeout 0 -f "$f" < /dev/null > "$log" 2>&1
  if grep -q "Traceback\|Error: \|^\[colab\].*rror" "$log"; then
    echo "== $(date +%H:%M:%S) FAILED $st (see $log)" | tee -a "$OUT/logs/timeline.txt"; exit 3
  fi
  echo "== $(date +%H:%M:%S) ok $st" | tee -a "$OUT/logs/timeline.txt"
  if [ "$st" != 00_setup ]; then
    colab download -s "$S" "/content/ut_results_${SLUG}.tar.gz" "$OUT/ut_results_${SLUG}.tar.gz" < /dev/null >/dev/null 2>&1 \
      && tar -xzf "$OUT/ut_results_${SLUG}.tar.gz" -C "$OUT" && echo "   fetched results"
  fi
done
