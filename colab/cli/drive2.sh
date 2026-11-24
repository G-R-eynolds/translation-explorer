#!/usr/bin/env bash
# Kernel stages with hard limits (21-22 Nov: a lost CLI connection once hung `colab exec` for 18 h).  For each stage:
# run `colab exec` under `timeout`; if the client dies (timeout / "Connection was lost") the kernel may still be
# working, so poll for the stage's marker file on the VM until the stage's limit; halt on any Python traceback.
#   ./drive2.sh "00_setup:20" "C1:15" "C2:45" "D:40" "Z:10"        (stage:limit in minutes)
set -uo pipefail
S="${SESSION:-ut}"; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/../../results_colab"
SLUG=unsloth_Llama_3_3_70B_Instruct_bnb_4bit; R="/content/ut_results/$SLUG"
declare -A MARK=([00_setup]="" [C1]="/content/cache/$SLUG/C_hi_held.pt" [C2]="$R/C_spectrum_hi.json" [D]="$R/D_mediation.json" [Z]="$R/SUMMARY.md")
mkdir -p "$OUT/logs"
vm_has() { echo "import os; print('HAS' if os.path.exists('$1') else 'NO')" | timeout 120 colab exec -s "$S" --timeout 90 2>/dev/null | grep -q HAS; }
for item in "$@"; do
  st="${item%%:*}"; lim="${item##*:}"; f="$(ls "$HERE"/${st}*.py | head -1)"; log="$OUT/logs/k_${st}.log"; t0=$(date +%s)
  echo "== $(date +%H:%M:%S) START $st (limit ${lim} min)" | tee -a "$OUT/logs/timeline.txt"
  timeout $((lim * 60)) colab exec -s "$S" --timeout 0 -f "$f" < /dev/null > "$log" 2>&1; rc=$?
  if grep -q "Traceback" "$log" && ! grep -q "Connection was lost" "$log"; then
    echo "== $(date +%H:%M:%S) FAILED $st (traceback, $log)" | tee -a "$OUT/logs/timeline.txt"; exit 3; fi
  if [ $rc -ne 0 ] || grep -q "Connection was lost" "$log"; then
    m="${MARK[$st]}"; [ -z "$m" ] && { echo "== client lost during $st and no marker to wait for"; exit 4; }
    echo "== $(date +%H:%M:%S) client lost during $st (rc $rc); waiting for $m" | tee -a "$OUT/logs/timeline.txt"
    until vm_has "$m"; do
      [ $(( $(date +%s) - t0 )) -gt $((lim * 60 + 600)) ] && { echo "== $(date +%H:%M:%S) GAVE UP on $st" | tee -a "$OUT/logs/timeline.txt"; exit 5; }
      sleep 120; done
  fi
  echo "== $(date +%H:%M:%S) ok $st" | tee -a "$OUT/logs/timeline.txt"
  if [ "$st" != 00_setup ]; then
    timeout 600 colab download -s "$S" "/content/ut_results_${SLUG}.tar.gz" "$OUT/ut_results_${SLUG}.tar.gz" < /dev/null >/dev/null 2>&1 \
      && tar -xzf "$OUT/ut_results_${SLUG}.tar.gz" -C "$OUT" && echo "   fetched results"
  fi
done
