#!/usr/bin/env bash
# The weekend queue (20 Nov 2026): every model through steps 1-3, a grid of second back-translators, then figures
# and the report.  One failed step is logged and skipped, never fatal, so the queue always reaches the end.
#   remote/run.sh sentq sent/queue.sh          # progress: remote/status.sh sentq 40 ; summary: logs/sentq_steps.log
set -uo pipefail
GEN=(3b 8b qwen7b aya8b mistral7b)         # generators (main models), in priority order
BT=(aya8b qwen7b 8b 3b)                    # back-translators used as second opinions on every generator
STEPS=logs/sentq_steps.log
mkdir -p logs; : > "$STEPS"
step() {   # step <label> <command...>
  local t0=$SECONDS
  echo "===== $(date '+%a %H:%M') START $1" | tee -a "$STEPS"
  if "${@:2}"; then s=ok; else s="FAILED ($?)"; fi
  echo "===== $(date '+%a %H:%M') $s $1 ($(( (SECONDS - t0) / 60 )) min)" | tee -a "$STEPS"
}
step "prefetch weights" python - <<'EOF'
from huggingface_hub import snapshot_download
from ut.common import MODELS
for k in ["8b", "qwen7b", "aya8b", "mistral7b"]:
    snapshot_download(MODELS[k]["hf_id"], allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])
    print("have", k, flush=True)
EOF
full() {
  local M=$1
  step "$M ladder" python -m sent.ladder --model "$M" --dialect
  step "$M score" python -m sent.score --model "$M"
  step "$M analyze" python -W ignore -m sent.analyze --model "$M"
  step "$M spread" python -m sent.spread --model "$M" --dialect
  step "$M targeting" python -m sent.targeting --model "$M"
}
cross() {
  local M=$1
  for B in "${BT[@]}"; do
    [[ "$B" == "$M" ]] && continue
    step "$M candidates, back-translator $B" python -m sent.score --model "$M" --bt "$B" --rt-only
  done
  step "$M analyze (with second back-translators)" python -W ignore -m sent.analyze --model "$M"
  step "$M figures" python -W ignore -m sent.figures --model "$M"
}
full 3b; full 8b                          # the handoff's two main models first
cross 3b; cross 8b
for M in qwen7b aya8b mistral7b; do full "$M"; done
for M in qwen7b aya8b mistral7b; do cross "$M"; done
step "report" python -W ignore -m sent.report --models "${GEN[@]}"
echo "===== queue finished $(date)" | tee -a "$STEPS"
grep -c FAILED "$STEPS" | xargs -I{} echo "{} failed steps" | tee -a "$STEPS"
