#!/usr/bin/env bash
# Sentence-level untranslatability (../HANDOFF_sentence_untranslatability.md): one step for one model, in the
# handoff's §9 order.  Run on the desktop through remote/run.sh (it syncs, then runs in tmux under uv):
#
#   remote/run.sh sent1-3b  sent/run.sh 1 3b      # ladder                      ~10-15 min (3B, bf16)
#   remote/run.sh sent2-3b  sent/run.sh 2 3b      # score + analyse + figures   ~5-10 min.  STOP: duende heat map + validation
#   remote/run.sh sent3-3b  sent/run.sh 3 3b      # spread + targeting          ~10 min
#   remote/run.sh sent4-8b  sent/run.sh all 8b    # steps 1-3 again on the 8B   ~45-75 min (nf4; ~2-3x the 3B)
#   remote/run.sh sent5-8b  sent/run.sh bt 8b aya8b    # second back-translator on the 8B's candidates  ~10-15 min
#   remote/run.sh sent6     sent/run.sh 6 8b 3b 8b     # figures for the lead model + report over all models (CPU)
#   remote/pull.sh                                     # results/sent/ comes back to the laptop
#
# Time short?  Drop, in this order (§9): B7 (pass --rungs B0 B1 B2 B3 B4 B5 B6 to step 1), Mistral, the dialect
# items (they are off unless --dialect).  Never skip step 2's validation: if validation.json has "stop" entries,
# read them before running anything else.
set -euo pipefail
step="$1"; M="$2"; shift 2
case "$step" in
  1)   python -m sent.ladder --model "$M" "$@" ;;
  2)   python -m sent.score --model "$M"
       python -m sent.analyze --model "$M"
       python -m sent.figures --model "$M" ;;
  3)   python -m sent.spread --model "$M"
       python -m sent.targeting --model "$M"
       python -m sent.figures --model "$M" ;;
  all) "$0" 1 "$M" "$@"; "$0" 2 "$M"; "$0" 3 "$M" ;;
  bt)  for B in "$@"; do python -m sent.score --model "$M" --bt "$B" --rt-only; done
       python -m sent.analyze --model "$M" ;;
  6)   python -m sent.figures --model "$M"
       python -m sent.report --models "$M" "$@" ;;
  smoke)  # laptop CPU, ~6 min: the whole chain on 5 sentences with Qwen2.5-0.5B (numbers are meaningless)
       export SENT_LENGTH_TOKENIZER=Qwen/Qwen2.5-0.5B-Instruct
       S="es_duende es_tren de_brueder it_caffe es_gustar"
       python -m sent.ladder --model qwen05 --n 2 --only $S --batch 16
       python -m sent.score --model qwen05
       python -m sent.analyze --model qwen05
       python -m sent.spread --model qwen05 --n 4 --only $S
       python -m sent.targeting --model qwen05
       python -m sent.figures --model qwen05
       python -m sent.report --models qwen05 ;;
  *)   echo "unknown step $step (1 2 3 all bt 6 smoke)"; exit 1 ;;
esac
