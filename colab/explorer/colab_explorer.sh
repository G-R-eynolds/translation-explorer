#!/usr/bin/env bash
# Run the explorer queue (Phases 1, 3, 4 + report/offline export) on a Colab GPU via the official Colab CLI.
#
#   colab/explorer/colab_explorer.sh up        # new GPU session (GPU=A100 by default) + HF token upload
#   colab/explorer/colab_explorer.sh launch    # clone repo on the VM, install deps, start the queue detached
#   colab/explorer/colab_explorer.sh poll      # tail the remote log (safe to call repeatedly)
#   colab/explorer/colab_explorer.sh fetch     # download results into untranslatables/results + figures_for_dad
#   colab/explorer/colab_explorer.sh down      # colab stop (ALWAYS run this; compute units are money)
#
# HF token: export HF_TOKEN=... before `up` (google/gemma-3-4b-pt is gated). It is uploaded as a file, never echoed
# and never sent through `colab exec`, so it does not end up in the session history.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
S="${SESSION:-explorer}"
GPU="${GPU:-A100}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RDIR=/content/translation-explorer
RLOG=$RDIR/untranslatables/results/explorer/g3_4b_pt/colab_run.log

case "${1:-}" in
  up)
    colab new -s "$S" --gpu "$GPU" </dev/null
    echo 'import torch; p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/2**30,1), "GiB")' \
      | colab exec -s "$S" --timeout 120
    if [[ -n "${HF_TOKEN:-}" ]]; then
      t="$(mktemp)"; chmod 600 "$t"; printf '%s' "$HF_TOKEN" > "$t"
      echo 'import pathlib; pathlib.Path("/root/.cache/huggingface").mkdir(parents=True, exist_ok=True)' \
        | colab exec -s "$S" --timeout 60
      colab upload -s "$S" "$t" /root/.cache/huggingface/token </dev/null; rm -f "$t"
      echo "HF token uploaded"
    else
      echo "WARNING: HF_TOKEN not set; gated google/gemma-3-4b-pt will fail to download" >&2
    fi
    ;;
  launch)
    colab exec -s "$S" --timeout 900 -f "$HERE/remote_launch.py" </dev/null
    ;;
  poll)
    printf 'import subprocess\nprint(subprocess.run("tail -n %s %s; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)\n' \
      "${N:-25}" "$RLOG" | colab exec -s "$S" --timeout 120
    ;;
  fetch)
    printf 'import subprocess\nprint(subprocess.run("cd %s && tar -czf /content/explorer_results.tgz untranslatables/results/explorer/g3_4b_pt figures_for_dad/duende_workbench_offline.html && ls -la /content/explorer_results.tgz", shell=True, capture_output=True, text=True))\n' \
      "$RDIR" | colab exec -s "$S" --timeout 300
    colab download -s "$S" /content/explorer_results.tgz "$REPO/explorer_results.tgz" </dev/null
    tar -xzf "$REPO/explorer_results.tgz" -C "$REPO" && rm -f "$REPO/explorer_results.tgz"
    echo "results extracted into $REPO"
    ;;
  down)
    colab stop -s "$S" </dev/null || true
    colab sessions </dev/null || true
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
