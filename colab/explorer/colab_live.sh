#!/usr/bin/env bash
# Run or train the v2 Live Co-Translation Workbench on Colab for quantized Gemma 3 12B or 27B.
#
# Usage:
#   HF_TOKEN=... MODEL=g3_12b_pt GPU=L4 colab/explorer/colab_live.sh up
#   MODEL=g3_12b_pt colab/explorer/colab_live.sh train      # trains doc-prompt axes/LoRA/ReFT/ForkHead (~8-15 min)
#   MODEL=g3_12b_pt colab/explorer/colab_live.sh serve      # starts live server on Colab port 8765
#   colab/explorer/colab_live.sh tunnel                     # forwards http://127.0.0.1:8765 -> Colab!
#   MODEL=g3_12b_pt colab/explorer/colab_live.sh fetch      # downloads trained checkpoints into repo
#   colab/explorer/colab_live.sh down                       # ALWAYS stop when done!
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

S="${SESSION:-live_g3}"
MODEL="${MODEL:-g3_12b_pt}"
GPU="${GPU:-L4}"
PORT="${PORT:-8765}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RDIR=/content/translation-explorer
RES=$RDIR/untranslatables/results/explorer/$MODEL

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
      echo "HF token uploaded."
    else
      echo "WARNING: HF_TOKEN not set; gated google/gemma-3-*-pt will fail to download." >&2
    fi
    cat <<'PY' | colab exec -s "$S" --timeout 600
import pathlib, subprocess
R = pathlib.Path("/content/translation-explorer")
if (R / ".git").exists():
    subprocess.run(["git", "-C", str(R), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/G-R-eynolds/translation-explorer.git", str(R)], check=True)
subprocess.run("pip install -q -U transformers accelerate bitsandbytes pyyaml scipy scikit-learn", shell=True, check=True)
print("Colab VM ready.")
PY
    ;;
  train)
    cat <<PY | colab exec -s "$S" --timeout 1800
import os, pathlib, subprocess
U = pathlib.Path("$RDIR/untranslatables")
LOG = U / "results" / "explorer" / "$MODEL" / "train_g3_scale.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, MODEL="$MODEL", PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1")
cmd = f"cd {U} && bash explorer/queues/train_g3_scale.sh > {LOG} 2>&1; echo __EXIT_\$? >> {LOG}"
p = subprocess.Popen(cmd, shell=True, env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Launched training PID", p.pid, "log:", LOG)
PY
    ;;
  serve)
    cat <<PY | colab exec -s "$S" --timeout 300
import os, pathlib, subprocess
U = pathlib.Path("$RDIR/untranslatables")
OUT = U / "results" / "explorer" / "$MODEL"
LOG = OUT / "server.log"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run("pkill -f 'explorer serve' || true", shell=True)
args = ["python", "-m", "explorer", "serve", "--mode", "live", "--model", "$MODEL", "--host", "127.0.0.1", "--port", "$PORT"]
for flag, p in [
    ("--global-axes", OUT / "phase3" / "global_axes.npz"),
    ("--lora", OUT / "phase3" / "multiref_lora.pt"),
    ("--reft", OUT / "phase3" / "spectral_reft.pt"),
    ("--fork-head", OUT / "fork_head.pt"),
]:
    if p.exists():
        args += [flag, str(p)]
env = dict(os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
cmd = "cd " + str(U) + " && " + " ".join(args) + f" > {LOG} 2>&1"
p = subprocess.Popen(cmd, shell=True, env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Started live workbench PID", p.pid, "->", " ".join(args))
print("Log:", LOG)
PY
    ;;
  tunnel)
    echo "Opening SSH port forward: http://127.0.0.1:${PORT}/ -> Colab session '${S}'"
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ProxyCommand="colab ssh --proxy-mode -s ${S}" \
        -N -L "${PORT}:127.0.0.1:${PORT}" root@colab
    ;;
  fetch)
    LOCAL="$REPO/untranslatables/results/explorer/$MODEL"
    mkdir -p "$LOCAL/phase3"
    REMOTE_TGZ="/content/explorer_${MODEL}.tgz"
    LOCAL_TGZ="$(mktemp "/tmp/explorer_${MODEL}_XXXXXX.tgz")"
    trap 'rm -f "$LOCAL_TGZ"' EXIT
    PACK_OK=0
    if cat <<PY | colab exec -s "$S" --timeout 300
import pathlib, tarfile
res = pathlib.Path("$RES")
out_tgz = pathlib.Path("$REMOTE_TGZ")
targets = [
    "samples",
    "branches",
    "steering",
    "phase3",
    "summary.json",
    "report.html",
    "fork_head.pt",
    "train_g3_scale.log",
    "server.log",
]
added = []
with tarfile.open(out_tgz, "w:gz") as tf:
    for rel in targets:
        p = res / rel
        if p.exists():
            tf.add(p, arcname=rel)
            added.append(rel)
    if res.is_dir():
        for mp in sorted(res.glob("map_*.json")):
            tf.add(mp, arcname=mp.name)
            added.append(mp.name)
print("Packed", len(added), "entries into", out_tgz, "->", added)
PY
    then
      if colab download -s "$S" "$REMOTE_TGZ" "$LOCAL_TGZ" </dev/null; then
        tar -xzf "$LOCAL_TGZ" -C "$LOCAL"
        PACK_OK=1
        echo "import pathlib; pathlib.Path('$REMOTE_TGZ').unlink(missing_ok=True)" \
          | colab exec -s "$S" --timeout 60 >/dev/null 2>&1 || true
      fi
    fi
    rm -f "$LOCAL_TGZ"
    trap - EXIT
    if [[ "$PACK_OK" != "1" ]]; then
      echo "Tarball pack/download unavailable; falling back to direct file downloads..." >&2
      for f in fork_head.pt summary.json report.html map_duende_flamenco.json train_g3_scale.log server.log; do
        colab download -s "$S" "$RES/$f" "$LOCAL/$f" </dev/null || true
      done
      for f in global_axes.npz multiref_lora.pt spectral_reft.pt; do
        colab download -s "$S" "$RES/phase3/$f" "$LOCAL/phase3/$f" </dev/null || true
      done
    fi
    echo "Fetched outputs into $LOCAL"
    ;;
  down)
    colab stop -s "$S" </dev/null || true
    colab sessions </dev/null || true
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
