# Executed inside the Colab kernel by colab_explorer.sh (`colab exec -f`).
# Clones/updates the public repo, installs deps, and starts the explorer queue as a DETACHED process so a dropped
# CLI connection cannot kill it (lesson from the 22 Nov 70B run: `colab exec` can hang without returning).
import os
import pathlib
import subprocess

REPO_URL = "https://github.com/G-R-eynolds/translation-explorer.git"
R = pathlib.Path("/content/translation-explorer")
U = R / "untranslatables"
LOG = U / "results" / "explorer" / "g3_4b_pt" / "colab_run.log"

if (R / ".git").exists():
    subprocess.run(["git", "-C", str(R), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(R)], check=True)
print(subprocess.run(["git", "-C", str(R), "log", "-1", "--oneline"], capture_output=True, text=True).stdout)

subprocess.run("pip install -q -U transformers accelerate pyyaml scipy scikit-learn", shell=True, check=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

tok = pathlib.Path.home() / ".cache" / "huggingface" / "token"
print("HF token present:", tok.exists())

running = subprocess.run(["pgrep", "-f", "run_all_desktop.sh"], capture_output=True, text=True).stdout.strip()
if running:
    print("already running, PIDs:", running)
else:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1",
               OMP_NUM_THREADS="8")
    cmd = f"cd {U} && bash explorer/queues/run_all_desktop.sh > {LOG} 2>&1; echo __EXIT_$? >> {LOG}"
    p = subprocess.Popen(cmd, shell=True, env=env, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("launched PID", p.pid, "log", LOG)
