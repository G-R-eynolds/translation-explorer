# HANDOFF: Run the 70B untranslatability experiments on Colab through the official Colab CLI

21 Nov 2026. You are Claude Code on the user's Linux laptop, in `Translation Experiments/`. The goal is to run the experiments in `colab/cli/` on a Colab 80 GB GPU **from the terminal**, using Google's official `google-colab-cli` (released June 2026; `colab ssh` merged July 2026). Bring the results back to `results_colab/`, then write up what they show.

**Context:**
- Read `HANDOFF_internals.md`, especially §0 and §3–§5, for what the experiments mean.
- The scripts here are generated from the same source as `colab/untranslatability_internals_colab.ipynb`, so the notebook is the readable version and the scripts are the runnable one.
- The scripts have only been tested end to end on a tiny random model on CPU, never on a GPU. **Expect to fix small things.** Keep your fixes in the scripts and note them in `colab/cli/CHANGES.md`.

## 0. Rules
- **Use only the official CLI** (`colab new / install / exec / download / ssh / stop`). Do **not** use third-party tunnels (`colab_ssh`, ngrok, cloudflared): Google prohibits them on Colab and using them risks the user's account.
- **Compute units are the user's money.** Every session must end with `colab stop`, including after errors. `run_all.sh` has an EXIT trap for this. When you drive stages by hand, check `colab sessions` at the end and stop anything left over. Never leave a GPU session idle while you think: if you need a long pause, stop the session and accept the re-setup cost.
- **Ask the user before** creating the first paid (A100/H100) session, and again if a run looks like exceeding **25 compute units** in total. The A100 80 GB costs ≈7.5 units/h and the H100 more. The full run is estimated at 2–2.5 h on the A100 80 GB.

## 1. Install and authenticate (laptop)
```bash
uv tool install google-colab-cli      # or: pip install -U google-colab-cli
colab version
colab --help                          # check the subcommands and flags below exist; adapt run_all.sh if any differ
```
Authenticate as the CLI's README describes; the first `colab new` may open a browser for Google sign-in, so ask the user to complete it. The account must be the one with Colab Pro and the compute units.

## 2. Rehearsal (required): Qwen 0.5B on a T4, ~10 min, ~0.2 units
```bash
cd "colab/cli" && chmod +x run_all.sh
DRY=1 ./run_all.sh
```
This exercises every stage: logit lens, ladder generation, round-trip scoring, corpus streaming from Hugging Face, XLM-R alignment, covariance collection, CCA, probes, mediation and summary. Results land in `results_colab/Qwen_Qwen2_5_0_5B_Instruct/`.

**Check:**
- Every stage printed `PACKED`.
- There are figures in `figures/` and a `SUMMARY.md`.
- The log (`results_colab/run_*.log`) has no Python tracebacks.

Note that `colab exec` might return exit status 0 even when the Python code raised. **Grep the log for `Traceback`.** If it does return 0 on errors, don't trust `run_all.sh` to stop on failure: drive stages by hand (§3).

Things that may need fixing on real hardware:
- **Europarl** may fail to load with current `datasets` (the code falls back to OPUS-100 only; that's fine, but note it).
- **spaCy model download:** if it fails, the code falls back to regex words and the morphology probes are skipped. Try `colab exec` with `import spacy.cli; spacy.cli.download("es_core_news_sm")`.
- **transformers versions:** layer outputs being a tuple vs a tensor is handled in the hooks. If generation complains about sampling flags, drop `top_k=None` and the like.

## 3. Main run: Llama-3.3-70B (4-bit), stage by stage, with stop points
Drive it by hand so that each stage can be checked before moving on, and keep the session alive between stages (**the kernel state matters**: Part D needs the fitted bases that C2 holds in GPU memory).

```bash
S=ut
colab new -s $S --gpu A100 --high-mem
echo 'import torch; print(torch.cuda.get_device_properties(0))' | colab exec -s $S
```
**If the card is 40 GB**, run `colab stop -s $S`, then `colab new -s $S --gpu H100 --high-mem`. `00_setup.py` also refuses to continue on a card under 75 GB.

```bash
colab install -s $S -r requirements.txt
colab exec -s $S -f 00_setup.py              # ~10 min: downloads ~40 GB and loads the model
colab exec -s $S -f A_detour.py              # ~3 min
colab download -s $S /content/ut_results_unsloth_Llama_3_3_70B_Instruct_bnb_4bit.tar.gz ../../results_colab/   # then untar
```
After each stage, download, untar into `results_colab/`, and **look at the numbers before going on**.

| stage | file | time (A100 80 GB, est.) | stop and check |
|---|---|---|---|
| setup | `00_setup.py` | 10–15 min | GPU name and memory printed; model loads |
| A | `A_detour.py` | 3 min | P(English) rises in the middle layers and Italian overtakes late. If English never rises, say so |
| B | `B_roundtrip.py` | 15–25 min | **Stop point (tell the user):** the *duende* line (leftover duende / gusta / mucho vs last time's 0.50 / 0.23 / 0.01), sense recovery for *duende*, the category table, hot-word AUROC. Read some ladder outputs in `B_ladder.json`: do the B3/B5/B6 prompts produce what they ask for? |
| C1 | `C1_collect.py` | 40–60 min | prints progress every 100 batches; produces covariance files in `/content/cache` (on the VM, not downloaded) |
| C2 | `C2_spectrum.py` | 10–20 min | **Stop point:** see below |
| D | `D_mediation.py` | 10–15 min | **Stop point:** flip fractions and the understood-vs-translated gap |
| Z | `Z_summary.py` | 1 min | `SUMMARY.md` |

Then `colab log -s $S -o ../../results_colab/session.ipynb` and **`colab stop -s $S`**.

**Long calls:** C1 runs ~20–30 min per language pair. If `colab exec` times out client-side on long calls, check whether the kernel keeps running (`colab status`, then a quick `colab exec` that prints `acc[C_LAYERS[0]].n` or checks the cache files). Otherwise run C1 with `PAIRS=['es']` first (`echo "PAIRS=['es']" | colab exec -s $S`), then `PAIRS=['es','hi']`. Already-collected pairs are skipped.

**If the kernel or session dies:**
- Results on the VM's disk survive a kernel restart but not a lost VM. Stages skip anything whose JSON already exists.
- After a kernel restart, rerun `00_setup.py`, then the stages still to do. Note that **D needs C1 then C2 to have run in the same kernel** (C1 is fast if its cache exists).
- After a lost VM, start again from setup. Copy any downloaded JSONs back up only if you want to skip finished stages: `colab upload` if the CLI has it, otherwise rerun.

**Optional, SSH:** `colab ssh -s $S` gives an interactive shell, handy for `nvidia-smi`, watching `/content/cache`, or tailing progress. Use `ProxyCommand colab ssh --proxy-mode -s ut` in `~/.ssh/config` for `ssh colab`. If it returns HTTP 404, the runtime doesn't expose SSH: carry on with `exec`.

## 4. What to check at the C2 stop point (this decides whether Part C means anything)
Report these, in order:
1. **The spectrum is real.** At the best layer, the number of shared directions above the shuffled-pair null, and the gap between held-out and in-sample top-50 ρ. A huge gap means overfitting; say so.
2. **U picks out hard words.** AUROC of U and U_raw for hot vs ordinary content words, **compared with the #tokens baseline**. If U doesn't beat the baseline, U may just be measuring how rare or how split-up a word is. Say this plainly; it's the main threat.
3. **The father's sentence:** U and half-life for *duende*, *gusta*, *mucho*.
4. **English bias:** E_en←x vs E_x←en for es and hi, raw and standardised. Is the sign consistent across layers?
5. **Probes:** the shared-direction share for gender, number, mood and tense against chance. Prediction: gender is mostly private and tense mostly shared.
6. **Agreement with the round trip** (Spearman).

If the spectrum isn't above the null, or U fails against the baseline, **still run D**, but flag that its "private part" is then hard to interpret.

## 5. Write-up (`results_colab/REPORT.md`)
- One section per part. Each section's first line says whether the result was positive, null or mixed, with the statistic.
- Compare with the earlier 8B results (`untranslatables/results/sent/report.md`) wherever the same quantity exists.
- **Captions must not overclaim.** Three auto-generated captions overclaimed last time. Every claim in the write-up must cite the number it rests on.
- Add a short **Colab cost log**: units used per stage (from `colab status`, or elapsed time × rate).
- End with a recommendation for the next step (see `HANDOFF_internals.md` §10): which parts are worth scaling further (Qwen2.5-72B as a second 70B model; Gemma 2/3 27B with Gemma Scope for the SAE experiment, which is not in these scripts yet), and which should be dropped.

## 6. Second model (only if the user asks after seeing the Llama results)
`MODEL_ID=unsloth/Qwen2.5-72B-Instruct-bnb-4bit`. It's stronger on Hindi, so it's the better test of the Hindi–English bias. Same procedure. Results go in their own folder, named by model.
