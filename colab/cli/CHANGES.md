# Changes to the Colab CLI scripts (made by Claude Code while running them)

- `run_all.sh`: `colab exec` (CLI v0.8.0) fails after 30 s without output by default; added `--timeout 0` to the setup
  and stage calls, which print nothing for minutes at a time (model download, collection).
- Laptop: installed `keyrings.alt` into the CLI's tool environment so the OAuth token is stored between calls.
- `run_all.sh`: `colab status -s NAME` exits 0 even when the session does not exist, so session creation was skipped; it now checks `colab sessions` for the name.
- `C2_spectrum.py`: probe crash (`IndexError: invalid index to scalar variable`) because `RidgeClassifier.coef_[0]` is a
  scalar when `coef_` is 1-D in the installed scikit-learn; now `np.ravel(clf.coef_)`.
- `C2_spectrum.py`: also reports `n_strong`, the number of held-out directions with rho > 0.71 (rho^2 > 0.5: more than half
  kept per round trip). In the rehearsal the handoff's "shared" count (held-out rho above the 99th percentile of the
  shuffled-pair null, ~0.07) saturated at 884-896 of 896 dimensions for es, which leaves D's "private" part only a few
  dimensions; the stricter count is reported alongside, the definition used by D is unchanged.
- Confirmed in the rehearsal: `colab exec` exits 0 when the cell raises, so every stage's log is grepped for Traceback.
- `drive.sh` (new): runs stages one at a time with `--timeout 0`, greps each stage log for tracebacks (exec exits 0 on errors) and halts on failure with the session left up; fetches results after every stage; timeline in results_colab/logs/timeline.txt.
- 21 Nov, first A100 session: the CLI connection dropped during C1 (Hindi corpus step) and `colab exec --timeout 0` hung
  for ~18 h before raising "Connection was lost"; the VM was gone by then (12 units used) and the C1 cache with it.
- `run_bg.py` (new): runs setup, A, B and C1 as a detached process on the VM (started from a short `colab exec` with
  `subprocess.Popen(..., start_new_session=True)`), so a dropped connection cannot stall or kill the long stage; it is
  polled every 4 min with time-limited calls.
- `drive2.sh` (new): kernel stages (setup, C1 cache check, C2, D, Z) under hard `timeout`s; after a lost connection it
  polls the VM for the stage's output file instead of trusting the client.
- `strict_pass.py` (new): after C2 and D, re-runs the C2 probes and D in the same kernel with "shared" = held-out rho > 0.71; outputs suffixed `_strict`. Reason: the null-based "shared" set saturated (8188-8191 of 8192 for es), making the probe shared-share 1.00 by construction and leaving D a ~2-dimensional private part.
- `strict_pass.py` caveat: D skips when `D_mediation.json` exists, so the strict D was run separately with `have()` redirected to the `_strict` file name.
