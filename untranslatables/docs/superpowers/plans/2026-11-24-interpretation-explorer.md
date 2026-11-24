# Interpretation explorer, step 1: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether `gemma-3-4b-pt` holds distinct readings of hard Spanish texts, and whether a Fisher-natural direction added from the cursor onward steers generation into a chosen reading.

**Architecture:** A new package `explorer/` in the untranslatables repo. It reuses the sampling and scoring primitives of `blind/fisher.py`, the Gram-matrix spectral solver `blind/spectrum.Comparator`, and `ut.arch`. The stages run per text and can be resumed. Results go to `results/explorer/<model>/` and are summarised in a static HTML report.

**Tech stack:** Python 3.12, torch 2.14, transformers 5.18, numpy, scipy, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-interpretation-explorer-design.md`

## Global constraints

- Model key `g3_4b_pt` = `google/gemma-3-4b-pt`, bf16. Replace the image encoder with `torch.nn.Identity()`.
- N = 64 samples per text, temperature 1, `top_k=0`, `top_p=1`, max new tokens = 3·|text tokens| + 16.
- A reading is a cluster with at least 5% of the samples. Choose k ∈ [2, 6] by silhouette; use k = 1 if the best silhouette is below 0.1.
- The stored layer is the 50% layer (17 for Gemma 3 4B). Natural direction: ε relative 1e-2. Target KL ∈ {0.25, 0.5, 1, 2} nats.
- Steering adds αv at positions ≥ b, decode steps included. The "at b only" variant is also tested.
- Success = lands in the target reading AND fluency (log-probability per token under the unsteered model) is within 1.5 nats of that reading's median.
- Decision rule: at least half of the hard texts show 2 or more readings, AND steering succeeds in at least 60% of (text, ordered reading pair) cases at some target KL ≤ 1.
- Never invent labels. Clusters are labelled from the texts, in `readings_labels.yaml`.

---

### Task 1: Material, frame and loader

**Files:**
- Create: `explorer/__init__.py`, `explorer/data/texts.yaml`, `explorer/frame.py`
- Modify: `ut/common.py`, to add `"g3_4b_pt": dict(hf_id="google/gemma-3-4b-pt", quant=None, family="gemma3", base=True)`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Produces:
  - `load_texts() -> list[dict]` (keys `id, text, kind, note`);
  - `build_prompt(tok, text) -> (ids: list[int], start: int)` (`start` = index of the first output position);
  - `cut(tok, y: list[int]) -> list[int]` (truncates at "\n\n" or "Spanish:");
  - `load(model_key) -> (tok, model)` (requires_grad off, image encoder stubbed).

- [ ] Step 1: write the tests.

  ```python
  def test_frame_and_cut(tiny_tok):
      ids, start = build_prompt(tiny_tok, "Hola.")
      assert start == len(ids) and tiny_tok.decode(ids).rstrip().endswith("English:")
      y = tiny_tok("I am here.\n\nSpanish: x", add_special_tokens=False)["input_ids"]
      assert tiny_tok.decode(cut(tiny_tok, y)).strip() == "I am here."

  def test_texts():
      t = load_texts()
      assert len(t) == 8
      assert sum(x["kind"] == "control" for x in t) == 1
  ```

- [ ] Step 2: run the tests. Expected: FAIL (module missing).
- [ ] Step 3: implement `frame.py` and `texts.yaml`. The few-shot examples must be plain, public-domain sentences.
- [ ] Step 4: run the tests. Expected: PASS.
- [ ] Step 5: commit.

### Task 2: Cursor steering hook

**Files:**
- Create: `explorer/steer.py`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Produces: `CursorHook(model, layer)` with
  - `.set(vec: Tensor[d] | None, b: int, only_at_b: bool = False)`
  - `.remove()`

  It adds `vec` to the block-`layer` output at absolute positions ≥ b (only b if `only_at_b`). It tracks absolute positions across KV-cache decode steps through an internal counter, reset whenever a call's sequence length is above 1.

- [ ] Step 1: write the test. With a tiny model, record the block outputs with and without the hook over a prefill of length 6 (b = 3) plus 2 decode steps, then check:
  - the difference is `vec` at positions 3, 4, 5 and at both decode steps;
  - the difference is 0 at positions 0–2;
  - with `only_at_b`, the difference is nonzero only at position 3.
- [ ] Step 2: run the test. Expected: FAIL.
- [ ] Step 3: implement. The position counter is `self.pos = seq_len` after a prefill call, and `+= 1` after each decode call.
- [ ] Step 4: run the test. Expected: PASS.
- [ ] Step 5: commit.

### Task 3: Sampling and scoring

**Files:**
- Create: `explorer/sample.py`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Consumes: `blind.fisher.generate_ids`, `blind.fisher.seq_logp`, `frame.cut`, `steer.CursorHook`.
- Produces:
  - `sample(model, tok, ids, n, max_new, seed, temperature=1.0) -> list[list[int]]` (already cut);
  - `score(model, ids, ys) -> np.ndarray` (summed log p per continuation, no grad);
  - `steered(model, tok, hook, ids, prefix, vec, b, max_new, n=1, temperature=0.0, only_at_b=False) -> list[list[int]]`.

- [ ] Step 1: test on the tiny model.
  - `score` of a sampled y equals the sum of the token log-probabilities from a manual forward pass.
  - `steered` with `vec=None` equals greedy `sample` at temperature 0.
- [ ] Step 2: run the tests. Expected: FAIL.
- [ ] Step 3: implement.
- [ ] Step 4: run the tests. Expected: PASS.
- [ ] Step 5: commit.

### Task 4: Readings

**Files:**
- Create: `explorer/readings.py`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Produces:
  - `embed(model, ids, ys) -> np.ndarray (n, d)`: mean last-layer state over the y tokens, L2-normalised;
  - `cluster(X, min_share=0.05, kmax=6, sil_min=0.1) -> dict(labels, k, silhouette, readings: list[int])`: the cluster ids with at least 5% of the samples, ordered by size;
  - `centroids(X, labels, readings) -> np.ndarray`;
  - `assign(x, C) -> int`.

- [ ] Step 1: test.
  - Three planted Gaussian clusters on the unit sphere (sizes 30/20/14) give k = 3 and three readings.
  - Isotropic noise gives k = 1.
- [ ] Step 2: run the tests. Expected: FAIL.
- [ ] Step 3: implement with sklearn `AgglomerativeClustering(metric="cosine", linkage="average")` and `silhouette_score`.
- [ ] Step 4: run the tests. Expected: PASS.
- [ ] Step 5: commit.

### Task 5: Directions

**Files:**
- Create: `explorer/directions.py`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Consumes: `blind.spectrum.Comparator`.
- Produces:
  - `grads_at(model, ids_prefix, conts, layer, b, batch=8) -> np.ndarray (n, d)`: the gradient of log p(cont | prefix) with respect to the layer-ℓ residual at position b, through an additive zero leaf;
  - `natural(GA, GB, eps=1e-2) -> v (unit)`;
  - `plain(GA, GB) -> v (unit)`;
  - `alpha_for(G_all, v, kl) -> float`: α = sqrt(2·kl / vᵀĜv).

  `natural` solves (Ĝ + ε·tr(Ĝ)/r·I) x = (ḡ_A − ḡ_B) in the span of the rows of [GA; GB], with Ĝ the empirical Fisher of all rows.

- [ ] Step 1: test. On random GA, GB (20×50 each), `natural` equals the dense solve projected onto the row span, after normalisation.
- [ ] Step 2: run the test. Expected: FAIL.
- [ ] Step 3: implement. `grads_at` uses a forward hook on `ut.arch.layers(model)[layer]` that adds a zero leaf of shape (B, d) at position b.
- [ ] Step 4: run the test. Expected: PASS.
- [ ] Step 5: commit.

### Task 6: Branches

**Files:**
- Create: `explorer/branches.py`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Produces:
  - `timeline(model, ids, y_ref) -> dict(entropy: list, fisher_top: list)`. The Fisher top eigenvalue at position t: the output Fisher at the last-layer residual is Wᵀ(diag(p) − ppᵀ)W. Its top eigenvalue comes from 20 power iterations using matrix-vector products only.
  - `choose_b(y_ref, members_by_reading: dict[int, list[list[int]]], fisher_top) -> (b: int, pair: tuple | None, why: str)`. This returns the earliest t where at least one member of each of two readings shares `y_ref[:t]` and their tokens at t differ; otherwise the argmax of `fisher_top`.

- [ ] Step 1: test `choose_b` on synthetic token lists:
  - readings `{0: [[1,2,3]], 1: [[1,2,9]]}` with `y_ref=[1,2,3]` give b = 2;
  - with no shared prefix and the first tokens differing, b = 0;
  - test `timeline` shapes on the tiny model.
- [ ] Step 2: run the tests. Expected: FAIL.
- [ ] Step 3: implement.
- [ ] Step 4: run the tests. Expected: PASS.
- [ ] Step 5: commit.

### Task 7: Pipeline, report, end-to-end test, launch

**Files:**
- Create: `explorer/run.py`, `explorer/report.py`, `explorer/__main__.py`, `explorer/queue.sh`, `explorer/README.md`
- Test: `tests/test_explorer.py` (end to end on the tiny model, 2 texts, N = 8)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `python -m explorer run --model g3_4b_pt [--n 64] [--texts ids] [--layer 17]`
  - `python -m explorer report --model g3_4b_pt`
  - outputs as in the spec.

Pipeline per text (each step skipped if its output exists):
1. Sample, embed, cluster; write `samples/<id>.json`.
2. Compute the timeline and b.
3. For each ordered pair of readings (A, B), and each variant in {natural, plain} × {cursor, at_b}:
   - compute the direction;
   - for each target KL: greedy continuation plus 8 samples at T = 0.7;
   - assign each continuation to a reading, check fluency, flag success.
4. Write `summary.json`, applying the decision rule.

- [ ] Step 1: write the end-to-end test. Run the pipeline on the tiny model with two texts and N = 8. Assert that every output file exists, and that `summary.json` has the keys `n_hard_with_2plus`, `steer_success_rate`, `pass`.
- [ ] Step 2: run the test. Expected: FAIL.
- [ ] Step 3: implement `run.py`, `report.py` and `__main__.py`.
- [ ] Step 4: run the full test file. Expected: PASS.
- [ ] Step 5: check that `google/gemma-3-4b-pt` is accessible on the desktop. Sync, then launch `remote/run.sh explorer bash explorer/queue.sh`, and watch the first text.
- [ ] Step 6: commit.
