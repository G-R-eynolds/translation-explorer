# Interpretation explorer, step 1: do readings exist, and can we steer into them?

24 Nov 2026. Approach C, staged: first test feasibility without fine-tuning; fine-tune (approach B) only where this
falls short.

## Goal

The end goal is a tool. You ask a model to translate a new, hard text and watch, in real time, the interpretations it
is weighing. You choose one at the cursor, and the generation follows it.

This step answers the question that decides how to build that tool, using the base model as it stands:
- Does the model hold several distinct readings of a hard text?
- Can a direction found from the model's own sensitivities steer the live generation into a chosen one?

## Scope

**In scope:**
- Reading discovery.
- Branch-point timelines.
- Directions between readings.
- A steering test from the cursor onward.
- A static HTML report per text.

**Out of scope:** the interactive real-time UI (step 2), fine-tuning (step 3, if needed), and any instruct model run. The code supports instruct models, but this run uses the base model only, as decided on 24 Nov.

## Model

The model is `google/gemma-3-4b-pt` (base, bf16, about 8 GB), on the desktop RTX 4070 (12 GB).

A base model has no chat template, so translation uses a fixed few-shot frame:

```
Spanish: <example 1>
English: <translation 1>

Spanish: <example 2>
English: <translation 2>

Spanish: <TEXT>
English:
```

The two examples are short public-domain prose sentences with plain translations, chosen so they don't push towards a reading of the test texts. Generation stops at a blank line or at the next "Spanish:".

The image encoder is replaced with a parameter-free stub, as in `blind/`; on 12 GB it otherwise pushed the driver into system-RAM fallback.

## Material (`explorer/data/texts.yaml`)

There are 9 texts (8 hard, 1 control), each with an id, the text, a kind (`hard` or `control`) and a note:
- **3 duende sentences:** the flamenco context, the goblin context, and the bare sentence.
- **3 short public-domain stanzas:** Bécquer (Rima XXI and one more), Antonio Machado, Rosalía de Castro, each 2–4 lines.
- **1 ambiguous prose sentence.**
- **1 plain control sentence,** expected to have essentially one reading.

## Method

1. **Sampling.** Draw N = 64 translations per text at temperature 1, with no top-k or top-p (the true distribution), max 3·|text tokens| + 16 new tokens. Store the token ids and the summed log-probability of each.

2. **Readings.**
   - Represent each translation by the base model's mean final-layer state over the translation tokens, with the source shown in front of it, so the representation is of meaning in context.
   - L2-normalise, then cluster agglomeratively (cosine distance, average linkage).
   - Choose k between 2 and 6 by silhouette, with k = 1 when the best silhouette is below 0.1.
   - A **reading** is a cluster holding at least 5% of the samples.
   - Report: the number of readings, the entropy of the cluster masses, and the 3 most probable members of each cluster.
   - Clusters are labelled afterwards, by reading the texts only, in `readings_labels.yaml`. The code never invents labels.

3. **Branch points.**
   - Along the most probable sample of the largest reading (the reference translation), compute at each output position t:
     1. the next-token entropy;
     2. the top eigenvalue of the output Fisher at the last-layer residual of position t, as a fork-strength measure;
     3. the reading divergence: for each reading, the mean log-probability of its members' continuations, scored on the reference prefix up to t. The spread across readings is reported only at positions where members share that prefix (see the note below).
   - The **branch point** used for steering is the first reference position where two readings' greedy continuations diverge in meaning. Concretely, it is the earliest t where the reference prefix is shared by at least one member of each of two readings and their next tokens differ. If there is none, use the position of maximum Fisher eigenvalue.

   *Note.* Members of different readings often split at the very first output token. Then the branch point is t = 0, the start of the English, and steering acts on the whole output. That is a valid and common case.

4. **Directions between readings.**
   - At the branch position b and the stored layer ℓ (default: the 50% layer, 17 of 34), take per-sample gradients of log p(y_rest | prefix) with respect to the residual at position b. Here y_rest is each member's continuation from b. Members whose prefix differs from the reference before b are rescored on the reference prefix plus their own continuation.
   - The direction towards reading A, away from B: g_A − g_B, where g_X is the mean gradient over X's members.
   - **Natural direction:** v = (Ĝ + εI)⁻¹ (g_A − g_B), with Ĝ the empirical Fisher over all members at b, computed in the span of the gradients as in `blind/spectrum.py`, with ε relative 1e-2. The "plain" variant g_A − g_B is also tested.
   - Normalise v to unit length. α is calibrated so that the predicted KL ½α²vᵀĜv equals a target t ∈ {0.25, 0.5, 1, 2} nats.

5. **Steering test.**
   - Starting from the reference prefix up to b, add αv to the layer-ℓ residual at position b **and every later position**, decode steps included ("from the cursor onward"). Decode greedily, and also take 8 samples at temperature 0.7.
   - Assign each continuation to the nearest reading centroid, using the same representation as in step 2.
   - **Success:** the continuation lands in the target reading, and its log-probability per token under the *unsteered* model is within 1.5 nats of the median of that reading's own members (the fluency guard).
   - Also tested: the simpler variant, which adds the vector at position b only, with no later positions. If it works as well, the tool uses it (it is simpler and cheaper).

6. **Decision rule (pre-registered here).** Step 1 passes if both hold:
   - at least half of the 8 hard texts show 2 or more readings;
   - steering reaches the target reading in at least 60% of (text, ordered reading pair) cases at some target KL ≤ 1 nat, without failing the fluency guard.

   The control should show 1 reading. If it doesn't, the clustering is splitting on surface wording, and the report says so before anything else.

## Outputs (`results/explorer/<model>/`)

- `samples/<text>.json`: texts, ids, log-probabilities and cluster assignments.
- `readings.json`
- `branches.json`: per-position entropy, Fisher eigenvalue and reading spread; the chosen b.
- `steering.jsonl`: every steered text, its assigned reading, its fluency and its success flag.
- `report.html`: per text, the reading cards, the branch timeline (an inline SVG strip under the reference translation) and side-by-side steering cards.
- `summary.json`: the decision-rule numbers.

## Components (`untranslatables/explorer/`)

| file | does | depends on |
|---|---|---|
| `frame.py` | few-shot prompt building; stop rules; the base-model loader (with the image-encoder stub) | `ut.common`, `ut.arch` |
| `sample.py` | sampling, and continuation scoring with and without steering | `frame`, `blind.fisher` (sampling primitives, `seq_logp`) |
| `steer.py` | `CursorHook`: adds a vector at positions ≥ b, decode steps included | `ut.arch` |
| `readings.py` | translation representations, clustering, silhouette choice of k | numpy, scipy, sklearn |
| `branches.py` | entropy, output-Fisher eigenvalue, divergence, choice of b | `sample`, `steer` |
| `directions.py` | per-sample gradients at b; plain and natural directions; α calibration | `blind.spectrum.Comparator` |
| `run.py` | the stages: sample → readings → branches → directions+steer → summary; resumable per text | all of the above |
| `report.py` | the HTML report | outputs only |

Tests (`tests/test_explorer.py`, CPU, on the tiny model):
- the prompt frame and stop rules;
- `CursorHook` adds the vector at exactly the positions ≥ b, decode steps included, and nowhere else;
- clustering recovers planted clusters;
- the natural direction equals the dense solve on random data;
- the pipeline runs end to end on two texts.

## Error handling

- Each stage writes its output per text, and is skipped if the output exists.
- An out-of-memory error halves the batch, as in `blind/`.
- A text where no reading is found, or that has a single reading, goes in the report with "1 reading" and no steering section; that is a valid result, not an error.

## Honesty

- Readings are clusters of the model's own samples. They are the readings the base model weighs, not all possible readings.
- Labels come only from reading the texts.
- The steering test measures reachability within the model's own distribution. Success does not mean the reading is good.
