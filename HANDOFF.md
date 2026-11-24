# HANDOFF: Translation lattices with a local Llama, and a principled "differs by meaning"

## 1. What this is

This is a side project feeding ideas into a humanities grant bid (AHRC) on AI translation and **untranslatability**. In translation studies (Cassin, Sakai, Liu), untranslatables are culturally or imaginatively embedded meanings with no ready equivalent, which keep provoking retranslation. The bid asks what a translation model would look like if it *explored* those points instead of flattening them into one "best" output.

The output we want is a **translation lattice**:
- sample many line-for-line translations of a poem;
- group each line's renderings by **meaning**, not wording;
- draw a graph that narrows where samples agree and branches where they genuinely diverge, with a central "most likely" route.

A prototype already exists (see §2). In it, the "same meaning?" judgement was made by asking an LLM. **The goal now is to replace that with a quantity grounded in the model's own probabilities and hidden states, and validate it.** No LLM-as-judge loops in the core metric.

Test poem: G. G. Belli, *Er giorno der giudizzio* (1831), a Romanesco sonnet in the public domain.

## 2. What already exists (translation-lattice.zip)

```
lattice/
  data/source.json     # the sonnet, 14 lines; lines 7-14 NOT yet verified against a print edition
  data/samples.json    # 12 hand-written demo translations, 4 stances x 3 (literal, rhymed, scots, london)
  data/clusters.json   # hand-made meaning clusters per line, with gloss + note
  pipeline.py          # Anthropic-API version of sampling + LLM-judge clustering (superseded by this work)
  build.py             # samples + clusters -> out/lattice.json + out/index.html (interactive SVG lattice)
  template.html        # page template; build.py injects the data
```

Schemas, which keep `build.py` working unchanged:
- `samples.json`: `{sampler, stances:{name:desc}, samples:[{id, stance, lines:[n strings]}]}`
- `clusters.json`: `{clusterer, lines:[{line, note, clusters:[{id, gloss, members:[sample ids]}]}]}`

Every sample must appear in exactly one cluster per line. `build.py` asserts this.

Keep the 12 demo samples and hand-made clusters as a **weak reference**. They were written by the same model that made the clusters, so they're biased. Don't treat them as ground truth.

## 3. Compute environment

- **Laptop**: project files live here. Run Claude Code here.
- **GPU**: a Windows desktop with an RTX 4070 (12 GB VRAM), running WSL2 Ubuntu. It's reached over Tailscale as `ssh desktop-gpu`. This setup has worked since Oct 2026.
- The user's dissertation code (`~/Documents/Fourth Year/Dissertation/code`) already drives the GPU this way. **Look there first for existing conventions**: sync scripts, the Python env on the desktop, HF cache location, any Llama weights already downloaded. Reuse them rather than inventing new ones.
- Check before building: `ssh desktop-gpu nvidia-smi`, the Python/torch/CUDA versions on the desktop, free disk space, and whether an HF token with Llama licence access is configured.

Put this project in a new folder on the laptop (ask the user where; suggest `~/projects/translation-lattice`). Unzip the prototype into it. Rsync to the desktop to run jobs, and pull results back.

### Model choice
- **Primary: `meta-llama/Llama-3.1-8B-Instruct` in 4-bit (bitsandbytes nf4)**, about 6 GB. That leaves room for batching and `output_hidden_states`. bf16 (16 GB) does not fit.
- Fallback / full-precision internals: `meta-llama/Llama-3.2-3B-Instruct` in bf16 (~6.5 GB).
- Second model for the epistemic analysis in §8: `Qwen/Qwen2.5-7B-Instruct` in nf4.
- Use plain HF `transformers` for everything that needs log-probs or hidden states. vLLM is optional and only for sampling speed. Don't add it unless sampling is the bottleneck.

**Sanity check first:** an 8B model's Romanesco may be weak. Before anything else, ask it to paraphrase each source line into standard Italian and save the output for the user to eye. Record this as a known limitation. Weak understanding is *epistemic* uncertainty, and §8 exists to separate it from genuine multiplicity in the source.

## 4. Phase 1: sampling

- Prompt for strictly line-for-line translations, numbered `1.` to `14.`. Parse them, and retry up to 3 times if the line count is wrong. Log the failure rate.
- Use 4 stances (reuse the `stances` descriptions in `samples.json`) × 12 samples each at T=1.0, top-p 0.95, giving **N=48**. Start with N=24 while debugging.
- Write `data/samples_llama.json` in the same schema. Record the model, quantisation, temperature, seed and prompt in `sampler`.
- Also log the **per-token log-probs** of each sample under its own prompt. They're cheap and useful later.

## 5. Phase 2: measuring "differs by meaning"

### 5.1 The idea
Two renderings of a line mean the same thing if they **commit the reader to the same things about the source**. Differences in wording, dialect, register or rhyme shouldn't count.

We make that operational with a **style-invariant channel**: back-translation into standard Italian. A Scots line and a London line that make the same interpretive choice should produce the *same distribution* of Italian back-translations. A line that reads *fora a chi tocca* as "Next!" (queue) should produce a different one from a line that reads it as "Arise, you who are called" (summons).

### 5.2 Metric M1: back-translation Jensen–Shannon divergence (primary)

For line *i* and renderings r₁…r_N, define
q_j(b) = p_θ(b | BT-prompt(r_j)), a distribution over Italian strings b.

**Important: the BT prompt must NOT contain the source text**, or every rendering just copies the source back and all distances collapse to 0. It may include the neighbouring *English* lines of the same sample for context.

Distance: d_ij = JSD(q_i ‖ q_j) ∈ [0, ln 2].

Monte Carlo estimator:
1. Sample m back-translations from each q_j (start with m=6, T=1.0). Pool them as B.
2. Teacher-force every b ∈ B under every rendering's BT prompt. This gives the score matrix L[j, b] = log q_j(b), the **sum** of token log-probs, with no length normalisation. Normalisation isn't needed because JSD compares two distributions on the *same* string, so length effects cancel.
3. With M = ½(q_i + q_j), so that log M(b) = logsumexp(L[i,b], L[j,b]) − log 2:
   JSD_ij ≈ ½·mean_{b∈B_i}[L[i,b] − log M(b)] + ½·mean_{b∈B_j}[L[j,b] − log M(b)].
4. Clamp to [0, ln 2]. Report the MC standard error by bootstrapping over b.

Efficiency: for each j, cache the KV of the BT prompt and score all b ∈ B as continuations. At N=48 and m=6, that's 48 × 288 short continuations per line. Benchmark one line before running all 14.

### 5.3 Metric M2: hidden-state distance with style erased (secondary; also the interpretability view)

- **Hold context fixed.** Each rendering originally followed different previous lines. Re-embed every rendering of line *i* after a **common prefix**: the source poem plus the central-route lines 1…i−1 (any fixed choice will do; record it). This counterfactual splice removes context confounds.
- Collect residual-stream states at every layer, mean-pooled over the rendering's tokens. Call them h_j^ℓ.
- **Erase stance** with LEACE (EleutherAI `concept-erasure`, Belrose et al. 2023). It's the provably least-squares-optimal linear erasure of a concept. Fit it per layer on all lines' renderings, with the stance labels.
- Distance: cosine distance on the erased states. Pick ℓ by the validation in §6. Report all layers.

### 5.4 Baselines (to beat)
- **B0 lexical**: char-n-gram TF-IDF cosine. `build.py` already computes it; on the demo data it matched neither meaning nor stance.
- **B1 raw M2**: hidden states *without* LEACE. Measures how much style contaminates representations.
- **B2 NLI**: bidirectional entailment with an off-the-shelf NLI model (e.g. `microsoft/deberta-large-mnli`). This is the semantic-entropy clustering of Kuhn, Gal & Farquhar (2023). It's a classifier, not an LLM loop, and it's the standard method in the literature to compare against.

### 5.5 From distances to a lattice
- Kernel: K_ij = 1 − d_ij / ln 2 for M1, or 1 − cosine distance for M2.
- **Continuous measure (no hard clusters):** Kernel Language Entropy (Nikitin et al. 2024). With ρ = K / tr K, take the von Neumann entropy −tr(ρ log ρ). This is the per-line **untranslatability score**. It's high where meanings genuinely spread.
- **Clusters for the lattice:** spectral clustering on K, choosing k by the eigengap of the normalised Laplacian, k ∈ {1,…,5}.
  - *k=1 test:* if every pairwise d_ij is below the style-noise threshold τ (§6), the line converges.
- Semantic entropy over the clusters, H = −Σ p_c log₂ p_c, as in the prototype.
- **Glosses without an LLM:** label each cluster by its medoid rendering (min total distance), plus its size. Leave `note` empty. The human-facing notes come later, from a person.
- Write `data/clusters_m1.json` (and `_m2`, `_nli`) in the prototype schema, then run `build.py`. Make `build.py` take the samples and clusters paths as CLI args if it doesn't already.

## 6. Phase 3: validating the metric (this is what makes it sound)

A metric of "same meaning" has to pass two tests:

1. **Style invariance.** Take renderings and have the model *restyle* them while keeping the meaning: same line into Scots / London / formal / plain. Spot-check that these preserve meaning and drop failures. These **style-only pairs** should have small d.
   - **τ = 95th percentile of d over the style-only pairs.** This calibrates "differs by meaning" as *more different than restyling alone produces*.
2. **Meaning sensitivity.** Build **minimal meaning edits**: change one interpretive choice and keep everything else (cellar → basement is *not* one; cellar → hell *is*; "Next!" → "Arise, you who are called"). Also include hand-picked cross-cluster pairs from the demo clusters. These should have d > τ.

Report, for each metric (M1, M2 at each layer, B0, B1, B2):
- the separation between the two distributions (AUROC);
- agreement with the hand-made clusters (adjusted Rand index), labelled as weak reference;
- agreement among the metrics themselves.

The headline result is whether M1 and/or erased-M2 separate meaning from style better than lexical and raw-embedding baselines. If they don't, say so plainly. That's a finding too.

Keep the control pairs in `data/controls.json` (id, line, a, b, type ∈ {style, meaning}, provenance) so they can be reused for other poems.

## 7. Phase 4: interpretability (layer sweep)
Use the M1 clusters as labels and run over layers ℓ:
- **meaning separation**: between-cluster / within-cluster distance of the raw h^ℓ;
- **style decodability**: linear-probe accuracy for stance from h^ℓ (cross-validated);
- **source-side view**: logit lens on the *source* tokens of the untranslatable spans (*fora a chi tocca*, *filastrocca*, *a pecorone*, *bbona sera*). At which layer does the model's internal "nearest English word" settle, and does it settle on one reading? Wendler et al. (2024), "Do Llamas Work in English?", is the relevant prior work on an English-like pivot in middle layers.

Plot all three against layer on one figure. The interesting claim to test: meaning separates in middle layers while style is still decodable, and LEACE removes the second without destroying the first.

## 8. Phase 5 (stretch): genuine multiplicity vs the model not knowing
Sample and score with a second model (Qwen2.5-7B). Per line, take the total semantic entropy of the pooled samples minus the mean per-model entropy. That's the mutual information, the disagreement *between models* (epistemic). Lines where both models branch the same way are better candidates for genuine untranslatability than lines where they disagree. This distinction matters for the grant's argument: under-resourced languages will mostly show epistemic uncertainty, and that mustn't be presented as "mystery".

## 9. Deliverables
1. `samples_llama.json`, `clusters_{m1,m2,nli}.json`, `controls.json`
2. `out/index.html` lattices for M1 (main) and the baselines
3. `results.md`: validation table (AUROC, τ, ARI), per-line KLE/semantic entropy, the layer-sweep figure, a short paragraph per branching line naming the competing readings by their medoids, and the limitations
4. Clean scripts: `sample.py`, `backtranslate.py`, `score.py`, `metrics.py`, `controls.py`, `layers.py`, plus a `README`

## 10. Guardrails
- Debug at small scale (N=24, m=4, 3 lines) before full runs. Print timing estimates before launching anything over ~15 minutes.
- Fix seeds and record every prompt verbatim in the outputs.
- Don't commit model weights or HF tokens. Results JSON is fine.
- Don't "fix" the poem text yourself. Lines 7–14 need checking against a print edition by the user.
- Stop and ask the user if: the model can't produce line-aligned output reliably (>30% failures), M1 is too slow even after KV caching, or the validation shows no metric beats the lexical baseline.
