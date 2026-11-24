# HANDOFF: Untranslatability inside the model. Round-trip spectra, features that don't cross over, and causal mediation

21 Nov 2026. This follows `HANDOFF_sentence_untranslatability.md` and reuses `untranslatables/` (especially `ut/` and `sent/`). It runs on the **desktop RTX 4070 (12 GB)** through the same `remote/` workflow, on models somewhat bigger than before (9–14B in nf4). **What this run finds decides what we scale up on Colab** (big GPUs, Jupyter only). §11 says how to keep the code easy to move there.

## 0. Why, and what changed since last time

**The project so far** (see `results/sent/report.md` and `results/3b/`):
- **Round-trip recoverability (d_RT) works** at sentence level. Categories separate (8B: control 0.21 vs untranslatable 0.93 doublings). Different back-translators agree (Spearman 0.70–0.75). The father's sentence comes out in the expected order: *duende* 0.50 > *gusta* 0.23 > *mucho* 0.01.
- **The concept-layer optimal-transport distance (d_CL) is nearly flat** at sentence level. We are replacing it (see §3).
- **Copying loophole:** in the 8B, all of B2's ("keep the word") gain was copying. With source words masked, the gain is negative.
- **The targeting test was null.**
- **The English detour replicates:** hard content looks closer to English in the detour layers than in the concept layer (8B −0.021 vs control −0.009).
- **Duende is translated as "elf"**: all 12 sampled senses were folkloric, and the coinage was "duendismo". Aya and Mistral borrowed "duende" instead.

**The user now wants a definition grounded in the model's internals.** Three ideas, all defined precisely in §3–§5:
1. **Round-trip operator spectrum (N1):** untranslatability as what decays under repeated linear round trips between the two languages' representations. The decay rates are the squared canonical correlations.
2. **SAE features that don't cross over (N2):** interpretable features active on a source span with no counterpart on its translation.
3. **Causal mediation (N3):** meaning the model *has* in the source language (it affects Spanish-side behaviour) but that does *not* flow into its translation.

**This run also replicates the most important earlier results on the bigger models (§2), with the known flaws fixed.**

**Audience:**
- The formal definitions are for the user, a maths/CS student.
- The figures in §9 are for his father, a linguist working on an AHRC bid about translation (Cassin's *Dictionary of Untranslatables*, critique of print-era one-to-one equivalence, distortion of low-resource languages through English). **He is not mathematical.**

## 1. Models

| key | HF id | quant | approx. VRAM | role |
|---|---|---|---|---|
| `gemma9b` | `google/gemma-2-9b-it` | nf4 (try 8-bit for SAE work, §4.1) | ~7 GB | **main**: SAEs exist (Gemma Scope), so N1, N2, N3 and R2 all run here |
| `qwen14b` | `Qwen/Qwen2.5-14B-Instruct` | nf4 | ~9.5 GB | the biggest that fits; strong multilingual model including Hindi; N1, R1, R2 |
| `nemo12b` | `mistralai/Mistral-Nemo-Instruct-2407` | nf4 | ~8 GB | a European model the father named (Mistral); N1, R1, R2 |
| `ayax8b` | `CohereForAI/aya-expanse-8b` | nf4 | ~6 GB | less English-centric comparison; N1, R1, back-translation check |
| `8b` | `meta-llama/Llama-3.1-8B-Instruct` | nf4 | ~6 GB | **bridge** to the earlier results; N1 only (plus Llama Scope cross-check in N2 if time allows) |
| `sarvam2b` (optional) | `sarvamai/sarvam-1` | bf16 | ~5 GB | Indic-focused base model (the father named Sarvam); N1 on hi–en only |

**Notes:**
- **Check first** what is current: if **Gemma Scope 2 / Gemma 3** SAEs exist for a ≤12B model and cover the residual stream at many layers, prefer `google/gemma-3-12b-it` as the main model (it fits in nf4). Otherwise use Gemma 2 9B. Say which you picked and why.
- Gemma, Llama and possibly FLORES are gated. **Check licence acceptance before anything else**, and substitute and say so if one is missing.
- Add these to the registry in `ut/common.py`. Keep the pinned chat date and seeds. For Qwen 3–style models, if you substitute them, turn thinking off.

## 2. Replications (Part A): the most important earlier results, now on bigger models and with the flaws fixed

### R1. English detour (logit lens)
Rerun `latent_lang.py` on `gemma9b`, `qwen14b`, `nemo12b` and `ayax8b`. Report the layer profile of P(English) vs P(target language) for the translation task, as in Wendler et al. Cheap (~5–10 min per model). This is the backbone of the "models think in English" story, so it should hold up on the bigger models.

### R2. Sentence round trip, with the loophole fixed
Reuse `sent/` (ladder, d_RT, spread, figures). Run it on `gemma9b` and `qwen14b`, plus `nemo12b` if time allows, with these changes:

1. **Masking is the default.** When scoring d_RT, source-word tokens (and close cognates: normalised edit distance ≤ 0.3 to a source word of ≥4 letters) that appear in t are replaced with a neutral placeholder `[·]` before scoring. Report the unmasked score as a secondary.
2. **Borrowing is its own labelled rung.** Rename B2 to `B2_borrow` and report it separately. **Its gain is reported as "what borrowing buys"**, not folded into the distortion–length curve. The curve D_s(L) is computed over the non-borrowing rungs only, plus a second curve that includes borrowing. This turns the copying loophole into a result: how much of the gap a loan closes, which echoes the word-level "borrowing closes the gap" figure.
3. **Intended-sense recovery.** Add `intended_sense` to each hot span in `data/sentences.yaml`: a one-line English gloss of the sense meant *in this sentence*. For example, *duende* = "the mysterious, almost demonic power of a great artistic performance (Lorca's sense)", not "elf". Define:
   - **sense recovery** = log p(gloss | "In the original text, the word for '⟨span⟩' meant:" with the context being t) − the same with s as context, teacher-forced. Values near 0 mean t carries the intended sense as well as the source does.
   - Report sense recovery for B1 and B3–B6.
   - This directly tests the "elf" problem.
4. **Drop** d_CL (the optimal-transport version, replaced by N1's span U), the targeting test (null), B7 sampling beyond one greedy coinage, and the dialect items.
5. **Keep:** the heat map of the father's sentence, the category table, spread over hot spans (20 samples, NLI clusters), and validation §6.2–6.3 of the previous handoff (ladder monotone; hot-span AUROC).

### R3 (optional, 15 min CPU). The loanword-neighbour finding, pre-registered
At word level, untranslatables seemed to be filed next to English loanwords, but the loanword list was chosen after the fact. Write the list **before running** (e.g. ~30 English words borrowed from es, it, fr, de, pt: *siesta*, *fiesta*, *bravura*, *finesse*, *angst*, *zeitgeist*, *macho*, …, chosen from a published list of loanwords, with the source named). Rerun the neighbour test on `gemma9b`. Record the list's provenance in the results.

## 3. N1. Round-trip operator spectrum (core experiment)

### 3.1 Definition (put this verbatim in `results/internals/DEFINITIONS.md`)

**Paired representations.** Fix a layer ℓ. Take a parallel corpus of language X (e.g. Spanish) and Y (English), with word alignments. Each aligned word pair i gives a pair of vectors (xᵢ, yᵢ):
- xᵢ ∈ ℝᵈ is the residual-stream state of the X word in its own sentence (mean over the word's subword tokens), with the sentence fed raw (BOS + sentence, no prompt);
- yᵢ is the same for the aligned Y word in the Y sentence;
- both are centred per language.

The covariances are Σxx, Σyy, Σxy, with ridge Σxx(λ) = Σxx + λ·(tr Σxx/d)·I, and the same for Σyy.

**Round trip.** The least-squares maps are A = Σyx Σxx⁻¹ (X→Y) and B = Σxy Σyy⁻¹ (Y→X). The round-trip operator on X is M = BA.

**Identity.** Σxx^{-½} M Σxx^{½} = K Kᵀ, where K = Σxx^{-½} Σxy Σyy^{-½}. Write the SVD as K = U P Vᵀ, where P = diag(ρ₁ ≥ ρ₂ ≥ …). Then:
- **M's eigenvalues are ρₖ², the squared canonical correlations;**
- its eigenvectors, in whitened coordinates, are the canonical directions uₖ.

In words: one linear round trip keeps a share ρₖ² of direction k, and n round trips keep ρₖ²ⁿ. Directions with ρ ≈ 1 are a shared core that survives indefinitely. Everything else keeps decaying ("never stops being translated", Cassin).

**Per-span untranslatability.** For a span with (mean) state x:
- whiten it, z = Σxx^{-½} x, with coefficients c = Uᵀz;
- **retention** Rₙ(x) = Σₖ ρₖ²ⁿ cₖ² / Σₖ cₖ²;
- **U(x) = 1 − R₁(x)**, the share lost in one round trip;
- **half-life** h(x) = the real n at which Rₙ = ½, by interpolation in n, capped at 50.
- **Token-weighted span version:** U_S = Σ_{i∈S} wᵢ U(xᵢ) / Σ wᵢ, with the round-trip information weights wᵢ from R2, so function words count little.

**Raw-metric check.** Whitening gives every direction equal weight, including low-variance noise directions. So also report U_raw(x) = ‖x − Mx‖² / ‖x‖² in centred raw coordinates. Treat agreement between U and U_raw as part of the result.

**English bias (asymmetry).** The spectrum is symmetric (X and Y share the ρₖ), but how much of each language's variance lives in the shared core is not:
- **E_X = tr(Σxy Σyy⁻¹ Σyx) / tr(Σxx) = Σₖ ρₖ² ‖Σxx^{½}uₖ‖² / tr Σxx**, the share of X's representational variance predictable from Y;
- **E_Y** is the same with X and Y swapped;
- **bias = E_Y − E_X**. If positive, English keeps more of itself in the shared space than the other language does.

Report this in raw coordinates and in per-language standardised coordinates (each dimension z-scored), because anisotropy differs between languages.

**The corpus defines the practice.** A and B are learned from existing human translations, so U is untranslatability *relative to conventional translation practice as embodied in that corpus*. **Say this in every write-up.** It is the formal handle on the father's critique.

### 3.2 Estimation
- **Streaming sufficient statistics.** CCA only needs Σxx, Σyy and Σxy. Accumulate them on the GPU in float32 per batch and add them into float64 totals, at each stored layer. Don't store full activations for the training split.
- **Held-out set:** store 20k aligned pairs per layer in fp16 for evaluation.
- **Layers:** every 4th layer, plus the layer with peak sharing (found from a quick pass), plus the detour layers from R1.
- **Ridge λ** is chosen per layer to maximise the sum of the top-50 **held-out** canonical correlations, i.e. corr(aₖᵀx, bₖᵀy) on the test pairs with directions fitted on train. **Report held-out ρ everywhere.** In-sample canonical correlations are biased upwards.
- **Null:** mismatched pairs (each x with the y of a random other pair) give a null spectrum. Only directions whose held-out ρ exceeds the null's 99th percentile count as "shared". Report how many there are.
- **Robustness:**
  - drop BOS and the first token;
  - repeat with the top-5 "massive activation" dimensions removed;
  - repeat with PCA to 512 dimensions per language before CCA.
- **Alignment:** SimAlign (`itermax`, `xlm-roberta-base`) or awesome-align, at word level, keeping 1-to-1 links only. Spot-check 30 alignments by eye per pair and record the precision estimate.

### 3.3 Data

| pair | fit corpus | evaluation | why |
|---|---|---|---|
| es–en | 50k sentence pairs: ½ Europarl (`Helsinki-NLP/europarl`), ½ OPUS-100 (`Helsinki-NLP/opus-100`), lengths 8–40 words | `data/sentences.yaml` (es items) + FLORES devtest | interpretable; ties to all the earlier results |
| hi–en | 50k from OPUS-100 en–hi (plus Samanantar if accessible) | Hindi test sentences (§6) + FLORES | the unequal pair: the father's distortion argument and the Indian context of the bid |
| es–hi (via FLORES, multiway) | FLORES dev+devtest, ~2k sentences, at word level (~25k pairs), PCA-128 | — | **is English the hub?** Compare the es–hi spectrum with es–en and hi–en *on the same sentences* |
| bho (Bhojpuri) | none (too little data) | FLORES `bho_Deva`, scored with the hi–en maps | a low-resource variety scored with a neighbour's maps; prediction: higher U than Hindi on the same sentences |

Mixing two domains matters: parliament proceedings alone would make "conventional practice" mean parliamentary register. Record the domain mix in the results.

### 3.4 Questions to answer, with pre-registered predictions (append to `PREDICTIONS.md` **before running**)
1. **Spectrum shape across depth:** how many shared directions does each layer have? Prediction: sharing peaks in the middle layers and coincides with or neighbours the detour layers.
2. **Does U pick out untranslatables?**
   - AUROC of span U for annotated hot spans vs control content words;
   - the father's sentence order (*duende* > *gusta* > *mucho*);
   - U by category;
   - Spearman correlation with the R2 round-trip U across sentences (internal vs behavioural agreement).
3. **English bias:** E_en − E_es and E_en − E_hi for each model. Predictions: positive, larger for hi than es, and smaller for Aya (and Sarvam on hi) than for Gemma, Llama and Qwen.
4. **English as hub:** number of shared directions and mean ρ² for es–hi vs es–en on FLORES. Prediction: es–hi < es–en, even though Spanish and Hindi are both "far" from each other and from English in different ways.
5. **What lives in the private directions?** Use spaCy (`es_core_news_md`) or stanza for Hindi morphology. Label Spanish words with Gender, Number, Mood=Sub, Tense (imperfect vs preterite), formal address (*usted*), clitic dative/experiencer constructions (*me gusta*, *se me olvidó*), and ser vs estar. For each feature:
   - fit a logistic probe on whitened Spanish states;
   - report the **shared share**: the fraction of the probe's weight energy on shared directions, against the random-direction expectation r/d;
   - report probe accuracy using shared-only vs private-only coordinates.
   - Predictions: grammatical gender is mostly private; number and tense mostly shared; subjunctive and ser/estar partly private; the dative experiencer is private.
   - For Hindi: gender, honorific (*aap*/*tum*), the ergative *ne*, and dative-subject constructions (*mujhe … pasand hai*, structurally parallel to *gustar*: a gift for the father).
6. **Is linear enough?** Fit a 2-layer MLP for X→Y and Y→X (held-out R², early stopping). Report how much of the "lost" variance the MLP recovers. If it recovers a lot, the linear U overstates untranslatability (the meaning was carried, just not in a straight line). Say so plainly.

### 3.5 Cost
For 50k pairs × 2 languages × ~30 tokens per sentence, that's ~3M tokens per pair per model. Forward passes only, about 15–25 min per pair per model on the 4070 in nf4. Covariances at d ≈ 3.5–5k in float64 come to 100–200 MB per matrix per layer. Fine.

## 4. N2. SAE features that don't cross over

### 4.1 Setup
- Use `sae_lens`. Check what's in the pretrained list for the main model.
  - For Gemma 2 9B: Gemma Scope residual-stream SAEs (`gemma-scope-9b-pt-res-canonical`, all layers) and the IT SAEs at layers 9, 20, 31 (`gemma-scope-9b-it-res-canonical`).
  - Prefer IT SAEs where they exist, and say which you used.
- **Layers:** one in the shared peak from N1, one detour layer, one late layer.
- **Quantisation check:** SAEs were trained on bf16 activations. On 200 English sentences, report the SAE's reconstruction fraction of variance explained and its L0 under nf4 vs the nominal values. If they degrade badly, run Gemma in 8-bit for N2 (it should fit at short sequence lengths) or fall back to the 2B Gemma for N2 only. Report which you did.

### 4.2 Definitions
For SAE feature f and word state x: aᶠ(x) ≥ 0 is the activation (summed over the word's subword tokens).

- **Language-identity features:** features whose activation separates es from en text with AUROC > 0.95 over the corpus. Tag them and **exclude them** from every untranslatability score: they mark *which language*, not *what meaning*.
- **Transfer of feature f** over aligned word pairs:
  - τᶠ = Σᵢ min(aᶠ(xᵢ), aᶠ(yᵢ)) / Σᵢ aᶠ(xᵢ), the share of f's source activation matched by f itself on the aligned target word;
  - **partner transfer** τᶠ* = max(τᶠ, best correlation over features g of aᶠ(x) with aᵍ(y) across aligned pairs). Meaning may move to a *different* English feature.
  - Use τᶠ* as the main score.
- **Span score:** U_SAE(x) = Σ_f aᶠ(x)(1 − τᶠ*) / Σ_f aᶠ(x), over non-language-identity features.
- **Instance view** (for the cards): for one source span and its translation, list the features that **persist** (active on both), **vanish** (active on source, absent on translation) and **appear** (absent on source, active on translation).

### 4.3 Link to N1 (an important validation)
For each feature f, take its decoder direction d_f. Express it in Spanish whitened canonical coordinates and compute its **ρ-profile** ρ̄ᶠ = Σₖ ρₖ² cₖ² / Σₖ cₖ². Report the Spearman correlation between ρ̄ᶠ and τᶠ* across features. If the two definitions agree at the feature level, the low-ρ subspace has an interpretable content, namely the low-τ features. List the 30 most frequent features with low ρ̄ and low τ*, with their labels.

**Labels:**
- if Neuronpedia's API is reachable from the desktop, fetch its explanation for each listed feature;
- otherwise, show the top-10 activating corpus contexts.

Never invent labels.

### 4.4 Cards
1. *Me gusta mucho el duende* → its greedy B1 translation ("I really like the elf" or whatever the model writes). Show features on *duende* that persist, vanish and appear, and the same for *gusta*.
2. **Contrast sentence:** *El duende del cuento se escondió debajo de la cama* (goblin sense). Prediction: folklore features fire on both *duende* sentences, while performance/art features fire only on the flamenco one and then vanish in translation.
3. One control (*el tren sale a las ocho*).
4. One Hindi example (§6).

### 4.5 Bridge (optional, if time)
Repeat 4.2–4.3 on `8b` with Llama Scope (trained on the base model, so note the caveat).

## 5. N3. Causal mediation: what the model knows but doesn't translate

### 5.1 The decomposition
From N1, take the top-r shared canonical directions for the source language (r = the number above the null). The canonical variates are uncorrelated and unit-variance, so each centred source state splits as:
- x = x_sh + x_pr, where **x_sh = Σ_{k≤r} (aₖᵀx) Σxx aₖ** (with aₖ = Σxx^{-½}uₖ) and x_pr = x − x_sh;
- check numerically that this oblique projection is idempotent.

Interventions act on the residual stream at layer ℓ (the N1 peak layer, plus one detour layer as a second run), at the source-span token positions only. Use plain forward hooks or `nnsight`; both work with bnb 4-bit.

**Mean-ablation of a component:** set it to its corpus mean, which is 0 after centring (so add back the language mean).

### 5.2 Two effects per span and component V ∈ {shared, private}
- **Translation effect E_trans(V):** the change in teacher-forced log-probability of the model's own greedy B1 translation, per target token, inside the translation prompt. Also record whether greedy decoding changes, and show the new output.
- **Source-language effect E_mono(V):** the change in log-probability of a Spanish-side task that depends on the span's meaning:
  - (a) the original sentence's continuation tokens after the span;
  - (b) the **intended-sense check in Spanish**: "En esta frase, '⟨span⟩' significa: ⟨Spanish gloss⟩", teacher-forced on the gloss.
- Normalise both by the effect of ablating the *whole* state (shared + private).

**Question:** do hot spans have private components that matter for the Spanish tasks (high E_mono) but barely touch the translation (low E_trans)? That is the precise version of "**the model understands it, and its translation drops it**". Plot E_mono(private) against E_trans(private) for hot spans vs control words. Report the per-category means with bootstrap CIs.

### 5.3 Interchange interventions on sense minimal pairs (`data/sense_pairs.yaml`, new)
Same word, two senses, two contexts. The prediction is that the sense difference lives (partly) in the private component. Patch the **private** component of the word from context 2 into the run on context 1, then separately the **shared** component, and measure:
- the log-odds between sense-specific English renderings, teacher-forced (e.g. "goblin/elf" vs "soul/magic/spirit of performance");
- the same log-odds on the Spanish sense question.

**Pairs to include (Claude Code drafts them; the user checks them):**
- *duende*: fairy-tale goblin vs flamenco spirit.
- **Gender-marked sense pairs**, where gender carries the meaning and English must say it in a different word: *el cura / la cura* (priest / cure), *el capital / la capital* (money / capital city), *el orden / la orden* (order as tidiness / order as a command), *el frente / la frente* (front / forehead), *el cometa / la cometa* (comet / kite).
  - This is a sharp test. N1 predicts that gender is private, yet a correct translation needs it.
  - If patching the private component flips the translation (priest ↔ cure), the private subspace is not *dropped*; it is *converted* nonlinearly into lexical choice.
  - If it doesn't flip, the translation channel runs through the shared part.
  - Either result is informative. Report it plainly.
- **ser/estar** (*es aburrido* / *está aburrido*: boring / bored).
- **Formal address** (*tú* / *usted*, rendered as "you" either way): a case where the difference *should* vanish in translation. A positive control for "dropped".
- Hindi: *aap* / *tum*, likewise.

### 5.4 Stretch
Learn the minimal translation channel directly with distributed alignment search (`pyvene`, rotation-based interchange intervention) and compare the learned subspace with the CCA shared subspace (principal angles). Only do this if 5.2–5.3 show a clear effect.

## 6. Test data additions. **Stop point 1: the user (and his father) check these, then continue**
- `data/sentences.yaml`:
  - add `intended_sense` to every hot span;
  - add **12 Hindi sentences**:
    - 6 `untranslatable_in_context` (e.g. *jugaad*, *viraha*, *abhimaan*, *man*/*mann* as heart-mind, *nazar*, *apnapan*);
    - 3 `construction` (*mujhe … pasand hai*, *mujhe … lagta hai*, a *ne*-ergative);
    - 3 `control`.
  - Devanagari, with romanisation and annotated hot spans in a `notes` field. Invented sentences only.
- `data/sense_pairs.yaml`, as in §5.3, ~15 pairs.
- `data/loanwords_preregistered.yaml` (R3), with the source of the list.

Mark the Hindi items clearly as **drafts for the father to check**. Neither Claude Code nor the user is a Hindi speaker.

## 7. Validation and stop points
1. **Before N1 interpretation:**
   - the held-out spectrum is clearly above the null at the chosen layer;
   - alignment precision is ≥ 0.8 on the spot check;
   - U separates hot spans from control words with AUROC > 0.6 on es.
   - If U fails, **stop and report**; don't move on to N3, which builds on the N1 subspace.
2. **Before N2 interpretation:** SAE reconstruction under the chosen quantisation is within 10% of nominal FVE (fraction of variance explained). Language-identity features are excluded. Report how many.
3. **Before N3:** the oblique-projection checks pass, and whole-state ablation has a large effect (a sanity floor).

## 8. Outputs
`results/internals/<model>/`:
- `n1_spectrum_<pair>_L<ℓ>.json`: held-out ρ, null, λ, the count of shared directions, E_X, E_Y;
- `n1_span_scores.jsonl`: U, U_raw, h per span, token and layer;
- `n1_probes.json`;
- `n1_mlp.json`;
- `n2_features.parquet`: per-feature τ, τ*, ρ̄, language-ID flag, label or contexts;
- `n2_cards.json`;
- `n3_ablation.jsonl`;
- `n3_interchange.jsonl`: including all generated texts;
- `replication_R1.json`, `replication_R2/` (same schema as `results/sent/<model>/`).

Also, at the top level:
- `results/internals/DEFINITIONS.md`;
- `results/internals/report.md`: each pre-registered prediction marked confirmed, refuted or unclear, with the statistic;
- **`results/internals/COLAB_RECOMMENDATION.md`** (§10).

Store the canonical bases (Uₖ, aₖ, ρₖ) as `.npz` so they can be reused without recomputation.

## 9. Figures (PNG + SVG; same visual language as `figures_for_dad/` and `results/sent/figures/`)
**Style:**
- Inter for labels, Charter italic for foreign words;
- teal #2f7d6d = easy, coral #c4572e = hard, purple #5b4fc4 = loan/borrowing, grey #d9d6cf = not carried across.

Every title states the finding as a sentence, every figure has a one-line "how to read this", and there are no symbols in the father's figures. **Captions are checked against the statistic they summarise.** Last time, three auto-captions overclaimed. Write the caption text into `captions.json` along with the number it relies on.

**For the father:**
1. **What survives the round trip.** Retention curves Rₙ over n = 0…10 round trips for *duende*, *gusta*, *saudade*, *jugaad*, *tren* and *hermano*, with half-lives marked. Title it by the result, e.g. "*Duende* loses half its meaning in N trips; *tren* never does".
2. **English keeps more of itself.** Paired bars of E_Y and E_X for es–en and hi–en, for each model. Plus one bar group for es–hi via FLORES (the "is English the hub?" question).
3. **Opening up *duende*.** A feature card: what survives into the English, what is lost, what the English adds (labelled features), beside the goblin-sense contrast sentence.
4. **The model knows more than it translates.** The E_mono vs E_trans scatter, with *el cura / la cura* and *tú / usted* called out as worked examples.
5. **Bigger models, same picture?** The father's sentence heat map (masked round trip) for the 8B (previous run) vs `gemma9b` vs `qwen14b`, with intended-sense recovery shown under each.

**For the user (technical, symbols allowed):**
- spectrum (held-out ρₖ against k) by layer, with the null band;
- shared-direction count against depth, overlaid with the R1 detour profile;
- probe shared-share by grammatical feature;
- ρ̄ᶠ vs τᶠ* scatter;
- linear vs MLP held-out R²;
- U (N1) vs U (round trip) across sentences.

## 10. What this run should tell us about Colab (`COLAB_RECOMMENDATION.md`)
Answer these explicitly, with numbers:
1. Did **N1** pass validation, and did U rank untranslatables sensibly? Did the spectrum or bias change systematically with model size (8B → 9B → 12B → 14B)? If the effects grow with scale, the 70B-class models on Colab (Llama-3.3-70B, Qwen2.5-72B; A100 80 GB in 4-bit) are worth it.
2. Did **N2** agree with N1 (ρ̄ vs τ*)? If yes, the Colab case is Gemma 2 27B or Gemma 3 27B with its Gemma Scope SAEs.
3. Did **N3** show the dissociation (high E_mono, low E_trans for hot spans)? If yes, Colab should run it on the biggest model plus distributed alignment search.
4. **What failed or was flat**, so we don't scale it.
5. **Estimated Colab cost** per experiment (GPU-hours on A100/H100), and which cached artifacts (covariances, canonical bases) can be computed once and reused.

## 11. Code organisation (so it moves to Jupyter painlessly)
- New package `untranslatables/internals/` with modules `data.py`, `align.py`, `collect.py` (streaming covariances and held-out pairs), `cca.py` (pure numpy/torch, CPU-runnable), `spans.py`, `probes.py`, `sae.py`, `mediate.py`, `figures.py`, `report.py`.
- Every stage is a function `run(cfg: dict) -> paths`. The CLI wrappers and `run_stage.sh` entries call those functions.
- **No hard-coded WSL paths:** the data, cache and results roots come from env vars, with defaults.
- **Make each stage resumable** (skip if the output exists and the config hash matches), because Colab sessions get cut off.
- Keep GPU stages (collect, SAE activations, mediation) separate from CPU stages (CCA, probes, figures), so that on Colab the expensive part runs once and the analysis can be repeated anywhere.
- Add `untranslatables/internals/NOTEBOOK_SKETCH.md`: the sequence of cells a Colab notebook would need (install, mount Drive, set env vars, call `run()` per stage). **Don't build the notebook yet.**

## 12. Schedule (can span several evenings; print ETAs before each stage)

| step | what | model(s) | rough time | stop? |
|---|---|---|---|---|
| 0 | licences, registry, env (`sae_lens`, `simalign`, `pot` already there, spaCy models, stanza hi), corpora download, test data §6, `PREDICTIONS.md` additions | CPU | 45 min | **yes: the user checks the sentences, sense pairs and predictions** |
| 1 | R1 detour on 4 models; R2 on `gemma9b` | GPU | 60 min | **yes: duende heat map, sense recovery, detour profiles** |
| 2 | N1 collection: es–en and hi–en on `gemma9b`; alignment | GPU | 60–80 min | |
| 3 | N1 analysis (CCA, null, spans, probes, MLP) on `gemma9b` | CPU | 30 min | **yes: spectrum, U validation, English bias** |
| 4 | N1 collection and analysis on `qwen14b`, `nemo12b`, `ayax8b`, `8b`, (`sarvam2b` hi–en); FLORES es–hi and Bhojpuri | GPU+CPU | 3–4 h | |
| 5 | N2 on `gemma9b` | GPU | 60 min | |
| 6 | N3 ablations and interchange on `gemma9b` | GPU | 60 min | **yes: dissociation plot and minimal-pair results** |
| 7 | R2 on `qwen14b` (and `nemo12b`) | GPU | 60–90 min | |
| 8 | figures, report, `COLAB_RECOMMENDATION.md` | CPU | 30 min | |

If time runs short, drop in this order: `sarvam2b`, R3, the MLP check, `nemo12b` in R2, §4.5, §5.4. **Never drop the null spectrum, the held-out evaluation or the validation stop points.**

## 13. Honesty
- Every number is **model-relative and corpus-relative**. "Inside model M, relative to the translation conventions in corpus C…" belongs in every caption, in plain words for the father.
- **Canonical correlations overfit**: only held-out values are reported.
- **"Private" does not mean "meaningless" or "untranslatable" by itself.** It means "not linearly predictable from the aligned English word". N3 and the MLP check are there to test what it actually means. Keep the three readings apart: dropped, converted nonlinearly, or carried elsewhere in the sentence.
- **Word alignment drops what has no 1-to-1 counterpart.** Constructions like *me gusta* are exactly where this bites. Say so, and report the share of hot-span words that got an alignment at all. Unaligned hot words are themselves evidence.
- **The Hindi materials are drafts** until the father or another speaker has checked them.
- If something is null, say so in the first line of its section. Last time's nulls (targeting, steering) were as useful as the positives.
