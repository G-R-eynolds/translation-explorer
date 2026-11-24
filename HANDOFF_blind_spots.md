# HANDOFF: Translation blind spots. Untranslatability as a Fisher-information spectrum

23 Nov 2026. For Claude Code on the user's laptop, in `Translation Experiments/`. Runs on the **desktop RTX 4070 (12 GB)** through the existing `untranslatables/remote/` workflow (`ssh desktop-gpu`, WSL). This **replaces the word-level round-trip spectrum** (`HANDOFF_internals.md` N1, Colab results in `results_colab/`) with a definition that works for any span of text. Put the new code in `untranslatables/blind/` and the outputs in `untranslatables/results/blind/<model>/`.

## 0. Why

The user wants a more elegant, more general definition than the word-level canonical-correlation one, which needed word alignment and a parallel corpus and only scored single words. His father is a linguist writing an AHRC bid about translation (Cassin's *Dictionary of Untranslatables*, critique of print-era one-to-one equivalence, the distortion of low-resource languages through English). His father's questions:
- What does translation into English lose?
- Is English special?
- "What counts as translation": could other practices (glossing, commentary, thick translation) carry more?

The definition below answers all three with one object. The father is not mathematical, so the figures (§8) must explain themselves. The formal parts are for the user (maths/CS).

**Lessons from earlier runs** that apply here (see `results_colab/REPORT.md` and `untranslatables/results/sent/report.md`):
- Short sentences are ambiguous. *Me gusta mucho el duende* was read by the 70B as "goblin" before any translation happened, so add context where a sense matters.
- Auto-generated captions overclaimed three times. Every caption must cite the number it rests on.
- Single-layer, single-word interventions barely moved the translation. Here directions are found over **all source positions at once**.

## 1. The definition

**Setup.**
- s is a source text (any length) in language A (Spanish unless stated).
- ℓ is a layer.
- h ∈ ℝ^{|P|·d} is the residual stream at layer ℓ over the source positions P, flattened.
- A **channel** c is a prompt template that asks the model to express s in some way, and so defines an output distribution p_c(y | h) over output texts y.

**One template, only the language varies.** All channels use the same instruction, **placed after the text**:

> ⟨chat user turn⟩ {s}\n\nExpress the text above in {L}, in your own words, keeping exactly its meaning. Reply with the text only.

- With L = Spanish (the source language), the channel is a **paraphrase**: the model's own re-expression in the source language. This is the reference channel, src.
- With L = English, Italian, French or Portuguese, it is a **translation** into L.

The model is causal and the instruction comes after s, so **the source-position states h are bit-identical across all channels.** Every channel is a different function of the same h. **Assert this in code**: max |Δh| = 0 across channel prompts at every stored layer. If the chat template puts anything channel-specific before s, restructure until the assert passes.

**Fisher information of a channel.** Nudging h by δ changes the output distribution by

  KL(p_c(·|h+δ) ‖ p_c(·|h)) ≈ ½ δᵀ G_c δ,  where G_c = E_{y∼p_c(·|h)}[ ∇_h log p_c(y|h) ∇_h log p_c(y|h)ᵀ ].

G_c says which directions in the model's representation of s make a difference to that channel's output.

**Blind spots.** Solve the generalised eigenproblem

  **G_src v = λ (G_L + εI) v.**

- An eigenvector with large λ is a change in the model's representation of s that changes how the model would *say s in Spanish* but barely changes how it would *say s in L*. It is a distinction the source language draws that translation into L does not register: **a blind spot of translating into L.**
- Directions with λ ≪ 1 are the reverse: what the translation is sensitive to but the paraphrase is not. Report these too; they may be surface or noise.

**Untranslatability of s into L (in bits):**

  **U_L(s) = ½ Σᵢ log₂ max(1, λᵢ / λᵢ⁽⁰⁾)**,

where λ⁽⁰⁾ is the null spectrum (§2.3), i.e. what the same comparison gives for two independent sample sets of the *same* channel. Also report λ₁ / λ₁⁽⁰⁾ and the number of eigenvalues above the null's 95th percentile (the **number of blind directions**).

**Span version.** Restrict δ to the positions of a span S (slice the gradients to S before forming G). This gives U_L(s; S) for any word, phrase or clause, and a per-token heat map when S is one token. No alignment and no word lists are needed.

**Practice version.** Replace the L-channel's instruction with another translational practice into L (§5, E3). Then U_{L,practice}(s) asks which practices close which blind spots. This is the father's "what counts as translation", made measurable.

**Why the paraphrase is the reference and not a copy:** a verbatim copy is sensitive to surface form (spelling, word choice) that a translation legitimately ignores, which would make everything look untranslatable. "In your own words" asks for meaning. As a diagnostic, also compute a **copy channel** ("Repeat the text above exactly"). Directions that are blind for L *and* strong in copy but weak in the paraphrase are surface, not meaning; report what share of U they account for.

## 2. Estimation

### 2.1 Gradients
- Load the model with all parameters `requires_grad_(False)`.
- At layer ℓ, a forward hook takes the residual output, makes the slice at positions P a leaf tensor with `requires_grad=True`, and splices it back in. Store several layers in one pass: the Gemma Scope 2 "detailed" depths (25%, 50%, 65%, 85%). Read the exact layer indices from the Gemma Scope 2 repo for the chosen model.
- For each channel, sample K outputs y ∼ p_c(·|h) at **temperature 1, no top-p or top-k** (the Fisher needs the model's true distribution). Use max_new_tokens = 2·|s tokens| + 16, and stop at end-of-turn. A capped output is still a well-defined distribution, so this is fine.
- Then teacher-force each y, take log p_c(y|h) summed over output tokens, backpropagate, and read ∇_h at P for each stored layer. Batch the samples (8–16 per batch); that uses little memory at these lengths.
- K = 64 per channel by default. Store the gradients (K × |P|·d per layer and channel; small) in `.npz`, so all the analysis is CPU-only and repeatable.

### 2.2 Eigenproblem in the low-rank span
- With finite samples, each Ĝ has rank ≤ K. Let Q be an orthonormal basis for the span of all gradients involved (src and L, ≤ 2K dimensions). Form A = QᵀĜ_src Q and B = QᵀĜ_L Q + ε·(tr Ĝ_L / rank)·I, and solve with `scipy.linalg.eigh(A, B)`.
- Sweep ε relative ∈ {1e-3, 1e-2, 1e-1} and report stability. The headline uses 1e-2 unless validation says otherwise.

### 2.3 Null and uncertainty (essential: finite-sample Fisher estimates are rank-deficient and can manufacture "blind" directions)
- **Null spectrum λ⁽⁰⁾:** split the src samples into halves a and b and solve Ĝ_src^a v = λ (Ĝ_src^b + εI) v. Use matched sizes, so every real comparison is also src half a against L half a (K/2 vs K/2).
- Repeat over 20 random splits. Report the median and 90% interval of every statistic.
- **Spanish-into-Spanish check:** a second paraphrase wording (e.g. "Rewrite the text above in Spanish with different words…") must give a U near the null. If not, prompt sensitivity dominates: stop and report.

### 2.4 Check the linearisation (finite differences)
For the top two blind directions and the top two "seen" directions of a few items:
- Move h → h + αv, with α set so that the predicted KL_src = ½α² vᵀG_src v equals 0.5 nats.
- Measure the actual KL for both channels by Monte Carlo: E_{y∼p(·|h)}[log p(y|h) − log p(y|h+αv)] over 64 fresh samples.
- Report predicted vs measured KL (scatter, Spearman). The measured λ ratio should track the eigenvalue.

### 2.5 What the blind directions mean
For each top blind direction v:
- **Steering:** greedy-decode both channels at h + αv and h − αv (α as in 2.4, plus 2α). Expected result: the Spanish paraphrase changes in a specific way while the translation into L stays the same. **Show the texts**: these steering cards are the most persuasive output of the whole project.
- **SAE features:** encode h ± αv with the Gemma Scope 2 residual SAE at that layer (IT version, width 16k or 262k, medium L0; check the `sae_lens` pretrained list for exact release and ID names). List the features whose activation changes most, per position. Fetch Neuronpedia explanations if the API is reachable; otherwise use top-activating contexts from a small Spanish corpus. **Never invent labels.**

## 3. Models (desktop 4070, 12 GB)

| key | model | precision | role |
|---|---|---|---|
| `g3_4b` | `google/gemma-3-4b-it` | bf16 (~8.6 GB) | **main**. Gemma Scope 2 has SAEs at every layer. Loads as a multimodal model: use the text path (`Gemma3ForConditionalGeneration` → language model, or `AutoModelForCausalLM` if it gives the text model) and check that generation works |
| `g3_12b` | `google/gemma-3-12b-it` | nf4 | scale check on E1/E2 (gradients through bnb 4-bit layers work for activations) |
| `3b` | `meta-llama/Llama-3.2-3B-Instruct` | bf16 | bridge to all earlier results |
| `ayax8b` | `CohereForAI/aya-expanse-8b` | nf4 | less English-centric comparison |

Gemma 3 is gated on Hugging Face: check the licence has been accepted. If `g3_4b` with gradients doesn't fit at some layer, lower the batch size before anything else. Reuse `ut/common.py` (registry, loading, ETAs) and `remote/`.

## 4. Test material (`blind/data/*.yaml`). **Stop point 1: the user checks it**

### 4.1 Built-in blind spots (E1). The distinction is certain, and the target language decides whether it survives
Write about 6 natural variants of each type. Predictions go in `PREDICTIONS.md` **before running**.

| type | example | dropped into | kept into |
|---|---|---|---|
| speaker/subject gender on adjectives | *Estoy cansada.* / *¿Estás lista?* | en | it, pt, fr (*fatiguée*, *prête*) |
| gender on a profession noun | *Soy médica.* | en, fr (*médecin*) | it (*dottoressa*), pt (*médica*) |
| plural vs singular "you" | *¿Queréis café?* vs *¿Quieres café?* | en | it, fr, pt (Brazilian *vocês*) |
| formal vs informal "you" | *¿Usted quiere café?* | en | it (*Lei*), fr (*vous*, conflated with plural), pt |
| ser vs estar (negative control) | *Juan está aburrido.* | **none** (bored/boring) | all |
| lexical but preserved (negative control) | *El tren sale a las ocho.* | none | all |
| aspect (exploratory) | *Cuando era niño, jugaba al fútbol.* | en? ("used to" can carry it) | it, fr, pt |

Plus the Spanish-into-Spanish check (§2.3) on every item.

### 4.2 Sentences (E2)
- `untranslatables/data/sentences.yaml` (Spanish items, plus pt/it/de with the paraphrase in their own language).
- The father's sentence, **plus context-disambiguated versions**:
  - *Cuando canta Camarón, tiene duende; me gusta mucho el duende.* (flamenco)
  - *En el cuento, el duende se escondió debajo de la cama.* (goblin)
- **Two short passages** (3–4 sentences each, invented), one ordinary and one dense with culture-bound terms, to show the definition working on paragraphs.

### 4.3 Atlas corpus (E4)
200 varied Spanish sentences of 8–25 words, from FLORES devtest or OPUS (record the source). Exclude near-duplicates.

## 5. Experiments

**E0. Plumbing (g3_4b):**
- the identical-h assert;
- gradient check against finite differences on one item;
- the null spectrum on controls;
- the Spanish-into-Spanish check.

**E1. Validation on built-in blind spots** (g3_4b, then 3b, ayax8b, g3_12b). Targets en, it, fr, pt. Pre-registered tests:
1. For gender, plural-you and formal-you items, U_en > U_it (and > U_pt) on the same items. Use a paired Wilcoxon test across items, and give the median difference with a bootstrap CI.
2. For *Soy médica*-type items, U_fr ≈ U_en > U_it. French also drops the gender.
3. The negative controls (ser/estar, plain sentences) give U near the null into every language.
4. **The steering card for the top blind direction actually flips the distinction in the Spanish paraphrase** (*cansada* ↔ *cansado*, *queréis* ↔ *quieres*, *usted* ↔ *tú*) while the English stays the same, in at least half the items of each type. Count by reading the texts, and list them all in the output.

**Stop point 2:** show the E1 table, three steering cards and the null comparison to the user. If tests 1–4 mostly fail, stop: the definition isn't measuring what we want.

**E2. Our sentences** (g3_4b; plus 3b for comparison):
- U into en, it, fr, pt for every sentence, by category;
- per-token heat maps into English;
- the father's sentence and its context versions: is *duende* hot only in the flamenco context?
- steering cards for the top blind direction of the father's sentences;
- for the passages, U per sentence within the passage and for the whole passage. **Is a passage's U less than the sum of its sentences' U?** Context may disambiguate.

**E3. Practices** (g3_4b; father's sentences, 6 untranslatables, 4 controls, 2 built-in items per type). L = English with these templates:
- plain (as above);
- keep untranslatable words in the original;
- add a short gloss in brackets after hard words;
- add a footnote;
- thick translation (one paragraph that opens up the meaning).

Longer outputs have larger Fisher simply because they are longer, so report each result two ways:
- **raw**;
- **length-normalised**: divide each channel's Ĝ by its mean output length in tokens.

The question is which practices shrink which blind directions. Re-project each practice's Fisher onto the plain translation's top blind directions and report the fraction of each direction that becomes "seen".

**E4. Atlas** (g3_4b, into English; only if E1 passed):
1. For each of the 200 sentences, take the top 3 blind directions above the null.
2. Characterise each by:
   - its position profile (which tokens carry it);
   - its SAE feature-change signature (§2.5);
   - one steering pair (paraphrase at h ± αv).
3. Cluster the directions by SAE signature (e.g. Jaccard over the top-20 changed features, then hierarchical clustering) and read each cluster's steering pairs.
4. Write `atlas.md`: one section per cluster with the recurring distinction in plain words, 3 example sentences with their steering pairs, how many sentences contain it, and which target languages keep it (rerun those sentences into it, fr and pt). Label clusters only from what the steering texts show.

**E5. Models** (E1 and E2 on 3b, ayax8b, g3_12b):
- Grammatical blind spots (gender, plural-you, formality) should be properties of English and **similar across models**.
- Lexical and cultural ones may differ.
- Report the split. "Structural untranslatability" (shared across models) vs "model-induced" (varies) is a distinction the father will care about.

## 6. Outputs (`results/blind/<model>/`)
- `grads/` (`.npz` per item, layer and channel);
- `spectra.jsonl` (λ, null, U, number of blind directions, per item × target × layer × ε);
- `heatmaps.json`;
- `fd_check.json`;
- `steering.jsonl` (all texts);
- `sae.jsonl`;
- `practices.json`;
- `atlas.md` plus `atlas.json`;
- `validation.json` (E0, and E1 tests 1–4 with statistics);
- top level: `results/blind/report.md`, every claim with its number, each section opening with positive / null / mixed.

## 7. Schedule (several evenings; print ETAs; resumable stages; skip what exists)

| step | what | rough time on the 4070 | stop? |
|---|---|---|---|
| 0 | env (`sae_lens`, scipy), licences, test material, `PREDICTIONS.md` | 30 min CPU | **yes: material and predictions** |
| 1 | E0 + E1 on g3_4b (~42 items × 6 channels × 64 samples) | 40–60 min | **yes: validation** |
| 2 | E2 on g3_4b | 40 min | |
| 3 | E3 practices | 30 min | |
| 4 | E1 + E2 on 3b, ayax8b, g3_12b | 2–3 h | |
| 5 | E4 atlas (200 sentences, English only) | 2–3 h | **yes: before writing it up** |
| 6 | figures and report | 30 min CPU | |

If time is short, drop in this order: g3_12b, the aspect items, E3's keep-word practice, then E4 down to 100 sentences. **Never drop the null, the Spanish-into-Spanish check or the finite-difference check.**

## 8. Figures (PNG + SVG; same visual language as `figures_for_dad/`: Inter labels, Charter italic for foreign words, teal = carried, coral = lost)
Each title states the finding, each figure has one line saying how to read it, and there are no symbols in the father's figures.

1. **What English can't see.** For each built-in type, U into English, Italian, French and Portuguese (dots per item, median bars, the null band). Expected headline, if it holds: "English loses the speaker's gender; Italian keeps it."
2. **Steering cards.** For 4–6 items, the original, then the Spanish paraphrase moved along the blind direction, then the English translation moved along the same direction: "the Spanish changes, the English doesn't." Text only, beautifully set. This is the one figure that needs no explanation.
3. **The father's sentence, with and without context.** Heat maps into English, flamenco vs goblin context, plus the steering card for *duende*.
4. **What different practices recover.** For a few sentences, the share of the plain translation's blind directions that each practice makes visible.
5. **The atlas.** A one-page table of the recurring blind spots of Spanish → English, with an example each and the share of sentences containing it.
6. (For the user) the spectra against the null, the finite-difference check, layer dependence and ε stability.

## 9. Honesty
- **It's local:** the definition describes small changes around this text's representation, not every possible misreading. Say so.
- **It's model-relative and prompt-relative.** The Spanish-into-Spanish and wording checks bound the prompt effect. Report them next to every headline.
- **Low-rank Fisher estimates can invent blind spots.** Only directions above the null count.
- **"Blind" means the model's translation output is insensitive** to that change of the model's internal representation. It's a statement about this model's translating, not about what any translation could do. E3 is the evidence on the latter.
- If the built-in blind spots aren't found (E1), say so in the first line of the report. Don't go on to the atlas.
