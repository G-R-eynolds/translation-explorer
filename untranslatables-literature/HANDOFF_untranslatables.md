# HANDOFF (v2): Untranslatables inside multilingual models: shared representations, English bias, and two kinds of uncertainty

v2, 18 Nov 2026. This replaces v1. It has been rewritten after a literature review: see `literature/` next to this file. **Read `literature/NOTES_*.md` before starting**; they explain most of the design choices below. Bracketed numbers like [2402.10588] are arXiv IDs listed in `literature/papers.csv`.

## 0. Purpose and audience

This feeds ideas to the user's father, a linguist working on an AHRC bid about AI translation and **untranslatability**. The relevant ideas from translation studies:
- Cassin: untranslatables are words that keep being retranslated rather than settling on an equivalent.
- Sakai: translation creates the borders between languages that it seems to cross.
- Liu: in translation between unequal languages, the weaker one is "forcibly transformed".

The father is **not mathematical**: every result needs a picture he can read without equations (§9).

The previous experiment (`translation-lattice/`, see its `results.md`) taught two things. First, model ignorance swamps everything: a 3B model can't read Romanesco. Second, measures that are sensitive to wording mistake rewording for a change of meaning. Hence: **single words, well-resourced languages first, ignorance controlled explicitly.**

Two threads, sharing one word list and one set of models:

**Thread 1. What is an untranslatable made of, and is the "universal" space English-shaped?** Inside the model's middle-layer concept space, rebuild a word like *saudade* from the best few English words. The result is a "recipe" (longing, nostalgia, absence…) plus a **leftover** that English can't reach. Run the same rebuild in the other direction, and measure how strongly the space leans toward English. Then measure how that lean distorts the recipes.

**Thread 2. "The model doesn't know" vs "the word genuinely has many readings".** Separate *ignorance* (epistemic) from *multiplicity* (aleatoric) by **manipulating the input** with glosses and sense clarifications, and by comparing models. Validate on a 2×2 set where the right answer is known.

## 1. What the literature changes (short version; detail in the notes)

| Finding | Consequence here |
|---|---|
| Middle layers hold a concept space that leans English [2402.10588], [2502.15603] | Recipes measured in that space may look *too good* in English. The bias has to be measured and corrected for (§6). |
| Smaller models are more English-shaped; larger ones drift to language-specific subspaces [2505.13141] | Compare Llama 3.2-3B with 3.1-8B as a falsifiable size test. |
| Aya-23 is the least English-routed model; Japanese-centric models think partly in Japanese [2502.15603], [2507.20279], [2408.10811] | Add **Aya-23-8B** as a less English-centric comparison. |
| Rare concepts are poorly aligned [2601.22851]; non-English words cost ~1.5× the tokens [2502.15603] | **Control frequency and token count explicitly.** Otherwise "untranslatable" just means "rare". |
| Closest prior work: gap words have lower nearest-neighbour similarity; unaligned spaces beat Procrustes-aligned; one frame only; no frequency control [2605.24310] | Replicate their baseline first; centre but don't align; use multiple frames; control frequency. |
| Senses are linear superpositions recoverable by sparse coding [1601.03764] | Theoretical footing for the recipe. |
| Concepts are better treated as "usage clouds" than points [2610.08303] | Decompose **each context separately**. Recipe variability across contexts becomes a measurement and links to thread 2. |
| Averaging a concept's latent across languages helps translation [2411.08745] | Add a **patching readout**: inject *saudade*'s latent into an English translation prompt and see what comes out. |
| Aleatoric estimates are unreliable when epistemic uncertainty is high [2505.23506]; the additive entropy/MI split is questionable [2209.03302] | Remove ignorance *first*. Report diagnostics separately and never sum them [2603.24967]. |
| Clarifying the input separates ambiguity from the rest [2311.08718]; dictionary hints help rare words [2302.07856] | Two input manipulations: a **knowledge gloss** (targets ignorance) and a **sense clarification** (targets multiplicity). |
| Instruction tuning makes models overconfident in low-resource languages [2601.01362] | Expect *confident flattening* on dialect items; a knowledge check is mandatory. |

## 2. Environment (unchanged)

- **Code location.** Code lives on the laptop in `~/Documents/Fourth Year/Translation Experiments/untranslatables/`. Reuse `../translation-lattice/remote/` (sync, run, pull, status), plus its uv env and HF cache conventions.
- **GPU.** Runs happen on the RTX 4070 (12 GB) under WSL2, reached via `ssh desktop-gpu`.
- **Models:**

| model | role | precision |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | small, already working | bf16 |
| `meta-llama/Llama-3.1-8B-Instruct` | main | nf4 |
| `Qwen/Qwen2.5-7B-Instruct` | different training mix, already downloaded | nf4 |
| `CohereForAI/aya-23-8B` | least English-centric reference | nf4 |

- **Licences.** Last time the Llama-3.1 HF licence wasn't accepted. Check it, and check Aya's licence (non-commercial; research use is fine). **Ask the user to accept any gated licence.** Don't work around it.
- **Order.** Develop everything on the 3B, then run the 8B, Qwen and Aya.
- **Reuse code where it exists.** Use the `llm-latent-language` code from [2402.10588] (github.com/epfl-dlab/llm-latent-language) for the latent-language measurements, adapting it rather than rewriting. Use EleutherAI `concept-erasure` for LEACE.

## 3. Word sets (`data/words.yaml`). Stop and show the user before going further

Every entry needs these fields: `word`, `language`, `category`, `gloss_en` (short, dictionary-style), `senses` (a list of distinct readings, for untranslatables and polysemous words), `source` (a citation or URL the user can check; `TODO` if unknown, **never invented**), and `notes`.

**Languages:** Phase 1 is pt, de, it, fr, es. Phase 2 (flagged lower-confidence) is ru, ja, hi, and Romanesco for thread 2.

| category | purpose | seed examples (the father may swap in entries from Cassin) |
|---|---|---|
| `untranslatable` | the target | pt *saudade*; de *Sehnsucht*, *Heimat*, *Fernweh*, *Weltschmerz*, *Gemütlichkeit*; it *sprezzatura*, *magari*, *menefreghismo*, *abbiocco*; fr *dépaysement*, *flâner*, *l'esprit de l'escalier*, *retrouvailles*; es *sobremesa*, *duende*, *madrugada*, *estrenar* |
| `ordinary` | frequency-matched control | *chuva*, *mesa*, *Fenster*, *pane*, *maison*, *ventana* … |
| `emotion_plain` | controls for "abstract or emotional" | *tristeza*, *Freude*, *gioia*, *colère*, *nostalgia* |
| `polysemous` | known many-readings words with clear equivalents per sense | it *tempo* (time / weather), fr *temps*, de *Schloss* (castle / lock), es *muñeca* (doll / wrist) |
| `loanword` | words English borrowed | *Schadenfreude*, *Angst*, *Zeitgeist*, *ennui*, *déjà vu*, *hygge* |
| `english_untranslatable` / `english_ordinary` | for the reverse direction | *cosy*, *serendipity*, *awkward*, *privacy*, *wistful*, *home*; *rain*, *table*, *sadness*, *bread* |
| `dialect_known_gloss` | thread-2 "unknown, one reading" | Romanesco *schertri* (skeletons), *bbiocca* (broody hen), *purcini* (chicks), with glosses from **Belli's own footnotes** (the user can supply these) |
| `dialect_untranslatable` | thread-2 "unknown, many readings" | Romanesco *e bbona sera* (good night / that's that), *a ppecorone*, *fora a chi tocca* |

**Distinction-collapse items (`data/distinctions.yaml`, for §6.3).** Sentence pairs in two *non-English* languages that share a distinction English lacks:
- fr *savoir/connaître* → es *saber/conocer*;
- pt *ser/estar* → es *ser/estar*;
- it *tempo* (weather) → es *tiempo*.

Each needs matched controls where English *does* make the distinction. Aim for 40 items per pair, written by Claude Code and **checked by the user**.

**Before running anything**, write `PREDICTIONS.md` and commit it. Include the literature's predictions explicitly:
- the 3B is more English-shaped than the 8B;
- Aya is less English-shaped than Llama;
- English recipes leave smaller leftovers than the reverse direction for *ordinary* words;
- dialect items show low spread with wrong content until glossed.

## 4. Step 0: familiarity gate (both threads)

1. For each word, prompt in its own language for a definition, then prompt for an English explanation. Also record the word's token count, its frequency (`wordfreq`), and its mean token log-prob in natural sentences.
2. Build `results/familiarity_check.html`, a tick-box page (understood: yes / partly / no) for the user or his father. Until it's checked, use a provisional keyword-overlap flag against `gloss_en`, and say so in the report.
3. Words that fail are excluded from thread-1 headline figures but kept in an appendix. In thread 2, failures are the *point* (they belong to the "unknown" row).

## 5. Thread 1 method: recipes in the concept space

### 5.1 Representations as usage clouds
- **Contexts per word.** 10 contexts per word, all written in the word's own language:
  - 4 short templates (e.g. pt `A palavra "saudade" quer dizer`);
  - for emotion words, 2 frames following [2307.01370] and [2605.24310]: "I feel X" / "I am X" in that language;
  - 4 natural example sentences, model-generated, saved, and shown in the appendix.
- **Per-context vector.** At layer ℓ, take the residual-stream state averaged over the word's own tokens. Keep **all 10 vectors** (the usage cloud) as well as their mean.
- **Remove language identity.** Subtract the language centroid, computed over that language's vocabulary pool at the same layer [2004.05160], then L2-normalise. **Don't Procrustes-align**: [2605.24310] found unaligned worked better for gap detection.

### 5.2 Vocabulary pools
- ~8,000 content words per language from `wordfreq`, **frequency-matched across languages and with the same pool size**. Embed them with the template contexts only, and cache.
- When decomposing a word, exclude the word itself, its cognates and its borrowed forms from the pool, and log every exclusion.
- **Loanword sub-test:** decompose each loanword with and without its English loan in the pool.

### 5.3 Layer choice
Pick the layer by **translation retrieval**. For ~300 ordinary dictionary pairs (MUSE dictionaries; record the source), measure how often the correct translation is the nearest neighbour at each layer. Use the best layer and report its two neighbours. Plot retrieval by layer, since it doubles as a "where the shared space lives" figure.

### 5.4 Recipe (non-negative greedy decomposition)
- **Fitting.** Use non-negative orthogonal matching pursuit (NNLS refit after each addition), up to k = 10. Record the explained share R²(k), the chosen words and their weights.
- **Recipe card** = the words chosen by k = 5, with shares. **Leftover** = 1 − R²(5).
- **Usage-cloud version.** Run the same decomposition on each of the 10 context vectors separately. Report:
  - the **stable core**: ingredients that appear in ≥7 of 10 contexts;
  - **recipe variability**: the mean pairwise Jaccard distance between the per-context ingredient sets.

  High variability means the word reads differently in different contexts. That is a geometric hint of multiplicity, and it is checked against thread 2.
- **Chance baselines.** Run the same procedure on random unit vectors and on random same-language words. Report leftovers relative to the null distribution, using permutation calibration [2602.14486].

### 5.5 Baselines to beat
- **B-NN:** CSLS nearest-neighbour similarity to the English pool, replicating [2605.24310]. Can the recipe leftover separate `untranslatable` from `ordinary` / `emotion_plain` better than B-NN (AUROC, frequency-matched)?
- **B-freq / B-tok:** frequency alone and token count alone, as predictors of "untranslatable". **If either matches the leftover, say so plainly.**

### 5.6 Patching readout: what the model "would say"
Following [2411.08745]:
1. Build few-shot word-translation prompts into English (5 examples, pt→en and so on).
2. Take *saudade*'s middle-layer latent (at the last token of the word, from the source-language prompt) and patch it into the target prompt at layers j…L.
3. Sample 50 English outputs and record their distribution.

Compare that distribution with the recipe. Is the top recipe ingredient what the model outputs? Then repeat with the latent averaged over several source-language contexts, where those exist (cognates, or the word itself used in other languages' text). This also tests the "rare concepts align poorly" warning from [2601.22851].

## 6. Thread 1: measuring the English bias, and how it bends the results

### 6.1 Latent-language share (does it "think" in English?)
- Adapt the [2402.10588] code. Use word-translation prompts between **non-English** pairs (fr→es, de→it, pt→fr) with single-token targets where possible.
- For each layer, compute P(English version of the answer) vs P(target-language answer) under the logit lens. Add the tuned lens [2303.08112] if pretrained lenses exist for the model.
- **Output:** one curve per model of "share of the model's working answer that is English, by layer".

### 6.2 Coverage asymmetry (is the space built around English?)
- At the concept layer, take the top-k principal directions of the English pool, and measure the share of Portuguese pool variance they capture. Then do the reverse. Use equal-size, frequency-matched pools and several values of k.
- Report the asymmetry with bootstrap CIs, per language pair and per model.
- **Caveat** [1805.03620]: spaces aren't globally isomorphic, so also compute the asymmetry *within frequency bands*.

### 6.3 Distinction collapse (does meaning get lost passing through English?)
- Translate the `distinctions.yaml` items X→Y between non-English languages, at greedy decoding and with samples.
- Measure the **error rate on the shared distinction** (e.g. *connaître* → *saber* is an error). Compare it with matched controls where English keeps the distinction.
- **The excess error rate is the behavioural English-bias score.**
- Use the logit lens to check whether the English collapsed word ("know") appears in the middle layers on error items more than on correct items.
- This is the concept-spilling phenomenon of [2601.12549], made controlled.

### 6.4 Causal test: steering toward or away from English
- Steering vector = mean last-token activation on parallel English inputs minus the mean on non-English ones, at middle layers, as in [2505.13141]. Apply it with a multiplier γ ∈ {−3…+3}.
- Measure how γ changes: (a) the distinction-collapse error rate; (b) recipe leftovers for untranslatables. If pushing *away* from English raises leftovers, the English lean was *hiding* untranslatability.
- **Watch for off-distribution damage**: check output fluency, and report any γ where the outputs degrade.

### 6.5 Correcting the recipes for the bias
- **Within-direction normalisation.** Untranslatability score = leftover(word) − median leftover of `ordinary` words in the same direction and the same frequency band.
- **Pivot comparison.** Also decompose each untranslatable into es, it and de pools. Report leftover by pivot language. **A word can be untranslatable into English but not into Spanish.** That relational picture is Cassin's and Sakai's point.
- **Layer dependence.** Plot each untranslatable's leftover against the layer-wise English share from §6.1. If leftovers shrink exactly where English share peaks, the model is assimilating the word to English.
- **Model comparison.** Compare all of the above across 3B, 8B, Qwen and Aya.

## 7. Thread 2 method: ignorance vs multiplicity

### 7.1 Translation spread
- **Sampling.** For each word, use 5 natural source-language contexts (for dialect items, take them from Belli's sonnets). Sample 20 English translations of the word per context at T = 1.0.
- **Measuring spread.** Cluster the samples by meaning using bidirectional NLI entailment (`microsoft/deberta-large-mnli`; it was the best meaning judge last run, at 0.98 AUROC). Then compute two quantities:
  - **H**, the semantic entropy, in bits [2302.09664];
  - **KLE**, the kernel version [2405.20003].
- **Translation-studies name.** In the report, call H the *translation entropy* of the word, to connect with translation-process research [2603.12050].

### 7.2 Input manipulations (the core design; one factor at a time [2603.24967])

| condition | prompt adds | should reduce |
|---|---|---|
| `bare` | nothing | — |
| `gloss` | a short **knowledge gloss**: definition plus one example, from a dictionary or Belli's footnotes; **never** a translation | ignorance |
| `sense_k` (one per listed sense) | a **sense clarification**: "here it means …" | multiplicity |
| `context+` | 2–3 extra sentences of surrounding text | ambiguity that context can resolve |

Diagnostics per word, kept separate and never summed:
- **Ignorance** = H(bare) − H(gloss), bootstrap CI.
- **Multiplicity** = H(gloss), the spread left once the model knows the word. Check it with I(output; sense) over the `sense_k` conditions, following the clarification-ensembling formula of [2311.08718], adapted.
- **Context-resolvability** = H(gloss) − H(gloss + context+).

**Never put the expected English translation in a gloss.** That would just measure copying.

### 7.3 Converging signals of ignorance (they should agree; report agreement)
1. **Gloss sensitivity** (§7.2).
2. **Cross-model disagreement.** Total variation distance between the reading distributions of Llama-8B, Qwen and Aya [2604.17112]. Note the caveat about uneven competence.
3. **Small vs large.** Compare 3B and 8B: is the 3B unsure where the 8B is confident [2402.03563]?
4. **Iterative-prompting instability.** Show the model its previous answer and ask again. Does it flip [2406.02543]?

### 7.4 A human reference for multiplicity
- For the untranslatables, collect English renderings from several bilingual dictionaries and from published translations. **The user and his father can supply these; don't scrape copyrighted texts.**
- For Belli lines, use the published translations the user owns.
- Report the model's multiplicity next to the human spread (the "human production variability" idea from [2305.11707]).

### 7.5 Validation on the 2×2 set (§3 categories)
Each cell should land in its own corner:
- `ordinary`: low ignorance, low multiplicity;
- `untranslatable` / `polysemous`: low ignorance, high multiplicity;
- `dialect_known_gloss`: high ignorance, low multiplicity;
- `dialect_untranslatable`: high ignorance, *and* high multiplicity once glossed.

**Report a confusion table.** If the cells don't separate, the method isn't ready, and that is itself a finding.

### 7.6 Stretch: internal signatures
Train linear probes on middle-layer states (one per layer) to predict two labels from the 2×2 set: "unfamiliar" and "many readings". Linear probes decode ambiguity well [2506.01205]. Do the two probes find *different* directions (low cosine between probe weights)? If they do, check how each relates to the recipe leftover and recipe variability from §5.4. That is the bridge between the two threads.

## 8. Controls and honesty (put these in the report)
- **Phrasing.** Say "inside this model…", never "the meaning of the word is…".
- **Confounds.** Frequency, token count and familiarity are possible confounds for every thread-1 result. Show the results with these matched or regressed out, or say plainly that they couldn't be.
- **Additivity.** Diagnostics are reported separately and are **not** an additive partition of uncertainty [2209.03302], [2603.24967].
- **Human checks.** All model-generated lists (contexts, distinction items, glosses) are saved and listed for the user to check. Don't invent sources.
- **Negative results.** If the English-bias effects or the 2×2 separation fail, **report that plainly**. Both outcomes interest the father.

## 9. Figures: the deliverable for the father

Shared rules:
- every title states its finding as a sentence;
- every figure has a plain "how to read this" caption with no symbols;
- grey always means "leftover / what English can't reach";
- one accent colour per category, held consistent everywhere;
- big type, PNG and SVG.

**Thread 1**
1. **Recipe cards.** One per word: the word, its gloss and an example sentence, then one bar of English ingredients plus a grey leftover. Mark the **stable core** solid and context-dependent ingredients hatched. Lead with *saudade* next to *tristeza*.
2. **"How many English words does it take?"** curves, with a chance band.
3. **Neighbourhood sketch.** A local PCA showing the word *between* its ingredients. Caption it honestly as a flattened sketch.
4. **Leftover ranking.** Sorted bars by category, with a chance band and a hatch for words that failed the familiarity check.
5. **"What the model would say".** The patching readout (§5.6) as a small bar chart of English outputs, next to the recipe.
6. **"Into English vs into Spanish".** Leftover per pivot language for each untranslatable, as a dot strip. Shows untranslatability as relational.

**English bias**
7. **"Does it think in English?"** English share by layer, one line per model, with Aya highlighted.
8. **"Which way does the space lean?"** A coverage-asymmetry dumbbell per language and model.
9. **"Lost through English".** Distinction-collapse error rates, as bars, against controls. Include 2–3 example sentences in plain text.
10. **"Push it away from English".** The effect of steering on distinction errors and on leftovers.

**Thread 2**
11. **The uncertainty map.** The headline: x = ignorance (how much the gloss helps), y = multiplicity (the spread left after the gloss). One labelled dot per word, coloured by 2×2 category, with quadrant labels in plain words ("model doesn't know it", "genuinely many readings", "both", "neither").
12. **Before and after the gloss.** For 4–6 example words, the meaning clusters of the model's translations as stacked bars in the bare, gloss and sense conditions. Use *bbona sera* if it behaves.
13. **Model vs humans.** Multiplicity against the human rendering spread.

Everything goes into **`results/report.html`**:
- a 5-sentence plain-English summary;
- the figures in the order above, each with its finding and a "how to read this" line;
- a "what this does and doesn't show" section;
- a technical appendix for the user (methods, exclusions, prompts, familiarity table, all model-generated lists), with **citations to the arXiv IDs in `literature/papers.csv`** wherever a method is borrowed.

## 10. Order of work and stop points
1. `words.yaml`, `distinctions.yaml`, contexts, glosses, and `PREDICTIONS.md`. **Stop: show the user.**
2. Familiarity gate (§4) on the 3B. **Stop: the user or his father ticks `familiarity_check.html`.**
3. **Thread 1 core on the 3B:** pools, layer choice, recipes, baselines, patching readout. Then figures 1–5. **Stop: show the *saudade*/*tristeza* card and the baseline comparison.**
4. **English bias on the 3B and 8B:** §6.1–6.3, then figures 7–9.
5. **Thread 2 on the 8B:** §7.1–7.5, then figures 11–13. **Stop: show the uncertainty map.**
6. Add Qwen and Aya across everything. Then §6.4–6.5, figures 6 and 10, and the §7.6 stretch.
7. `report.html` and `README.md`.

Suggested modules: `words.py`, `represent.py`, `pools.py`, `layers.py`, `recipe.py`, `baselines.py`, `patching.py`, `latent_lang.py`, `coverage.py`, `distinctions.py`, `steering.py`, `spread.py`, `manipulations.py`, `ignorance.py`, `probes.py`, `figures.py`, `report.py`.

**Ask the user about any ambiguity in §3, §5.6, §6.3 or §7.2** rather than improvising around a gap. Debug at small scale, and print a time estimate before any run longer than about 15 minutes.
