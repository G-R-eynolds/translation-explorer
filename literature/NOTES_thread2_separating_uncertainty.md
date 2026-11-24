# Thread 2: Separating "the model doesn't know" from "the word genuinely has many readings"

Synthesis of the corpus in `papers.csv` (thread = 2). arXiv IDs are in brackets. These are paraphrased notes for designing experiments.

## 1. The two kinds of uncertainty, and why the standard split is shaky

- **Definitions.** *Epistemic* uncertainty comes from missing knowledge and can in principle be reduced. *Aleatoric* uncertainty is irreducible spread in the true answer distribution [1910.09457]. In translation, intrinsic uncertainty (many valid translations of one source) was separated from noise-driven uncertainty early on. Models were found to *over*-spread probability mass [1803.00047]. Inputs with many valid references give flatter output distributions and more search errors [2204.00471].
- **Mapped onto our problem:**
  - epistemic = the model can't read Romanesco, or barely knows *saudade*;
  - aleatoric = *bbona sera* really does mean both "good night" and "that's that".
- **The textbook decomposition** (total entropy = expected conditional entropy [aleatoric] + mutual information [epistemic]) **has known problems**:
  - mutual information behaves more like a divergence than a measure of ignorance, and the additive split itself is questionable [2209.03302];
  - "reducible" depends on *how* you would reduce it. More data and a better mechanism are different routes, so epistemic splits into sample-reducible and mechanism-reducible parts [2606.12646];
  - the dichotomy may simply be too coarse [2412.20892].
- **Most important for us:** aleatoric estimates are only trustworthy when epistemic uncertainty is low; otherwise methods confuse the two [2505.23506]. *That is exactly what our Belli run showed.* With weak Romanesco, the measured "multiplicity" was mostly ignorance. **So: remove the ignorance first, then measure multiplicity.**

## 2. Practical toolkit

**Meaning-level spread.** Entropy over meaning clusters, built by bidirectional entailment [2302.09664]. Or the soft version, Kernel Language Entropy: the von Neumann entropy of a similarity kernel [2405.20003], with a spectral aleatoric/epistemic variant in [2509.22272]. NLI-based clustering scored 0.98 AUROC as a meaning judge in our own run.

**Vary one thing at a time.** "The Anatomy of Uncertainty" [2603.24967] estimates each source by changing one factor and holding the rest fixed:
- paraphrased prompts → input ambiguity;
- a LoRA-seed ensemble → knowledge gaps;
- decoding strategies → randomness.

They state outright that the components are diagnostics, not an additive partition. **Adopt this stance.**

**Clarify the input.** Input clarification ensembling [2311.08718] ensembles predictions over *clarified* inputs. The mutual information between the output and the clarification estimates ambiguity (aleatoric); the entropy left after clarifying is the remainder. **This is the template for our gloss test**, with one twist. We use two kinds of clarification:
- a **knowledge gloss** (what the word means) should remove *epistemic* uncertainty;
- a **sense clarification** (which reading is intended) should remove *aleatoric* uncertainty.

Their limitation applies: the clarifier bounds the method, and a clarifier that misses senses understates ambiguity.

Dictionary hints in the prompt measurably help rare words in low-resource MT [2302.07856]. Dictionary plus a few thousand parallel sentences take a model from 0 to 16 BLEU on an unseen language [2402.19167]. That is direct evidence that glosses reduce epistemic uncertainty. But grammar-book gains on MTOB [2309.16575] seem to come mostly from parallel examples, not explanations [2409.19151]. So **glosses should be short and concrete** (a definition plus an example), not prose grammar.

**Use other models as a reference.**
- A larger model stands in for ground truth: where the small model is unsure but the large one is confident, the uncertainty is epistemic. Linear probes on the small model's activations can predict this (~0.94 AUC in-distribution) [2402.03563].
- Cross-model disagreement as an epistemic signal flags confident failures. The caveat is that models of uneven competence or different style inflate disagreement [2604.17112].
- In our Llama-3B vs Qwen-7B run, branching lines were branched *differently* by the two models, which is the signature of ignorance.

**Iterative prompting** [2406.02543]: show the model its own previous answers. If it changes its mind easily, epistemic uncertainty is high; genuine multiplicity is stable under this. **Cheap, and worth adding as a fourth signal.**

**Use humans as the reference for aleatoric spread.** Measure model spread against variability across *human* productions [2305.11707], and against multiple references [1803.00047], [2204.00471]. For us that means multiple published translations of the same word or passage (Belli has many English translators) and multiple bilingual dictionaries for each untranslatable.

**Look inside.** Prompting fails to detect ambiguity reliably, but linear probes on hidden states decode it with high accuracy [2506.01205]. Intermediate layers also give better multilingual calibration than the final layer [2510.03136]. So it is plausible that *ambiguity* and *unfamiliarity* have separable internal signatures. That is a thread-1/thread-2 bridge.

## 3. What the low-resource literature predicts
- Calibration is systematically worse outside English and especially in low-resource languages [2510.03136], [2210.12265]. Instruction tuning makes models *overconfident* on low-resource languages [2601.01362].
- **Prediction:** on dialect texts, instruction-tuned models will often be *confidently* wrong, giving low spread with wrong content. That matches the "false agreement" we saw on *bbona sera* and *cantina/tetto*. **Spread alone cannot detect this**, so a knowledge check is needed.
- Dialect gaps are largest for dialects far from a well-resourced standard [2310.15135]. DialectBench gives a relative-gap metric [2403.11009].

## 4. Translation-studies bridge
Translation-solution entropy H(T|S) is already used as a measure of translation difficulty [2603.12050]. It descends from translation-process research on word translation entropy. Expressing our results as "the entropy of translations of a word, before and after a gloss" therefore speaks the language of translation studies, which matters for the bid. Human translations are also measurably less literal than LLM ones across 54 language pairs [2605.25686], [2305.16806]. That is another angle on "flattening".

## 5. Implications for our design

1. **Two manipulations, two quantities, per word.** Measure the meaning-level spread H of the model's translations under three conditions: no help; with a knowledge gloss; with each sense clarification.
   - *Ignorance* = the drop in H when the knowledge gloss is added.
   - *Multiplicity* = the H that remains after the knowledge gloss. The drop under sense clarification checks that it really is about senses.
   - Report the two as separate diagnostics, never summed [2603.24967], [2209.03302].
2. **A 2×2 validation set built so the answers are known:**

   | | model knows the word | model doesn't know it |
   |---|---|---|
   | one reading | ordinary words | rare or dialect words with a clear gloss (Romanesco *schertri* = skeletons; Belli's own footnotes supply glosses) |
   | many readings | documented untranslatables, polysemous words | dialect untranslatables (*bbona sera*, *a ppecorone*) |

   The method has to put each cell where it belongs before it is trusted on anything else.
3. **Four independent signals of ignorance, which should agree:** gloss sensitivity; cross-model disagreement (Llama, Qwen, Aya) [2604.17112]; small-vs-large disagreement (3B vs 8B) [2402.03563]; instability under iterative prompting [2406.02543].
4. **Human spread as the reference** for multiplicity: dictionary renderings and published translations [2305.11707].
5. **Internal signatures (stretch).** Train linear probes on middle-layer states to predict "unfamiliar" vs "ambiguous" from the 2×2 set [2506.01205], [2402.03563]. If two different directions emerge, that ties the uncertainty question back to thread 1's representations.
6. **Expect confident flattening** on dialect items [2601.01362], and test for it directly with the knowledge check.
