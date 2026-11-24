# HANDOFF: What is an "untranslatable" made of? Decomposing words inside a multilingual model

## 1. Purpose and audience

This feeds ideas into an AHRC bid on AI translation and **untranslatability** (Cassin: words that "never stop being (un)translated"; Sakai and Liu: translation builds the borders it crosses, and the weaker language tends to be "forcibly transformed").

The previous experiment (`../translation-lattice/`, see its `results.md`) taught two lessons:
- model ignorance swamped everything, because a small model can't read Romanesco;
- whole-poem lattices were hard to interpret.

So this experiment is deliberately **simple, single-word and well-resourced**.

**The final reader is a linguist who is not mathematical.** Every result must come with a picture that can be understood without equations. Treat the figures as the deliverable, not decoration (§7).

## 2. The idea, in one paragraph

Inside a multilingual LLM, the middle layers seem to hold a shared "concept space" across languages. Take a documented untranslatable, e.g. Portuguese *saudade*, and find its representation there. Then ask: **what is the best recipe of English words that rebuilds it?** For example: 40% longing + 25% nostalgia + 15% absence + …, plus a **leftover**, the part no combination of English words reaches.
- The recipe is a machine-made "thick translation", like a translator's footnote.
- The leftover is a simple untranslatability score.

Then **run it the other way**: rebuild English words from Portuguese (and German, Italian…) words. If English words always leave smaller leftovers than other languages' words, even for ordinary words, then the model's concept space is English-shaped. That is a measurable version of Liu's "forcible transformation". **This asymmetry is the central question.**

## 3. Environment (same as last time)

- Code lives on the laptop. Make a new folder `~/Documents/Fourth Year/Translation Experiments/untranslatables/`.
- Run on the RTX 4070 (12 GB) in WSL2 over `ssh desktop-gpu`. **Reuse** `../translation-lattice/remote/` (sync, run, pull, status) and its conventions, uv env and HF cache.
- Models:
  - `meta-llama/Llama-3.1-8B-Instruct` (nf4). Last time its HF licence wasn't accepted, so check. If it's still blocked, ask the user to accept it, and meanwhile use `meta-llama/Llama-3.2-3B-Instruct` (bf16).
  - `Qwen/Qwen2.5-7B-Instruct` (nf4, already downloaded) as the comparison model. Its training mix is less English-dominated (much Chinese), so whether *its* concept space is English-shaped too is part of the result.
- Only forward passes and hidden states are needed. Everything except the optional §6.5 is cheap.

## 4. Word sets (`data/words.yaml`)

Phase 1 languages: **Portuguese, German, Italian, French, Spanish**, all well resourced. Phase 2 (stretch, flagged as lower-confidence): Russian, Japanese, Hindi, Welsh.

Every entry has `word, language, category, gloss_en` (a short dictionary-style gloss, for the figures), `source`, and `notes`. Build these categories:

| category | what | examples (seed list; the user's father may swap in others) |
|---|---|---|
| `untranslatable` | widely documented "untranslatables" | pt *saudade*; de *Sehnsucht*, *Heimat*, *Fernweh*, *Weltschmerz*, *Gemütlichkeit*; it *sprezzatura*, *magari*, *menefreghismo*, *abbiocco*; fr *dépaysement*, *flâner*, *l'esprit de l'escalier*, *retrouvailles*; es *sobremesa*, *duende*, *madrugada*, *estrenar* |
| `ordinary` | everyday words with a clear English equivalent, frequency-matched where possible | pt *chuva*, *mesa*, *tristeza*; de *Freude*, *Fenster*, *Angst*(see loan); it *gioia*, *pane*; fr *colère*, *maison*; es *cansancio*, *ventana* |
| `emotion_plain` | emotion words *with* a near equivalent, controlling for "abstract/emotional" vs "untranslatable" | *tristeza* (sadness), *Freude* (joy), *nostalgia* (it/es/pt, nostalgia), *colère* (anger) |
| `loanword` | words English has borrowed | de *Schadenfreude*, *Angst*, *Zeitgeist*; fr *ennui*, *déjà vu*; da/no *hygge* |
| `english_untranslatable` | for the reverse direction: English words often called hard to translate | *cosy*, *serendipity*, *awkward*, *privacy*, *fair* (as in fair play), *home* (vs house), *mind*, *wistful* |
| `english_ordinary` | everyday English words | *rain*, *table*, *sadness*, *window*, *bread*, *anger* |

Rules:
- `source` must be a real citation or a URL the user can check (Cassin's *Dictionary of Untranslatables* where applicable). **Don't invent sources.** Leave `source: TODO` and let the user fill it.
- Keep the list modest: about 15 untranslatables per Phase-1 language, matched with about 15 ordinary and a few emotion words.
- Write a `PREDICTIONS.md` **before running anything**: which words you expect to have the largest and smallest leftovers, and the expected direction of the asymmetry. Commit it. This keeps the analysis honest.

## 5. Step 0: does the model know the word? (familiarity gate)

Last time's lesson: never measure untranslatability for words the model doesn't know.
- For each word, prompt in its own language for a one-sentence definition, then in English for an English explanation. Save both to `results/familiarity.csv`.
- Also record the word's mean token log-prob inside a natural sentence, and how many tokens it splits into.
- Make a **simple HTML checklist** (`results/familiarity_check.html`) the user or his father can tick: "model understood: yes / partly / no". Flag words marked "no" and exclude them from headline figures, but keep them in an appendix.
- Until a human has checked, use the model's English explanation vs `gloss_en` (rough keyword overlap) as a provisional flag. Say so in the report.

## 6. Method

### 6.1 Getting a word's representation
- For each word, use **5 short templates written in the word's own language**, e.g. pt `A palavra "saudade" quer dizer`, de `Das Wort „Sehnsucht" bedeutet`. Write templates for each language and ask the user to check them. Plus **5 natural example sentences** in that language: model-generated, saved and visible in the appendix.
- Representation at layer ℓ = mean of the residual-stream states over the word's own tokens, averaged across the 10 contexts. Keep the template-only and sentence-only versions too, as a robustness check.
- **Remove language identity.** Subtract the language's mean vector, computed over that language's vocabulary pool (§6.2) at the same layer. Then L2-normalise. Without this, "being Portuguese" dominates every comparison. (Removing per-language centroids is a known trick for cross-lingual alignment; mention it in the report's appendix.)

### 6.2 Vocabulary pools ("ingredient lists")
- For each language, take the top ~8,000 content words by frequency (`wordfreq` package). Lowercase, alphabetic only, drop stopwords. Represent each one by the same procedure, but **templates only**, for speed. Cache to disk.
- **Pools must be frequency-matched and the same size across languages**, or the asymmetry test is meaningless.
- **Exclude trivial matches.** When decomposing a word, remove the word itself and its obvious cognates or borrowed forms from the pool (e.g. remove English *saudade*, *schadenfreude*, *angst*, *nostalgia*, *ennui* when decomposing those words). Keep a log of exclusions.
- **Loanword sub-experiment.** Decompose *Schadenfreude*, *Angst*, *ennui* and *hygge* twice, with and without the English loan in the pool. Does borrowing the word "close the gap"?

### 6.3 Choosing the layer
Use a simple, explainable criterion: **translation retrieval**. For ~300 ordinary word pairs with dictionary translations (pt↔en etc.; take them from a bilingual dictionary, e.g. the MUSE dictionaries, and record the source), at each layer, how often is the correct translation the nearest neighbour in the other language's pool?
- Pick the layer with the best retrieval (expect the middle layers).
- Plot it, because it doubles as a nice picture of "where the concept space is".
- Run everything at that layer. Also report two neighbouring layers to show the results aren't fragile.

### 6.4 The recipe: non-negative greedy decomposition
For target vector x and pool vectors {e_i}:
1. Greedy forward selection with **non-negative** weights (non-negative orthogonal matching pursuit, or NNLS refitted after each addition), up to k = 10 words.
2. Record explained variance R²(k) = 1 − ‖x − Σ w_i e_i‖² / ‖x‖² after each addition. Record the chosen words and weights.
3. **Recipe for display** = the chosen words at k = 5, with weights rescaled to shares of the explained part. **Leftover** = 1 − R²(5).
4. **Chance baseline.** With 8,000 candidates in high dimension, *any* vector gets partly explained. Run the same procedure on (a) random unit vectors and (b) vectors of random words from the same language. Report every leftover **relative to chance**, and show the chance band on every figure.

The method is deliberately plain. Don't add sparsity penalties, learned dictionaries or other complexity unless this fails.

### 6.5 (Optional) behavioural check: the model's translations
For each word, in its 5 example sentences, sample 10 English translations of the word (T=1). Count the distinct renderings, and the entropy over meaning-groups (use an NLI model: last time it was the best meaning judge, AUROC 0.98). Then plot it against the leftover. Do the inside (geometry) and the outside (behaviour) agree?

### 6.6 The asymmetry test (the headline)
For each language pair (X, EN):
- leftover(X word → EN pool) for X's ordinary and untranslatable words;
- leftover(EN word → X pool) for the English ordinary and English-untranslatable words.

Compare matched categories: ordinary vs ordinary, untranslatable vs untranslatable. Report medians with bootstrap intervals over words.
- Check the confounds explicitly: the number of tokens a word splits into, word frequency, model familiarity. Show that the asymmetry survives matching on these, or say plainly that it doesn't.
- Repeat on Qwen2.5-7B. Is its space English-shaped too, or shaped some other way?

## 7. Figures: the main deliverable

Write for a humanities reader:
- every figure title states its **finding as a sentence**;
- each figure has a one-line "how to read this" caption in plain words;
- no symbols in captions (say "leftover", "share", "how much English can say");
- consistent colours: **grey always means "leftover / what English can't reach"**, and one accent colour for the untranslatables;
- big type, PNG and SVG both.

1. **Word recipe cards** (`fig_recipes`). One card per word: the word, its language, the dictionary gloss, and one example sentence. Below it, a single horizontal bar split into English "ingredients" (labelled segments with shares) plus a grey "leftover" segment. Show *saudade* next to *tristeza* first; that pair should explain the whole idea in one glance. Make a gallery of all words, sorted by leftover.
2. **"How many English words does it take?"** (`fig_curves`). Explained share (0–100%) against the number of English words used (1–10). Ordinary words shoot up and flatten high; untranslatables climb slowly and flatten lower. Show the chance band in light grey. Highlight 3–4 named words and keep the rest as faint lines.
3. **The neighbourhood map** (`fig_map`). For one word at a time, a 2D picture of the word and its nearest English words: a local PCA of the word plus its ~40 nearest pool words. Draw the untranslatable as a highlighted point, with lines to its recipe ingredients. Caption it honestly as "a flattened sketch; distances are approximate". It should show *saudade* sitting **between** longing, nostalgia and melancholy rather than on top of any of them.
4. **Leftover ranking** (`fig_ranking`). A horizontal bar per word, sorted, coloured by category (untranslatable / ordinary / emotion / loanword), with the chance band. Mark words that failed the familiarity check with a hatched pattern.
5. **Two directions** (`fig_asymmetry`), the headline figure. A dumbbell per language: left dot = "how much is left over when Portuguese words are rebuilt from English", right dot = "when English words are rebuilt from Portuguese". Separate rows for ordinary and untranslatable words, and both models side by side. Plain-English title, e.g. "Inside Llama, English can describe Portuguese more easily than Portuguese can describe English" (only if that's what the data shows).
6. **Borrowing** (`fig_loans`). For each loanword, the leftover with and without the borrowed English word in the pool: before → after arrows.
7. **Where the shared space lives** (`fig_layers`). Translation-retrieval accuracy by layer: one simple line with a shaded "concept layers" band.
8. *(If §6.5 is done)* **Inside vs outside** (`fig_behaviour`). Leftover (x) against how many different English translations the model actually produces (y), one labelled dot per word.

Assemble everything into **`results/report.html`**, a single self-contained page:
- a 5-sentence plain-English summary at the top;
- the figures in the order above, each with its finding and a "how to read this" line;
- a short "what this does and doesn't show" section;
- a technical appendix (method, maths, exclusions, templates, familiarity table) at the bottom, for the user, not the father.

## 8. Honesty and pitfalls (put these in the report)

- A model's internal representation is **not** "the meaning of the word". It is how one model, trained mostly on web English, files the word. Phrase findings as "inside this model…".
- A big leftover can mean the word is untranslatable, **or** that the model barely knows it, **or** that it is split into many tokens. §5 and the confound checks in §6.6 exist to tell these apart. Report the ones that remain ambiguous as ambiguous.
- Only include word lists, templates and dictionary pairs that a human can check. Record every model-generated sentence.
- If the asymmetry doesn't appear, or disappears once token count is matched, **say so plainly**. That's also a result worth telling a translation scholar.

## 9. Order of work

1. `words.yaml`, templates and `PREDICTIONS.md`. **Stop and show the user** before building the pools. His father may want to edit the list.
2. Familiarity gate (§5).
3. Pools and layer choice (§6.2–6.3), on the 3B first for speed.
4. Decompositions and chance baselines (§6.4), then figures 1–4 and 7. **Show the user the *saudade*/*tristeza* recipe card and the curves before going further.**
5. Asymmetry and loanwords (§6.6), figures 5–6, then Qwen.
6. Optional §6.5, then `report.html`.

Suggested files: `words.py`, `represent.py`, `pools.py`, `layers.py`, `decompose.py`, `asymmetry.py`, `behaviour.py`, `figures.py`, `report.py`, plus `README.md`.

Ask the user if anything in §4 or §6 is ambiguous. Don't silently improvise around a gap.
