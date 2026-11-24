# Thread 1: "Universal" representations, untranslatables, and English bias

Synthesis of the corpus in `papers.csv` (thread = 1 or method). arXiv IDs are in brackets. These are paraphrased notes for designing experiments, not quotations.

## 1. What the field currently believes

**A. There is a shared, roughly language-neutral "concept space" in the middle layers.**
- Activation patching on Llama-2 separates *which language to answer in* from *which concept to express*: the output language is fixed in earlier layers than the concept. Averaging a concept's hidden state across several source languages makes word translation slightly *better*, which is evidence for a language-agnostic concept representation [2411.08745].
- The "semantic hub" view says semantically equivalent inputs land near each other across languages and even modalities [2411.04986].
- The Platonic Representation Hypothesis generalises this: different models converge toward one statistical model of reality [2405.07987]. A synthetic-grammar theory explains why cross-lingual similarity peaks in the middle layers and grows with language proximity [2610.07168].
- Shared features exist for grammar too: sparse-autoencoder features on Llama-3-8B and Aya-23-8B fire across many languages [2501.06346].

**B. That space is not neutral. It leans English.**
- Logit-lens readouts of Llama-2 show a middle-layer phase where the English version of the answer is most probable, even for French→Chinese prompts. The authors read this as an English-biased concept space, not literal translation into English [2402.10588].
- In open-ended generation, content words (nouns, verbs) are routed through English latents far more than function words. English steering vectors work better than target-language ones. Routing varies by model: Gemma is the most English-routed, Aya the least [2502.15603].
- An English "default" signal appears in intermediate layers one layer before content becomes causally available [2611.12538]. Middle-layer English representations can causally fix non-English errors [2504.09378].
- **Model size matters, possibly in the opposite direction to intuition.** Smaller models work mostly in an English-like shared space; larger models drift into language-specific subspaces [2505.13141].
- **Training mix matters.** Japanese-centric models show *dual* Japanese+English latent languages [2408.10811]. Aya-23 activates typologically related languages instead of a single English pivot [2507.20279].
- **Behavioural traces of the bias:**
  - English grammar leaks into Spanish and Greek in mBERT [2210.05619]; LLM output in French and Chinese has an English "accent" [2410.15956].
  - Non-English emotion-word embeddings drift toward English ones in multilingual models, and explicit alignment pulls them further [2307.01370].
  - When asked for the meanings of polysemous words in other languages, models fall back on dominant-language meanings ("language spilling") [2601.12549].

**C. The shared space is weakest exactly where untranslatables live.**
- During EuroLLM training the shared space forms early, but **rare concepts stay poorly aligned**. Patching can even hurt them. Cross-language averaging also nudges polysemous words toward one dominant sense [2601.22851].
- The concept-patching work tested only simple picturable nouns and explicitly leaves culture-specific words open [2411.08745]. **That is the gap our experiments target.**
- Embedding spaces of different languages are not isomorphic, even for close pairs [1805.03620]. Any "universal geometry" claim needs local, not global, tests.

## 2. Closest prior work to the decomposition experiment

**Jung, Cohen and Bergen [2605.24310]** is the nearest neighbour to what we plan.
- Setup: Korean–English emotion adjectives in a fixed frame ("I am X" / "나는 X"), multi-token words averaged, embeddings mean-centred and normalised.
- Finding: words with no translation equivalent ("gap words") have lower nearest-neighbour similarity in the other language. Classifiers reach AUC ~0.81 (KO→EN) and ~0.76 (EN→KO).
- **Unaligned spaces beat Procrustes-aligned ones.**
- Not controlled: word frequency and tokenisation. Only one frame was used.
- Data: the EVOKE resource (Jung et al., 2026) supplies human gap labels. Check whether it is released; it would give us *ground-truth* gaps for validation.

**Arora et al. [1601.03764]**: word vectors are sparse linear superpositions of sense directions ("discourse atoms"), recoverable by sparse coding. This is the theoretical precedent for a *recipe* (non-negative sparse decomposition). Our twist is that the dictionary is restricted to one *other language's* vocabulary.

**Hsieh and Lian [2610.08303]** (position paper, largely untested): concepts are "usage clouds" of contextual embeddings, and some are "α-unalignable" (no map preserves both centroid and neighbourhood structure). The vocabulary is useful. Their proposed metrics (cycle consistency, intrinsic-dimension and density mismatch, a representation–intervention gap) give extra measurements we can borrow.

**Other ground-truth sources:**
- kinship lexical gaps across many languages [2204.05049];
- the Universal Knowledge Core's explicit gap annotations [2203.04723];
- LingoGap crowdsourcing [2410.23133];
- word-level alignment validated on kinship gaps [2410.07239].

## 3. Methods the literature hands us

| Need | Method | Source |
|---|---|---|
| Remove language identity before comparing | subtract per-language centroid | [2004.05160], [2603.02258], [2010.08275] |
| Remove a labelled concept (e.g. stance) linearly | LEACE | [2306.03819] |
| Read middle layers as words | logit lens (fine in mid/late layers); tuned lens is more reliable | [2402.10588], [2303.08112] |
| Causal test that a direction matters | activation patching; mean-difference steering vectors (EN − non-EN) | [2411.08745], [2505.13141], [2502.15603] |
| Chance baselines for similarity | permutation null calibration (similarity inflates with scale) | [2602.14486] |
| Elicit representations of emotion words | frames "I feel X" / "I am X" | [2307.01370], [2605.24310] |

## 4. Implications for our design (what changes because of this reading)

1. **Add a less English-centric model.** Aya-23-8B is the obvious third model, alongside Llama-3.1-8B and Qwen2.5-7B [2502.15603], [2507.20279]. The English-bias claim is only interesting comparatively.
2. **Compare model sizes in the same family** (Llama 3.2-3B vs 3.1-8B). [2505.13141] predicts the *smaller* model is more English-shaped. That is a sharp, falsifiable prediction.
3. **Don't Procrustes-align; just centre.** [2605.24310] found unaligned beats aligned for gap detection, and centring is the standard neutralisation [2004.05160].
4. **Control frequency and tokenisation explicitly.** Rare concepts are poorly aligned [2601.22851], and non-English words cost more tokens (~2 vs ~1.4 tokens per word for Llama-3.1-70B in [2502.15603]). Either could fake "untranslatability" or "English bias".
5. **Use several frames, not one.** This was a stated weakness of [2605.24310]. Treat each word as a *usage cloud* [2610.08303]: decompose each context separately, then report the mean recipe *and* how much the recipe varies across contexts. The variation links directly to thread 2.
6. **Replicate the nearest-neighbour gap baseline** (CSLS similarity) from [2605.24310] before showing anything new. The recipe leftover must beat it to be worth explaining.
7. **Add a concept-patching readout** in the style of [2411.08745]. Patch *saudade*'s middle-layer latent into a word-translation prompt targeting English, and record the distribution of English words the model produces. This is the model's own forced translation, read straight from its concept space. Compare it with the recipe.
8. **Turn "language spilling" into the distinction-collapse test** [2601.12549], [2309.11668]: pairs of non-English languages that share a distinction English lacks.

## 5. Outside arXiv (find separately; not in `pdfs/`)
- Cassin (ed.), *Dictionary of Untranslatables*: the humanities anchor for the word list.
- Jackson et al. (2019, *Science*), emotion semantics via colexification across ~2,500 languages; Thompson, Roberts and Lupyan (2020, *Nature Human Behaviour*), cross-linguistic semantic alignment. Both are classic human-side evidence that word meanings vary systematically across languages.
- Anthropic's "On the Biology of a Large Language Model" (2025, transformer-circuits.pub) has a section on multilingual circuits (shared vs language-specific features). Relevant to the "shared space" question.
- Translation-process research on *word translation entropy* (Carl, Schaeffer and colleagues): the translation-studies counterpart of our entropy measures.
