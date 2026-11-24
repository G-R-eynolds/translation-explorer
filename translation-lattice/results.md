# Results: a principled "differs by meaning" for translation lattices

Run: **`meta-llama/Llama-3.2-3B-Instruct`, bf16** (`runs/3b/`), 18 Nov 2026. The handoff's primary,
Llama-3.1-8B-Instruct, was not accessible (HF licence not accepted). Every script takes `--model 8b`.
Poem: Belli, *Er giorno der giudizzio*. Lines 7–14 are still unverified against a print edition.

## Headline

1. **The model-internal metrics separate meaning from style better than the lexical baseline, but an
   off-the-shelf NLI classifier separates them better still.** On the generated controls (195 meaning-preserving
   restyles vs 43 one-choice meaning edits), AUROC is: NLI 0.98; M1, back-translation JSD, 0.84;
   M1r, back-translation KL rate, 0.80; erased hidden states (M2) 0.35; raw hidden states (B1) 0.33, both *below chance*;
   lexical char-n-grams 0.52, which is chance.
2. **String-level back-translation JSD (M1 as specified) saturates.** For 13 of 14 lines the median pairwise
   JSD is exactly ln 2, so its per-line entropy is pinned at the ceiling, ln 48. It still ranks control pairs
   well (0.84), because restyles sometimes fall below ln 2. But it can't say *how far apart* two readings are.
   The per-token rate (M1r) fixes the saturation at a small cost in AUROC.
3. **On this model, most of the branching is epistemic.** The 3B can't read Romanesco
   (see "Sanity check" below). Lines 1, 11 and 14 converge, and line 5 (*filastrocca*) splits into two stable
   readings. Most other lines scatter into 30–40 one-sample "readings" that are misreadings, not
   interpretations. The lattice is honest about this, but it is a map of the model's ignorance more than
   of the poem's untranslatability. §8 confirms it: wherever the lines branch, Llama-3B and Qwen2.5-7B branch
   *differently* (mutual information between model and reading well above a permutation null, p < 0.001).
   They agree only where both converge.
4. **Hidden states measure wording, not meaning, and LEACE can't rescue them here.** Mean-pooled residual
   states put meaning-preserving restyles *further apart* than one-word meaning edits at every layer
   (AUROC 0.23–0.35). They correlate 0.84 with the lexical baseline across the lattice. LEACE takes the
   stance probe from 0.35 to chance (0.25–0.28), but control AUROC moves by only +0.02, because stance labels
   capture little of the wording variation that restyling introduces.
5. **None of the metrics recover the hand-made demo clusters well** (mean ARI 0.03–0.37 with each metric's own
   k; 0.39–0.51 when given the hand-made k). AUROC on the demo pairs is 0.57–0.71 for every metric, lexical included (0.66). The demo clusters are a weak,
   model-written reference, and they mostly separate subtle register-adjacent readings ("Next!" vs
   "out, whoever's turn it is"), the hardest case for every metric.

## Sanity check: does the model understand the source?

Asked for an Italian paraphrase, the 3B first answered in English. With the prompt itself in Italian it gave
Italian (`runs/3b/paraphrase_it.json`), but misread most dialect words: *schertri* (skeletons)→"scherzi"
(jokes), *bbiocca* (broody hen)→"bocca" (mouth), *ffora a cchi ttoca*→"fuori a toccare", *a ppecorone* (on
all fours)→"ai contadini". Qwen2.5-7B (`runs/qwen7b/paraphrase_it.json`) mostly copies the dialect back,
and also gives "scherzi" and *bbiocca*→"botte" (barrel). **Both models' Romanesco is weak.** This is
the central limitation: most measured multiplicity is the model not knowing.

## Phase 1: sampling

48 samples (4 stances × 12) at T=1.0, top-p 0.95, seed 1831, written to `samples_llama.json`, with
per-token log-probs under the model's own (untruncated) distribution in `samples_llama_logprobs.json`.
Line alignment failed 1 time in 49 (2.0%; retried). Format alignment isn't content alignment, though: a
few samples shift content by one line (e.g. RH02 renders source line 10 in slot 9). These show up as
spurious branches (below).

## Phase 2: the metrics, and what had to change

**M1 back-translation JSD.** The BT prompt holds the rendering plus its neighbouring English lines, labelled
"do NOT translate". With unlabelled context the 3B translated the neighbours too. It never holds the source.
m=6 back-translations per rendering, scored under all 48 BT prompts with the prompt's KV cache shared (≈1 min
per line on the RTX 4070). Two deviations, both needed to make it work:

* *T_bt = 0.7, not 1.0.* At T=1.0, ancestral samples from the 3B had junk tails ("cheIcono che rapisceLemma").
  Scoring uses the same tempered distribution, softmax(logits/0.7), with no top-p, so the Monte Carlo
  estimator remains exact for that q.
* *Saturation.* Sum-of-token log-probs make q_i and q_j almost disjoint at the string level: small per-token
  wording preferences ("scendere" vs "andare") add up over ~15 tokens. Median pairwise JSD = ln 2 on 13 of 14
  lines. Bootstrap SE ≈ 0.00–0.02, so this is not noise.

**M1r back-translation KL rate (added).** J_ij = ½ E_{b~q_i}[(log q_i(b) − log q_j(b))/|b|] + the reverse
term: the symmetric KL divergence *rate* between the two back-translation processes, in nats/token, from
the same score matrices. It doesn't saturate (pairwise values 0 to 8.7 nats/token). Kernel exp(−J/τ), so a pair
at style-noise distance has similarity 1/e.

**M2 hidden states with LEACE.** Every rendering of line *i* is re-embedded after one common prefix: the
translation prompt (neutral stance) plus the central-route lines 1…i−1 (`spine.json`, the medoid of the
largest M1r cluster per line). States are mean-pooled over the rendering's tokens at all 29 layers, and
LEACE (stance, 4 classes) is fitted per layer on the 672 lattice renderings. The layer was chosen by AUROC on
the controls, cross-fitted by line parity.

**Clustering.** As specified (spectral, eigengap k ≤ 5) it **chose k=1 on every line** for M1r and B0. These
kernels are dense: the 3B's renderings scatter rather than forming a few blocks. The lattice instead uses the
rule the τ calibration implies directly: **average-linkage clustering cut at τ**. Two groups are different
readings only if they are further apart, on average, than restyling alone produces. The eigengap results are
kept in `validation.json` (`k_alt`, `ari_hand_alt`).

## Phase 3: validation

Controls are in `runs/3b/controls.json`.
* **Style-only:** 28 whole-translation restyles (8 samples × the other styles of Scots / London / formal /
  plain), with no source in the prompt. I read all 252 non-identical restyled lines and dropped 57 that
  changed content. "Formal" and "London" embellished or reinterpreted: "good evening"→"that's a wrap",
  cellar→"gaff", "Lady"→"Mummy", and two restyles slipped a line (`data/controls_spotcheck_3b.json`). That
  leaves 195 style pairs.
* **Meaning:** 43 hand-written minimal edits, one interpretive choice each, rest verbatim (angels→devils,
  cellar→hell, "nursery rhyme"→"long procession", "jesters"→"skeletons", "good evening"→"good riddance").
  See `data/meaning_edits_3b.json`.
* For M1, the main style set puts the restyled line **in the original's context**, as the meaning edits are.
  A second set keeps the restyle's own restyled neighbours, which measures context leakage: M1 0.84→0.88,
  M1r 0.80→0.82. So context barely matters.
* **Demo block (weak reference):** 279 cross-cluster pairs and 645 same-cluster pairs from the hand clusters.

#### Validation

| metric | AUROC controls | AUROC own-context restyles | AUROC demo (weak) | tau | meaning edits > tau | ARI hand (tau-cut) | ARI hand (hand k) | ARI clusters~stance |
|---|---|---|---|---|---|---|---|---|
| M1 back-translation JSD (string level) | 0.84 | 0.88 | 0.71 | 0.693 | 0.14 | 0.37 | 0.51 | 0.01 |
| M1r back-translation KL rate (nats/token) | 0.80 | 0.82 | 0.68 | 2.280 | 0.16 | 0.17 | 0.50 | 0.01 |
| M2 hidden states, LEACE-erased, layer 8 | 0.35 (cross-fit 0.34) | 0.35 | 0.57 | 0.397 | 0.00 | 0.03 | 0.40 | 0.01 |
| B0 lexical char-n-gram | 0.52 | 0.52 | 0.66 | 0.671 | 0.00 | 0.11 | 0.50 | 0.01 |
| B1 hidden states, raw, layer 8 | 0.33 (cross-fit 0.30) | 0.33 | 0.57 | 0.412 | 0.00 | 0.03 | 0.39 | 0.01 |
| B2 NLI bidirectional entailment | 0.98 | 0.98 | 0.69 | 0.450 | 0.91 | 0.15 | 0.50 | 0.00 |

#### Per line (k = clusters after the tau cut; shared = clusters with >= 2 samples; top = largest cluster of 48)

| line | m1 KLE / H bits / k (shared, top) | m1r KLE / H bits / k (shared, top) | m2 KLE / H bits / k (shared, top) | b0 KLE / H bits / k (shared, top) | b1 KLE / H bits / k (shared, top) | b2 KLE / H bits / k (shared, top) |
|---|---|---|---|---|---|---|
| 1 | 1.55 / -0.00 / 1 (1, 48) | 0.79 / -0.00 / 1 (1, 48) | 0.29 / -0.00 / 1 (1, 48) | 0.92 / 0.15 / 2 (1, 47) | 0.30 / -0.00 / 1 (1, 48) | 0.41 / 0.15 / 2 (1, 47) |
| 2 | 3.86 / 4.14 / 32 (1, 17) | 3.64 / 5.17 / 40 (4, 4) | 2.69 / 4.51 / 29 (8, 7) | 3.46 / 4.70 / 32 (8, 7) | 2.70 / 4.43 / 28 (8, 7) | 3.36 / 4.83 / 36 (4, 8) |
| 3 | 3.87 / 4.13 / 30 (4, 16) | 3.65 / 5.24 / 41 (4, 4) | 2.49 / 3.97 / 23 (7, 9) | 3.39 / 4.30 / 25 (10, 6) | 2.50 / 3.90 / 22 (7, 9) | 3.19 / 4.72 / 31 (9, 4) |
| 4 | 3.87 / 5.08 / 39 (3, 5) | 3.47 / 5.15 / 39 (6, 4) | 2.33 / 2.94 / 19 (5, 25) | 3.41 / 4.42 / 26 (11, 6) | 2.35 / 2.84 / 18 (4, 25) | 3.20 / 4.69 / 31 (9, 6) |
| 5 | 3.73 / 1.39 / 10 (2, 38) | 2.95 / 3.08 / 19 (4, 21) | 2.23 / 3.36 / 15 (7, 10) | 2.90 / 2.78 / 12 (7, 18) | 2.23 / 3.25 / 13 (8, 10) | 2.25 / 2.83 / 17 (3, 21) |
| 6 | 3.84 / 3.74 / 28 (2, 20) | 3.58 / 5.18 / 40 (5, 5) | 2.47 / 3.67 / 23 (6, 17) | 3.29 / 3.99 / 25 (6, 13) | 2.47 / 3.47 / 21 (6, 18) | 3.10 / 4.64 / 31 (7, 6) |
| 7 | 3.86 / 4.32 / 33 (2, 15) | 3.68 / 5.26 / 42 (3, 5) | 2.86 / 4.71 / 32 (7, 6) | 3.52 / 4.60 / 29 (9, 4) | 2.86 / 4.63 / 31 (6, 5) | 3.25 / 4.64 / 34 (4, 10) |
| 8 | 3.82 / 3.14 / 20 (2, 16) | 3.42 / 4.68 / 31 (9, 5) | 2.13 / 2.37 / 11 (5, 25) | 3.35 / 4.28 / 25 (9, 7) | 2.14 / 2.37 / 11 (5, 25) | 3.56 / 5.17 / 39 (7, 4) |
| 9 | 3.77 / 0.91 / 5 (2, 40) | 3.02 / 3.34 / 15 (9, 14) | 1.78 / 1.08 / 6 (3, 39) | 2.89 / 2.58 / 12 (6, 24) | 1.79 / 0.84 / 5 (2, 41) | 2.96 / 3.92 / 20 (11, 9) |
| 10 | 3.58 / 1.01 / 8 (1, 41) | 2.61 / 2.17 / 11 (3, 27) | 1.64 / 1.46 / 7 (4, 35) | 2.66 / 2.37 / 11 (6, 26) | 1.64 / 1.01 / 5 (3, 39) | 2.02 / 2.01 / 11 (4, 31) |
| 11 | 3.60 / 0.83 / 6 (2, 42) | 2.20 / 1.39 / 8 (4, 37) | 1.23 / 0.78 / 5 (3, 42) | 1.89 / 0.73 / 4 (3, 42) | 1.23 / 0.78 / 5 (3, 42) | 1.62 / 1.37 / 6 (5, 36) |
| 12 | 3.86 / 3.86 / 22 (6, 8) | 3.51 / 4.95 / 34 (11, 3) | 2.59 / 3.55 / 17 (10, 10) | 3.13 / 3.53 / 17 (8, 12) | 2.59 / 3.55 / 17 (10, 10) | 2.96 / 4.05 / 23 (9, 10) |
| 13 | 3.87 / 3.07 / 18 (3, 19) | 3.44 / 4.69 / 31 (8, 5) | 2.77 / 4.54 / 28 (10, 5) | 3.32 / 4.06 / 21 (10, 6) | 2.77 / 4.26 / 23 (12, 5) | 2.82 / 3.82 / 25 (3, 14) |
| 14 | 3.65 / 0.68 / 5 (2, 43) | 2.67 / 2.19 / 13 (4, 30) | 1.76 / 1.62 / 11 (3, 36) | 2.61 / 1.73 / 10 (4, 34) | 1.78 / 1.49 / 10 (3, 37) | 1.51 / 1.19 / 8 (2, 39) |

#### Agreement among metrics on the lattice (Spearman of pairwise distances; mean ARI of clusterings)

| pair | Spearman | ARI |
|---|---|---|
| m1~m1r | 0.77 | 0.30 |
| m1~m2 | 0.64 | 0.41 |
| m1~b0 | 0.64 | 0.27 |
| m1~b1 | 0.65 | 0.40 |
| m1~b2 | 0.56 | 0.27 |
| m1r~m2 | 0.76 | 0.34 |
| m1r~b0 | 0.73 | 0.33 |
| m1r~b1 | 0.76 | 0.34 |
| m1r~b2 | 0.57 | 0.37 |
| m2~b0 | 0.84 | 0.35 |
| m2~b1 | 1.00 | 0.92 |
| m2~b2 | 0.50 | 0.22 |
| b0~b1 | 0.84 | 0.35 |
| b0~b2 | 0.53 | 0.37 |
| b1~b2 | 0.50 | 0.22 |


Reading the table:
* τ is the 95th percentile of each metric over the style pairs. "Meaning edits > τ" is the share of meaning
  edits that a lattice would call different. NLI calls 91% of them different. M1 calls 14% and M1r 16%:
  one changed word in a 10-word line moves the back-translation distribution less than restyling the whole
  line does.
* "ARI clusters~stance" ≈ 0 for every metric: no lattice's branches track the stance.

## Per line: where the 3B's samples branch

KLE (nats; max ln 48 = 3.87) and semantic entropy H (bits) from M1r; readings named by medoid.

| line | source | M1r: KLE / H | readings (M1r clusters with >= 3 of 48, medoid) | NLI agrees? |
|---|---|---|---|---|
| 1 | *Cuattro angioloni co le tromme in bocca* | 0.79 / 0.00 | **converges**: “Four angels with trumpets in their mouths” (48) | yes (47/48) |
| 2 | *se metteranno uno pe ccantone* | 3.64 / 5.17 | scattered: “will put one on each corner” (4), “will put themselves in one corner” (4); 40 clusters | yes, scattered |
| 3 | *a ssonà: poi co ttanto de voscione* | 3.65 / 5.24 | scattered (41 clusters); “to call: then with that much noise” (4) | yes |
| 4 | *cominceranno a ddì: ffora a cchi ttoca.* | 3.47 / 5.15 | scattered (39): “start to sing: outside where they touch” (4), “call: out to whoever touches” (3). The hand-cluster readings (“Next!”, “Arise, you who are called”) never appear | yes |
| 5 | *Allora vierà ssù una filastrocca* | 2.95 / 3.08 | **two stable readings**: “Then up will rise a nursery rhyme” (21) vs “At that, a little song will come up” (8). The *procession / long string* reading is absent | yes: 21 vs 11 |
| 6 | *de schertri da la terra a ppecorone,* | 3.58 / 5.18 | scattered (40): “of jesters from the earth to shepherds” (5). *schertri* = skeletons and *a ppecorone* = on all fours are never found | yes |
| 7 | *pe rripijjà ffigura de perzone,* | 3.68 / 5.26 | scattered (42): “To ripen the figure of a man” (5) | partly: NLI merges “ripping up a figure of man” (10) |
| 8 | *come purcini attorno de la bbiocca.* | 3.42 / 4.68 | “Like cherries around God's mouth” (5) vs “like cherubs around a child's breast” (5) vs a shifted line 9 (4); 31 clusters | yes, scattered |
| 9 | *E sta bbiocca sarà Ddio bbenedetto,* | 3.02 / 3.34 | “And that mouth will be blessed God” (14) vs “And that cradle will be blessed by God” (7) vs “that God's mouth will be blessed” (5) vs a shifted line 10 (3) | yes (mouth / belly / Virgin) |
| 10 | *che ne farà du' parte, bbianca, e nnera:* | 2.61 / 2.17 | **mostly converges**: “who will make two parts of it, white and black” (27 + 8 near-identical), plus a shifted line 11 (5) | yes (31) |
| 11 | *una pe annà in cantina, una sur tetto.* | 2.20 / 1.39 | **converges**: “one to go down to the cellar, one on the roof” (37); “one to stay [under the roof]” (3) | yes (36) |
| 12 | *All'urtimo usscirà 'na sonajjera* | 3.51 / 4.95 | scattered (34): bell (3), trumpets (3), “an angel's bell will ring out” (3) | yes: trumpet (10) vs bell (5) |
| 13 | *d'angioli, e, ccome si ss'annassi a lletto,* | 3.44 / 4.69 | scattered (31): “as they lie down to sleep” (4), “of angels, as they settle into bed” (4), a shifted line 14 (5) | yes |
| 14 | *smorzeranno li lumi, e bbona sera.* | 2.67 / 2.19 | **converges** on “they will extinguish the lights, and good evening” (30); “the lights will fade out” (5). The “and that's that” reading of *bbona sera* is absent | yes (39) |

So the 3B's lattice has one genuine two-way branch, line 5 (*filastrocca*: nursery rhyme vs little song).
Arguably even that is one reading (both are songs). The branching the grant cares about (*ffora a chi tocca* as
queue vs summons, *bbona sera* as leave-taking vs finality, the cellar/roof dome gloss) **does not appear at
all**. The model never reaches those readings. Several "branches" are samples whose content slipped by one
line (lines 8, 9, 10, 11, 13).

## Phase 4: layers (`runs/3b/results/layer_sweep.png`)

![layer sweep](runs/3b/results/layer_sweep.png)

* **Stance is weakly decodable at every layer** (grouped-CV probe 0.31–0.37; chance 0.25). LEACE, refitted
  inside each fold, takes it to chance (0.24–0.28) at every layer.
* **Meaning separation rises through the middle layers**: the between/within-cluster distance ratio, with
  M1r clusters as labels, goes from 2.3 (layers 2–5) to 3.7 (layer 13), then plateaus. **LEACE leaves it
  intact** (3.68 → 3.66 at layer 13). So the predicted pattern partly holds: meaning structure emerges in the
  middle layers, and erasing stance doesn't damage it. But "style still decodable there" is weak, because stance
  was barely decodable anywhere. The labels also come from M1r, so this measures agreement with M1r, not
  ground truth.
* **On the controls, no layer separates meaning from style** (bottom-left panel). Raw and erased AUROC peak
  at 0.35 around layer 8 and sit below 0.5 everywhere. The mid-layer "meaning" signal is real relative to
  M1r's clusters, but at the scale of a one-word edit versus a full restyle it is swamped by wording.
* **Logit lens (source side, `logit_lens.json`).** Asked what a span means in English, the layer-wise guess for
  the first word is noise until about layer 17. An English-like answer then emerges between layers 17 and 23,
  matching the English-pivot picture in Wendler et al. (2024):
  * *bbona sera* settles early and singly ("good", 0.83 by layer 20).
  * *filastrocca* briefly holds two candidates ("little" 0.66 / "nursery" 0.07 at layer 20, flipping to
    "nursery" 0.69 at 23).
  * *ffora a cchi ttoca* never settles: entropy is still 5 nats at layer 26, spread over "then / out / forth /
    come".
  * *bbiocca* moves through belly → lips → mouth, all wrong (it is a broody hen).
  * In the source-position view, *a ppecorone* decodes to "sheep / pasture / cattle" at layer 22. That is the
    etymological pull of *pecora*, and it explains the "to the sheep" misreadings in the samples.

  So the lens separates "one confident reading" (*bbona sera*), "two candidates" (*filastrocca*) and
  "unsettled" (*fora a chi tocca*). Whether those are right is a separate question: two of the four confident
  answers are wrong.

## Phase 5: two models (§8)

Qwen2.5-7B-Instruct (nf4) sampled 48 translations with the same prompts (alignment failures 1/49). The
96 pooled renderings per line were clustered with the τ-calibrated metrics: NLI, and M1r with the Llama-3B
back-translation scorer (m=4). MI = H(pooled) − mean H(per model), in bits, is compared to a 500-draw
permutation null over model labels (`runs/3b/results/epistemic_qwen7b.json`).

| line | NLI: H / MI / null / p | M1r: H / MI / null / p | reading |
|---|---|---|---|
| 1 | 0.58 / 0.10 / 0.04 / .008 | 0.17 / 0.02 / 0.02 / .49 | both converge (angels with trumpets) |
| 4 *ffora a cchi ttoca* | 3.77 / 0.81 / 0.38 / <.001 | 4.84 / 0.90 / 0.57 / <.001 | Qwen converges on “forth / out to who it concerns” (40/48); 3B scatters |
| 5 *filastrocca* | 4.34 / 0.86 / 0.41 / <.001 | 4.62 / 0.82 / 0.49 / <.001 | 3B “nursery rhyme”, Qwen “a rhyme / shall come forth a rhyme” |
| 8 *bbiocca* | 5.79 / 1.00 / 0.69 / <.001 | 4.82 / 0.95 / 0.47 / <.001 | Qwen “piglets around a box / bowl”, 3B “cherries around God's mouth” |
| 9 | 4.75 / 0.88 / 0.37 / <.001 | 3.75 / 0.69 / 0.30 / <.001 | Qwen “basket / bowl”, 3B “mouth” |
| 10 | 1.52 / 0.18 / 0.12 / .024 | 2.16 / 0.23 / 0.18 / .10 | both converge (two parts, white and black) |
| 11 | 1.21 / 0.18 / 0.09 / <.001 | 1.35 / 0.19 / 0.13 / .012 | both converge (cellar / roof) |
| 13 | 3.95 / 0.47 / 0.37 / .004 | 5.37 / 0.77 / 0.60 / <.001 | the most shared branching line: “as they settle into bed” is shared by both |
| 14 | 0.90 / 0.11 / 0.08 / .054 | 2.66 / 0.41 / 0.24 / <.001 | both converge (lights out, good evening) |

Every other line (2, 3, 6, 7, 12) looks like 4, 8 and 9: high entropy, and MI 0.12–0.39 bits above the
null (p < 0.001). **No line shows the signature of genuine multiplicity: high entropy with the models
branching the same way.** Where they converge they agree, and where they branch they branch differently.
This is the outcome the handoff anticipated for weak models on an under-resourced variety. It is good evidence
that the method can tell "the model doesn't know" apart from multiplicity, and that it mustn't be presented as
mystery. It is not yet evidence about Belli's untranslatables. That needs models that actually read Romanesco.

## Limitations

* **Model competence** (above) dominates everything. Rerun with 8B once the licence is accepted
  (`--model 8b`), and expect fewer scattered lines.
* **Controls are mine.** I wrote the meaning edits and did the restyle spot-check, as a model, in session.
  The controls are reusable, but a human should check them, especially before using τ on another poem.
* M1's sensitivity is low at the controls' scale: one-word meaning edits mostly fall under τ. The τ cut
  therefore merges near-identical readings and splits mainly on gross differences. On line 11 it also splits
  a small group of cellar/roof renderings from the main 37/48 one (a wording split it shouldn't make).
* Misaligned samples (content shifted by a line) appear as branches. A content-alignment check could filter
  them; that isn't done.
* The M2 layer is chosen on the same controls it is scored on. The cross-fitted AUROC (by line parity) is the
  honest number.
* Demo hand clusters were written by the model that wrote the demo samples. ARI against them is weak
  evidence either way.
* One run, one seed, one poem. The bootstrap SEs cover Monte Carlo noise over back-translations, not sampling
  noise over translations.
