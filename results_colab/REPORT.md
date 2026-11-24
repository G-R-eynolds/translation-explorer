# Untranslatability inside Llama-3.3-70B (4-bit): Colab run, 22 Nov 2026

Model: `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (80 layers, d = 8192) on a Colab A100 80 GB, driven through the official
Colab CLI. Design: `../HANDOFF_internals.md` §0, §3–§5; scripts: `../colab/cli/` (fixes in `CHANGES.md` there).
Every number here is **inside this one model**, and for Part C **relative to the conventional translations in the
corpus** (Europarl + OPUS-100 for es–en; OPUS-100 for hi–en). Raw outputs: `unsloth_Llama_3_3_70B_Instruct_bnb_4bit/`
(JSON, figures, the scripts' own `SUMMARY.md`); logs: `logs/`.

**Comparisons with the earlier 8B run** (`../untranslatables/results/sent/report.md`, Llama-3.1-8B) are indicative only:
the Colab scripts use their own, smaller sentence set and calibrate the tolerance on it (ε = 0.342 here vs 0.140 for the
8B), and Part A's English-share metric is computed here from the stored probabilities so that it matches the 8B's.

## A. English detour — positive

- The logit-lens **English share** (P(English) / (P(English) + P(Italian))) peaks at **0.947 at layer 54 of 80**; at the
  last layer Italian holds 0.948 of the probability and English 0.001 (28 words). The Llama-3.1-8B peaked at 0.93–0.94.
- The scripts' summary line "peak mean P(English) = 0.039 (layer 60)" is the *raw* probability, which is spread thin in the
  middle layers; it is not comparable with the 8B's share and should not be quoted as "weak".

## B. Sentence round trip (masked) — mixed

- **Hot words are only modestly hotter:** leftover AUROC, hot vs other content words = **0.611** (29 hot, 70 other).
  The 8B gave 0.726 (95% CI 0.637–0.812) on its own sentence set.
- **No category needs extra length:** with ε = 0.342 (90th percentile of control distortion at the standard rung), every
  category reaches ε at the standard translation (U = 0.00 doublings) except constructions (0.33). Only the threshold-free
  average distortion A separates them: untranslatable **0.063** vs control **0.022** (construction 0.080, idiom 0.010).
  The 8B: untranslatable 0.089 vs control 0.023, and U 0.93 vs 0.21.
- **What borrowing buys** (B1 − keep-the-word, unmasked / masked): untranslatable 0.101 / 0.013, construction
  0.120 / −0.046, control 0.079 / 0.076. As in the 8B, most of the untranslatables' gain disappears once the borrowed word is
  hidden.
- **The father's sentence:** standard translation **"I love the elf."** Round-trip leftover: duende **0.07**, gusta **1.00**,
  mucho **1.00** (8B: 0.50 / 0.23 / 0.01). Duende's low leftover is not success: "elf" maps straight back to *duende* in the
  dictionary sense. **Intended-sense recovery** for duende (Lorca's sense): standard **0.30**, keep the word −0.04, inline
  gloss 0.67, footnote −0.72, thick translation −0.46. The 70B, like the 8B, reads duende as a goblin: its B0 is "the
  goblin", B3 glosses it as "[mythical creature]", B6 describes "a mythical creature".
- The ladder prompts do what they ask (B3 brackets, B5 "Note:", B6 a paragraph, B2 italics), but meanings are often wrong
  (*sobremesa* → "overtime" in B3 and B6).

## C. Round-trip spectrum (N1) — mixed: the spectrum and U are real, the predicted English bias and probe pattern are not

1. **The spectrum is real, but the handoff's "shared" count is uninformative.** At the best layer (40 for both pairs) the
   held-out and in-sample top-50 ρ are nearly equal (es **45.2 / 45.7**, hi **43.0 / 43.0**): no overfitting. But the
   shared-direction count (held-out ρ above the 99th percentile of the shuffled-pair null, ≈0.03) saturates: **8190 of 8192**
   for es, 7288 for hi. Aligned words correlate a little in almost every direction. The informative count is the stricter one
   (held-out ρ > 0.71, i.e. more than half kept per round trip): es **318 / 325 / 500 / 407 / 280** at layers 20/32/40/48/60,
   hi **95 / 103 / 150 / 129 / 101**: sharing peaks mid-depth (layer 40), and Spanish shares about three times as much with
   English as Hindi does.
2. **U picks out hard words, and beats the token-count baseline (es).** Layer 40: AUROC hot vs other content words
   **U 0.78, U_raw 0.70, number-of-tokens baseline 0.45** (26 hot, 50 other words, 19 sentences); layer 60: U 0.82, U_raw 0.69.
   So U is not just measuring how split-up a word is. **Hindi is too small to judge:** U 0.83, U_raw 1.00 rest on **3** hot
   words (14 other, 6 sentences).
3. **The father's sentence does not separate:** layer 40, U duende **0.785**, gusta **0.794**, mucho **0.769**; half-life 0.5
   round trips for all three (layer 60: 0.858 / 0.857 / 0.827). In whitened coordinates every word loses most of itself in one
   linear round trip, and the differences between words are small.
4. **English bias has the opposite sign to the prediction, consistently.** bias = E(English predicted from x) − E(x predicted
   from English), held-out: es **−0.007, −0.011, −0.023, −0.029, −0.036** and hi **−0.043, −0.055, −0.098, −0.107, −0.116**
   at layers 20–60 (standardised: es −0.010 to −0.040, hi −0.045 to −0.120). The sign never changes, the gap grows with depth,
   and it is about 3–4 times larger for Hindi. So the other language is *more* predictable from English than English is from
   it. Read in the handoff's terms ("English keeps more of itself in the shared space" = positive), this **refutes** the
   prediction; read differently, non-English representations are largely English-explainable while English keeps extra
   variance of its own, which is still an English-centred asymmetry, and stronger for the lower-resource language. Both
   readings should be put to the father; the numbers are not in doubt.
5. **Probes (es, layer 40) — refuted as predicted, informative only with the strict definition.** With the null-based
   "shared" set every probe's shared share is 1.00 by construction (shared ≈ all dimensions), and private-only accuracy is at
   chance because "private" is two dimensions. With shared = held-out ρ > 0.71 (500 of 8192 dimensions, chance share 0.06):
   grammatical gender **0.24**, number **0.42**, subjunctive **0.10**, imperfect vs preterite **0.08**, noun vs verb (sanity)
   **0.65**. Prediction was "gender mostly private, tense mostly shared": gender is four times enriched on the *shared*
   directions, and tense is at chance (0.08 vs 0.06). Accuracy all / shared-only / private-only: gender 0.92 / 0.77 / 0.81,
   tense 0.84 / 0.76 / 0.81 — both readable from either subspace.
6. **Agreement with the round trip is weak:** Spearman of U with the round-trip A across es sentences **0.14** (layer 40),
   0.02 (layer 60), n = 19. The hi value (0.94) rests on 6 sentences.

## D. Mediation (N3) — null (and the test's own sanity check failed)

- **The sanity floor fails.** Mean-ablating a hot word's *whole* state at layer 40 changes the teacher-forced log-probability
  of the model's own translation by a median of only **0.0073** nats/token (control words 0.0010; layer 60: 0.0020). The
  translation barely depends on that word's state at a single layer, so ratios "private effect / whole effect" are mostly
  noise. (The scripts' headline "private-part share of harm: translation 82.04" is such a ratio over near-zero denominators;
  its 95% CI is −6.4 to 252.7. Do not quote it.)
- **With the strict split (private = 7692 of 8192 dimensions), no dissociation:** the private part carries most of whatever
  effect there is, on translation (median **0.86** of the whole effect for hot words, 0.65 for control) at least as much as on
  the model's explanation of the word (median **0.55**, control 0.43). "Understood but not translated" predicts the reverse.
  The explanation task is more sensitive (median whole effect 0.19 for hot, 0.08 for control words).
- **Interchange on sense minimal pairs is uninformative:** even patching the **whole** word state from the other sense moves the
  log-odds by at most **0.24** (frente, layer 40; the whole-state patch does not depend on the split, so this holds for both runs), and **no generated translation changes** (*el cura* stays
  "priest", *duende* stays "elf"). The positive control fails, so private vs shared cannot be compared. The sense cue (the
  article, the copula) is in other tokens, which the patch does not touch.

## Colab cost log

| session | what | time | units |
|---|---|---|---|
| rehearsal, T4 | Qwen 0.5B, all stages | 17:33–17:44, 21 Nov | ≈0.2 (est.) |
| A100 #1 | setup, A, B, C1-es; connection lost in C1-hi; the CLI hung until next day, the VM was gone | 18:12, 21 Nov – unknown | **12** (user's balance) |
| A100 #2 | background setup/A/B/C1 (106 min), kernel setup/C1-cache (1 min), C2 cell lost (30 min idle), C2+D+Z (30 min), strict probes + strict D (9 min) | 12:12–15:10, 22 Nov | **≈22** (2.97 h × 7.5/h, est.) |
| **total** | | | **≈34** — over the 25-unit cap agreed in the handoff; see below |

Per stage on the A100: setup 3 min (first load; 30 s from the VM's cache), A 5 s, B 3 min, C1 87 min (es 41, hi 48, incl.
alignment), C2 24 min, D 6 min, Z < 1 min, strict probes 1 min, strict D 5 min. **What went wrong:** the CLI's connection
dropped twice (once mid-C1, once at the start of C2), and `colab exec` neither returns nor errors when that happens; it
hung for 18 h the first time. The fix that worked: long stages run as a detached process (C1) or in a kernel thread (C2,
D, Z), polled over `colab ssh` / short `exec` calls with hard time limits. My estimate for the rerun (12–13 units) was
too low because C1 took longer than the handoff's estimate and 30 min were lost to the dropped C2 cell.

## Recommendation (HANDOFF_internals §10)

- **Keep, and scale: U (N1) as a word-level untranslatability score.** It beats the token-count baseline on es
  (0.78 vs 0.45) and is cheap once the covariances exist. Next: more annotated hot words (the hi test has 3, which decides
  nothing), and report U against frequency as well as token count.
- **Keep, with a changed definition: "shared".** Replace the null-based threshold (saturates at 8190/8192) with ρ > 0.71 or
  a ρ-weighted spectrum everywhere, including D. The handoff's definition should be amended before any further run.
- **Keep, and test on a second 70B: the English asymmetry.** Its sign is opposite to the prediction but perfectly consistent
  (every layer, both languages, raw and standardised, larger for Hindi). Qwen2.5-72B (stronger on Hindi) would show whether it
  is a property of Llama or of English-centred training in general. ≈14 units at today's C1 speed, if run with the
  detached-process driver from the start.
- **Drop in its current form: Part D.** The single-layer, single-word interventions barely move the translation (sanity
  floor failed) and the interchange's positive control fails. If revisited: patch the sense cue tokens too (article, copula),
  patch a band of layers rather than one, and measure on the generated text, not only log-probabilities.
- **Not tested here:** the SAE experiment (N2), which needs Gemma 2/3 27B with Gemma Scope and is not in these scripts.
- **Round trip (B):** on this model and sentence set the length-based U is flat (every category reaches ε at the standard
  translation); A and intended-sense recovery carry the signal. For the father's figures, sense recovery is the clearest
  measure of the "elf" problem (duende 0.30 at standard).
