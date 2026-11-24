# Pre-registered predictions (written 18 Nov 2026, before any run)

From the literature (`../literature/`, arXiv IDs in brackets) and the lattice run:

1. **Size:** Llama-3.2-3B is more English-shaped than Llama-3.1-8B: higher peak English share in §6.1, larger coverage asymmetry in §6.2 [2505.13141].
2. **Training mix:** Aya-23-8B is less English-shaped than both Llamas [2502.15603], [2507.20279]; Qwen2.5-7B sits between them (no strong prior).
3. **Direction:** for *ordinary* words, rebuilding a non-English word from English leaves a smaller leftover than rebuilding an English word from a non-English pool.
4. **Untranslatables:** leftover(untranslatable) > leftover(frequency-matched ordinary) and > leftover(emotion_plain) after frequency and token-count matching. *Risk:* B-freq or B-tok explains this as well as the leftover does [2601.22851].
5. **Baseline:** the recipe leftover separates untranslatable from ordinary at least as well as CSLS nearest-neighbour similarity (B-NN, [2605.24310], AUC ~0.76–0.81 there).
6. **Distinction collapse:** the error rate on english_lacks items (savoir/connaître, ser/estar) exceeds that on the english_keeps controls; error items show more middle-layer probability on the English collapsed word [2601.12549].
7. **Steering:** pushing away from English raises untranslatable leftovers and lowers distinction errors (if the lean hides untranslatability).
8. **Thread 2:** dialect items show low spread with wrong content until glossed (confident flattening [2601.01362]); the gloss lowers their entropy (high ignorance). Ordinary words: low ignorance, low multiplicity. Untranslatables / polysemous: low ignorance, high multiplicity that falls under sense clarification.
9. **Cross-model:** ignorance signals (gloss sensitivity, cross-model TVD, 3B-vs-8B, iterative flips) agree in rank across words.

Negative results on any of these will be reported as such.

---

# Internals run (`../HANDOFF_internals.md`), written 21 Nov 2026 before any internals run

Main model: **Gemma 3 12B IT** (`gemma12b`), chosen over Gemma 2 9B because Gemma Scope 2 has IT residual SAEs for it
(layers 12, 24, 31, 41, plus every layer in `resid_post_all`). Comparison models: Qwen2.5-14B, Mistral-Nemo-12B,
Aya-23-8B (substituting Aya Expanse 8B, whose licence is not accepted), Llama-3.1-8B, Sarvam-1 (hi only).
Each prediction gets an id; `results/internals/report.md` marks each confirmed / refuted / unclear with its statistic.

**R1 (detour).** P1. On every new model the logit-lens English share on non-English word translation peaks in the
middle-to-late layers above the target-language share (the Wendler pattern), with a peak share > 0.5.
**R2 (sentence round trip, masking default).** P2. Masked d_RT still separates categories: mean U(control) <
mean U(untranslatable) on gemma12b and qwen14b. P3. Borrowing (B2_borrow) buys a positive unmasked gain over B1 for
untranslatable sentences, larger than for controls. P4. Intended-sense recovery of the standard translation (B1) is
worse (more negative) for untranslatable hot spans than for control probe words, and B6 (thick translation) recovers
more of the intended sense than B1. P5. For *duende* the standard translation recovers the Lorca sense poorly
(sense recovery of B1 below the median of all hot spans).
**N1 (round-trip spectrum).**
- P6. Shared directions (held-out rho above the mismatched-pair null's 99th percentile) peak in the middle layers,
  at or within 4 layers of the R1 detour peak.
- P7. Span U separates annotated es hot-span words from control content words with AUROC > 0.6; the father's
  sentence orders duende > gusta > mucho; U(untranslatable) > U(control) by category; U correlates positively with
  the R2 round-trip U across es sentences (Spearman > 0).
- P8. English bias E_en - E_es > 0 and E_en - E_hi > 0 for every model (raw coordinates), with the hi gap larger
  than the es gap; the gap is smaller for Aya (and Sarvam on hi) than for Gemma, Llama and Qwen.
- P9. English is a hub: on the same FLORES sentences, es-hi has fewer shared directions and lower mean rho^2 than
  both es-en and hi-en.
- P10. Bhojpuri words scored with the hi-en maps have higher U than Hindi words on the same FLORES sentences.
- P11. Probes: grammatical gender is mostly private (shared share below r/d); number and tense mostly shared (above
  r/d); subjunctive and ser/estar partly private; the dative-experiencer construction private. Hindi: gender and the
  ne-ergative private, honorific aap/tum private, dative-subject private.
- P12. A 2-layer MLP recovers only a small part of the variance the linear map loses (held-out R^2 gain < 0.05); if it
  recovers more, the linear U overstates untranslatability and we will say so.
**N2 (SAE features).** P13. Feature-level agreement with N1: Spearman(rho-bar_f, tau*_f) > 0 across features.
P14. On *duende* (flamenco sense) art/performance features vanish in the English translation while folklore features
fire on both duende sentences; U_SAE separates hot spans from control words (AUROC > 0.6).
**N3 (mediation).** P15. Hot spans show the dissociation: the private component's effect on the Spanish-side sense
task (E_mono) exceeds its effect on the translation (E_trans) more than for control words. P16. Sense minimal pairs:
patching the private component moves the Spanish sense log-odds; for tu/usted (and aap/tum) it moves the Spanish
side but not the English (dropped); for gender pairs (el cura / la cura) whether it flips the English is open (no
directional prediction), and either result will be reported.

Negative results will be reported as such, in the first line of their section.
