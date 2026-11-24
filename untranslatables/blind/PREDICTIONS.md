# Blind spots: pre-registered predictions (HANDOFF_blind_spots.md §4.1, §5)

Written 23 Nov 2026, before any run. Not to be edited after E1 has run, except to add results below a dated line.

**Conventions.** Main model g3_4b. The headline layer is the 50% layer (17 for g3_4b; 24 for g3_12b; 14 for 3b; 16 for aya8b). The headline ε is 1e-2 relative. A result counts as **robust** if it also holds at at least 3 of the 4 stored layers and at ε = 1e-3 and 1e-1. U is the median over 20 matched splits. "Near the null" means U ≤ the 95th percentile of the split-null U distribution for that item.

**E0 (plumbing).**
- Source ids are identical across channels. Source-state Δ is 0, or at most 1e-3 of the state's RMS (bf16 kernel noise).
- Gradient vs finite differences: correlation > 0.9 at the best step size.
- Spanish-into-Spanish (second paraphrase wording): U near the null for at least 75% of items. If this fails, prompt sensitivity dominates: **stop and report**.

**E1 tests.**
1. Gender on adjectives, plural you and formal you (18 items pooled, and per type): U_en > U_it and U_en > U_pt. Paired one-sided Wilcoxon, p < 0.05; median difference with a bootstrap 95% CI.
2. Profession nouns: U_en > U_it and U_fr > U_it (Wilcoxon), with U_fr ≈ U_en (the CI of the difference includes 0).
3. Negative controls (ser/estar and lexical): U near the null into every language, for at least 75% of (item, language) pairs.
4. The steering card for the top blind direction into English flips the distinction in the Spanish paraphrase while the English stays the same, in at least half the items of each test type. The count is final only after reading the texts.
- Aspect (exploratory, no test): U_en > U_it, fr, pt, weaker than for gender.
- Expected direction for plural you into French: smaller than into English but above Italian, because *vous* conflates plural and formal. Same for formal you.

**E2.** *duende* has higher per-token U into English in the flamenco context than in the goblin context. The dense passage has a higher U than the ordinary one. A passage's U is less than the sum of its sentences' U.

**E3.** The gloss, footnote and thick practices make a larger share of the plain translation's blind directions "seen" (ρ ≥ 0.5) than keep-the-word does. Keep-the-word helps lexical items but not grammatical ones. This should hold after length normalisation.

**E5.** Grammatical blind spots (gender, plural you, formality) into English appear in all four models with similar ordering across types. Lexical and cultural ones vary more across models.

**Decision rule (stop point 2).** If tests 1–4 mostly fail, the definition is not measuring what we want: no atlas, and the report says so in its first line.
