# Definitions (internals run)

Every quantity here is **model-relative and corpus-relative**: inside model M, relative to the translation conventions
embodied in corpus C (Europarl + OPUS-100 for es-en; OPUS-100 + Samanantar for hi-en).

## N1. Round-trip operator spectrum (HANDOFF_internals §3.1, verbatim)

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


### Implementation notes (what the code does where the text leaves a choice)
- Layer convention: layer l = hidden_states[l] (0 = embeddings, l >= 1 = output of block l-1). Gemma Scope "layer k" = l = k+1.
- Word state = mean of the word's subword tokens; words = `sent.data.WORD` matches; pairs = SimAlign itermax 1-to-1 links.
- One prefix token (BOS, or a newline for tokenizers without BOS) precedes every sentence and is never used.
- rho_k used in retention, U and half-life is the **held-out** canonical correlation, clipped to [0, 1]; directions are
  ordered by in-sample rho. "Shared" directions: held-out rho above the 99th percentile of the mismatched-pair null's
  held-out rho (all directions pooled).
- Ridge: lam * tr(Sigma)/d added to both covariances, lam from {1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3} maximising the sum
  of the top-50 held-out rho.
- English bias is reported three ways: the in-sample formula above (raw and standardised), and held-out R^2.

## N2. SAE features (HANDOFF_internals §4.2, verbatim)

For SAE feature f and word state x: aᶠ(x) ≥ 0 is the activation (summed over the word's subword tokens).

- **Language-identity features:** features whose activation separates es from en text with AUROC > 0.95 over the corpus. Tag them and **exclude them** from every untranslatability score: they mark *which language*, not *what meaning*.
- **Transfer of feature f** over aligned word pairs:
  - τᶠ = Σᵢ min(aᶠ(xᵢ), aᶠ(yᵢ)) / Σᵢ aᶠ(xᵢ), the share of f's source activation matched by f itself on the aligned target word;
  - **partner transfer** τᶠ* = max(τᶠ, best correlation over features g of aᶠ(x) with aᵍ(y) across aligned pairs). Meaning may move to a *different* English feature.
  - Use τᶠ* as the main score.
- **Span score:** U_SAE(x) = Σ_f aᶠ(x)(1 − τᶠ*) / Σ_f aᶠ(x), over non-language-identity features.
- **Instance view** (for the cards): for one source span and its translation, list the features that **persist** (active on both), **vanish** (active on source, absent on translation) and **appear** (absent on source, active on translation).

## N3. Causal mediation (HANDOFF_internals §5.1-5.2, verbatim)

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

## R2. Intended-sense recovery

sense recovery = mean log p(gloss | "<t>\n\nIn the original text, the word for '<span>' meant:") - the same with the
source sentence s in place of t; teacher-forced on the gloss; t has source words and close cognates masked. Because the
gloss is English, an English t that contains the gloss's words can score above 0 ("better than the source"): compare
rungs and span kinds, not raw values against 0.
