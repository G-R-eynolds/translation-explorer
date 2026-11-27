# Untranslatability vs. Model Ignorance (Gemma 3 4B → 12B → 27B)

This note accompanies the three new figures in `figures_for_dad/` and tests two ideas side-by-side:

1. **Your father's intuition:** When a model knows a language well and hits a genuine untranslatable (*duende*, *saudade*, *sprezzatura*, *Sehnsucht*, *sobremesa*), it makes sharp contextual micro-shifts inside the source language and splits into coherent, steerable English clusters. When it is ignorant of a word or language (Romanesco dialect like *schertri* and *bbiocca*, or low-resource Welsh *hiraeth*, Basque *bihotz*, Yami *anito*), its source-side predictions are brittle and its steering is noisier.
2. **Hsieh & Lian (arXiv:2610.08303, Oct 2026):** Their paper formalises concepts as contextualised "usage clouds" and measures **Cycle-Consistency (CC)** (whether pairwise contextual distances survive round-trip translation) and the **Representation–Intervention Gap (RIG)** (whether encoded regularities can actually be steered).

---

## What the three figures show

### 1. `5_untranslatability_vs_ignorance.png` (Gemma 3 27B)
- **Horizontal axis (Source-Language Conceptual Sharpness):** Combines how cleanly the word's internal representation shifts across different sentences with how unsurprised the model is by the word in context.
- **Vertical axis (Cross-Lingual Dispersion & English Leftover):** Combines the NNOMP English-basis "leftover" share with how far apart the model's English translations scatter.
- **Marker size (Directional Steerability / Inverse RIG):** Larger markers mean the Output-Fisher top direction cleanly and fluently steers between distinct English readings.
- **Takeaway:**
  - *saudade* and *sobremesa* sit in the **top-right** (sharp in Portuguese/Spanish, high cross-lingual dispersion).
  - *volverán* and *bank* sit on the **far right** with lower cross-lingual leftover—confirming that *volverán* has high source precision and stylistic freedom without a deep conceptual gap.
  - Low-resource *anito* (Yami) and *schertri* (Romanesco) sit lower on sharpness with high source-language NLL (`7.68` and `9.00` nats vs. `3.73–4.78` for *Sehnsucht* / *duende* / *sobremesa*).

### 2. `6_scale_trajectories_4b_to_27b.png` (4B → 27B)
- Arrows trace each word from **Gemma 3 4B** (hollow tail) to **Gemma 3 27B** (solid head).
- Across the board, scaling up 4B → 27B increases cross-lingual structural nuance without collapsing genuine untranslatables into simple 1-to-1 equivalents.

### 3. `7_cycle_consistency_and_rig.png` (Testing Hsieh & Lian 2610.08303)
- Plots **Cycle-Consistency distortion (CC)** on the horizontal axis against our **Directional RIG proxy** on the vertical axis.
- **Key finding:** Low-resource ignorance items (*bbiocca* RIG `0.70`, *hiraeth* RIG `0.66`, *bihotz* RIG `0.56`) cluster toward the **top** (harder to steer cleanly), whereas well-represented concepts with multiple readings (*sprezzatura* RIG `0.00`, *bank* RIG `0.16`, *volverán* RIG `0.19`, *sobremesa* RIG `0.29`) are **far more causally steerable** even when Cycle-Consistency distortion is high (*sobremesa* CC `1.30`, RIG `0.29`).
- This refines Hsieh & Lian's claim: *cross-lingual unalignability (high CC) does not imply causal inoperability when you use Fisher-whitened directional steering on a well-known language.*

---

## Live Workbench Updates (`workbench.html` + `session.py`)
- **Stylistic Fork vs. Conceptual Gap:** The live frontier now computes `semantic_dispersion` ($\Delta$) across lookahead ribbons (`session.py` / `recipe_live.py`) to distinguish calm **Stylistic forks** (*volverán*, ochre) from **Conceptual gaps** (*duende*, cinnabar).
- **Inline English Ingredients & Leftover Bar:** On conceptual gaps (when `en_pool.npz` is present), the workbench displays the source word's top 5 English ingredients and `left over %` directly above the branches.
- **Sketch Space (Constellation Mode):** A **Sketch space** button lets you select 2+ competing branches and either **Keep as Chord** (`[balcony / ledge / sill]`) or **Blend Phrase** directly into the translation.
