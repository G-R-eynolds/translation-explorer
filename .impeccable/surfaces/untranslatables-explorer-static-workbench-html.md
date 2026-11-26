---
version: 1
slug: "untranslatables-explorer-static-workbench-html"
primary_target: "untranslatables/explorer/static/workbench.html"
related_targets: ["figures_for_dad/duende_workbench_offline.html"]
---

# Surface Brief: Translation Explorer Studio

- **Target:** `untranslatables/explorer/static/workbench.html` (also compiled into `figures_for_dad/duende_workbench_offline.html`)
- **Mode:** Operate (with Read-grade literary typography for facing poems and prose)
- **Audience & Job:** Researcher/translator stepping through literary translations word-by-word on desktop (local RTX 4070 / Colab L4) and sharing standalone offline HTML archives with family and colleagues.
- **Constraints:** Single self-contained zero-build HTML/CSS/JS file; preserve all `/api/*` and `window.__OFFLINE_DATA__` contracts, keyboard shortcuts (`Space`, `Tab`, `Backspace`, `1–6`, `Shift+1–6`, `C`, `S`, `H`, `Esc`), light & dark themes, and all 158 existing unit + Playwright E2E test selectors.

## Direction contract

**THESIS:** Replace stacked boxed cards and buried drawers with an open **Herbarium Cartographic Desk**—facing source and unfolding translation manuscripts on a calm mineral celadon-vellum spread, paired with a persistent docked right-hand Instrument Rail (Compass · Sliders · Paths) so steering never displaces the poem.

**OWN-WORLD:** Mineral celadon-vellum desk (`#EBF0EA`), crisp specimen-paper manuscripts (`#FAFCFA`), deep botanical forest-slate studio bar (`#162620`) with frost-white controls, primary spruce-teal (`#1B6B5C`) for steering and committed focus, mineral ochre (`#B36B1E`) for subtle uncertainty, cinnabar (`#B33A2E`) strictly for complex interpretive forks, and calibrated evening botanical dark mode (`#111916` / `#17221E` / `#5FD4BE`). Typography uses **Literata** (optical-size literary serif) for bilingual poetry and prose, **Public Sans** for crisp Operate UI controls, and **JetBrains Mono** strictly for tabular coordinates, probabilities, and keycaps.

**STORY:** The user pastes or picks a formatted poem, hits Start, and immediately sees facing source lines and the unfolding translation with zero card clutter; predictable words stay quiet and linear while genuine forks expand into crisp inline branch rows with depth-2 twig continuations (`→ word %`) directly beneath the cursor, and the docked Instrument Rail keeps the `<5 ms` engraved Compass Rose, continuous Tone Sliders, and transit-interchange Paths tree one click or keystroke away.

**FIRST VIEWPORT:** Full-width 48px deep botanical studio bar (`#162620`) carrying brand, status/model pills, compact language pair, rail toggles (`Sliders`, `Paths`, `Compass`), mode switch, and theme toggle. Directly below: a full-bleed three-zone open desk divided by 1px mineral hairlines—Left (32%): Source Folio (editable textarea or line-aligned reading view); Center (44%): Living Translation Canvas + quiet hairline transport bar + inline Adaptive Branch Fan with depth-2 twigs; Right (24%, collapsible/tabbed): Docked Instrument Rail housing the engraved Compass Rose, continuous Tone Sliders, and branching Paths tree. Primary action (`Start Co-Translating` / `Step Word`) sits in spruce-teal directly below the active manuscript.

**FORM:** Herbarium Cartographic Desk — candidate 5 of 7 on the ordered grounded list, `seed:67cf23a11b1a4f0d`.

**FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
