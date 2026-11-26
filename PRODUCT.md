# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Single-file zero-build HTML/CSS/JS (`untranslatables/explorer/static/workbench.html`), served by the local Python HTTP/SSE server (`untranslatables/explorer/server.py`) and compiled verbatim into standalone offline HTML showcases (`figures_for_dad/duende_workbench_offline.html`) via `window.__OFFLINE_DATA__`.

## Users

Primary user is the researcher/translator running Translation Explorer locally on a desktop browser (e.g., Windows + RTX 4070 with Gemma 3 12B NF4, or Colab L4/A100 with Gemma 3 27B NF4), exploring literary translations of Spanish/Portuguese/French/German poems and prose passages. Secondary audience is family and colleagues opening exported standalone offline HTML showcases in any browser with zero setup.

## Product Purpose

Turn LLM translation from a one-shot black box into a live, real-time co-translation studio. The user steps through a translation word-by-word or token-by-token alongside a quantized base model, seeing quiet linear flow on predictable words and rich interpretive alternatives—semantic branches, depth-2 sub-forks, 2D spectral steering, and continuous global tone/practice sliders—exactly where the poem genuinely forks.

## Positioning

Unlike standard chat translators or post-hoc editing tools, Translation Explorer operates directly inside the model's live persistent KV cache and residual stream: instant non-destructive rewind/redo, adaptive fork detection via a trained `ForkHead(n_dirs=2)` side-head (`<5 ms` 2D compass without backprop), and continuous multi-layer `SpectralReFT` tone axes—all preserving multi-stanza poetry formatting and line-by-line source↔target alignment.

## Operating Context

- Local desktop studio (wide screen, keyboard-first workflow: `Space`, `Tab`, `Backspace`, `1–6`, `Shift+1–6`, `C`, `S`, `H`, `Esc`, `Tab` indentation in source editor) during focused reading and translation sessions.
- Occasional sharing of self-contained offline `.html` files across models (`g3_27b_pt`, `g3_4b_pt`, `g3_12b_pt`) with saved Phase-1 archives, steering compasses, and recorded live co-translation sessions.
- Current pain points reported by the user after live testing:
  1. Too many boxy controls and stacked drawers competing with the poem and live translation text.
  2. Branches and depth-2 sub-forks are hard to scan and compare while stepping word-by-word.
  3. The 2D Spectral Compass and Global Tone Sliders feel buried or awkward to use alongside live stepping.
  4. The source poem and live translation do not yet feel like a calm, cohesive reading/writing surface.

## Capabilities and Constraints

- Must remain a single self-contained zero-build HTML/CSS/JS file (`untranslatables/explorer/static/workbench.html`) so live server mode and standalone offline HTML mode stay 100% unified.
- Must preserve all existing API endpoints (`/api/status`, `/api/session/*`, `/api/replay/*`), offline bootstrap (`window.__OFFLINE_DATA__`), multi-model switcher, keyboard shortcuts, and DOM contracts verified by the 158-test unit + Playwright E2E suite.
- Supports arbitrary-length multi-stanza formatted poems and paragraphs (`\n`, `\n\n`, indentation) with bounded scrolling and live source↔target line highlight sync.
- Must support both light and dark themes (`[data-theme="dark"]` / `[data-theme="light"]`).

## Brand Commitments

- Name: **Translation Explorer** (document title includes `Spectral Translation` for existing compatibility).
- Calm, scholarly, non-gimmicky voice—no AI hype, no decorative clutter, no nested card chrome.

## Evidence on Hand

- Live trained checkpoints & recorded sessions for `g3_4b_pt`, `g3_12b_pt`, and `g3_27b_pt` under `untranslatables/results/explorer/`.
- Multi-model standalone offline showcase at `figures_for_dad/duende_workbench_offline.html`.
- Complete quantitative comparison at `untranslatables/results/explorer/SCALE_COMPARISON.md`.

## Product Principles

1. **The poem and its unfolding translation own the viewport.** Controls, telemetry, and steering instruments serve the text and recede when unneeded.
2. **Adaptive visual calm.** Predictable words stay quiet; interpretive forks earn structured visual focus so branches and depth-2 sub-forks compare effortlessly at a glance.
3. **Instruments stay reachable without displacing the manuscript.** Compass, global tone sliders, and explored paths should work fluidly alongside stepping rather than pushing text off-screen in awkward stacked boxes.
4. **Keyboard speed, pointer grace.** Every stepping, branching, rewind/redo, and steering action feels instantaneous and crisp under both keyboard and mouse.
