---
name: Translation Explorer
description: Herbarium cartographic workbench for real-time literary co-translation and spectral steering.
colors:
  spruce-teal: "#186456"
  spruce-deep: "#135045"
  spruce-ink: "#ffffff"
  amber-ochre: "#965710"
  cinnabar-fork: "#a63227"
  cobalt-align: "#23527c"
  celadon-vellum: "#ebf0ea"
  specimen-paper: "#fafcfa"
  mineral-rail: "#f2f6f1"
  surface-hover: "#e5ece4"
  botanical-ink: "#17231e"
  slate-muted: "#465850"
  moss-subtle: "#51645c"
  hairline-rule: "#d3ddd4"
  hairline-strong: "#b7c6b9"
  botanical-chrome: "#15241e"
  chrome-ink: "#eef4f0"
  chrome-muted: "#9eb5aa"
  chrome-rule: "#253b32"
  chrome-surface: "#1e322a"
  dark-desk: "#0f1613"
  dark-paper: "#151f1b"
  dark-rail: "#111a16"
  dark-ink: "#e7efe9"
  dark-spruce: "#49b8a3"
typography:
  display:
    fontFamily: "Literata, Georgia, Times New Roman, serif"
    fontSize: "21px"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "-0.015em"
  headline:
    fontFamily: "Literata, Georgia, Times New Roman, serif"
    fontSize: "19.5px"
    fontWeight: 400
    lineHeight: 1.74
  title:
    fontFamily: "Public Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "13.5px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.005em"
  body:
    fontFamily: "Literata, Georgia, Times New Roman, serif"
    fontSize: "18.5px"
    fontWeight: 400
    lineHeight: 1.72
  label:
    fontFamily: "Public Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "12.5px"
    fontWeight: 500
    lineHeight: 1.2
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "11px"
    fontWeight: 500
    lineHeight: 1
rounded:
  sm: "4px"
  md: "6px"
  pill: "999px"
spacing:
  xs: "6px"
  sm: "10px"
  md: "16px"
  lg: "22px"
  xl: "28px"
components:
  button-primary:
    backgroundColor: "{colors.spruce-teal}"
    textColor: "{colors.spruce-ink}"
    rounded: "{rounded.sm}"
    height: "32px"
    padding: "0 11px"
  button-primary-hover:
    backgroundColor: "{colors.spruce-deep}"
    textColor: "{colors.spruce-ink}"
  button-secondary:
    backgroundColor: "{colors.specimen-paper}"
    textColor: "{colors.botanical-ink}"
    rounded: "{rounded.sm}"
    height: "32px"
    padding: "0 11px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.slate-muted}"
    rounded: "{rounded.sm}"
    height: "32px"
    padding: "0 11px"
  input-field:
    backgroundColor: "{colors.specimen-paper}"
    textColor: "{colors.botanical-ink}"
    rounded: "{rounded.sm}"
    height: "32px"
    padding: "0 10px"
  branch-row:
    backgroundColor: "transparent"
    textColor: "{colors.botanical-ink}"
    padding: "10px 8px"
  status-pill:
    backgroundColor: "{colors.chrome-surface}"
    textColor: "{colors.chrome-muted}"
    rounded: "{rounded.pill}"
    padding: "5px 10px"
---

# Design System: Translation Explorer

## Overview

**Creative North Star: "The Herbarium Cartographic Desk"**

Translation Explorer treats live literary co-translation like examining pressed botanical specimens on an archival mineral-paper desk beside engraved surveying instruments. The poetry and prose stay open, legible, and undisturbed across facing folios, while every steering tool—the 2D Spectral Compass, zero-centered bipolar tone sliders, and transit-interchange branch history—lives in a disciplined right-hand instrument rail that opens on demand and never competes with the words.

The visual voice rejects generic SaaS cards, heavy drop shadows, neon glows, and stacked dashboard chrome. Surface tone shifts (celadon-vellum desk, mineral rail, crisp specimen paper) and 1px hairline rules establish clear spatial hierarchy across both Daylight Vellum light mode and Evening Botanical dark mode, while a single committed dark botanical studio bar anchors the top of the workspace.

**Key Characteristics:**
- Open facing manuscripts separated by crisp 1px hairlines rather than nested card frames
- High-legibility literary serif typography (`Literata`) paired with restrained geometric UI chrome (`Public Sans`) and tabular monospace metrics (`JetBrains Mono`)
- Strictly semantic signal colors: spruce-teal for active steering and primary actions, amber-ochre for subtle choices, cinnabar for high-tension interpretive forks, and cobalt for bidirectional source-target line alignment
- Zero-build single-file HTML/CSS/JS delivery unified across both the live FastAPI studio and standalone offline archival exports

## Colors

A daylight mineral-vellum surface system anchored by a deep botanical studio header and calibrated semantic cartographic inks, with a full evening botanical dark theme.

### Primary
- **Spruce Teal** (`spruce-teal`): Primary action fill (`Start Co-Translating` before session start, `Step Word` once active, `Commit Word` in the compass), active live cursor caret, bipolar slider fill from center zero, and compass steering vector ray. Dark mode shifts to luminous `dark-spruce` for ≥6:1 legibility.
- **Spruce Deep** (`spruce-deep`): Hover state for primary buttons.

### Secondary
- **Cinnabar Fork** (`cinnabar-fork`): High-tension interpretive fork regime badge, entropy meter fill, fork-token underline on committed prose, and active token marker in offline replay.
- **Amber Ochre** (`amber-ochre`): Subtle interpretive choice regime badge and medium-tension meter fill.

### Tertiary
- **Cobalt Align** (`cobalt-align`): Bidirectional source-line and target-token alignment hover highlight so readers can trace verse correspondence without confusing alignment with forks or steering.

### Neutral
- **Celadon Vellum** (`celadon-vellum`): Outer studio desk backdrop (`dark-desk` in evening mode).
- **Specimen Paper** (`specimen-paper`): Primary reading and writing folio surface for the live translation canvas, active branch fan, and compass plate (`dark-paper` in evening mode).
- **Mineral Rail** (`mineral-rail`): Secondary structural surface for the open Source folio, unified transport strip, docked Instrument Rail, and replay compass rail (`dark-rail` in evening mode).
- **Botanical Ink** (`botanical-ink`): Primary prose and control ink (`dark-ink` in evening mode).
- **Slate Muted / Moss Subtle** (`slate-muted`, `moss-subtle`): Secondary metadata, line numbers, keyboard caption text, and ghost preview continuations, tuned to ≥4.5:1 contrast across both themes.
- **Hairline Rule / Strong Rule** (`hairline-rule`, `hairline-strong`): 1px structural dividers, branch row hairlines, bipolar slider tracks, and compass rings.
- **Botanical Chrome** (`botanical-chrome`, `chrome-surface`, `chrome-ink`, `chrome-muted`, `chrome-rule`): Committed dark studio bar framing the top navigation and mode controls across both light and dark modes.

### Named Rules
**The Single Primary Action Rule.** Only one solid Spruce Teal CTA occupies visual dominance in the main desk at a time: `Start Co-Translating` while composing source text, demoted to secondary `Restart` once live so `Step Word` commands focus.

**The Semantic Ink Rule.** Cinnabar, ochre, and cobalt are reserved strictly for fork tension, subtle choice, and cross-folio line alignment—never for decorative chrome or generic buttons.

## Typography

**Display & Reading Font:** Literata (with Georgia, Times New Roman, serif fallback)
**UI Chrome Font:** Public Sans (with system sans-serif fallback)
**Metrics & Keycap Font:** JetBrains Mono (with ui-monospace fallback)

**Character:** Literary warmth meets surveying precision. `Literata` carries all source poetry, live committed translation, ghost previews, and branch continuations with generous leading and crisp optical sizing, while `Public Sans` keeps controls quiet and scannable and `JetBrains Mono` aligns probabilities, entropies, coordinates, and keyboard shortcuts on tabular columns.

### Hierarchy
- **Display** (`400`, `21px`, `1.55`, italic): Archived source epigraph in the Replay / Offline Archive header.
- **Headline** (`400`, `19.5px`, `1.74`): Live translation canvas prose (`#translationCanvas`), preserving exact verse indentation and stanza breaks (`white-space: pre-wrap`).
- **Body** (`400`, `18.5px`, `1.72`): Facing source poem editor, aligned source folio, and offline continuous prose replay tokens.
- **Title** (`600`, `13.5px`, `1.2`): Column bar headings (`Source`, `Translation`, `Saved Explorer Archive`) and Instrument Rail drawer headers (`13px`).
- **Label** (`500`, `12.5px`, `1.2`): Buttons, language selectors, regime status badges, and slider pole labels (`11.5px`).
- **Mono** (`500–600`, `11px`, `1.0`, `tabular-nums`): Line numbers, fork probabilities, entropy readouts, compass coordinates, bipolar axis readouts, and `<kbd>` hints (`10px`).

### Named Rules
**The Prose-First Scale Rule.** Source and translation text stay at `18.5px–19.5px` Literata while all surrounding controls remain compact (`11.5px–13.5px` sans/mono), ensuring poetry dominates the eye before chrome.

**The Tabular Numerals Rule.** Every live metric—entropy (`H`), probability (`%`), coordinates `(x, y)`, and slider readout (`+0.6`)—uses `font-variant-numeric: tabular-nums` in `JetBrains Mono` so values never jitter horizontally during streaming or dragging.

## Layout

The workspace uses an open, full-bleed 3-zone grid constrained to `1640px` max-width and bounded by 1px hairline rules:

- **Botanical Studio Header** (`50px` min-height, sticky on desktop, `position: static` below `640px`): Brand mark + status pills on the left; language pair, `Sliders (S)`, `Paths (H)`, `Replay Archive`, and theme toggle on the right.
- **2-Column Facing Folios (Default):** `minmax(290px, 0.86fr)` Source folio (`mineral-rail`) facing `minmax(0, 1.14fr)` Live Translation desk (`specimen-paper`) across a single vertical hairline.
- **3-Column Desk with Docked Instrument Rail (`.rail-open`):** `minmax(270px, 0.78fr) minmax(0, 1.06fr) minmax(300px, 356px)` when Compass, Sliders, or Paths is toggled open.
- **Responsive Breakpoints:** At `≤1180px`, the Instrument Rail wraps cleanly into a full-width lower deck; at `≤860px`, the desk stacks into a single reading column; at `≤640px`, the header unsticks (`position: static`), keyboard `<kbd>` badges hide, and padding tightens to `14px–16px`.
- **Replay / Offline Archive Grid:** `minmax(0, 1.3fr) minmax(310px, 370px)` with a sticky right-hand 2D Steering Plane (`top: 68px`), collapsing to single column below `940px`.

## Elevation & Depth

This system is **flat by default** and conveys structural separation purely through tonal layering (`celadon-vellum` desk → `mineral-rail` secondary folios → `specimen-paper` active canvas) and crisp 1px hairline borders (`hairline-rule`). No drop shadows are applied to panels, drawers, or branch rows.

The only micro-shadows permitted are functional interactive feedback: the circular thumb of the bipolar slider (`0 1px 2px rgba(0, 0, 0, 0.24)`) and the high-contrast double-ring keyboard focus indicator (`0 0 0 2px var(--paper), 0 0 0 4px var(--spruce)`).

### Named Rules
**The Flat-By-Default Rule.** Never stack drop-shadowed cards inside a column. Separate open workspaces with surface tone shifts and 1px hairlines.

## Shapes

Restrained cartographic precision with tight corners (`4px` for buttons, inputs, branch keys, and history nodes; `6px` for editorial plates such as the compass frame, steered preview, and offline session cards; `999px` pills for status indicators, depth-2 twig chips, and probability bars).

All interactive icons use custom inline SVGs on a `16×16` grid with consistent `1.5px` stroke weight (`12×12` for branch twig arrows)—never raw typographic glyphs or mismatched emoji. Motion uses a single swift deceleration curve (`cubic-bezier(0.22, 1, 0.36, 1)`) at `120ms–200ms` and honors `prefers-reduced-motion: reduce`.

## Components

Tactile yet understated instruments calibrated for rapid keyboard and pointer co-translation.

### Buttons
- **Shape:** Tight crisp corners (`4px` radius, `32px` height in workspace, `30px` in studio bar).
- **Primary:** Solid Spruce Teal (`#186456`) with white text (`600` weight); darkens to Spruce Deep (`#135045`) on hover.
- **Secondary:** Specimen Paper fill with `1px` `hairline-rule` stroke and Botanical Ink text; subtle hover shift to `surface-hover` and `hairline-strong`.
- **Ghost:** Transparent at rest with `slate-muted` text, revealing hairline border on hover (`Copy`, `Mode: Word`, `Reset`, `Re-probe`).
- **Active Toggle:** Spruce wash fill + spruce border for open rail triggers (`Compass`, `Sliders`, `Paths`, active `Glide`).

### Chips & Pills
- **Status Pills:** Full-pill (`999px`) chrome badges in the top bar with a `6px` status dot (`Live` in luminous mint-spruce, `Offline Replay` in steel-blue).
- **Depth-2 Twig Chips:** Compact pill badges (`mineral-rail` fill, `1px` hairline) displaying the 2-step lookahead word in `12.5px` Literata beside its tabular mono probability, led by a `12×12` inline SVG branch twig.

### Cards / Containers
- **Open Facing Folios:** When a session is active, the Source folio sheds its textarea border and sits flush on `mineral-rail` directly beside the `specimen-paper` translation canvas.
- **Hairline Branch Fan:** Zero stacked boxes—each branch is a full-width `10px 8px` row divided by 1px bottom hairlines, showing its numbered color badge (`1–6`), bold lead word + muted continuation, inline depth-2 twigs, optional gloss, and right-aligned probability bar, with one quiet keyboard caption below the table.
- **Offline Session Spread:** Structured two-column facing `Source` (italic) / `Live Translation` spread above hairline `Explored Paths` rows and two-column `Global Tone Sliders` specimens.

### Inputs / Fields
- **Custom Word Input & Selects:** `32px` height, `4px` radius, `specimen-paper` fill, `1px` `hairline-rule`, shifting to `spruce-teal` border on focus with double-ring `:focus-visible`.
- **Bipolar Tone Sliders:** Custom transparent-track `<input type="range">` layered over a neutral hairline track with a `50%` center zero tick and a center-origin spruce fill bar that expands left for negative values, right for positive values, and collapses cleanly to `0%` width at neutral `0.0`.

### Navigation
- **Botanical Studio Bar:** Persistent dark forest-slate header (`#15241e`) separating global mode/language controls from the local desk transport strip (`Rewind`, `Redo`, `Step Word`, `Glide`, `Mode`, `↵`, `¶`, `Compass`, custom word input). On offline exports, live-only controls and their divider collapse cleanly away.

### Engraved Cartographic Compass Rose
- Circular SVG steering instrument (`viewBox="-1.45 -1.45 2.9 2.9"`) with concentric engraved rings (`r=0.5` dashed, `r=1.0` solid, `r=1.24` outer bezel), 16 radial compass ticks, clipped scatter clouds (`clipPath` at `r=1.20`), radial vector spokes, and outward-placed, vertically de-overlapped reading labels rendered via `color-mix(in srgb, <reading>, var(--ink))` with a `var(--paper)` stroke halo for ≥6:1 contrast in both light and dark modes.

### Transit-Interchange Branch Tree (Paths)
- Indented node list (`14px` per depth level) inside the docked Instrument Rail where each branched node (`depth > 0`) renders 1px hairline vertical stem and horizontal branch connectors (`::before` / `::after`) linking back to its parent path, with circular hollow/filled station dots and click-to-restore checkout.

## Do's and Don'ts

### Do:
- **Do** keep the active Source poem and live Translation canvas open and flush across a single 1px vertical hairline so both manuscripts read as a single facing spread.
- **Do** enforce the Single Primary Action Rule—demote `Start Co-Translating` to secondary `Restart` once a session is active so `Step Word` is the unmistakable primary action.
- **Do** render every bipolar axis slider from a true center zero tick (`0.0` = zero colored fill) so neutral axes never falsely appear half-pulled toward the negative pole.
- **Do** clip compass scatter dots inside the `r=1.20` bezel and mix reading label colors toward `var(--ink)` with a `var(--paper)` halo so dark mode stays crisp and legible.
- **Do** use custom stroke-1.5 inline SVGs for transport and drawer icons (`↵`, `¶`, sliders, twigs) while preserving accessible `aria-label` attributes and all keyboard shortcuts.

### Don't:
- **Don't** wrap every branch option or active source poem in a drop-shadowed card box—use 1px hairline rows and tonal surface layering instead.
- **Don't** repeat identical keyboard-hint copy on every branch row; place one quiet caption line beneath the branch list.
- **Don't** dump recorded offline sessions as raw plain text with ASCII `— Divider —` lines or flex-gapped subword chips that break words like `Camarón` or `duende` apart.
- **Don't** keep the top header sticky below `640px` where wrapped controls would consume vertical reading height on mobile screens.
- **Don't** introduce external build steps or CSS frameworks that break single-file standalone offline HTML export.
