# explorer/: interpretation explorer, step 1

- Spec: `../docs/superpowers/specs/2026-11-24-interpretation-explorer-design.md`
- Plan: `../docs/superpowers/plans/2026-11-24-interpretation-explorer.md`

Question: does the base model (`google/gemma-3-4b-pt`) hold several distinct readings of a hard text, and does a direction found from its own sensitivities, added from the cursor onward, steer it into a chosen one?

```
remote/run.sh explorer bash explorer/queue.sh     # desktop 4070
python -m explorer report --model g3_4b_pt        # results/explorer/g3_4b_pt/report.html
.venv-cpu/bin/python -m pytest -q tests/test_explorer.py
```

| file | does |
|---|---|
| `frame.py` | few-shot "Spanish: / English:" frame, stop rules, loader (image encoder stubbed) |
| `steer.py` | `CursorHook`: adds a vector at positions ≥ b, decode steps included (or at b only) |
| `sample.py` | sampling (true distribution, T = 1), scoring, steered continuation |
| `readings.py` | meaning-in-context embeddings, silhouette-chosen agglomerative clusters, readings (≥ 5% of samples) |
| `branches.py` | per-token entropy and output-Fisher top eigenvalue; the cursor where two readings split |
| `directions.py` | per-sample gradients at the cursor; natural (Fisher-whitened) and plain contrast directions; α from a target KL |
| `run.py` | the pipeline (resumable per text) and `summary.json` with the pre-registered decision rule |
| `report.py` | static HTML: reading cards, branch timeline, steering table |

**Labelling readings:** read the cards in the report, then write `data/readings_labels.yaml` (`{text_id: {reading_id: label}}`). The code never names a reading.

**Changes from the spec, made during the build:**
- **9 texts instead of 8:** the bare *duende* sentence was added.
- **Cursor convention:** "steer at output token t" acts on the state at absolute position start + t − 1, the state that predicts token t.
- **Reading representation, changed after the first three texts:** readings now come from a semantic sentence embedder (all-mpnet-base-v2), centred on each text's own samples, with the lowest-probability 10% of samples left out.
  - *What went wrong:* the spec's representation (the base model's own mean final-layer state) merged *spirit*, *charm* and *elf* into one cluster and split off only a garbage sample.
  - *Evidence for the change:* on the saved samples, the new representation separates spirit/soul, magic/fairy, charm/charisma and keep-*duende*/flame for the flamenco sentence, and elf/gnome/goblin/dwarf/fairy for the bare one.
  - *Caveat:* some clusters are synonym choices rather than different interpretations, and the report says so.
- **Fork rule, changed twice during the build:** the fork is now the token where the reference commits to its reading. That is the token of the reference most specific to its reading's members compared with the other readings' members ("magic"). The other readings' continuations are aligned to that point by sequence alignment.
  - *Why:* exact-prefix rules either fired on stray first tokens (t = 0) or never fired, because punctuation variants break exact prefixes.
- **Steering strength, changed after the first run:** α is now calibrated by measuring KL, by bisection, not by the single-position Fisher prediction. That prediction badly underestimated cursor mode, where the vector is added at every later position, and the text degenerated.
  - *Grids:* at_b mode uses KL at the cursor ∈ {0.25, 0.5, 1, 2} nats. Cursor mode uses mean per-token KL along the reference ∈ {0.05, 0.1, 0.25, 0.5}.
  - *Decision rule:* it uses the three lower levels of each grid.
