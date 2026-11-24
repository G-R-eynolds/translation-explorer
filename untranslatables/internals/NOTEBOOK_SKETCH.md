# Colab notebook sketch (not built yet; HANDOFF_internals §11)

Every stage is `run(cfg) -> paths`, resumable (skipped when its outputs exist and the config hash matches), so a
cut-off session just re-runs the same cell.

1. **Install**: `!git clone <repo> && cd untranslatables && pip install -e . bitsandbytes` (the repo has no package build;
   use `pip install torch transformers accelerate bitsandbytes sae-lens datasets pyarrow pandas scipy scikit-learn
   spacy stanza pot wordfreq stopwordsiso pyyaml matplotlib` and the es_core_news_md wheel).
2. **Mount Drive** and point the roots at it, so caches survive the session:
   `os.environ.update(UT_RESULTS="/content/drive/MyDrive/ut/results", UT_CACHE="/content/drive/MyDrive/ut/cache",
   UT_CORPUS="/content/drive/MyDrive/ut/corpora", HF_TOKEN=...)`.
3. **Corpora + alignment (once, reusable across models)**:
   `from internals import data, align; data.build_corpus({"pair": "es-en"}); data.build_corpus({"pair": "hi-en"});
   data.build_flores({}); align.run({"what": ["es-en", "hi-en", "flores"]})`.
4. **Per model (GPU)**: `replicate.r1({"model": M})`, `collect.quick({"model": M, "pair": P})` for both pairs, then
   `collect.collect({"model": M, "pair": P})`, `collect.eval_states({"model": M})`.
   On an A100 80 GB, 70B-class models in 4-bit fit; set `cfg["batch"]` higher.
5. **Analysis (CPU or GPU; repeatable anywhere from the cached sums)**: `cca.run`, `spans.run`, `cca.hub`, `probes.run`,
   `probes.mlp`.
6. **N2 / N3 (GPU, main model)**: `sae.run({"model": "gemma27b"...})` once a 27B registry entry and SAE layers are added;
   `mediate.run`, `mediate.interchange`.
7. **Figures and report**: `figures.run({})`, `report.run({})`.

Cached artifacts worth keeping on Drive: `cache/<model>/n1/<pair>/acc_L*.pt` (streamed covariances: everything in N1
can be recomputed from these plus the held-out pairs), `basis_L*.npz` (canonical bases, reused by N2 and N3), and the
alignments in `corpora/`.
