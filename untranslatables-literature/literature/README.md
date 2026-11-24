# Literature corpus: untranslatables, shared representations, English bias, uncertainty

Built 18 Nov 2026 to support the experiments in `../HANDOFF_untranslatables.md`.

| file | what |
|---|---|
| `papers.csv` | 71 arXiv papers: id, year, authors, title, thread (1 = representations/English bias, 2 = uncertainty, method), tags, and one line on why each matters |
| `references.bib` | BibTeX for all of them (authors marked TODO where the listing didn't show them) |
| `NOTES_thread1_universal_representations_and_english_bias.md` | synthesis plus design implications for thread 1 |
| `NOTES_thread2_separating_uncertainty.md` | synthesis plus design implications for thread 2 |
| `fetch_pdfs.py` | downloads every PDF into `pdfs/`, 3 s apart. **Run it on your own machine**: the cloud workspace couldn't reach arXiv |

```
cd literature && python fetch_pdfs.py          # ~4 minutes for all 71
python fetch_pdfs.py --thread 2                # just one thread
```

**Moving it to Google Drive:** once the Drive connector is connected in Claude, this whole folder can be uploaded there. Or drag it into Drive yourself.

**Caveats:**
- The papers were found through web search and read through their arXiv abstract or HTML pages. Thirteen were read in depth; the rest from their abstracts.
- Several are 2026 preprints and not peer-reviewed.
- Where authors show as "see arXiv", the listing didn't give them. Fill them in from the PDF.
- Key non-arXiv sources (Cassin, Jackson et al. 2019, Thompson et al. 2020, Anthropic's multilingual-circuits write-up, translation-process work on word translation entropy) are listed at the end of the thread-1 notes but are not downloaded.
