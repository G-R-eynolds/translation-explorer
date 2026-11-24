"""Download every paper in papers.csv from arXiv into ./pdfs/ (run on your own machine;
this workspace cannot reach arXiv). Polite: one request every 3 s, skips files already present.

    python fetch_pdfs.py            # all
    python fetch_pdfs.py --thread 1 # only thread 1 (or 2, method)
"""
import csv, time, sys, argparse, urllib.request
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--thread"); a = ap.parse_args()
here = Path(__file__).parent; out = here / "pdfs"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(here / "papers.csv")))
if a.thread: rows = [r for r in rows if r["thread"] == a.thread]
for r in rows:
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in r["title"])[:80].strip().replace(" ", "_")
    f = out / f"{r['arxiv_id']}_{safe}.pdf"
    if f.exists(): continue
    req = urllib.request.Request(r["pdf_url"], headers={"User-Agent": "lit-corpus/1.0 (personal research)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(f, "wb") as fh: fh.write(resp.read())
        print("ok ", r["arxiv_id"], r["title"][:70])
    except Exception as e:
        print("ERR", r["arxiv_id"], e, file=sys.stderr)
    time.sleep(3)
