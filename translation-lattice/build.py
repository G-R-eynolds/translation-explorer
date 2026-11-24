"""Build a translation lattice from line-aligned samples + per-line meaning clusters.

Inputs (data/): source.json, samples.json, clusters.json  (schemas as produced by pipeline.py)
Outputs: out/lattice.json, out/index.html

    python build.py                                                    # the demo data, as before
    python build.py --samples runs/3b/samples_llama.json --clusters runs/3b/clusters_m1r.json --out runs/3b/out/m1r
"""
import argparse, json, re, sys, itertools
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).parent
D = ROOT / "data"
ap = argparse.ArgumentParser()
ap.add_argument("--source", default=str(D / "source.json"))
ap.add_argument("--samples", default=str(D / "samples.json"))
ap.add_argument("--clusters", default=str(D / "clusters.json"))
ap.add_argument("--out", default=str(ROOT / "out"))
ap.add_argument("--display-top", type=int, default=0,
                help="display only: keep the N largest clusters with >= --display-min samples per line and merge "
                     "the rest into one 'other readings' node (0 = show every cluster)")
ap.add_argument("--display-min", type=int, default=3)
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

src = json.loads(Path(args.source).read_text())
smp = json.loads(Path(args.samples).read_text())
clu = json.loads(Path(args.clusters).read_text())
if args.display_top:
    for L in clu["lines"]:
        cs = sorted(L["clusters"], key=lambda c: -len(c["members"]))
        keep = [c for c in cs if len(c["members"]) >= args.display_min][:args.display_top]
        rest = [c for c in cs if c not in keep]
        if rest:
            keep.append(dict(id="unshared", members=[m for c in rest for m in c["members"]],
                             gloss=f"other readings ({sum(len(c['members']) for c in rest)} samples in {len(rest)} clusters)"))
        L["clusters"] = keep
    clu["clusterer"] += (f"; DISPLAY ONLY: top {args.display_top} clusters with >= {args.display_min} samples per line, "
                         "the rest merged into 'other readings' (counts and bits on this page use the merged view; "
                         "see results.md for the real ones)")
samples = smp["samples"]
ids = [s["id"] for s in samples]
stance = {s["id"]: s["stance"] for s in samples}
n_lines = len(src["lines"])

# ---- validate -------------------------------------------------------------
for s in samples:
    assert len(s["lines"]) == n_lines, f"{s['id']} has {len(s['lines'])} lines"
assign = {}  # (line, sample) -> cluster id
for L in clu["lines"]:
    seen = Counter(m for c in L["clusters"] for m in c["members"])
    missing = set(ids) - set(seen); dup = [k for k, v in seen.items() if v > 1]
    assert not missing and not dup, f"line {L['line']}: missing {missing} dup {dup}"
    for c in L["clusters"]:
        for m in c["members"]:
            assign[(L["line"], m)] = c["id"]

norm = lambda t: re.sub(r"[^\w\s']", "", t.lower()).split()

def jaccard_dist(a, b):
    A, B = set(norm(a)), set(norm(b))
    return 1 - len(A & B) / max(1, len(A | B))

# ---- per-line stats -------------------------------------------------------
rows = []
for L in clu["lines"]:
    i = L["line"]
    texts = [s["lines"][i] for s in samples]
    surface = len({" ".join(norm(t)) for t in texts})
    lex_div = float(np.mean([jaccard_dist(a, b) for a, b in itertools.combinations(texts, 2)]))
    sem = [assign[(i, x)] for x in ids]
    k = len(L["clusters"])
    p = np.array(list(Counter(sem).values())) / len(ids)
    sem_entropy = abs(float(-(p * np.log2(p)).sum()))
    # lexical baseline: cluster the same lines by word overlap into k groups
    if k > 1:
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts).toarray()
        lex = AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit_predict(X)
        ari_sem = adjusted_rand_score(sem, lex)
        ari_stance = adjusted_rand_score([stance[x] for x in ids], lex)
        # does stance predict the semantic branch?  purity of clusters wrt stance
        stance_sem = adjusted_rand_score([stance[x] for x in ids], sem)
    else:
        ari_sem = ari_stance = stance_sem = None
    rows.append(dict(line=i, surface_variants=surface, lexical_diversity=round(lex_div, 3),
                     meaning_clusters=k, semantic_entropy_bits=round(sem_entropy, 3),
                     lexical_vs_meaning_ARI=None if ari_sem is None else round(ari_sem, 2),
                     lexical_vs_stance_ARI=None if ari_stance is None else round(ari_stance, 2),
                     stance_vs_meaning_ARI=None if stance_sem is None else round(stance_sem, 2)))

# ---- lattice --------------------------------------------------------------
nodes, edges = [], Counter()
for L in clu["lines"]:
    i = L["line"]
    for c in sorted(L["clusters"], key=lambda c: -len(c["members"])):
        variants = [{"id": m, "stance": stance[m], "text": next(s for s in samples if s["id"] == m)["lines"][i]}
                    for m in c["members"]]
        nodes.append(dict(key=f"{i}:{c['id']}", line=i, id=c["id"], gloss=c["gloss"],
                          weight=len(c["members"]), variants=variants))
for s in samples:
    for i in range(n_lines - 1):
        edges[(f"{i}:{assign[(i, s['id'])]}", f"{i+1}:{assign[(i+1, s['id'])]}")] += 1
paths = {s["id"]: [f"{i}:{assign[(i, s['id'])]}" for i in range(n_lines)] for s in samples}
# the central route takes the largest real reading; an 'unshared readings' lump only if nothing is shared
spine = [max((n for n in nodes if n["line"] == i), key=lambda n: (n["id"] != "unshared", n["weight"]))["key"]
         for i in range(n_lines)]
# the "most likely" reading assembled from the modal cluster at each line, using the
# sample that sits in the modal cluster most often as the voice
order = {"literal":0,"rhymed":1,"scots":2,"london":3}
score = lambda x: sum(paths[x][i] == spine[i] for i in range(n_lines))
best = sorted(ids, key=lambda x: (-score(x), order.get(stance[x], 9)))[0]
print("modal-membership scores:", {x: score(x) for x in ids})

sampler = smp["sampler"]
if isinstance(sampler, dict):  # the local-model sampler records a dict; the page shows a one-line summary
    sampler = f"{sampler['model']}, T={sampler['temperature']}, top-p={sampler['top_p']}, seed={sampler['seed']}"
lattice = dict(source=src, stances=smp["stances"], sampler=sampler, clusterer=clu["clusterer"],
               notes={L["line"]: L["note"] for L in clu["lines"]},
               nodes=nodes, edges=[dict(a=a, b=b, w=w) for (a, b), w in edges.items()],
               paths=paths, spine=spine, spine_voice=best, stats=rows)
(OUT / "lattice.json").write_text(json.dumps(lattice, ensure_ascii=False, indent=1))

tmpl = (ROOT / "template.html").read_text()
(OUT / "index.html").write_text(tmpl.replace("/*__DATA__*/null", json.dumps(lattice, ensure_ascii=False)))

# ---- console summary --------------------------------------------------------
print(f"{'line':>4} {'surf':>4} {'lexdiv':>6} {'k':>2} {'H(bits)':>7} {'lex~mean':>8} {'lex~stance':>10} {'stance~mean':>11}")
for r in rows:
    f = lambda v: "  -" if v is None else f"{v:5.2f}"
    print(f"{r['line']+1:>4} {r['surface_variants']:>4} {r['lexical_diversity']:>6.2f} {r['meaning_clusters']:>2} "
          f"{r['semantic_entropy_bits']:>7.2f} {f(r['lexical_vs_meaning_ARI']):>8} {f(r['lexical_vs_stance_ARI']):>10} {f(r['stance_vs_meaning_ARI']):>11}")
print("spine voice:", best)
