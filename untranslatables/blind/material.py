"""Test material (§4): validation of the YAML files and the atlas corpus build (CPU).

  python -m blind material            check every set, print counts and problems
  python -m blind atlas_corpus        (re)build data/atlas.yaml from FLORES-101 Spanish devtest (seeded)

Atlas source: FLORES-101 devtest, Spanish (gsarti/flores_101 'spa', Hugging Face parquet conversion, stored in
data/sources/flores101_spa_devtest.parquet).  FLORES+ (openlanguagedata/flores_plus) is gated and access was not
granted on the desktop account; FLORES-101 devtest is the same professionally translated Wikinews / Wikivoyage /
Wikibooks material.
"""
import random
import re
import unicodedata
from collections import Counter

import yaml

from blind.common import BDATA, TARGETS, load_atlas, load_builtin, load_e2, load_e3


def _norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9ñ ]", "", "".join(c for c in s if not unicodedata.combining(c)))


def _shingles(s, n=5):
    s = _norm(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def build_atlas(cfg):
    import pandas as pd
    n_want, seed = cfg.get("n", 200), cfg.get("seed", 0)
    d = pd.read_parquet(BDATA / "sources" / "flores101_spa_devtest.parquet")
    d["nw"] = d.sentence.str.split().str.len()
    d = d[(d.nw >= 8) & (d.nw <= 25)]
    d = d[~d.sentence.str.contains(r"https?://|www\.", regex=True)]
    rng = random.Random(seed)
    rows = d.to_dict("records")
    rng.shuffle(rows)
    # round-robin over domains so that the 200 are varied (wikinews / wikivoyage / wikibooks), then near-duplicate
    # exclusion: character 5-gram Jaccard > 0.5 with an already chosen sentence.
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r)
    chosen, sh = [], []
    while len(chosen) < n_want and any(by.values()):
        for dom in sorted(by):
            if not by[dom] or len(chosen) >= n_want:
                continue
            r = by[dom].pop()
            s = _shingles(r["sentence"])
            if any(len(s & t) / len(s | t) > 0.5 for t in sh):
                continue
            chosen.append(r); sh.append(s)
    out = dict(source="FLORES-101 devtest, Spanish (gsarti/flores_101 spa, HF parquet conversion); "
                      f"8-25 words, no URLs, round-robin by domain, near-duplicates (char 5-gram Jaccard > 0.5) "
                      f"excluded, seed {seed}",
               sentences=[dict(id=f"at_{int(r['id']):04d}", text=r["sentence"], domain=r["domain"],
                               topic=r["topic"]) for r in sorted(chosen, key=lambda r: int(r["id"]))])
    p = BDATA / "atlas.yaml"
    p.write_text("# E4 atlas corpus (HANDOFF_blind_spots.md §4.3).  Built by `python -m blind atlas_corpus`.\n"
                 + yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200))
    print(f"atlas: {len(chosen)} sentences; domains {Counter(r['domain'] for r in chosen)}")
    return [p]


def check(cfg=None):
    issues = []
    b, types = load_builtin()
    for i in b:
        if i.group not in types:
            issues.append(f"{i.id}: unknown type {i.group}")
        try:
            i.spans()
        except ValueError as e:
            issues.append(str(e))
        if i.flip:
            if not re.search(rf"(?<!\w){re.escape(i.flip[0])}(?!\w)", i.text, re.I):
                issues.append(f"{i.id}: flip 'from' word {i.flip[0]!r} not in the text")
        elif i.group != "lexical":
            issues.append(f"{i.id}: needs a flip")
    sets = dict(e1=b, e2=load_e2(), e3=load_e3())
    try:
        sets["atlas"] = load_atlas()
    except FileNotFoundError:
        issues.append("data/atlas.yaml missing: run `python -m blind atlas_corpus`")
    for name, items in sets.items():
        ids = [i.id for i in items]
        for dup in {x for x in ids if ids.count(x) > 1}:
            issues.append(f"{name}: duplicate id {dup}")
        for i in items:
            if i.lang not in TARGETS:
                issues.append(f"{i.id}: no targets for source language {i.lang}")
            try:
                i.spans()
            except ValueError as e:
                issues.append(str(e))
        print(f"{name}: {len(items)} items  {dict(Counter(i.group for i in items))}")
    for p in [i for i in sets["e2"] if i.parts]:
        if len(p.parts) < 3:
            issues.append(f"{p.id}: a passage needs 3-4 sentences")
    print("problems:" if issues else "no problems found", *issues, sep="\n  ")
    return []
