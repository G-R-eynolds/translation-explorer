"""Phase 3 controls: style-only pairs and meaning-edit pairs -> runs/<model>/controls.json

    python controls.py --model 3b --restyle      # GPU: restyle whole translations line for line
    python controls.py --model 3b --assemble     # CPU: build controls.json

Style-only pairs.  Whole translations are restyled (Scots / London / formal / plain) WITHOUT the
source text in the prompt, so the model cannot "correct" interpretive choices.  Each restyled line
keeps its own restyled neighbours as context, exactly like a lattice rendering.  Restyles found to
change meaning on spot-check are listed in data/controls_spotcheck_<model>.json and dropped.

Meaning pairs.  (a) Minimal interpretive edits written by hand (data/meaning_edits_<model>.json):
one choice about the source changes, every other word stays, and the edited line keeps the original
line's context.  (b) Cross-cluster pairs of the hand-made demo clusters (weak reference).  Same-cluster
demo pairs are added as weak style pairs so the demo block is balanced; AUROC is reported with and
without the demo block.
"""
import argparse, itertools

from common import (DATA, MODELS, Timer, chat, describe, generate, load_json, load_model, parse_numbered,
                    run_dir, save_json)

STYLES = {
    "scots": "Lowland Scots vernacular",
    "london": "contemporary London vernacular",
    "formal": "formal, elevated literary English",
    "plain": "plain, simple, everyday modern English",
}

RESTYLE_PROMPT = """Here is an English translation of a poem, in {n} numbered lines.
Rewrite it in {style}, line for line. Keep exactly what each line says: the same people, things, actions, places and interpretive choices. Change only the wording, dialect and register. Do not correct, explain or reinterpret anything, even where a line seems odd. Keep exactly {n} numbered lines, "1." to "{n}.", line k rewriting line k.

{poem}

Reply with the {n} numbered lines only."""


def restyle(a):
    rd = run_dir(a.model)
    fam = MODELS[a.model]["family"]
    samples = load_json(rd / f"samples_{fam}.json")["samples"]
    by = {}
    for s in samples:
        by.setdefault(s["stance"], []).append(s)
    chosen = [s for v in by.values() for s in v[:a.per_stance]]
    tok, model = load_model(a.model)
    jobs = [(s, st) for s in chosen for st in STYLES if st != s["stance"]]
    out, fails = [], 0
    for s, st in jobs:
        n = len(s["lines"])
        prompt = RESTYLE_PROMPT.format(n=n, style=STYLES[st], poem="\n".join(f"{i + 1}. {l}" for i, l in enumerate(s["lines"])))
        lines = None
        for attempt in range(3):
            texts, _, _ = generate(model, tok, [chat(tok, prompt)], n=1, max_new_tokens=700,
                                   temperature=0 if attempt == 0 else 0.7, seed=attempt)
            lines = parse_numbered(texts[0][0], n)
            if lines:
                break
            fails += 1
        if not lines:
            print("restyle failed:", s["id"], st); continue
        out.append(dict(id=f"{s['id']}~{st}", source_sample=s["id"], source_stance=s["stance"], style=st,
                        lines=lines, decoding="greedy" if attempt == 0 else f"T=0.7 retry {attempt}"))
        print(s["id"], "->", st, "|", lines[3], "|", lines[10], flush=True)
    save_json(rd / "restyles.json", dict(model=describe(a.model), prompt_template=RESTYLE_PROMPT, styles=STYLES,
                                         parse_failures=fails, restyles=out))


def item(iid, sample, line, text, lines, stance, family, source):
    return dict(id=iid, sample=sample, line=line, text=text, prev=lines[line - 1] if line > 0 else None,
                next=lines[line + 1] if line + 1 < len(lines) else None, stance=stance, family=family, source=source)


def assemble(a):
    rd = run_dir(a.model)
    fam = MODELS[a.model]["family"]
    samples = {s["id"]: s for s in load_json(rd / f"samples_{fam}.json")["samples"]}
    rs = load_json(rd / "restyles.json")
    spot = load_json(DATA / f"controls_spotcheck_{a.model}.json")
    drop = {(d["restyle"], d["line"]) for d in spot["drop"]}
    edits = load_json(DATA / f"meaning_edits_{a.model}.json")["edits"]
    items, pairs = {}, []

    def add(it):
        items.setdefault(it["id"], it)
        return it["id"]

    for r in rs["restyles"]:
        s = samples[r["source_sample"]]
        for i, t in enumerate(r["lines"]):
            if (r["id"], i) in drop or t.strip() == s["lines"][i].strip():
                continue
            fam_id = f"{s['id']}:{i}"
            ia = add(item(f"{s['id']}:{i}", s["id"], i, s["lines"][i], s["lines"], s["stance"], fam_id, "sample"))
            ib = add(item(f"{r['id']}:{i}", r["id"], i, t, r["lines"], r["style"], fam_id, "restyle"))
            pairs.append(dict(id=f"style:{ib}", line=i, a=ia, b=ib, type="style", set="controls_ownctx",
                              provenance=f"restyle of {s['id']} into {r['style']} by {rs['model']} (no source in prompt); "
                                         "restyled line keeps its own restyled neighbours as BT context"))
            # Same restyled line in the ORIGINAL line's context: comparable to the meaning edits, which
            # also share context.  This is the main style set; the own-context set measures context leakage.
            ic = add(item(f"{r['id']}@ctx:{i}", f"{r['id']}@ctx", i, t, s["lines"], r["style"], fam_id, "restyle-samectx"))
            pairs.append(dict(id=f"style:{ic}", line=i, a=ia, b=ic, type="style", set="controls",
                              provenance=f"restyle of {s['id']} into {r['style']} by {rs['model']}, placed in the original's context"))
    for k, e in enumerate(edits):
        s = samples[e["sample"]]
        i = e["line"]
        assert e["original"].strip() == s["lines"][i].strip(), (e, s["lines"][i])
        fam_id = f"{s['id']}:{i}"
        ia = add(item(f"{s['id']}:{i}", s["id"], i, s["lines"][i], s["lines"], s["stance"], fam_id, "sample"))
        ib = add(item(f"{s['id']}#e{k}:{i}", f"{s['id']}#e{k}", i, e["edited"], s["lines"], s["stance"], fam_id, "edit"))
        pairs.append(dict(id=f"meaning:{ib}", line=i, a=ia, b=ib, type="meaning",
                          provenance="hand-written minimal interpretive edit (Claude, in session): " + e["change"]))
    # demo block: distances come from the 'demo' set (no extra items to score)
    demo = load_json(DATA / "clusters.json")
    for L in demo["lines"]:
        i = L["line"]
        lab = {m: c["id"] for c in L["clusters"] for m in c["members"]}
        for x, y in itertools.combinations(sorted(lab), 2):
            same = lab[x] == lab[y]
            pairs.append(dict(id=f"demo:{x}-{y}:{i}", line=i, a=f"{x}:{i}", b=f"{y}:{i}", set="demo",
                              type="style" if same else "meaning",
                              provenance=f"demo hand clusters ({'same' if same else 'different'} cluster; weak reference)"))
    for p in pairs:
        p.setdefault("set", "controls")
    save_json(rd / "controls.json", dict(
        description=__doc__, model=describe(a.model), restyle_prompt=rs["prompt_template"],
        spotcheck=spot, items=list(items.values()), pairs=pairs))
    from collections import Counter
    print(Counter((p["set"], p["type"]) for p in pairs), len(items), "items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--per-stance", type=int, default=2)
    ap.add_argument("--restyle", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    if a.restyle:
        with Timer("restyle"):
            restyle(a)
    if a.assemble:
        assemble(a)
