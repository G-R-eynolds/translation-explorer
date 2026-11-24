"""Sample line-aligned translations from Claude and cluster each line by meaning.

Writes data/samples.json and data/clusters.json in the schema build.py expects.

    pip install anthropic scikit-learn numpy
    export ANTHROPIC_API_KEY=...
    python pipeline.py --per-stance 5          # sample + cluster
    python build.py                            # lattice + out/index.html

Swap the poem by editing data/source.json (any number of lines).
"""
import argparse, json, re, concurrent.futures as cf
from pathlib import Path
import anthropic

ROOT = Path(__file__).parent; D = ROOT / "data"
client = anthropic.Anthropic()

STANCES = {
    "literal": "Close, line-for-line English; no rhyme",
    "rhymed":  "Formal English verse, attempting rhyme",
    "scots":   "Lowland Scots vernacular",
    "london":  "Contemporary London vernacular",
}

def ask(model, prompt, temperature, max_tokens=4000):
    r = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                               messages=[{"role": "user", "content": prompt}])
    txt = r.content[0].text
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))

def sample_one(model, src, stance, desc, temperature):
    n = len(src["lines"])
    poem = "\n".join(f"{i+1}. {l}" for i, l in enumerate(src["lines"]))
    prompt = f"""Translate this {src['language']} poem ({src['author']}, "{src['title']}") into English.
Stance: {desc}.
Keep it strictly line-for-line: exactly {n} lines, line k of your translation renders line k of the source.
Make your own interpretive choices; do not hedge or give alternatives.

{poem}

Reply with only JSON: {{"lines": [ ...{n} strings... ]}}"""
    for _ in range(3):
        try:
            out = ask(model, prompt, temperature)
            if len(out["lines"]) == n:
                return out["lines"]
        except Exception as e:
            print("retry:", e)
    raise RuntimeError(f"could not get {n} aligned lines for {stance}")

def cluster_line(model, src, i, renderings):
    numbered = "\n".join(f"{k}: {t}" for k, t in renderings.items())
    prompt = f"""Source poem ({src['language']}):
{chr(10).join(src['lines'])}

We are looking at line {i+1}: "{src['lines'][i]}"
Here are independent English renderings of that line, keyed by id:
{numbered}

Group the renderings into clusters that MEAN the same thing (same interpretive choice about the source), ignoring differences of wording, dialect, register or rhyme that don't change what is being said. Two renderings belong together only if each entails the other as a reading of the source line.
Use as few clusters as honestly possible; one cluster is fine if they all agree.
For each cluster give a short id, and a gloss like "'Next!' (queue, waiting room)".
If there is more than one cluster, add a one-to-two sentence note on what in the source makes it branch (idiom, pun, cultural reference, ambiguity). Distinguish genuine multiplicity in the source from renderings that are simply mistaken.

Reply with only JSON:
{{"note": "...", "clusters": [{{"id": "...", "gloss": "...", "members": ["id", ...]}}]}}"""
    out = ask(model, prompt, 0.0)
    seen = [m for c in out["clusters"] for m in c["members"]]
    missing = set(renderings) - set(seen)
    if missing:  # judge dropped someone: give each a singleton rather than lose data
        out["clusters"] += [{"id": f"x{m}", "gloss": "(unassigned by judge)", "members": [m]} for m in missing]
    out["line"] = i
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--judge", default="claude-sonnet-5-5")
    ap.add_argument("--per-stance", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--skip-sampling", action="store_true", help="re-cluster existing samples.json")
    a = ap.parse_args()
    src = json.loads((D / "source.json").read_text())

    if not a.skip_sampling:
        jobs, samples = [], []
        with cf.ThreadPoolExecutor(8) as ex:
            for s, desc in STANCES.items():
                for k in range(a.per_stance):
                    jobs.append((f"{s[0].upper()}{k+1}", s, ex.submit(sample_one, a.model, src, s, desc, a.temperature)))
            for sid, s, f in jobs:
                samples.append({"id": sid, "stance": s, "lines": f.result()})
        (D / "samples.json").write_text(json.dumps({"sampler": f"{a.model}, T={a.temperature}",
            "stances": STANCES, "samples": samples}, ensure_ascii=False, indent=1))
        print(f"sampled {len(samples)} translations")

    smp = json.loads((D / "samples.json").read_text())
    with cf.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(cluster_line, a.judge, src, i, {s["id"]: s["lines"][i] for s in smp["samples"]})
                for i in range(len(src["lines"]))]
        lines = [f.result() for f in futs]
    (D / "clusters.json").write_text(json.dumps({"clusterer": f"{a.judge} as meaning-equivalence judge",
        "lines": lines}, ensure_ascii=False, indent=1))
    print("clustered; now run: python build.py")
