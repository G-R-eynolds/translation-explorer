"""M1 step 1: sample m back-translations into standard Italian for every rendering ("unit").

The BT prompt contains the rendering and its neighbouring ENGLISH lines from the same translation,
and never the source text (otherwise every rendering copies the source back and distances collapse).

Sets of units:
  lattice   every (sample, line) of runs/<model>/samples_<family>.json
  demo      the 12 hand-written demo samples (data/samples.json), for ARI vs the hand clusters
  controls  items of runs/<model>/controls.json (style-only and meaning-edit pairs)
  pool      lattice samples of this model + --pool-with model's samples (for §8)

    python backtranslate.py --model 3b --set lattice --m 6
"""
import argparse

from common import DATA, MODELS, Timer, chat, describe, generate, load_json, load_model, run_dir, save_json

BT_PROMPT = """Translate one line of an English poem into standard Italian.

{context}Line to translate:
{text}

Reply with the Italian translation of that one line only, on a single line, with no quotes or notes."""


def bt_prompt(prev, text, nxt):
    ctx = ""
    if prev or nxt:
        ctx = "Context, the neighbouring lines (do NOT translate these):\n"
        ctx += (f"before: {prev}\n" if prev else "") + (f"after: {nxt}\n" if nxt else "") + "\n"
    return BT_PROMPT.format(context=ctx, text=text)


def units_from_samples(samples, group_prefix="line", lines=None, tag=""):
    out = []
    for s in samples:
        L = s["lines"]
        for i in (lines if lines is not None else range(len(L))):
            out.append(dict(uid=f"{tag}{s['id']}:{i}", sample=f"{tag}{s['id']}", group=f"{group_prefix}{i}", line=i,
                            text=L[i], prev=L[i - 1] if i > 0 else None, next=L[i + 1] if i + 1 < len(L) else None,
                            stance=s["stance"]))
    return out


def subset_samples(samples, n):
    """First n samples, taken round-robin across stances (for debug runs)."""
    if not n:
        return samples
    by = {}
    for s in samples:
        by.setdefault(s["stance"], []).append(s)
    out, k = [], 0
    while len(out) < n:
        for v in by.values():
            if k < len(v) and len(out) < n:
                out.append(v[k])
        k += 1
    return out


def build_units(model_key, which, lines=None, n_samples=None, pool_with=None):
    rd = run_dir(model_key)
    fam = MODELS[model_key]["family"]
    if which == "lattice":
        return units_from_samples(subset_samples(load_json(rd / f"samples_{fam}.json")["samples"], n_samples),
                                  lines=lines)
    if which == "demo":
        return units_from_samples(load_json(DATA / "samples.json")["samples"], "demo", lines=lines)
    if which == "controls":
        items = load_json(rd / "controls.json")["items"]
        return [dict(uid=it["id"], sample=it["sample"], group=it["family"], line=it["line"], text=it["text"],
                     prev=it["prev"], next=it["next"], stance=it["stance"]) for it in items
                if lines is None or it["line"] in lines]
    if which == "pool":
        other = load_json(run_dir(pool_with) / f"samples_{MODELS[pool_with]['family']}.json")["samples"]
        mine = load_json(rd / f"samples_{fam}.json")["samples"]
        return (units_from_samples(mine, "pool", lines=lines, tag=f"{model_key}/")
                + units_from_samples(other, "pool", lines=lines, tag=f"{pool_with}/"))
    raise ValueError(which)


def set_name(which, pool_with):
    return f"pool_{pool_with}" if which == "pool" else which


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--set", default="lattice", choices=["lattice", "demo", "controls", "pool"])
    ap.add_argument("--pool-with", default=None, choices=MODELS)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0, help="T_bt; score.py reads it from the bt file")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lines", type=str, default=None, help="comma-separated 0-based line indices (debug)")
    ap.add_argument("--n-samples", type=int, default=None, help="use only this many samples (debug)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    lines = [int(x) for x in a.lines.split(",")] if a.lines else None
    units = build_units(a.model, a.set, lines, a.n_samples, a.pool_with)
    tok, model = load_model(a.model)
    prompts = [chat(tok, bt_prompt(u["prev"], u["text"], u["next"])) for u in units]
    print(f"{len(units)} units x m={a.m}")
    # Pure ancestral sampling (T=1, no top-p/top-k): the MC JSD estimator needs samples from exactly
    # the distribution that score.py teacher-forces.
    with Timer("back-translation sampling"):
        texts, _, trunc = generate(model, tok, prompts, n=a.m, max_new_tokens=80, temperature=a.temperature,
                                   top_p=1.0, seed=a.seed, batch=96)
    n_tr = sum(sum(t) for t in trunc)
    for u, t, tr in zip(units, texts, trunc):
        u["bt"] = [x.strip() for x in t]
        u["bt_truncated"] = tr
    print(f"truncated (hit max_new_tokens): {n_tr}/{len(units) * a.m}")
    name = set_name(a.set, a.pool_with) + a.suffix
    save_json(run_dir(a.model) / "results" / f"bt_{name}.json",
              dict(model=describe(a.model), m=a.m, temperature=a.temperature, top_p=1.0, top_k=None, seed=a.seed,
                   prompt_template=BT_PROMPT, n_truncated=n_tr, units=units))
    for u in units[:3]:
        print(u["text"], "->", u["bt"][:3])


if __name__ == "__main__":
    main()
