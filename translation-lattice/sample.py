"""Phase 1: sample line-for-line translations (and the Romanesco -> Italian sanity check).

    python sample.py --model 3b --paraphrase            # runs/3b/paraphrase_it.json
    python sample.py --model 3b --per-stance 12         # runs/3b/samples_llama.json (+ _logprobs.json)
"""
import argparse

from common import (DATA, NUM_RE, Timer, chat, describe, generate, load_json, load_model, MODELS, parse_numbered,
                    run_dir, save_json, score_continuations)

SAMPLE_PROMPT = """Translate this {language} sonnet by {author}, "{title}" ({date}), into English.
Stance: {desc}.
Translate strictly line for line: your translation must have exactly {n} numbered lines, "1." to "{n}.", and line k must render source line k. Make your own interpretive choices; do not give alternatives, notes or commentary.

{poem}

Reply with the {n} numbered English lines only."""

# Written in Italian: with an English instruction the 3B model answered in English.
PARAPHRASE_PROMPT = """Questo è un sonetto in romanesco (il dialetto di Roma dell'Ottocento) di {author}, "{title}" ({date}).
Riscrivi ogni verso in italiano standard moderno, verso per verso, in modo che lo capisca chi non conosce il romanesco. Mantieni la numerazione da "1." a "{n}.". Rispondi solo con i {n} versi numerati, in italiano.

{poem}"""


def poem_block(src):
    return "\n".join(f"{i + 1}. {l}" for i, l in enumerate(src["lines"]))


def fill(tmpl, src, **kw):
    return tmpl.format(language=src["language"], author=src["author"], title=src["title"], date=src["date"],
                       n=len(src["lines"]), poem=poem_block(src), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    ap.add_argument("--per-stance", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1831)
    ap.add_argument("--max-tries", type=int, default=3, help="attempts per sample (1 + retries)")
    ap.add_argument("--paraphrase", action="store_true")
    a = ap.parse_args()
    out = run_dir(a.model)
    src = load_json(DATA / "source.json")
    n = len(src["lines"])
    tok, model = load_model(a.model)

    if a.paraphrase:
        prompt = fill(PARAPHRASE_PROMPT, src)
        with Timer("paraphrase"):
            texts, _, _ = generate(model, tok, [chat(tok, prompt)], n=1, temperature=0, max_new_tokens=800)
        raw = texts[0][0]
        save_json(out / "paraphrase_it.json", dict(model=describe(a.model), decoding="greedy", prompt=prompt,
                                                    raw=raw, lines=parse_numbered(raw, n), source=src["lines"]))
        print(raw)
        return

    stances = load_json(DATA / "samples.json")["stances"]
    fam = MODELS[a.model]["family"]
    samples, logprobs, attempts_log = [], [], []
    prompts = {s: fill(SAMPLE_PROMPT, src, desc=d) for s, d in stances.items()}
    for si, (s, desc) in enumerate(stances.items()):
        ptxt = chat(tok, prompts[s])
        need, k, attempt = a.per_stance, 0, 0
        while need > 0 and attempt < a.max_tries:
            # Over-generate a little on retries so a few failures don't need another round.
            with Timer(f"{s} attempt {attempt}: {need} samples"):
                texts, ids, trunc = generate(model, tok, [ptxt], n=need, temperature=a.temperature,
                                             top_p=a.top_p, seed=a.seed + 1000 * attempt + 17 * si,
                                             max_new_tokens=700, batch=12)
            for t, g, tr in zip(texts[0], ids[0], trunc[0]):
                lines = parse_numbered(t, n)
                attempts_log.append(dict(stance=s, attempt=attempt, ok=lines is not None, truncated=tr,
                                         raw=t if lines is None else None))
                if lines is None or need == 0:
                    continue
                k += 1; need -= 1
                sid = f"{s[:2].upper()}{k:02d}"  # LI, RH, SC, LO (L would collide)
                samples.append(dict(id=sid, stance=s, lines=lines))
                logprobs.append(dict(id=sid, ids=g, raw=t))
            attempt += 1
        if need:
            print(f"WARNING: {s}: {need} samples still missing after {a.max_tries} attempts")

    # Per-token log-probs of each accepted sample under its own prompt (raw model distribution, T=1,
    # no top-p), scored on the exact generated token ids.
    with Timer("logprobs"):
        for lp in logprobs:
            s = next(x["stance"] for x in samples if x["id"] == lp["id"])
            ptxt = chat(tok, prompts[s])
            total, per_tok = score_continuations(model, tok, ptxt, [lp["raw"]], append_stop=True, per_token=True)
            lp.update(stance=s, total_logprob=float(total[0]), tokens=[tok.decode([t]) for t in lp["ids"]] + ["<eot>"],
                      token_logprobs=per_tok[0])
            # per-line sums: token -> raw output line (by newlines before it) -> numbered line k
            raw_lines = lp["raw"].split("\n")
            num_of = {r: int(m.group(1)) - 1 for r, l in enumerate(raw_lines) if (m := NUM_RE.match(l))}
            line_lp, pos = [0.0] * n, 0
            for t, v in zip(lp["tokens"][:-1], per_tok[0]):
                k = num_of.get(lp["raw"][:pos].count("\n"))
                if k is not None and 0 <= k < n:
                    line_lp[k] += v
                pos += len(t)
            lp["line_logprob_approx"] = line_lp
            del lp["ids"]

    n_ok = sum(x["ok"] for x in attempts_log)
    fail_rate = 1 - n_ok / max(1, len(attempts_log))
    print(f"line-alignment failures: {len(attempts_log) - n_ok}/{len(attempts_log)} = {fail_rate:.1%}")
    sampler = dict(model=describe(a.model), hf_id=MODELS[a.model]["hf_id"], quantisation=MODELS[a.model]["quant"] or "bf16",
                   temperature=a.temperature, top_p=a.top_p, top_k=None, seed=a.seed, per_stance=a.per_stance,
                   max_tries=a.max_tries, prompt_template=SAMPLE_PROMPT, prompts=prompts,
                   chat_template_note="tokenizer chat template, date_string pinned (common.CHAT_DATE)",
                   alignment_failure_rate=round(fail_rate, 4), attempts=len(attempts_log))
    save_json(out / f"samples_{fam}.json", dict(sampler=sampler, stances=stances, samples=samples))
    save_json(out / f"samples_{fam}_logprobs.json", logprobs)
    save_json(out / f"samples_{fam}_attempts.json", attempts_log)
    print(f"wrote {len(samples)} samples")


if __name__ == "__main__":
    main()
