"""Thread 2 sampling: translation spread under input manipulations (handoff §7.1-7.2), plus the iterative-
prompting re-asks used as an ignorance signal (§7.3, [2406.02543]).

For every thread-2 word (ordinary, untranslatable, polysemous, dialect_*) and each of its contexts, the model
is asked how it would translate the word *as used in that text* into English, 20 samples at T=1, under:
  bare           the text alone
  gloss          + a knowledge gloss in the source language (definition + example; never an English translation)
  gloss+sense_k  + the gloss and the clarification for sense k (words with >= 2 listed senses)
  gloss+context  + the gloss and 2-3 sentences of preceding text (contexts that have `extra`)
One factor changes at a time [2603.24967].  Iterative prompting: in the bare condition the first sample is
shown back to the model, which is asked again (5 samples); the flip rate is computed in ignorance.py.

    python spread.py --model 8b          # -> results/<model>/spread_samples.json
"""
import argparse

from ut.common import LANG_EN, MODELS, Timer, chat, describe, generate, load_model, out_dir, save_json
from ut.represent import unmark
from ut.words import THREAD2, load_words

ASK = 'How would you translate "{form}" as used in this text into English? Reply with the English word or short phrase only.'
AGAIN = ("Look at the word in this text again and give your translation once more. "
         "Reply with the English word or short phrase only.")


def build_prompt(w, text, form, gloss=False, clarify=None, extra=None):
    lang = LANG_EN[w.language]
    parts = []
    if gloss:
        parts.append(f"Dictionary note ({lang}): {w.gloss_src}")
    if clarify:
        parts.append(f"Note: {clarify}")
    parts.append(f"{lang} text: {(extra + ' ') if extra else ''}{text}")
    parts.append(ASK.format(form=form) + (f" Answer in English, even though the note is in {lang}." if gloss or clarify else ""))
    return "\n\n".join(parts)


def conditions(w):
    """Yields (condition name, context index, user prompt, sense index or None)."""
    for k, marked, extra, sense in w.t2_contexts():
        text, (s, e) = unmark(marked)
        form = text[s:e]
        yield "bare", k, build_prompt(w, text, form), sense
        yield "gloss", k, build_prompt(w, text, form, gloss=True), sense
        if len(w.senses) >= 2:
            for j, sn in enumerate(w.senses):
                yield f"gloss+sense_{j}", k, build_prompt(w, text, form, gloss=True, clarify=sn["clarify"]), sense
        if extra:
            yield "gloss+context", k, build_prompt(w, text, form, gloss=True, extra=extra), sense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=MODELS)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-again", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="word ids to run (pilots / smoke tests)")
    a = ap.parse_args()
    words = [w for w in load_words() if w.category in THREAD2 and (not a.only or w.id in a.only)]
    jobs = [(w, c, k, p, s) for w in words for c, k, p, s in conditions(w)]
    tok, model = load_model(a.model)
    print(f"{len(jobs)} prompts x {a.n} samples")
    with Timer("sampling"):
        outs = generate(model, tok, [chat(tok, p) for _, _, _, p, _ in jobs], n=a.n, temperature=1.0,
                        max_new_tokens=16, batch=128, seed=a.seed, stop_at_newline=True)
    bare = [(i, j) for i, j in enumerate(jobs) if j[1] == "bare"]
    again_prompts = [chat(tok, [{"role": "user", "content": j[3]}, {"role": "assistant", "content": outs[i][0]},
                                {"role": "user", "content": AGAIN}]) for i, j in bare]
    with Timer("iterative prompting"):
        again = generate(model, tok, again_prompts, n=a.n_again, temperature=1.0, max_new_tokens=16, batch=128,
                         seed=a.seed + 1, stop_at_newline=True)
    again_of = {i: (outs[i][0], ag) for (i, _), ag in zip(bare, again)}
    rows = [dict(word_id=w.id, word=w.word, language=w.language, category=w.category, cell=list(THREAD2[w.category]),
                 condition=c, context=k, sense_of_context=s, prompt=p, samples=outs[i],
                 again=dict(first=again_of[i][0], answers=again_of[i][1]) if i in again_of else None)
            for i, (w, c, k, p, s) in enumerate(jobs)]
    save_json(out_dir(a.model) / "spread_samples.json", dict(model=describe(a.model), n=a.n, temperature=1.0,
                                                             ask=ASK, again=AGAIN, rows=rows))


if __name__ == "__main__":
    main()
