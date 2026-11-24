"""Step 0: familiarity gate (handoff §4).

    python familiarity.py --model 3b      # GPU -> results/<model>/familiarity.json + familiarity_check.html

For every word: a definition asked for in its own language and an English explanation (greedy), its token
count, wordfreq Zipf frequency, and mean token log-prob inside its natural contexts.  Until a human ticks the
page, a PROVISIONAL flag is used: overlap between the English explanation and gloss_en (content words of
4+ letters, compared on their first 5 letters).  Ticks saved from the page go to
data/familiarity_ticks_<model>.json and override the provisional flag everywhere (status()).
"""
import argparse, html, json, re

import numpy as np

from ut.common import DATA, LANG_EN, MODELS, Timer, chat, describe, generate, load_json, load_model, out_dir, save_json
from ut.lm import span_logprob, token_logprobs
from ut.represent import unmark
from ut.words import load_words

DEF_PROMPT = {
    "pt": 'O que significa a palavra "{w}"? Responde numa ou duas frases, em português.',
    "de": 'Was bedeutet das Wort „{w}“? Antworte in ein oder zwei Sätzen auf Deutsch.',
    "it": 'Che cosa significa la parola "{w}"? Rispondi in una o due frasi, in italiano.',
    "fr": 'Que signifie le mot « {w} » ? Réponds en une ou deux phrases, en français.',
    "es": '¿Qué significa la palabra "{w}"? Responde en una o dos frases, en español.',
    "en": 'What does the word "{w}" mean? Answer in one or two sentences.',
    "rom": 'Che cosa significa l\'espressione romanesca "{w}"? Rispondi in una o due frasi, in italiano.',
}
EN_PROMPT = 'What does the {lang} word "{w}" mean? Explain in one or two sentences in English.'
STOP = set("that this with from which have being their there about into word means mean meaning used refers "
           "something someone often also when where what your more very such them they then than other".split())


def content_stems(text):
    return {t[:5] for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in STOP}


def provisional(expl, gloss):
    g = content_stems(gloss)
    ov = len(g & content_stems(expl)) / max(1, len(g))
    return ("yes" if ov >= 0.34 else "partly" if ov > 0 else "no"), round(ov, 3)


def status(model_key):
    """word id -> 'yes' | 'partly' | 'no', with the source ('human' or 'provisional')."""
    fam = load_json(out_dir(model_key) / "familiarity.json")
    p = DATA / f"familiarity_ticks_{model_key}.json"
    ticks = load_json(p)["ticks"] if p.exists() else {}
    return {wid: (ticks[wid], "human") if wid in ticks else (r["provisional"], "provisional")
            for wid, r in fam["words"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    a = ap.parse_args()
    from wordfreq import zipf_frequency
    words = load_words()
    tok, model = load_model(a.model)
    p_def = [chat(tok, DEF_PROMPT[w.language].format(w=w.word)) for w in words]
    p_en = [chat(tok, EN_PROMPT.format(lang=LANG_EN[w.language], w=w.word)) for w in words]
    with Timer("definitions"):
        d_own = generate(model, tok, p_def, temperature=0, max_new_tokens=80, batch=16)
        d_en = generate(model, tok, p_en, temperature=0, max_new_tokens=80, batch=16)
    nat = [(w, unmark(c)) for w in words for c in w.contexts]
    with Timer("context log-probs"):
        lps = token_logprobs(model, tok, [t for _, (t, _) in nat])
    span_lp = {}
    for (w, (t, sp)), (offs, lp) in zip(nat, lps):
        v = span_logprob(offs, lp, sp)
        if v is not None:
            span_lp.setdefault(w.id, []).append(v)
    out = {}
    for w, own, en in zip(words, d_own, d_en):
        flag, ov = provisional(en[0], w.gloss_en)
        out[w.id] = dict(word=w.word, language=w.language, category=w.category, gloss_en=w.gloss_en,
                         definition_own=own[0], explanation_en=en[0],
                         n_tokens=len(tok(" " + w.word, add_special_tokens=False)["input_ids"]),
                         zipf=zipf_frequency(w.word, w.language) if w.language in LANG_EN and w.language != "rom" else None,
                         mean_token_logprob=float(np.mean(span_lp[w.id])) if w.id in span_lp else None,
                         provisional=flag, overlap=ov)
    rd = out_dir(a.model)
    save_json(rd / "familiarity.json", dict(model=describe(a.model), prompts=dict(own=DEF_PROMPT, en=EN_PROMPT),
                                            decoding="greedy", provisional_rule="overlap>=0.34 yes, >0 partly, 0 no",
                                            words=out))
    (rd / "familiarity_check.html").write_text(page(a.model, out))
    print("flags:", {k: sum(r["provisional"] == k for r in out.values()) for k in ("yes", "partly", "no")})


def page(model_key, rows):
    esc = html.escape
    trs = []
    for wid, r in rows.items():
        radios = "".join(f'<label><input type="radio" name="{esc(wid)}" value="{v}"> {v}</label>'
                         for v in ("yes", "partly", "no"))
        trs.append(f"""<tr><td><b>{esc(r['word'])}</b><div class=m>{esc(r['language'])} · {esc(r['category'])}</div></td>
<td>{esc(r['gloss_en'])}</td><td>{esc(r['definition_own'])}</td><td>{esc(r['explanation_en'])}</td>
<td class=m>{r['provisional']}</td><td class=r>{radios}</td></tr>""")
    return f"""<!doctype html><meta charset=utf-8><title>Familiarity check: {esc(MODELS[model_key]['hf_id'])}</title>
<style>body{{font:15px/1.45 system-ui,sans-serif;margin:24px;color:#1f1f1e;background:#faf9f6}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #e6e5e0;padding:8px;vertical-align:top;text-align:left}}
th{{position:sticky;top:0;background:#faf9f6}}.m{{color:#6b6a63;font-size:13px}}.r label{{display:block;white-space:nowrap}}
textarea{{width:100%;height:140px;font:12px monospace}}button{{font-size:15px;padding:6px 14px;margin:8px 8px 8px 0}}</style>
<h1>Does the model understand these words?</h1>
<p>Model: <b>{esc(describe(model_key))}</b>. For each word, read the model's own definition and its English explanation,
and tick whether it <b>understood</b> the word. The provisional column is a crude automatic guess (word overlap with
the dictionary gloss); please overrule it freely. When done, press <b>Export</b>, then save the text as
<code>data/familiarity_ticks_{model_key}.json</code> (or send it to the user).</p>
<p>Your name (optional): <input id=who></p>
<table><tr><th>word</th><th>dictionary gloss</th><th>model's definition (own language)</th><th>model's English explanation</th><th>provisional</th><th>understood?</th></tr>
{''.join(trs)}</table>
<button onclick="exp()">Export</button><button onclick="navigator.clipboard&&navigator.clipboard.writeText(document.getElementById('out').value)">Copy</button>
<textarea id=out placeholder="exported ticks appear here"></textarea>
<script>
function exp(){{const t={{}};document.querySelectorAll('input[type=radio]:checked').forEach(r=>t[r.name]=r.value);
document.getElementById('out').value=JSON.stringify({{model:{json.dumps(model_key)},checker:document.getElementById('who').value,
date:new Date().toISOString().slice(0,10),ticks:t}},null,1);}}
</script>"""


if __name__ == "__main__":
    main()
