"""Every prompt used by the sentence experiments, in one place (the outputs record each prompt verbatim).

The budget ladder (handoff §3): rungs B0-B7, one translational practice each, in English.  The prompts never
say which parts of the sentence are hard: the model decides.
"""
import re

from ut.common import LANG_EN

RUNGS = {
    "B0": dict(practice="word-for-word", max_new_tokens=80, prompt=(
        "Give a word-for-word English gloss of this {lang} sentence: one English word for each {lang} word, "
        "in the original word order, even if the result is not good English. Reply with the gloss only.")),
    "B1": dict(practice="standard translation", max_new_tokens=80, prompt=(
        "Translate this {lang} sentence into natural English, as a single sentence. "
        "Reply with the translation only.")),
    "B2_borrow": dict(practice="borrow (keep the word)", max_new_tokens=80, prompt=(
        "Translate this {lang} sentence into English, but keep any word or phrase that has no good English "
        "equivalent in the original {lang}, in italics (*like this*). Reply with the translation only.")),
    "B3": dict(practice="inline gloss", max_new_tokens=120, prompt=(
        "Translate this {lang} sentence into English. After each word or phrase that is hard to translate, add a "
        "short gloss in square brackets (at most 8 words) saying what it means. Reply with the translation only.")),
    "B4": dict(practice="alternatives", max_new_tokens=120, prompt=(
        "Translate this {lang} sentence into English. For each part that is hard to translate, give 2 or 3 "
        "alternative English renderings separated by \" / \". Reply with the translation only.")),
    "B5": dict(practice="footnote", max_new_tokens=140, prompt=(
        "Translate this {lang} sentence into natural English, then add one footnote (at most 40 words) on what "
        "the English translation loses. Reply in this format: the translation, a blank line, then the footnote "
        "starting with \"Note:\".")),
    "B6": dict(practice="thick translation", max_new_tokens=280, prompt=(
        "Translate this {lang} sentence into natural English, then write one paragraph (at most 120 words) that "
        "opens up the meanings, connotations and grammar of the original that the English cannot carry. Reply in "
        "this format: the translation, a blank line, then the paragraph.")),
    "B7": dict(practice="coinage", max_new_tokens=140, prompt=(
        "Translate this {lang} sentence into English. For the part that is hardest to translate, invent a new "
        "English term, define it in one sentence, and use it in the translation. Reply in this format: first the "
        "line \"Term: <the term> = <one-sentence definition>\", then a blank line, then the translation using "
        "the term.")),
}
RUNG_ORDER = list(RUNGS)
PRACTICE = {r: v["practice"] for r, v in RUNGS.items()}
PRACTICE["B2"] = "keep the word"          # name of B2_borrow in the 19-20 Nov runs (results/sent/)
BORROW = {"B2", "B2_borrow"}


def lang_name(lang):
    return LANG_EN[lang].split(" (")[0] if lang != "rom" else "Romanesco"


def rung_prompt(rung, sent):
    """The user turn for one rung.  Romanesco is named with its gloss so the model knows what it is."""
    lang = lang_name(sent.lang)
    head = RUNGS[rung]["prompt"].format(lang=lang)
    label = LANG_EN[sent.lang] if sent.lang == "rom" else lang
    return f"{head}\n\n{label}: {sent.text}"


_PREFIX = re.compile(r"^\s*(?:english(?: translation)?|translation|gloss|answer)\s*:\s*", re.I)


def strip_source(text, source):
    """Replace verbatim copies of the whole source sentence (with or without its final punctuation, any case)
    by '…', so the round trip cannot simply copy it back.  Returns (text, number of copies removed)."""
    core = source.strip().rstrip(".!?…,;:")
    pat = re.compile(re.escape(core) + r"[.!?…,;:]?", re.I) if len(core) >= 8 else None
    return (pat.subn("…", text) if pat else (text, 0))


def clean(text, source=None):
    """What gets scored: the model's text with markdown emphasis, a leading 'Translation:' label, wrapping
    quotes and (if `source` is given) verbatim copies of the source sentence removed.  Internal line breaks are
    kept (footnotes and paragraphs are part of the candidate)."""
    t = text.strip()
    if source:
        t = strip_source(t, source)[0]
    t = _PREFIX.sub("", t)
    t = re.sub(r"\*\*?|__?(?=\w)|(?<=\w)__?", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > 1 and t[0] in "\"“'" and t[-1] in "\"”'" and t.count(t[0]) == 1 + (t[0] == t[-1]):
        t = t[1:-1].strip()
    return t


# ------------------------------------------------------------------ round trip (§2.2)
def backtranslate_prompt(lang, english):
    """User turn of the back-translation task; the source sentence is then teacher-forced as the reply.
    english = "" gives the no-help baseline (the original's own predictability)."""
    L = lang_name(lang)
    return f"Translate this English text into {L}.\n\nEnglish: {english}"


def copy_prompt(lang, source):
    """The oracle: the most any text can supply about the source (the copy task)."""
    L = lang_name(lang)
    return f"Translate this {L} text into {L}.\n\n{L}: {source}"


# ------------------------------------------------------------------ spread (§1) and targeting (§3)
EXTRACT = ("Here is a {lang} sentence and an English translation of it.\n\n{lang}: {source}\nEnglish: {english}\n\n"
           "Which words of the English translation render the {lang} \"{span}\"? Reply with those English words "
           "only, copied exactly from the translation. If nothing renders it, reply \"(none)\".")

GLOSS = ("Here is a {lang} sentence.\n\n{lang}: {source}\n\nIn at most 8 English words, what does the {lang} "
         "word \"{word}\" mean in this sentence? Do not repeat the {lang} word. Reply with the gloss only.")


def extract_prompt(sent, english, span):
    L = lang_name(sent.lang)
    return EXTRACT.format(lang=L, source=sent.text, english=english, span=span)


def gloss_prompt(sent, word):
    L = lang_name(sent.lang)
    return GLOSS.format(lang=L, source=sent.text, word=word)
