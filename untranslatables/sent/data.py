"""Sentence test set: loading, hot-span offsets, word segmentation, centring texts, output paths.

    python -m sent.data          # validate data/sentences.yaml and print a summary table
"""
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ut.common import DATA, ROOT, load_yaml
from ut.represent import unmark

CATEGORIES = ["untranslatable_in_context", "construction", "idiom", "control", "dialect"]
CAT_TARGET_N = {"untranslatable_in_context": 12, "construction": 8, "idiom": 8, "control": 10}
LANGS = ["es", "it", "pt", "fr", "de", "hi", "rom"]
SRC_GLOSS_LANGS = {"es", "hi"}       # languages whose spans also need a source-language sense gloss (N3)
CENTROID_LANG = {"rom": "it"}            # Romanesco has no centring texts of its own: it borrows Italian's (recorded)
# \w alone would split Devanagari words at their vowel signs and viramas (categories Mn/Mc)
_W = r"[\w\u0900-\u0963\u0966-\u097F]"     # danda (U+0964-5) is punctuation
WORD = re.compile(rf"{_W}+(?:['’]{_W}+)*", re.UNICODE)

# |t| is measured with ONE fixed tokenizer for every model, so the L axis is comparable across models.
# CPU smoke tests (no access to the gated Llama tokenizer) override it with SENT_LENGTH_TOKENIZER.
LENGTH_TOKENIZER = os.environ.get("SENT_LENGTH_TOKENIZER", "meta-llama/Llama-3.2-3B-Instruct")


def sent_dir(model_key, *sub):
    """results/sent/<model> unless SENT_DIR_TEMPLATE says otherwise (the internals run's R2 writes to
    results/internals/<model>/replication_R2)."""
    d = ROOT / os.environ.get("SENT_DIR_TEMPLATE", "results/sent/{model}").format(model=model_key)
    for s in sub:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig_dir(model_key):
    d = ROOT / os.environ.get("SENT_FIG_TEMPLATE", "results/sent/figures/{model}").format(model=model_key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


@dataclass
class Sentence:
    id: str
    lang: str
    category: str
    text: str
    hot: list = field(default_factory=list)
    probe: list = field(default_factory=list)
    notes: str = ""
    source: str = ""
    optional: bool = False
    gloss: str = None
    senses: dict = field(default_factory=dict)
    draft: bool = False

    @property
    def is_control(self):
        return self.category == "control"

    def _find(self, spec):
        sub, _, k = spec.partition("#")
        k = int(k or 1)
        start = -1
        for _ in range(k):
            start = self.text.find(sub, start + 1)
            if start < 0:
                raise ValueError(f"{self.id}: span {spec!r} not in text")
        return start, start + len(sub)

    def spans(self):
        """Hot spans (or the probe span for controls): [{text, start, end, kind}]."""
        out = []
        for kind, specs in (("hot", self.hot), ("probe", self.probe)):
            for spec in specs:
                s, e = self._find(spec)
                sense = self.senses.get(spec.partition("#")[0], {})
                out.append(dict(text=self.text[s:e], start=s, end=e, kind=kind, sense_en=sense.get("en"),
                                sense_src=sense.get("src")))
        return out

    def words(self):
        """[(start, end, word)] for every word of the text."""
        return [(m.start(), m.end(), m.group()) for m in WORD.finditer(self.text)]

    def content_words(self):
        """Words that are not stopwords of the sentence's language (Romanesco uses Italian's list), plus every
        word inside an annotated span (stopwordsiso lists e.g. Italian 'magari' as a stopword)."""
        sw = stopwords(CENTROID_LANG.get(self.lang, self.lang))
        sp = self.spans()
        def in_span(a, b):
            return any(a < x["end"] and b > x["start"] for x in sp)
        return [w for w in self.words() if (fold(w[2]) not in sw and not w[2].isdigit()) or in_span(w[0], w[1])]

    def record(self):
        return dict(id=self.id, lang=self.lang, category=self.category, text=self.text, spans=self.spans(),
                    notes=self.notes, source=self.source, draft=self.draft)


_SW = {}


def stopwords(lang):
    if lang not in _SW:
        import stopwordsiso
        _SW[lang] = {fold(w) for w in stopwordsiso.stopwords(lang)}
    return _SW[lang]


def load_sentences(dialect=False, only=None, path=None, langs=None):
    raw = load_yaml(path or DATA / "sentences.yaml")["sentences"]
    out = [Sentence(**r) for r in raw]
    out = [s for s in out if (dialect or not s.optional) and (not only or s.id in only) and (not langs or s.lang in langs)]
    return out


def validate(sents):
    issues = []
    ids = [s.id for s in sents]
    for d in {i for i in ids if ids.count(i) > 1}:
        issues.append(f"duplicate id {d}")
    for s in sents:
        if s.lang not in LANGS:
            issues.append(f"{s.id}: unknown lang {s.lang}")
        if s.category not in CATEGORIES:
            issues.append(f"{s.id}: unknown category {s.category}")
        if not s.id.startswith(s.lang + "_"):
            issues.append(f"{s.id}: id should start with '{s.lang}_'")
        try:
            sp = s.spans()
        except ValueError as e:
            issues.append(str(e)); continue
        if s.is_control and (s.hot or len(s.probe) != 1):
            issues.append(f"{s.id}: a control needs exactly one probe and no hot spans")
        if not s.is_control and not s.hot:
            issues.append(f"{s.id}: needs at least one hot span")
        for x in sp:
            if not x["sense_en"]:
                issues.append(f"{s.id}: span {x['text']!r} has no senses.en")
            if s.lang in SRC_GLOSS_LANGS and not x["sense_src"]:
                issues.append(f"{s.id}: span {x['text']!r} has no senses.src ({s.lang} needs one)")
        if len(s.content_words()) < 2:
            issues.append(f"{s.id}: fewer than 2 content words")
        for x in sp:
            if not any(a <= x["start"] < b for a, b, _ in s.words()) or not any(a < x["end"] <= b for a, b, _ in s.words()):
                issues.append(f"{s.id}: span {x['text']!r} does not start and end on word boundaries")
    return issues


def centring_texts(lang):
    """Plain running text used to estimate a language's mean residual state (d_CL centring, §2.1 step 2):
    the natural contexts and 'extra' lines of the word-level data (data/words/<lang>.yaml), never the test set."""
    p = DATA / "words" / f"{CENTROID_LANG.get(lang, lang)}.yaml"
    out = []
    for w in load_yaml(p)["words"]:
        for c in w.get("contexts", []) + w.get("extra", []):
            try:
                out.append(unmark(c)[0])
            except ValueError:
                out.append(c)
    return list(dict.fromkeys(out))


def main():
    sents = load_sentences(dialect=True)
    issues = validate(sents)
    from collections import Counter
    cats = Counter(s.category for s in sents)
    langs = Counter((s.category, s.lang) for s in sents)
    print(f"{len(sents)} sentences")
    for c in CATEGORIES:
        per = ", ".join(f"{l} {langs[(c, l)]}" for l in LANGS if langs[(c, l)])
        tgt = CAT_TARGET_N.get(c)
        print(f"  {c:28s} {cats[c]:3d}{'' if tgt is None else f' (handoff: {tgt})':14s}  {per}")
    for s in sents:
        print(f"  {s.id:24s} {s.text:62s} hot/probe: {[x['text'] for x in s.spans()]}")
    for l in sorted({CENTROID_LANG.get(s.lang, s.lang) for s in sents} | {"en"}):
        if not (DATA / "words" / f"{l}.yaml").exists():
            print(f"  centring texts {l}: none (only needed by the old concept-layer score)"); continue
        print(f"  centring texts {l}: {len(centring_texts(l))}")
    print("\n".join(["ISSUES:"] + issues) if issues else "no issues")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
