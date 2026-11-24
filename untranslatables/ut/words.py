"""Word set: loading, validation, and the context lists every experiment uses.

    python -m ut.words            # validate data/words/*.yaml and print the frequency-matching table

All word data lives in data/words/<lang>.yaml (one entry per word; schema in data/README.md).
"""
import re
import sys
import unicodedata
from dataclasses import dataclass, field

from ut.common import DATA, load_yaml
from ut.represent import unmark

CATEGORIES = ["untranslatable", "ordinary", "emotion_plain", "polysemous", "loanword",
              "english_untranslatable", "english_ordinary", "emotion_plain_en",
              "dialect_known_gloss", "dialect_untranslatable"]
THREAD2 = {"ordinary": ("known", "one"), "untranslatable": ("known", "many"), "polysemous": ("known", "many"),
           "dialect_known_gloss": ("unknown", "one"), "dialect_untranslatable": ("unknown", "many")}
PHASE1 = ["pt", "de", "it", "fr", "es"]
N_NATURAL_T1 = 4          # thread 1 uses the first 4 natural contexts (handoff §5.1); thread 2 uses all 5


@dataclass
class Word:
    word: str
    language: str
    category: str
    gloss_en: str
    senses: list
    source: str
    notes: str
    contexts: list
    templates: list
    emotion: bool = False
    frames: list = field(default_factory=list)
    gloss_src: str = ""
    extra: list = field(default_factory=list)
    sense_of_context: list = None
    exclude_from_pool: list = field(default_factory=list)
    freq_match_for: str = None
    loan_en: str = None
    context_source: list = None

    @property
    def id(self):
        return f"{self.language}:{self.word}"

    def t1_contexts(self):
        """Thread-1 usage cloud: [(kind, marked_text)]: 4 templates, 2 emotion frames (emotion words), 4 natural."""
        out = [("template", t.format(w=f"[[{self.word}]]")) for t in self.templates]
        out += [("frame", f) for f in (self.frames if self.emotion else [])]
        out += [("natural", c) for c in self.contexts[:N_NATURAL_T1]]
        return out

    def t2_contexts(self):
        """Thread-2 contexts: (index, marked_text, extra_or_None, sense_index_or_None)."""
        return [(k, c, self.extra[k] if k < len(self.extra) else None,
                 self.sense_of_context[k] if self.sense_of_context else None) for k, c in enumerate(self.contexts)]


def load_words(langs=None, categories=None):
    words = []
    for p in sorted((DATA / "words").glob("*.yaml")):
        d = load_yaml(p)
        if langs and d["language"] not in langs:
            continue
        for e in d["words"]:
            e = dict(e)
            e.setdefault("senses", [])
            w = Word(language=d["language"], templates=d["templates"], **e)
            if categories is None or w.category in categories:
                words.append(w)
    return words


def by_id(words):
    return {w.id: w for w in words}


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


def leak_words(w):
    """English content words of gloss_en that also appear in gloss_src or a sense clarification -- a
    possible leak of the expected translation into a 'knowledge gloss' (handoff §7.2). Cognates that are the
    same word in both languages (e.g. 'nostalgia') will show up; they are reported for a human to judge."""
    eng = {x for x in re.findall(r"[a-z]{4,}", w.gloss_en.lower())} - {"with", "that", "from", "into", "something",
                                                                        "someone", "often", "also", "being", "their"}
    src = _strip_accents(" ".join([w.gloss_src] + [s.get("clarify", "") for s in w.senses]))
    return sorted(x for x in eng if re.search(rf"\b{x}\b", src))


def validate(words):
    issues = []
    seen = set()
    for w in words:
        if w.id in seen:
            issues.append(f"{w.id}: duplicate")
        seen.add(w.id)
        if w.category not in CATEGORIES:
            issues.append(f"{w.id}: unknown category {w.category}")
        for kind, c in w.t1_contexts():
            try:
                clean, (s, e) = unmark(c)
            except ValueError as ex:
                issues.append(f"{w.id}: {ex}")
        if w.category in THREAD2:
            if not w.gloss_src:
                issues.append(f"{w.id}: thread-2 item without gloss_src")
            if len(w.contexts) < 5 and not w.category.startswith("dialect"):
                issues.append(f"{w.id}: {len(w.contexts)} contexts (<5)")
            if w.category in ("untranslatable", "polysemous", "dialect_untranslatable") and len(w.senses) > 1 \
                    and not all(s.get("clarify") for s in w.senses):
                issues.append(f"{w.id}: sense without a clarification")
            if (lk := leak_words(w)):
                issues.append(f"{w.id}: gloss/clarification shares English words with gloss_en {lk} (check for leakage)")
        if w.sense_of_context is not None and len(w.sense_of_context) != len(w.contexts):
            issues.append(f"{w.id}: sense_of_context length != contexts")
        if w.emotion and w.category not in ("dialect_known_gloss",) and len(w.frames) != 2:
            issues.append(f"{w.id}: emotion word needs 2 frames")
        if "TODO" in w.source and not w.source.startswith("TODO"):
            issues.append(f"{w.id}: malformed source")
    return issues


def freq_table(words):
    from wordfreq import zipf_frequency
    ids = by_id(words)
    rows = []
    for w in words:
        if w.freq_match_for:
            tgt = ids.get(f"{w.language}:{w.freq_match_for}")
            if tgt is None:
                rows.append((w.id, None, None, "TARGET MISSING")); continue
            zw, zt = zipf_frequency(w.word, w.language), zipf_frequency(tgt.word, tgt.language)
            rows.append((tgt.id, round(zt, 2), w.id, round(zw, 2), round(zw - zt, 2)))
    return rows


def main():
    words = load_words()
    from collections import Counter
    print("words:", len(words), dict(Counter(w.category for w in words)))
    print("by language:", dict(Counter(w.language for w in words)))
    issues = validate(words)
    print(f"\n{len(issues)} issues")
    for i in issues:
        print("  -", i)
    print("\nfrequency matches (untranslatable/target zipf -> control zipf, delta):")
    for r in freq_table(words):
        flag = "  <-- |delta| > 0.3" if len(r) == 5 and abs(r[4]) > 0.3 else ""
        print("  ", r, flag)
    return 1 if any("unknown category" in i or "need exactly one" in i for i in issues) else 0


if __name__ == "__main__":
    sys.exit(main())
