"""Few-shot translation frame for a base model, stop rules, texts, loader (spec §Model, §Material)."""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SHOTS = [("La casa está cerca del río.", "The house is near the river."),
         ("Mañana iremos al mercado con mis padres.", "Tomorrow we will go to the market with my parents.")]
STOPS = ("\n\n", "Spanish:")

# Document/poem-preserving frame: explicit tags allow blank lines (\n\n), stanza breaks,
# indentation, and unusual punctuation inside translations without premature stopping.
DOC_STOPS = ("</translation>", "<source", "\n<source")

# Parallel multi-language exemplars: (literary prose, two-stanza indented em-dash poem).
_DOC_EXEMPLARS = {
    "spanish": (
        "La casa está cerca del río; por la tarde se oye el agua entre los árboles.",
        "En la plaza callada\n  cae la lluvia lenta.\n\nNadie cruza el umbral\n  —solo el viento.",
    ),
    "english": (
        "The house is near the river; in the afternoon you can hear the water through the trees.",
        "In the quiet square\n  the slow rain falls.\n\nNo one crosses the threshold\n  —only the wind.",
    ),
    "portuguese": (
        "A casa fica perto do rio; à tarde ouve-se a água entre as árvores.",
        "Na praça silenciosa\n  cai a chuva lenta.\n\nNinguém cruza o limiar\n  —apenas o vento.",
    ),
    "french": (
        "La maison est près de la rivière ; le soir, on entend l'eau entre les arbres.",
        "Sur la place silencieuse\n  tombe la pluie lente.\n\nPersonne ne franchit le seuil\n  —seulement le vent.",
    ),
    "german": (
        "Das Haus steht nahe am Fluss; am Nachmittag hört man das Wasser zwischen den Bäumen.",
        "Auf dem stillen Platz\n  fällt der langsame Regen.\n\nNiemand tritt über die Schwelle\n  —nur der Wind.",
    ),
    "italian": (
        "La casa è vicina al fiume; nel pomeriggio si sente l'acqua tra gli alberi.",
        "Nella piazza silenziosa\n  cade la pioggia lenta.\n\nNessuno varca la soglia\n  —solo il vento.",
    ),
}

_LANG_ALIASES = {
    "es": "spanish",
    "español": "spanish",
    "espanol": "spanish",
    "en": "english",
    "pt": "portuguese",
    "português": "portuguese",
    "portugues": "portuguese",
    "fr": "french",
    "français": "french",
    "francais": "french",
    "de": "german",
    "deutsch": "german",
    "it": "italian",
    "italiano": "italian",
}

DOC_SHOTS = list(zip(_DOC_EXEMPLARS["spanish"], _DOC_EXEMPLARS["english"]))


def _canonical_lang(lang):
    key = str(lang or "").strip().lower()
    return _LANG_ALIASES.get(key, key)


def get_doc_shots(src_lang="Spanish", tgt_lang="English"):
    """Return `(shot_src_lang, shot_tgt_lang, shots)` matched to the requested language pair when available."""
    sk = _canonical_lang(src_lang)
    tk = _canonical_lang(tgt_lang)
    if sk in _DOC_EXEMPLARS and tk in _DOC_EXEMPLARS and sk != tk:
        return src_lang, tgt_lang, list(zip(_DOC_EXEMPLARS[sk], _DOC_EXEMPLARS[tk]))
    return "Spanish", "English", list(DOC_SHOTS)


def load_texts():
    return yaml.safe_load((HERE / "data" / "texts.yaml").read_text())["texts"]


def prompt_text(text):
    head = "".join(f"Spanish: {s}\nEnglish: {e}\n\n" for s, e in SHOTS)
    return head + f"Spanish: {text}\nEnglish:"


def build_prompt(tok, text):
    """-> (ids, start): ids of the frame (with BOS if the tokenizer adds one), start = first output position."""
    ids = tok(prompt_text(text))["input_ids"]
    return ids, len(ids)


def doc_prompt_text(text, src_lang="Spanish", tgt_lang="English"):
    """Formatting-preserving few-shot prompt supporting poems, stanzas, indentation, and paragraphs."""
    shot_src, shot_tgt, shots = get_doc_shots(src_lang=src_lang, tgt_lang=tgt_lang)
    head = "".join(
        f"<source lang=\"{shot_src}\">\n{s}\n</source>\n<translation lang=\"{shot_tgt}\">\n{e}\n</translation>\n\n"
        for s, e in shots
    )
    clean_text = text.strip("\r\n")
    return head + f"<source lang=\"{src_lang}\">\n{clean_text}\n</source>\n<translation lang=\"{tgt_lang}\">\n"


def build_doc_prompt(tok, text, src_lang="Spanish", tgt_lang="English"):
    """-> (ids, start): ids of the formatting-preserving document frame."""
    ids = tok(doc_prompt_text(text, src_lang=src_lang, tgt_lang=tgt_lang))["input_ids"]
    return ids, len(ids)


def _partial_stop_pos(text, stops):
    """Return earliest index of a trailing partial XML stop tag in `text`, or -1 if none.

    Carefully avoids false matches on plain newlines ('\\n', '\\n\\n') or inline '<' (e.g. 'a < b').
    """
    if not text:
        return -1
    best = -1
    n = len(text)
    for st in stops:
        max_k = min(len(st) - 1, n)
        for k in range(max_k, 0, -1):
            if text.endswith(st[:k]):
                if k == 1:
                    if st[0] != "<":
                        continue
                    if n > 1 and text[-2] not in "\n\r":
                        continue
                pos = n - k
                if best == -1 or pos < best:
                    best = pos
                break
    return best


def _cut_with_stops(tok, y, stops, strip_partial_tags=False):
    """Truncate a continuation at the first stop string (token-level, excluding stop tokens).

    Fast-path runs in a single decode call when no stop occurs; binary-searches the exact token
    boundary in O(log N) decodes when a stop (or trailing partial XML tag) is detected.
    """
    if not y:
        return []
    full = tok.decode(y)
    first_pos = -1
    for st in stops:
        idx = full.find(st)
        if idx != -1 and (first_pos == -1 or idx < first_pos):
            first_pos = idx
    if first_pos == -1 and strip_partial_tags:
        first_pos = _partial_stop_pos(full, stops)
    if first_pos == -1:
        return list(y)
    if first_pos == 0:
        return []

    lo, hi = 0, len(y)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        s_mid = tok.decode(y[:mid])
        if len(s_mid) <= first_pos and not any(st in s_mid for st in stops):
            lo = mid
        else:
            hi = mid - 1
    while lo > 0:
        s_lo = tok.decode(y[:lo])
        if len(s_lo) <= first_pos and not any(st in s_lo for st in stops):
            break
        lo -= 1
    return list(y[:lo])


def cut(tok, y):
    """Truncate a continuation at the first blank line or 'Spanish:' (legacy single-sentence frame)."""
    return _cut_with_stops(tok, y, STOPS, strip_partial_tags=False)


def cut_doc(tok, y):
    """Truncate a document/poem continuation at '</translation>' or '<source', preserving blank lines."""
    return _cut_with_stops(tok, y, DOC_STOPS, strip_partial_tags=True)


def load(model_key):
    """Model with gradients off and Gemma 3's image encoder replaced by a parameter-free stub (saves VRAM)."""
    import gc
    import torch
    from ut.common import load_model
    tok, model = load_model(model_key)
    model.requires_grad_(False)
    inner = getattr(model, "model", None)
    for name in ("vision_tower", "multi_modal_projector"):
        if getattr(inner, name, None) is not None:
            setattr(inner, name, torch.nn.Identity())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tok, model
