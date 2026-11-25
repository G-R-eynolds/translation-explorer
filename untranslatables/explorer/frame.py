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
DOC_SHOTS = [
    (
        "La casa está cerca del río; por la tarde se oye el agua entre los árboles.",
        "The house is near the river; in the afternoon you can hear the water through the trees.",
    ),
    (
        "En la plaza callada\n  cae la lluvia lenta.\n\nNadie cruza el umbral\n  —solo el viento.",
        "In the quiet square\n  the slow rain falls.\n\nNo one crosses the threshold\n  —only the wind.",
    ),
]


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
    # Keep Spanish->English shots as structural exemplars (they teach tag + stanza preservation even for other langs)
    head = "".join(
        f"<source lang=\"Spanish\">\n{s}\n</source>\n<translation lang=\"English\">\n{e}\n</translation>\n\n"
        for s, e in DOC_SHOTS
    )
    clean_text = text.strip("\r\n")
    return head + f"<source lang=\"{src_lang}\">\n{clean_text}\n</source>\n<translation lang=\"{tgt_lang}\">\n"


def build_doc_prompt(tok, text, src_lang="Spanish", tgt_lang="English"):
    """-> (ids, start): ids of the formatting-preserving document frame."""
    ids = tok(doc_prompt_text(text, src_lang=src_lang, tgt_lang=tgt_lang))["input_ids"]
    return ids, len(ids)


def _cut_with_stops(tok, y, stops):
    """Truncate a continuation at the first stop string (token-level, excluding stop tokens)."""
    for i in range(1, len(y) + 1):
        s = tok.decode(y[:i])
        if any(st in s for st in stops):
            j = i
            while j > 0 and any(st in tok.decode(y[:j]) for st in stops):
                j -= 1
            return list(y[:j])
    return list(y)


def cut(tok, y):
    """Truncate a continuation at the first blank line or 'Spanish:' (legacy single-sentence frame)."""
    return _cut_with_stops(tok, y, STOPS)


def cut_doc(tok, y):
    """Truncate a document/poem continuation at '</translation>' or '<source', preserving blank lines."""
    return _cut_with_stops(tok, y, DOC_STOPS)


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
