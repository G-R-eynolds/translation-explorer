"""Few-shot translation frame for a base model, stop rules, texts, loader (spec §Model, §Material)."""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SHOTS = [("La casa está cerca del río.", "The house is near the river."),
         ("Mañana iremos al mercado con mis padres.", "Tomorrow we will go to the market with my parents.")]
STOPS = ("\n\n", "Spanish:")


def load_texts():
    return yaml.safe_load((HERE / "data" / "texts.yaml").read_text())["texts"]


def prompt_text(text):
    head = "".join(f"Spanish: {s}\nEnglish: {e}\n\n" for s, e in SHOTS)
    return head + f"Spanish: {text}\nEnglish:"


def build_prompt(tok, text):
    """-> (ids, start): ids of the frame (with BOS if the tokenizer adds one), start = first output position."""
    ids = tok(prompt_text(text))["input_ids"]
    return ids, len(ids)


def cut(tok, y):
    """Truncate a continuation at the first blank line or 'Spanish:' (token-level: the shortest prefix whose decoded
    text contains a stop string, minus the stop tokens)."""
    for i in range(1, len(y) + 1):
        s = tok.decode(y[:i])
        if any(st in s for st in STOPS):
            j = i
            while j > 0 and any(st in tok.decode(y[:j]) for st in STOPS):
                j -= 1
            return y[:j]
    return list(y)


def load(model_key):
    """Model with gradients off and Gemma 3's image encoder replaced by a parameter-free stub (12 GB card)."""
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
