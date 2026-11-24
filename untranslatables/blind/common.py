"""Paths, layers, channels (prompt templates), items, resumable-stage helper (HANDOFF_blind_spots.md).

Roots (env vars override):
  UT_BLIND_RESULTS  results/blind            per model: results/blind/<model>/...
  UT_BLIND_CACHE    cache/blind              per model: gradients cache/blind/<model>/grads/<set>/<item>/<channel>.npz
The gradients live under cache/ (not results/) so that remote/pull.sh never drags gigabytes back to the laptop; all
analysis stages run where the gradients are.

Layer convention: "layer k" = the output of decoder block k (0-based) = HF hidden_states[k+1] = Gemma Scope "layer k".
"""
import functools
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ut.common import DATA, LANG_EN, ROOT, load_yaml, save_json

BLIND = Path(__file__).resolve().parent
BDATA = BLIND / "data"
RESULTS = Path(os.environ.get("UT_BLIND_RESULTS", ROOT / "results" / "blind"))
CACHE = Path(os.environ.get("UT_BLIND_CACHE", ROOT / "cache" / "blind"))

# Stored layers: the Gemma Scope 2 "detailed" depths (25/50/65/85%).  For the Gemma models these are read from the
# google/gemma-scope-2-*-it repos (resid_post/layer_<k>_width_16k_l0_medium exists only at these layers); for the
# others they are round(f * n_layers) at the same fractions.
LAYERS = {"g3_4b": [9, 17, 22, 29], "g3_12b": [12, 24, 31, 41], "3b": [7, 14, 18, 24], "aya8b": [8, 16, 21, 27],
          "qwen05": [6, 12, 16, 20], "tiny": [0, 1]}
HEADLINE_FRAC = 0.5          # pre-registered headline layer: the 50% one (PREDICTIONS.md)
SAE = {   # model -> (sae_lens release, id template, neuronpedia model, neuronpedia source template)
    "g3_4b": ("gemma-scope-2-4b-it-res", "layer_{k}_width_16k_l0_medium", "gemma-3-4b-it", "{k}-gemmascope-2-res-16k"),
    "g3_12b": ("gemma-scope-2-12b-it-res", "layer_{k}_width_16k_l0_medium", "gemma-3-12b-it", "{k}-gemmascope-2-res-16k"),
}
K_DEFAULT = 64
EPS_REL = [1e-3, 1e-2, 1e-1]
EPS_HEAD = 1e-2
N_SPLITS = 20


def layers_for(model):
    return LAYERS[model]


def headline_layer(model):
    ls = LAYERS[model]
    return ls[1] if len(ls) >= 2 else ls[0]


# ------------------------------------------------------------------ channels
# One template; only the language varies.  The instruction comes AFTER the text, so the source-position states are the
# same for every channel (asserted in E0).
EXPRESS = "\n\nExpress the text above in {L}, in your own words, keeping exactly its meaning. Reply with the text only."
REWRITE = "\n\nRewrite the text above in {L} with different words, keeping exactly its meaning. Reply with the text only."
COPY = "\n\nRepeat the text above exactly. Reply with the text only."
PRACTICES = {   # E3: translational practices into English (plain = the 'en' channel itself)
    "p_keep": "\n\nExpress the text above in English, keeping exactly its meaning. Where a word has no real English "
              "equivalent, keep that word in the original language. Reply with the text only.",
    "p_gloss": "\n\nExpress the text above in English, keeping exactly its meaning. After each word or phrase that is "
               "hard to translate, add a short explanation in square brackets. Reply with the text only.",
    "p_foot": "\n\nExpress the text above in English, keeping exactly its meaning. Mark each word or phrase that is hard "
              "to translate with an asterisk and add a footnote after the translation explaining it. Reply with the "
              "text only.",
    "p_thick": "\n\nExpress the text above in English as a thick translation: one paragraph that translates it and opens "
               "up its full meaning, including what a plain translation would lose. Reply with the text only.",
}
LONG_CHANNELS = {"p_gloss", "p_foot", "p_thick"}     # may need more room than 2|s|+16 (see max_new_tokens)


def instruction(channel, src_lang):
    """The text placed after the source text for a channel name.
      src     paraphrase in the source language (the reference channel)
      src2    a second paraphrase wording (Spanish-into-Spanish check, §2.3)
      copy    verbatim copy (surface diagnostic)
      <lang>  translation into that language (en, it, fr, pt, es, de)
      p_*     a practice into English (E3)"""
    if channel == "src":
        return EXPRESS.format(L=LANG_EN[src_lang])
    if channel == "src2":
        return REWRITE.format(L=LANG_EN[src_lang])
    if channel == "copy":
        return COPY
    if channel in PRACTICES:
        return PRACTICES[channel]
    if channel in LANG_EN:
        return EXPRESS.format(L=LANG_EN[channel])
    raise KeyError(channel)


def max_new_tokens(channel, n_src_tokens, long_factor=6):
    """2|s|+16 (§2.1).  E3's gloss/footnote/thick practices ask for more text than the source; with the plain cap they
    would be cut off mid-gloss, so they get long_factor*|s|+48 (logged in CHANGES.md)."""
    if channel in LONG_CHANNELS:
        return long_factor * n_src_tokens + 48
    return 2 * n_src_tokens + 16


TARGETS = {"es": ["en", "it", "fr", "pt"], "pt": ["en", "it", "fr", "es"], "it": ["en", "fr", "pt", "es"],
           "de": ["en", "it", "fr", "pt"]}


# ------------------------------------------------------------------ items
@dataclass
class Item:
    id: str
    lang: str
    text: str
    set: str
    group: str                       # type (E1) or category (E2) or 'atlas'
    hot: list = field(default_factory=list)
    flip: list = None
    context: str = None
    parts: list = None               # passages: [(start, end)] char spans of the sentences
    parent: str = None               # passage sentences run alone: the passage id
    source: str = None

    def spans(self):
        """[(name, start, end)] char spans of the hot substrings ('x#2' = second occurrence)."""
        out = []
        for h in self.hot or []:
            sub, n = (h.rsplit("#", 1)[0], int(h.rsplit("#", 1)[1])) if re.search(r"#\d+$", h) else (h, 1)
            pos = -1
            for _ in range(n):
                pos = self.text.find(sub, pos + 1)
                if pos < 0:
                    raise ValueError(f"{self.id}: hot span {h!r} not in text")
            out.append((h, pos, pos + len(sub)))
        return out


def load_builtin():
    d = load_yaml(BDATA / "builtin.yaml")
    return [Item(id=r["id"], lang="es", text=r["text"], set="e1", group=r["type"], hot=r.get("hot", []),
                 flip=r.get("flip")) for r in d["items"]], d["types"]


def load_e2():
    """data/sentences.yaml (es/pt/it/de, not optional) + the father's context versions + passages (whole and per
    sentence)."""
    raw = load_yaml(DATA / "sentences.yaml")["sentences"]
    out = [Item(id=r["id"], lang=r["lang"], text=r["text"], set="e2", group=r["category"],
                hot=list(r.get("hot") or r.get("probe") or []), source=r.get("source"))
           for r in raw if r["lang"] in TARGETS and not r.get("optional")]
    ex = load_yaml(BDATA / "extra.yaml")
    out += [Item(id=r["id"], lang=r["lang"], text=r["text"], set="e2", group=r["category"], hot=r.get("hot", []),
                 context=r.get("context")) for r in ex["sentences"]]
    for p in ex["passages"]:
        text = " ".join(p["sentences"])
        parts, pos = [], 0
        for s in p["sentences"]:
            a = text.index(s, pos); parts.append((a, a + len(s))); pos = a + len(s)
        out.append(Item(id=p["id"], lang=p["lang"], text=text, set="e2", group=p["category"], hot=p.get("hot", []),
                        parts=parts))
        for j, s in enumerate(p["sentences"]):
            hot = [h for h in p.get("hot", []) if h in s]
            out.append(Item(id=f"{p['id']}_s{j + 1}", lang=p["lang"], text=s, set="e2", group=p["category"] + "_sent",
                            hot=hot, parent=p["id"]))
    return out


E3_SENTENCES = ["es_duende", "es_duende_flamenco", "es_duende_cuento",
                "es_sobremesa", "pt_saudade", "de_fernweh", "it_sprezzatura", "pt_jeitinho", "de_gemuetlichkeit",
                "es_tren", "es_casa", "it_negozio", "de_tisch"]


def load_e3():
    e2 = {i.id: i for i in load_e2()}
    b, _ = load_builtin()
    out = [Item(**{**e2[k].__dict__, "set": "e3"}) for k in E3_SENTENCES]
    seen = {}
    for i in b:                       # 2 built-in items per type
        if seen.get(i.group, 0) < 2:
            seen[i.group] = seen.get(i.group, 0) + 1
            out.append(Item(**{**i.__dict__, "set": "e3"}))
    return out


def load_atlas(n=None):
    d = load_yaml(BDATA / "atlas.yaml")
    items = [Item(id=r["id"], lang="es", text=r["text"], set="atlas", group="atlas", source=d["source"])
             for r in d["sentences"]]
    return items[:n] if n else items


def load_set(name, **kw):
    return {"e1": lambda: load_builtin()[0], "e2": load_e2, "e3": load_e3, "atlas": lambda: load_atlas(kw.get("n"))}[name]()


def channels_for(item, set_name, langs=None):
    """Channels collected for an item in a set (src first)."""
    if set_name == "e3":
        return ["src", "en", *PRACTICES]
    if set_name == "atlas":
        return ["src", *(langs or ["en"])]
    tg = langs or TARGETS[item.lang]
    return ["src", "src2", "copy", *tg]


# ------------------------------------------------------------------ io
def res_dir(model, *sub):
    d = RESULTS.joinpath(model, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def grad_path(model, set_name, item_id, channel):
    d = CACHE / model / "grads" / set_name / item_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{channel}.npz"


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_np) + "\n")


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _np(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def cfg_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("force",)}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def stage(name):
    """Resumable stage: skipped when <res>/<model>/.done/<name>[_<set>].json records the same config hash and its
    outputs exist (unless cfg['force']).  GPU stages are additionally resumable per (item, channel) file."""
    def deco(fn):
        @functools.wraps(fn)
        def run(cfg):
            model = cfg.get("model", "_global")
            tag = "_".join(str(cfg[k]) for k in ("set",) if k in cfg)
            mark = res_dir(model, ".done") / f"{name}{'_' + tag if tag else ''}.json"
            h = cfg_hash(cfg)
            if mark.exists() and not cfg.get("force"):
                m = json.loads(mark.read_text())
                if m.get("hash") == h and all(Path(p).exists() for p in m.get("outputs", [])):
                    print(f"[skip] {name} {tag} ({model}): done {m['when']}", flush=True)
                    return [Path(p) for p in m["outputs"]]
            t0 = time.time()
            print(f"[stage] {name} {tag} ({model}) {time.strftime('%H:%M')}", flush=True)
            outs = fn(cfg) or []
            save_json(mark, dict(hash=h, cfg=cfg, outputs=[str(p) for p in outs], when=time.strftime("%a %d %b %H:%M"),
                                 minutes=round((time.time() - t0) / 60, 1)))
            print(f"[done] {name} {tag} ({model}) in {(time.time() - t0) / 60:.1f} min", flush=True)
            return outs
        run.stage_name = name
        return run
    return deco
