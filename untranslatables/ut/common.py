"""Paths, model registry, loading, chat templating, batched generation, timing.

Every experiment script takes --model (a key of MODELS) and writes to results/<model>/.
"""
import hashlib, json, os, random, time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"          # activation memmaps; large, never synced back to the laptop

MODELS = {
    # key: hf_id, quantisation, chat-template quirks.  'tiny' and 'qwen05' exist only for CPU smoke tests.
    "3b": dict(hf_id="meta-llama/Llama-3.2-3B-Instruct", quant=None, family="llama"),
    "8b": dict(hf_id="meta-llama/Llama-3.1-8B-Instruct", quant="nf4", family="llama"),
    "qwen7b": dict(hf_id="Qwen/Qwen2.5-7B-Instruct", quant="nf4", family="qwen"),
    "aya8b": dict(hf_id="CohereForAI/aya-23-8B", quant="nf4", family="cohere"),
    "mistral7b": dict(hf_id="mistralai/Mistral-7B-Instruct-v0.3", quant="nf4", family="mistral"),
    # internals run (HANDOFF_internals.md); gemma12b is the main model (Gemma Scope 2 has IT SAEs for it)
    "gemma12b": dict(hf_id="google/gemma-3-12b-it", quant="nf4", family="gemma3"),
    "gemma9b": dict(hf_id="google/gemma-2-9b-it", quant="nf4", family="gemma2"),
    "qwen14b": dict(hf_id="Qwen/Qwen2.5-14B-Instruct", quant="nf4", family="qwen"),
    "nemo12b": dict(hf_id="mistralai/Mistral-Nemo-Instruct-2407", quant="nf4", family="mistral"),
    "ayax8b": dict(hf_id="CohereForAI/aya-expanse-8b", quant="nf4", family="cohere"),   # licence NOT accepted (403)
    "sarvam2b": dict(hf_id="sarvamai/sarvam-1", quant=None, family="llama", base=True),
    # blind spots run (HANDOFF_blind_spots.md); g3_4b is the main model
    "g3_4b": dict(hf_id="google/gemma-3-4b-it", quant=None, family="gemma3"),
    "g3_4b_pt": dict(hf_id="google/gemma-3-4b-pt", quant=None, family="gemma3", base=True),
    "g3_12b": dict(hf_id="google/gemma-3-12b-it", quant="nf4", family="gemma3"),
    "qwen05": dict(hf_id="Qwen/Qwen2.5-0.5B-Instruct", quant=None, family="qwen", cpu_ok=True),
    "tiny": dict(hf_id="hf-internal-testing/tiny-random-LlamaForCausalLM", quant=None, family="llama", cpu_ok=True),
}
CHAT_DATE = "26 Jul 2024"       # Llama's template stamps today's date; pinned for reproducibility
LANG_NAMES = {"pt": "Português", "de": "Deutsch", "it": "Italiano", "fr": "Français", "es": "Español",
              "en": "English", "rom": "Romanesco", "ru": "Русский", "ja": "日本語", "hi": "हिन्दी", "da": "Dansk"}
LANG_EN = {"pt": "Portuguese", "de": "German", "it": "Italian", "fr": "French", "es": "Spanish", "en": "English",
           "rom": "Romanesco (the 19th-century dialect of Rome)", "ru": "Russian", "ja": "Japanese", "hi": "Hindi",
           "da": "Danish"}


# ------------------------------------------------------------------ io
def out_dir(model_key, *sub):
    d = ROOT / "results" / model_key
    for s in sub:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(model_key):
    d = CACHE / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_np_default))


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_yaml(p):
    return yaml.safe_load(Path(p).read_text())


def stable_hash(*parts):
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------ models
def load_model(model_key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[model_key]
    cuda = torch.cuda.is_available()
    if not cuda and not spec.get("cpu_ok"):
        raise SystemExit(f"{model_key} needs the GPU desktop; on CPU use --model qwen05 or tiny")
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16 if cuda else torch.float32, device_map="cuda" if cuda else "cpu")
    if spec["quant"] == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **kw).eval()
    model.key = model_key
    return tok, model


def n_layers(model):
    from ut.arch import n_layers as _n
    return _n(model)


def describe(model_key):
    s = MODELS[model_key]
    return f"{s['hf_id']} ({s['quant'] or 'bf16'})"


def chat(tok, user, assistant_prefix=""):
    """`user` is a string (one turn) or a list of {"role", "content"} messages ending with a user turn."""
    msgs = [{"role": "user", "content": user}] if isinstance(user, str) else user
    if tok.chat_template is None:            # tiny test model: plain text
        return "".join(f"{m['role'].title()}: {m['content']}\n" for m in msgs) + "Assistant: " + assistant_prefix
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE)
    return txt + assistant_prefix


def stop_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|im_end|>", "<|END_OF_TURN_TOKEN|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def generate(model, tok, prompts, n=1, max_new_tokens=32, temperature=1.0, seed=0, batch=32, stop_at_newline=False):
    """Sample n completions per prompt text (prompts already chat-formatted). Pure ancestral sampling at
    temperature T (no top-p/top-k) unless T == 0 (greedy). Returns texts[prompt][k]."""
    import torch
    tok.padding_side = "left"
    stops = stop_ids(tok)
    jobs = [(p, k) for p in range(len(prompts)) for k in range(n)]
    out = [[] for _ in prompts]
    for c in range(0, len(jobs), batch):
        chunk = jobs[c:c + batch]
        seed_all(seed * 100003 + c)
        enc = tok([prompts[p] for p, _ in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=stops)
        kw.update(dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0
                  else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
        with torch.no_grad():
            gen = model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:].tolist()
        for (p, _), g in zip(chunk, gen):
            cut = next((i for i, t in enumerate(g) if t in stops), len(g))
            t = tok.decode([x for x in g[:cut] if x != tok.pad_token_id], skip_special_tokens=True)
            out[p].append(t.split("\n")[0].strip() if stop_at_newline else t.strip())
    return out


# ------------------------------------------------------------------ timing
class Progress:
    """Prints elapsed time and an ETA; call .tick(n_done_units). Prints a warning when the projected
    total exceeds 15 minutes (handoff guardrail)."""

    def __init__(self, label, total):
        self.label, self.total, self.t0, self.warned = label, total, time.time(), False

    def tick(self, done):
        el = time.time() - self.t0
        eta = el / max(done, 1) * (self.total - done)
        if not self.warned and done and el + eta > 900:
            print(f"[eta] {self.label}: projected {(el + eta) / 60:.0f} min total (>15 min)", flush=True)
            self.warned = True
        print(f"[{self.label}] {done}/{self.total}  {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        print(f"[time] {self.label}: {time.time() - self.t:.1f}s", flush=True)


def env_int(name, default):
    return int(os.environ.get(name, default))
