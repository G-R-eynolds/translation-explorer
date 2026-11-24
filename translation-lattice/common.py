"""Shared helpers: model registry and loading, fixed chat templating, batched sampling, and
prefix-cached teacher-forced scoring of continuations.

Every script takes --model (a key of MODELS); outputs go to runs/<model>/.
"""
import copy, json, os, random, re, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MODELS = {
    # Primary for now (cached on the desktop, full-precision internals).
    "3b": dict(hf_id="meta-llama/Llama-3.2-3B-Instruct", quant=None, family="llama"),
    # Handoff's primary; needs the licence accepted on HF.  nf4 changes activations: flag in results.
    "8b": dict(hf_id="meta-llama/Llama-3.1-8B-Instruct", quant="nf4", family="llama"),
    # Second model for the epistemic analysis (§8).
    "qwen7b": dict(hf_id="Qwen/Qwen2.5-7B-Instruct", quant="nf4", family="qwen"),
}
# Llama's chat template stamps today's date into the system prompt; pin it so prompts are reproducible.
CHAT_DATE = "26 Jul 2024"


def run_dir(model_key):
    d = ROOT / "runs" / model_key
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d


def load_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_model(model_key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if spec["family"] == "llama" else tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if spec["quant"] == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **kw).eval()
    return tok, model


def describe(model_key):
    s = MODELS[model_key]
    return f"{s['hf_id']} ({s['quant'] or 'bf16'})"


def chat(tok, user, assistant_prefix=None):
    """Render a single-turn chat prompt as text, ready for generation (or for an assistant prefix)."""
    txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                  add_generation_prompt=True, date_string=CHAT_DATE)
    return txt + (assistant_prefix or "")


def end_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def stop_token(tok):
    """The token that ends an assistant turn (appended when scoring a whole response)."""
    for t in ("<|eot_id|>", "<|im_end|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            return i
    return tok.eos_token_id


def generate(model, tok, prompts, n=1, max_new_tokens=600, temperature=1.0, top_p=1.0, seed=0, batch=16):
    """Sample n completions for each prompt text. Returns (texts[prompt][k], gen_ids[prompt][k], truncated)."""
    import torch
    tok.padding_side = "left"
    texts, ids_out, trunc = [[] for _ in prompts], [[] for _ in prompts], [[] for _ in prompts]
    stops = end_ids(tok)
    # Expand (prompt, k) pairs into flat jobs so batches stay full.
    jobs = [(p, k) for p in range(len(prompts)) for k in range(n)]
    for c in range(0, len(jobs), batch):
        chunk = jobs[c:c + batch]
        seed_all(seed * 100003 + c)
        enc = tok([prompts[p] for p, _ in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                  eos_token_id=stops)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p, top_k=0)
        else:
            kw.update(temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        for (p, _), g in zip(chunk, gen):
            hit = [i for i, t in enumerate(g) if t in stops]
            cut = hit[0] if hit else len(g)
            g = [t for t in g[:cut] if t != tok.pad_token_id]
            ids_out[p].append(g)
            texts[p].append(tok.decode(g, skip_special_tokens=True))
            trunc[p].append(not hit)
    return texts, ids_out, trunc


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def score_continuations(model, tok, prefix_text, conts, batch=48, append_stop=True, per_token=False, temperature=1.0):
    """log p(cont | prefix) for each continuation string: SUM of token log-probs (no length normalisation).

    The prefix is run once and its KV cache is shared by every continuation.  With temperature T the
    scores are under softmax(logits / T), i.e. exactly the distribution generate() samples from at T.  If append_stop, the
    end-of-turn token is scored too, so this is the probability of the complete response string.
    """
    import torch
    pre = torch.tensor(encode(tok, prefix_text), device=model.device)
    stop = stop_token(tok)
    cont_ids = [encode(tok, c) + ([stop] if append_stop else []) for c in conts]
    with torch.no_grad():
        o = model(input_ids=pre[None], use_cache=True)
        cache0, first_logits = o.past_key_values, o.logits[0, -1].float() / temperature
        first_lp = torch.log_softmax(first_logits, -1)
        P = pre.shape[0]
        sums, toks = [], []
        for c in range(0, len(cont_ids), batch):
            chunk = cont_ids[c:c + batch]
            B, Lm = len(chunk), max(len(x) for x in chunk)
            ids = torch.full((B, Lm), tok.pad_token_id, device=model.device)
            m = torch.zeros((B, Lm), device=model.device, dtype=torch.long)
            for b, x in enumerate(chunk):
                ids[b, :len(x)] = torch.tensor(x, device=model.device); m[b, :len(x)] = 1
            cache = copy.deepcopy(cache0)
            cache.batch_repeat_interleave(B)
            mask = torch.cat([torch.ones((B, P), device=model.device, dtype=torch.long), m], 1)
            # Run the trunk only, then project to the vocabulary just at real tokens, in chunks: the full
            # (B, L, 128k) fp32 logits tensor does not fit next to the model on 12 GB.
            h = model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True).last_hidden_state
            tgt = ids[:, 1:]
            valid = m[:, 1:].bool()
            hv, tv = h[:, :-1][valid], tgt[valid]
            lpv = torch.empty(len(tv), device=model.device)
            for s0 in range(0, len(tv), 2048):
                lg = model.lm_head(hv[s0:s0 + 2048]).float() / temperature
                lpv[s0:s0 + 2048] = lg.gather(1, tv[s0:s0 + 2048, None])[:, 0] - torch.logsumexp(lg, -1)
            lp_rest = torch.zeros(tgt.shape, device=model.device)
            lp_rest[valid] = lpv
            lp = torch.cat([first_lp[ids[:, 0]][:, None], lp_rest], 1) * m
            sums += lp.sum(1).tolist()
            if per_token:
                toks += [lp[b, :len(x)].tolist() for b, x in enumerate(chunk)]
            del h, cache
    return (np.array(sums), toks) if per_token else np.array(sums)


NUM_RE = re.compile(r"^\s*\**\s*(\d{1,2})\s*[.)]\**\s*(.*?)\s*$")


def parse_numbered(text, n):
    """Parse 'k. text' lines. Returns list of n strings, or None if lines 1..n aren't each present once."""
    got = {}
    for raw in text.splitlines():
        mm = NUM_RE.match(raw)
        if not mm:
            continue
        k, t = int(mm.group(1)), mm.group(2).strip().strip('"“”').strip()
        if k in got or not t:
            return None
        got[k] = t
    if sorted(got) != list(range(1, n + 1)):
        return None
    return [got[k] for k in range(1, n + 1)]


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        print(f"[time] {self.label}: {time.time() - self.t:.1f}s", flush=True)
