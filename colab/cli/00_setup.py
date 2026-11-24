# 00_setup.py: run once per Colab session (via `colab exec -f`), before any stage.
# Settings can be overridden by executing assignments first, e.g.  echo "MODEL_ID = '...'" | colab exec -s ut
g = globals()
MODEL_ID    = g.get("MODEL_ID", "unsloth/Llama-3.3-70B-Instruct-bnb-4bit")   # alt: unsloth/Qwen2.5-72B-Instruct-bnb-4bit; dry run: Qwen/Qwen2.5-0.5B-Instruct
DRY_RUN     = g.get("DRY_RUN", any(s in MODEL_ID for s in ["0.5B", "1.5B", "tiny"]))
OUT_ROOT    = g.get("OUT_ROOT", "/content/ut_results")       # on the VM; fetched with `colab download` (see pack())
LOCAL_CACHE = g.get("LOCAL_CACHE", "/content/cache")         # big covariance files; stay on the VM
RUN_A, RUN_B, RUN_C, RUN_D = g.get("RUN_A", True), g.get("RUN_B", True), g.get("RUN_C", True), g.get("RUN_D", True)
PAIRS       = g.get("PAIRS", ["es", "hi"])
N_SENT_FIT  = g.get("N_SENT_FIT", 2000 if DRY_RUN else 30000)
N_SENT_HELD = g.get("N_SENT_HELD", 200 if DRY_RUN else 1200)
LAMBDAS     = g.get("LAMBDAS", [1e-3, 1e-2, 1e-1])
SEED        = g.get("SEED", 0)
CHAT_DATE   = "26 Jul 2024"
MIN_GPU_GB  = g.get("MIN_GPU_GB", 0 if DRY_RUN else 75)

import subprocess, sys, matplotlib
matplotlib.use("Agg")
import torch
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0); print(f"GPU: {_p.name}, {_p.total_memory/1e9:.0f} GB")
    if _p.total_memory / 1e9 < MIN_GPU_GB:
        raise RuntimeError(f"This GPU has {_p.total_memory/1e9:.0f} GB; the 4-bit 70B model needs an 80 GB card. "
                           "Stop this session and create one with --gpu H100 (or an 80 GB A100).")
else:
    print("WARNING: no GPU visible")
try:
    import spacy; spacy.load("es_core_news_sm")
except Exception:
    subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm", "-q"], check=False)


import os, re, json, math, time, random, difflib, gc
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

def slug(s): return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
OUT = os.path.join(OUT_ROOT, slug(MODEL_ID)); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
CACHE = os.path.join(LOCAL_CACHE, slug(MODEL_ID)); os.makedirs(CACHE, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def _np(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.tolist()
    raise TypeError(type(o))
def save_json(name, obj):
    with open(os.path.join(OUT, name), "w") as f: json.dump(obj, f, indent=1, default=_np, ensure_ascii=False)
def load_json(name):
    with open(os.path.join(OUT, name)) as f: return json.load(f)
def have(name): return os.path.exists(os.path.join(OUT, name))

class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t = time.time(); print(f"▶ {self.label} ..."); return self
    def __exit__(self, *a): print(f"✓ {self.label}: {time.time()-self.t:.0f}s")

def auroc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
def rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort"); r = np.empty(len(a))
    sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j+1] == sa[i]: j += 1
        r[order[i:j+1]] = (i + j) / 2 + 1; i = j + 1
    return r
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3: return float("nan")
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1])
def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); m = rng.choice(x, (n, len(x))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))

# Visual language shared with the earlier figures
EASY, HARD, LOAN, GREY, INK = "#2f7d6d", "#c4572e", "#5b4fc4", "#d9d6cf", "#2b2b2b"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight")
    plt.show(); plt.close(fig)
print("results ->", OUT)

LANG_NAME = {"es": "Spanish", "pt": "Portuguese", "it": "Italian", "de": "German", "hi": "Hindi", "en": "English"}

SENTENCES = [
    # --- untranslatable words in context ---
    dict(id="duende", lang="es", cat="untranslatable", text="Me gusta mucho el duende.",
         hot={"duende": "the mysterious, almost demonic power and emotional intensity of a great artistic performance, as in flamenco (Lorca's sense)",
              "gusta": "it pleases me: the thing is the subject and I am the one affected, like having a taste for it"}),
    dict(id="sobremesa", lang="es", cat="untranslatable", text="Después de comer nos quedamos dos horas de sobremesa.",
         hot={"sobremesa": "the relaxed time spent talking at the table after a meal has finished"}),
    dict(id="madrugada", lang="es", cat="untranslatable", text="Llegamos a casa de madrugada, muertos de sueño.",
         hot={"madrugada": "the small hours, between midnight and dawn"}),
    dict(id="estrenar", lang="es", cat="untranslatable", text="Hoy estreno los zapatos que me regalaste.",
         hot={"estreno": "I am using or wearing something for the very first time"}),
    dict(id="verguenza", lang="es", cat="untranslatable", text="Sentí vergüenza ajena durante todo su discurso.",
         hot={"vergüenza ajena": "embarrassment felt on behalf of someone else who is embarrassing themselves"}),
    dict(id="tertulia", lang="es", cat="untranslatable", text="Los jueves vamos a la tertulia del café de la esquina.",
         hot={"tertulia": "a regular informal gathering of people for conversation, often literary or political"}),
    # --- constructions ---
    dict(id="olvido", lang="es", cat="construction", text="Se me olvidó la llave en el coche.",
         hot={"Se me olvidó": "I forgot it, but framed as something that happened to me by accident, as if the key forgot itself on me"}),
    dict(id="estar_aburrido", lang="es", cat="construction", text="Juan está aburrido en la reunión.",
         hot={"está aburrido": "is bored, a temporary state (not 'is boring')"}),
    dict(id="ser_aburrido", lang="es", cat="construction", text="Juan es aburrido en las reuniones.",
         hot={"es aburrido": "is boring, a lasting characteristic (not 'is bored')"}),
    dict(id="ojala", lang="es", cat="construction", text="Ojalá llueva mañana por la tarde.",
         hot={"Ojalá": "a strong hope or wish, 'if only', originally 'may God will it'"}),
    # --- idioms ---
    dict(id="pelo", lang="es", cat="idiom", text="Creo que me estás tomando el pelo.",
         hot={"tomando el pelo": "teasing me, pulling my leg"}),
    dict(id="nubes", lang="es", cat="idiom", text="Mi hermana siempre está en las nubes.",
         hot={"en las nubes": "daydreaming, absent-minded"}),
    dict(id="pelos", lang="es", cat="idiom", text="Mi abuela no tiene pelos en la lengua.",
         hot={"no tiene pelos en la lengua": "speaks very frankly, without holding anything back"}),
    # --- controls ---
    dict(id="tren", lang="es", cat="control", text="El tren sale a las ocho de la mañana.", hot={}),
    dict(id="hermano", lang="es", cat="control", text="Mi hermano vive en un piso pequeño.", hot={}),
    dict(id="pan", lang="es", cat="control", text="Compré pan y leche en la tienda.", hot={}),
    dict(id="reunion", lang="es", cat="control", text="La reunión empieza a las diez en punto.", hot={}),
    dict(id="gatos", lang="es", cat="control", text="Tengo dos gatos negros y un perro.", hot={}),
    dict(id="invierno", lang="es", cat="control", text="En invierno hace mucho frío en el norte.", hot={}),
    # --- other European languages (Part B only) ---
    dict(id="saudade", lang="pt", cat="untranslatable", text="Tenho muita saudade da minha avó.",
         hot={"saudade": "a deep, bittersweet longing for someone or something absent, that may never return"}),
    dict(id="magari", lang="it", cat="construction", text="Magari fosse vero!",
         hot={"Magari": "if only (a wistful wish)"}),
    dict(id="fernweh", lang="de", cat="untranslatable", text="Im Winter habe ich immer Fernweh.",
         hot={"Fernweh": "an ache to travel to faraway places, the opposite of homesickness"}),
    # --- Hindi (DRAFTS: to be checked by a native speaker) ---
    dict(id="hi_pasand", lang="hi", cat="construction", draft=True, text="मुझे चाय बहुत पसंद है।",
         hot={"मुझे": "to me (dative experiencer: the tea is liked by me, like Spanish 'me gusta')"}),
    dict(id="hi_jugaad", lang="hi", cat="untranslatable", draft=True, text="उसने जुगाड़ से पुरानी गाड़ी ठीक कर दी।",
         hot={"जुगाड़": "a clever improvised fix made with whatever is at hand"}),
    dict(id="hi_mann", lang="hi", cat="untranslatable", draft=True, text="आज पढ़ाई में मेरा मन नहीं लग रहा।",
         hot={"मन": "heart-mind: the seat of feeling, will and attention together"}),
    dict(id="hi_apnapan", lang="hi", cat="untranslatable", draft=True, text="इस घर में मुझे बहुत अपनापन लगता है।",
         hot={"अपनापन": "a warm sense of belonging and intimate familiarity, of being among one's own"}),
    dict(id="hi_train", lang="hi", cat="control", draft=True, text="ट्रेन सुबह आठ बजे चलती है।", hot={}),
    dict(id="hi_bhai", lang="hi", cat="control", draft=True, text="मेरा भाई दिल्ली में रहता है।", hot={}),
]

# Sense minimal pairs for Part D: same word, two contexts, two senses.
# The English candidates differ only in the critical word, so their log-odds measures which sense the translation carries.
SENSE_PAIRS = [
    dict(id="cura", word="cura", a="Me preocupa el cura.", b="Me preocupa la cura.",
         en_a="I am worried about the {}.", en_b="I am worried about the {}.", sense_a="priest", sense_b="cure"),
    dict(id="cometa", word="cometa", a="Miramos el cometa.", b="Miramos la cometa.",
         en_a="We looked at the {}.", en_b="We looked at the {}.", sense_a="comet", sense_b="kite"),
    dict(id="pendiente", word="pendiente", a="Vi el pendiente.", b="Vi la pendiente.",
         en_a="I saw the {}.", en_b="I saw the {}.", sense_a="earring", sense_b="slope"),
    dict(id="corte", word="corte", a="Me sorprendió el corte.", b="Me sorprendió la corte.",
         en_a="The {} surprised me.", en_b="The {} surprised me.", sense_a="cut", sense_b="court"),
    dict(id="frente", word="frente", a="Señaló el frente.", b="Señaló la frente.",
         en_a="He pointed at the {}.", en_b="He pointed at the {}.", sense_a="front", sense_b="forehead"),
    dict(id="papa", word="papa", a="Me gusta el papa.", b="Me gusta la papa.",
         en_a="I like the {}.", en_b="I like the {}.", sense_a="pope", sense_b="potato"),
    dict(id="aburrido", word="aburrido", a="Juan es aburrido.", b="Juan está aburrido.",
         en_a="Juan is {}.", en_b="Juan is {}.", sense_a="boring", sense_b="bored"),
    dict(id="duende_ctx", word="duende", a="En ese cuento infantil, lo que más me gusta es el duende.",
         b="En ese cante flamenco, lo que más me gusta es el duende.",
         en_a="In that children's story, what I like most is the {}.",
         en_b="In that flamenco song, what I like most is the {}.", sense_a="goblin", sense_b="soul"),
]

SID = {S["id"]: S for S in SENTENCES}

# Part A: Spanish -> Italian word translation, with the English equivalent for the logit lens
DETOUR_SHOTS = [("perro", "cane"), ("ventana", "finestra"), ("mesa", "tavolo"), ("leche", "latte")]
DETOUR_WORDS = [("árbol", "albero", "tree"), ("pájaro", "uccello", "bird"), ("manzana", "mela", "apple"),
    ("cuchillo", "coltello", "knife"), ("calle", "strada", "street"), ("niño", "bambino", "child"),
    ("nube", "nuvola", "cloud"), ("queso", "formaggio", "cheese"), ("llave", "chiave", "key"),
    ("zapato", "scarpa", "shoe"), ("camisa", "camicia", "shirt"), ("mantequilla", "burro", "butter"),
    ("cerdo", "maiale", "pig"), ("caballo", "cavallo", "horse"), ("hermano", "fratello", "brother"),
    ("pescado", "pesce", "fish"), ("cuchara", "cucchiaio", "spoon"), ("tenedor", "forchetta", "fork"),
    ("silla", "sedia", "chair"), ("puerta", "porta", "door"), ("ciudad", "città", "city"),
    ("rodilla", "ginocchio", "knee"), ("hielo", "ghiaccio", "ice"), ("lluvia", "pioggia", "rain"),
    ("nieve", "neve", "snow"), ("hoja", "foglia", "leaf"), ("lengua", "lingua", "tongue"),
    ("oreja", "orecchio", "ear"), ("huevo", "uovo", "egg"), ("ajo", "aglio", "garlic"),
    ("cerebro", "cervello", "brain"), ("espejo", "specchio", "mirror")]

from transformers import AutoTokenizer, AutoModelForCausalLM
with Timer(f"load {MODEL_ID}"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    _dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=_dtype, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=_dtype, device_map="auto")
    model.eval()

def _get_attr_path(obj, path):
    for p in path.split("."): obj = getattr(obj, p)
    return obj
def find_decoder(m):
    for base in ["model", "model.model", "model.language_model", "transformer"]:
        try:
            b = _get_attr_path(m, base)
            layers = getattr(b, "layers", None) or getattr(b, "h", None)
            norm = getattr(b, "norm", None) or getattr(b, "ln_f", None)
            if layers is not None and norm is not None: return b, layers, norm
        except AttributeError: pass
    raise ValueError("could not find decoder layers")

BASE, LAYERS_MOD, FINAL_NORM = find_decoder(model)
LM_HEAD = model.get_output_embeddings()
N_LAYERS = len(LAYERS_MOD)
D_MODEL = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
INPUT_DEV = model.get_input_embeddings().weight.device
BOS = [tok.bos_token_id] if tok.bos_token_id is not None else []
print(f"{N_LAYERS} layers, d={D_MODEL}, BOS={BOS}, input device {INPUT_DEV}")
if torch.cuda.is_available(): print(f"GPU memory in use: {torch.cuda.memory_allocated()/1e9:.1f} GB")

WORD_RE = re.compile(r"[\wऀ-ॣ०-ॿ]+(?:['’]\w+)?")
def words_with_spans(text): return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]
def enc(text, offsets=False): return tok(text, add_special_tokens=False, return_offsets_mapping=offsets)
def ntok(text): return len(enc(text)["input_ids"])

def chat_prompt(user, prefix=""):
    msgs = [{"role": "user", "content": user}]
    try: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE)
    except Exception: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + prefix

def tokens_in_span(offsets, a, b):
    return [i for i, (x, y) in enumerate(offsets) if y > x and y > a and x < b]

def find_phrase(text, phrase):
    m = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.I)
    return (m.start(), m.end()) if m else None

def positions_at(prompt, a, b):
    return tokens_in_span(enc(prompt, offsets=True)["offset_mapping"], a, b)

def positions_in_prompt(prompt, sentence, phrase=None, span=None):
    """Token positions of `phrase` (or of the character span `span`) inside `sentence`, inside `prompt`."""
    base = prompt.rfind(sentence); assert base >= 0, "sentence not in prompt"
    sp = span or find_phrase(sentence, phrase); assert sp is not None, f"{phrase!r} not in {sentence!r}"
    return positions_at(prompt, base + sp[0], base + sp[1])

class _Stop(Exception): pass

def make_edit_hook(edits_by_b):
    """edits_by_b: {batch index: [(positions LongTensor, fn)]}; fn maps states [k,d] -> new states [k,d]."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1: return None              # cached decoding step: leave untouched
        for b, items in edits_by_b.items():
            for pos, fn in items:
                pos = pos.to(h.device)
                if int(pos.max()) < h.shape[1]:
                    h[b, pos] = fn(h[b, pos]).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook

def _register_edits(edits_for_batch):
    """edits_for_batch: list (one per batch row) of lists of (layer, positions, fn)."""
    per_layer = defaultdict(lambda: defaultdict(list))
    for b, items in enumerate(edits_for_batch):
        for (layer, pos, fn) in (items or []):
            per_layer[layer][b].append((torch.as_tensor(pos, dtype=torch.long), fn))
    return [LAYERS_MOD[layer - 1].register_forward_hook(make_edit_hook(dict(eb))) for layer, eb in per_layer.items()]

@torch.no_grad()
def score(prompts, conts, batch=8, edits=None):
    """Teacher-forced log p of each continuation given its prompt.
    edits: optional list (per item) of [(layer, prompt token positions, fn)]; 'layer l' = hidden_states[l] = output of block l-1.
    Returns per item: dict(lp=array over continuation tokens, offsets=char offsets of those tokens in the continuation)."""
    res = [None] * len(prompts)
    for i in range(0, len(prompts), batch):
        idx = list(range(i, min(i + batch, len(prompts))))
        seqs, spans, offs = [], [], []
        for j in idx:
            p = enc(prompts[j])["input_ids"]; c = enc(conts[j], offsets=True)
            seqs.append(p + c["input_ids"]); spans.append((len(p), len(p) + len(c["input_ids"]))); offs.append(c["offset_mapping"])
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
        for b, s in enumerate(seqs): ids[b, :len(s)] = torch.tensor(s); am[b, :len(s)] = 1
        handles = _register_edits([edits[j] for j in idx]) if edits is not None else []
        try:
            out = BASE(input_ids=ids.to(INPUT_DEV), attention_mask=am.to(INPUT_DEV))
        finally:
            for h in handles: h.remove()
        H = out.last_hidden_state
        for b, j in enumerate(idx):
            s, e = spans[b]
            if e <= s: res[j] = dict(lp=np.zeros(0), offsets=[]); continue
            logits = LM_HEAD(H[b, s - 1:e - 1].to(LM_HEAD.weight.device)).float()
            tgt = ids[b, s:e].to(logits.device)
            res[j] = dict(lp=torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0].cpu().numpy(), offsets=offs[b])
        del out, H
    return res

@torch.no_grad()
def generate(prompts, max_new_tokens=64, batch=16):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(prompts), batch):
        e = tok(prompts[i:i + batch], add_special_tokens=False, return_tensors="pt", padding=True).to(INPUT_DEV)
        g = model.generate(**e, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(r, skip_special_tokens=True).strip() for r in g[:, e["input_ids"].shape[1]:]]
    tok.padding_side = "right"
    return outs

@torch.no_grad()
def generate_with_edits(prompt, edits, max_new_tokens=40):
    """Greedy generation for one prompt, with edits applied to the prompt positions (prefill only)."""
    e = tok(prompt, add_special_tokens=False, return_tensors="pt").to(INPUT_DEV)
    handles = _register_edits([edits])
    try:
        g = model.generate(**e, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id)
    finally:
        for h in handles: h.remove()
    return tok.decode(g[0, e["input_ids"].shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def capture(ids, am, layers):
    """Residual states at `layers` (dict layer -> [B,T,d]); stops the forward pass after the deepest one."""
    store, deepest = {}, max(layers)
    def mk(l):
        def hook(m, i, o):
            store[l] = (o[0] if isinstance(o, tuple) else o).detach()
            if l == deepest: raise _Stop()
        return hook
    hs = [LAYERS_MOD[l - 1].register_forward_hook(mk(l)) for l in layers]
    try: BASE(input_ids=ids, attention_mask=am)
    except _Stop: pass
    finally:
        for h in hs: h.remove()
    return store

@torch.no_grad()
def capture_prompt_states(prompt, layer):
    e = tok(prompt, add_special_tokens=False, return_tensors="pt").to(INPUT_DEV)
    return capture(e["input_ids"], e["attention_mask"], [layer])[layer][0]

def encode_raw(text):
    e = enc(text, offsets=True)
    return BOS + e["input_ids"], [(0, 0)] * len(BOS) + list(e["offset_mapping"])

print("helpers ready")

RUNGS = ["B0", "B1", "B2", "B3", "B5", "B6"]
RUNG_NAME = {"B0": "word for word", "B1": "standard", "B2": "keep the word", "B3": "inline gloss",
             "B5": "footnote", "B6": "thick translation"}
NON_BORROW = ["B0", "B1", "B3", "B5", "B6"]
MAXTOK = {"B0": 64, "B1": 64, "B2": 72, "B3": 120, "B5": 150, "B6": 260}

def rung_user(r, s, L):
    n = LANG_NAME[L]
    return {
        "B0": f"Give a word-for-word English gloss of this {n} sentence: one English word for each {n} word, in the original order. Reply with the gloss only.\n\n{s}",
        "B1": f"Translate this {n} sentence into natural English. Reply with the translation only.\n\n{s}",
        "B2": f"Translate this {n} sentence into English, but keep any word that has no good English equivalent in the original {n}, in italics with *asterisks*. Reply with the translation only.\n\n{s}",
        "B3": f"Translate this {n} sentence into English. After each word or phrase that is hard to translate, add a short gloss in square brackets (at most 8 words). Do not quote the {n} words. Reply with the translation only.\n\n{s}",
        "B5": f"Translate this {n} sentence into English, then add one footnote (at most 40 words) explaining what the translation loses. Do not quote the {n} words. Format: the translation, a blank line, then 'Note: ...'.\n\n{s}",
        "B6": f"Translate this {n} sentence into English, then write one paragraph (at most 120 words) opening up the meanings, connotations and grammar of its hard parts, so that an English reader understands what the original says. Do not quote the {n} words.\n\n{s}",
    }[r]



def hot_word_index(S):
    idx = set()
    for span in S["hot"]:
        sp = find_phrase(S["text"], span)
        if sp: idx |= {k for k, (_, a, b) in enumerate(words_with_spans(S["text"])) if b > sp[0] and a < sp[1]}
    return idx




import tarfile
def pack():
    """Bundle everything under OUT into one archive for `colab download`."""
    p = f"{OUT_ROOT.rstrip('/')}_{slug(MODEL_ID)}.tar.gz"
    with tarfile.open(p, "w:gz") as t: t.add(OUT, arcname=slug(MODEL_ID))
    print("PACKED", p)


print("setup done:", MODEL_ID)
