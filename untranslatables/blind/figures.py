"""Figures (HANDOFF_blind_spots.md §8), PNG + SVG, plus captions.json with the number each caption rests on.

For the father (no symbols; teal = carried, coral = lost; Charter italic for foreign words):
  1 what English can't see      2 steering cards       3 duende with and without context
  4 what practices recover      5 the atlas (only once the clusters have labels)
For the user: 6a spectra vs null, 6b finite-difference check, 6c layer dependence, 6d ε stability.
Titles are chosen from the statistic (a neutral title when the numbers don't support the strong one).
A figure whose inputs are missing is skipped and says so.
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blind.common import EPS_HEAD, RESULTS, headline_layer, layers_for, read_jsonl, res_dir, stage
from sent.figures import CORAL, GREY, GRID, INK, MUTED, SANS, SERIF, SURF, TEAL, style

LANG = {"en": "English", "it": "Italian", "fr": "French", "pt": "Portuguese", "src2": "Spanish, reworded",
        "copy": "copy"}
TYPE = {"gender_adj": "speaker's gender\n(cansada / cansado)", "profession": "gender of a profession\n(médica / médico)",
        "plural_you": "plural 'you'\n(queréis / quieres)", "formal_you": "formal 'you'\n(usted / tú)",
        "ser_estar": "ser / estar\n(control)", "lexical": "ordinary sentence\n(control)",
        "aspect": "ongoing past\n(jugaba / jugué)"}
CAPTIONS = {}


def J(m, name):
    p = RESULTS / m / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, d, name, title, how, claim, audience):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK, family=SANS)
    fig.text(0.012, 0.012, how, ha="left", va="bottom", fontsize=10.5, color=MUTED, family=SANS, wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}", dpi=160, facecolor=SURF)
    plt.close(fig)
    CAPTIONS[name] = dict(title=title, how_to_read=how, claim=claim, audience=audience)
    print("wrote", name)


def skip(name, why):
    print(f"skip {name}: {why}")


def whole(m, S, layer=None, eps=EPS_HEAD):
    return [r for r in read_jsonl(RESULTS / m / f"spectra_{S}.jsonl") if r["scope"] == "whole" and r["eps"] == eps
            and (layer is None or r["layer"] == layer)]


# ------------------------------------------------------------------ 1
def f1(d, m):
    k = headline_layer(m)
    rows = whole(m, "e1", k)
    if not rows:
        return skip("1_what_english_cant_see", "no spectra_e1")
    types = [t for t in TYPE if any(r["group"] == t for r in rows)]
    langs = ["en", "it", "fr", "pt"]
    fig, axes = plt.subplots(1, len(types), figsize=(2.3 * len(types) + 1, 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    med = {}
    for ax, t in zip(axes, types):
        style(ax)
        for j, L in enumerate(langs):
            U = [r["U"]["med"] for r in rows if r["group"] == t and r["target"] == L]
            nul = [r["U_null"]["hi"] for r in rows if r["group"] == t and r["target"] == L]
            if not U:
                continue
            med[(t, L)] = float(np.median(U))
            col = CORAL if L == "en" else TEAL
            ax.bar(j, np.median(nul), width=0.8, color=GREY, alpha=0.45, zorder=1)
            ax.scatter(j + np.linspace(-0.18, 0.18, len(U)), U, s=16, color=col, alpha=0.75, zorder=3)
            ax.hlines(np.median(U), j - 0.32, j + 0.32, color=col, lw=3, zorder=4)
        ax.set_xticks(range(len(langs)), [LANG[L] for L in langs], rotation=45, ha="right", fontsize=9)
        ax.set_title(TYPE[t], fontsize=10, color=INK)
    axes[0].set_ylabel("how much the translation misses (bits)", fontsize=10)
    fig.subplots_adjust(bottom=0.2, top=0.8, left=0.07, right=0.98)
    gram = ["gender_adj", "plural_you", "formal_you"]
    en = np.median([med[(t, "en")] for t in gram if (t, "en") in med] or [np.nan])
    it = np.median([med[(t, "it")] for t in gram if (t, "it") in med] or [np.nan])
    V = J(m, "validation.json") or {}
    t1 = V.get("headline", {}).get("t1_pass", False)
    title = ("English loses the speaker's gender and the kind of 'you'; Italian keeps them" if t1 and en > it
             else "How much each language's translation misses, by kind of distinction")
    save(fig, d, "1_what_english_cant_see", title,
         "Each dot is one Spanish sentence; the bar is the typical value. Grey shows what chance alone produces. "
         "Higher = the translation reacts less to a change that the Spanish rewording reacts to.",
         dict(median_U_en_grammatical=en, median_U_it_grammatical=it, layer=k, medians={f"{a}|{b}": v for (a, b), v in med.items()}),
         "father")


# ------------------------------------------------------------------ 2
def _card(ax, y, r, langs=("src", "en")):
    lab = {"src": "Spanish, said again", "en": "English", "it": "Italian"}
    ax.text(0.0, y, r["text"], family=SERIF, style="italic", fontsize=13, color=INK, va="top")
    y -= 0.06
    for L in langs:
        tx = r.get(f"{L}_texts")
        if not tx:
            continue
        a, b = tx[1], tx[3]
        same = a.strip() == b.strip()
        ax.text(0.0, y, lab[L], fontsize=9.5, color=MUTED, va="top", family=SANS)
        ax.text(0.2, y, "\n".join(textwrap.wrap(a, 52)), fontsize=10.5, color=INK, va="top", family=SERIF)
        ax.text(0.6, y, "\n".join(textwrap.wrap(b, 52)), fontsize=10.5, color=TEAL if same else CORAL, va="top",
                family=SERIF)
        y -= 0.045 * max(len(textwrap.wrap(a, 52)), len(textwrap.wrap(b, 52)), 1) + 0.01
    return y - 0.03


def f2(d, m):
    st = read_jsonl(RESULTS / m / "steering_e1.jsonl")
    if not st:
        return skip("2_steering_cards", "no steering_e1")
    good = [r for r in st if r.get("flip") and r["flip"]["flipped_en_unchanged"]]
    pick, seen = [], set()
    for r in sorted(good, key=lambda r: -r["lam"]) + sorted(st, key=lambda r: -r["lam"]):
        if r["group"] not in seen and r["group"] in ("gender_adj", "profession", "plural_you", "formal_you", "lexical"):
            pick.append(r); seen.add(r["group"])
    pick = pick[:5]
    fig, ax = plt.subplots(figsize=(13, 2.2 * len(pick) + 1.4))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.2, 0.95, "nudged one way", fontsize=10, color=MUTED, family=SANS)
    ax.text(0.6, 0.95, "nudged the other way", fontsize=10, color=MUTED, family=SANS)
    y = 0.9
    for r in pick:
        y = _card(ax, y, r, ("src", "en", "it"))
    n_ok = sum(1 for r in st if r.get("flip") and r["flip"]["flipped_en_unchanged"])
    n_f = sum(1 for r in st if r.get("flip") and r["group"] in ("gender_adj", "profession", "plural_you", "formal_you"))
    title = "The Spanish changes, the English doesn't" if n_ok >= max(1, n_f / 2) else \
        "Nudging the model along its blind direction: what changes"
    save(fig, d, "2_steering_cards", title,
         "We nudge the model's inner picture of each sentence in one direction and the other, then ask it to say the "
         "sentence again in Spanish and in English. Coral = the text changed; teal = it stayed the same.",
         dict(auto_flips_with_english_unchanged=n_ok, of_test_items=n_f, items=[r["item"] for r in pick]), "father")


# ------------------------------------------------------------------ 3
def f3(d, m):
    H = J(m, "heatmaps.json")
    k = str(headline_layer(m))
    ids = ["es_duende_flamenco", "es_duende_cuento", "es_duende"]
    if not H or not all(i in H and k in H[i]["layers"] for i in ids[:2]):
        return skip("3_duende_context", "heat maps for the duende sentences missing")
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.8))
    fig.subplots_adjust(top=0.88, bottom=0.1)
    vmax = max(max(H[i]["layers"][k]["en"]["U"]) for i in ids if i in H)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("h", [SURF, "#f3d2c4", CORAL])
    dval = {}
    names = {"es_duende_flamenco": "flamenco context", "es_duende_cuento": "fairy-tale context", "es_duende": "no context"}
    for ax, i in zip(axes, ids):
        ax.axis("off")
        if i not in H:
            continue
        toks, U = H[i]["tokens"], H[i]["layers"][k]["en"]["U"]
        x = 0.0
        ax.text(0, 0.9, names[i], fontsize=10, color=MUTED, family=SANS, transform=ax.transAxes)
        for t, u in zip(toks, U):
            w = 0.012 * max(len(t), 2) + 0.01
            ax.add_patch(plt.Rectangle((x, 0.15), w - 0.004, 0.55, color=cmap(u / max(vmax, 1e-9)),
                                       transform=ax.transAxes))
            ax.text(x + w / 2, 0.42, t.strip() or "·", ha="center", va="center", family=SERIF, style="italic",
                    fontsize=12, color=INK, transform=ax.transAxes)
            x += w
        dv = [u for t, u in zip(toks, U) if "duend" in t]
        dval[i] = max(dv) if dv else None
    fl, cu = dval.get("es_duende_flamenco"), dval.get("es_duende_cuento")
    title = ("Duende is a blind spot only when it means the art, not the goblin" if fl and cu and fl > 1.5 * cu
             else "Where English translation misses most: duende in two contexts")
    save(fig, d, "3_duende_context", title,
         "Darker = English translation misses more of what that word contributes (compared with saying it again in "
         "Spanish).", dict(duende_U_flamenco=fl, duende_U_goblin=cu, duende_U_bare=dval.get("es_duende"), layer=k),
         "father")


# ------------------------------------------------------------------ 4
def f4(d, m):
    P = J(m, "practices.json")
    if not P:
        return skip("4_practices", "no practices.json")
    order = ["plain", "p_keep", "p_gloss", "p_foot", "p_thick"]
    lab = {"plain": "plain translation", "p_keep": "keep the word", "p_gloss": "gloss in brackets",
           "p_foot": "footnote", "p_thick": "thick translation"}
    rows = [r for r in P if r["item"] in ("es_duende_flamenco", "es_sobremesa", "pt_saudade", "b_gender_adj_1",
                                          "b_plural_you_1", "es_tren")]
    if not rows:
        return skip("4_practices", "items missing")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.subplots_adjust(top=0.86, bottom=0.14)
    style(ax)
    w = 0.8 / len(order)
    meds = {}
    for j, c in enumerate(order):
        v = [r["practices"].get(c, {}).get("seen_frac_lennorm") for r in rows]
        v = [x if x is not None else np.nan for x in v]
        ax.bar(np.arange(len(rows)) + j * w, v, w * 0.92, color=TEAL if c != "plain" else GREY, alpha=0.4 + 0.12 * j)
        meds[c] = float(np.nanmedian([r["practices"].get(c, {}).get("seen_frac_lennorm", np.nan) for r in P]))
    ax.set_xticks(np.arange(len(rows)) + 0.4, [r["item"].split("_", 1)[1] for r in rows], family=SERIF, style="italic")
    ax.legend([lab[c] for c in order], frameon=False, fontsize=9, ncol=5, loc="upper left")
    ax.set_ylabel("share of what plain translation misses that becomes visible")
    best = max((c for c in order if c != "plain"), key=lambda c: meds[c])
    title = (f"A {lab[best]} recovers most of what plain translation misses"
             if meds[best] > meds["plain"] + 0.1 else "Different ways of translating recover similar amounts")
    save(fig, d, "4_practices", title, "Taller = more of the plain translation's blind spots are seen by that "
         "practice (adjusted for its longer text).", dict(median_seen_frac_lennorm=meds), "father")


# ------------------------------------------------------------------ 5
def f5(d, m):
    A = J(m, "atlas.json")
    if not A or not A["clusters"] or any(c["label"] is None for c in A["clusters"][:8]):
        return skip("5_atlas", "atlas missing or clusters not yet labelled (stop point 5)")
    cl = A["clusters"][:10]
    fig, ax = plt.subplots(figsize=(13, 0.9 * len(cl) + 1.5))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, len(cl) + 1)
    for j, c in enumerate(cl):
        y = len(cl) - j
        ax.text(0, y, c["label"], fontsize=11.5, color=INK, family=SANS, va="center")
        ax.text(0.42, y, c["examples"][0]["text"][:70], fontsize=10.5, family=SERIF, style="italic", va="center")
        ax.text(0.93, y, f"{100 * c['share_of_sentences']:.0f}%", fontsize=11.5, color=CORAL, va="center")
    save(fig, d, "5_atlas", "What Spanish-to-English translation keeps missing",
         "Each row is a kind of distinction the English translation does not register; the number is the share of "
         f"{A['n_sentences']} ordinary sentences where it appears.",
         dict(clusters=[(c["label"], c["share_of_sentences"]) for c in cl]), "father")


# ------------------------------------------------------------------ 6 (technical)
def f6(d, m):
    rows = whole(m, "e1")
    if not rows:
        return skip("6_technical", "no spectra_e1")
    k = headline_layer(m)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=0.9, bottom=0.09, hspace=0.3)
    ax = axs[0, 0]; style(ax)
    for r in [r for r in rows if r["layer"] == k and r["target"] == "en"][:12]:
        col = CORAL if r["group"] in ("gender_adj", "plural_you", "formal_you", "profession") else TEAL
        ax.semilogy(r["lam_med"], color=col, alpha=0.6)
        ax.semilogy(r["lam0_q95"], color=GREY, alpha=0.5, ls="--")
    ax.set_title("λ_i (es→en; coral = grammatical, teal = other) vs null q95 (dashed)", fontsize=10)
    ax = axs[0, 1]; style(ax)
    fd = J(m, "fd_check.json")
    if fd:
        for r in fd["rows"]:
            ax.scatter(r["pred_kl_src"], r["meas_kl_src"], color=CORAL, s=18)
            ax.scatter(r["pred_kl_en"], r["meas_kl_en"], color=TEAL, s=18)
        ax.plot([0, 1], [0, 1], color=GREY, lw=1)
        ax.set_title(f"finite differences: predicted vs measured KL (Spearman {fd['spearman_pred_meas']:.2f})", fontsize=10)
    ax = axs[1, 0]; style(ax)
    for t in TYPE:
        ys = [np.median([r["U"]["med"] for r in rows if r["layer"] == L and r["group"] == t and r["target"] == "en"]
                        or [np.nan]) for L in layers_for(m)]
        ax.plot(layers_for(m), ys, marker="o", label=t)
    ax.legend(fontsize=8, frameon=False); ax.set_title("median U_en by layer", fontsize=10)
    ax = axs[1, 1]; style(ax)
    allr = [r for r in read_jsonl(RESULTS / m / "spectra_e1.jsonl") if r["scope"] == "whole" and r["layer"] == k
            and r["target"] == "en"]
    for e, c in zip([1e-3, 1e-2, 1e-1], [GREY, CORAL, TEAL]):
        v = sorted(r["U"]["med"] for r in allr if r["eps"] == e)
        ax.plot(v, color=c, label=f"ε={e:g}")
    ax.legend(fontsize=8, frameon=False); ax.set_title("ε stability: sorted U_en over items", fontsize=10)
    save(fig, d, "6_technical", f"Technical checks ({m}, layer {k})", "For the user: spectra against the split-half "
         "null, the finite-difference check, layer dependence and ε stability.", {}, "user")


@stage("figures")
def run(cfg):
    m = cfg.get("model", "g3_4b")
    d = res_dir(m, "figures")
    for f in (f1, f2, f3, f4, f5, f6):
        try:
            f(d, m)
        except Exception as e:                 # one broken figure must not stop the others
            print(f"figure {f.__name__} failed: {type(e).__name__}: {e}")
    p = d / "captions.json"
    p.write_text(json.dumps(CAPTIONS, ensure_ascii=False, indent=1, default=float))
    return [p]
