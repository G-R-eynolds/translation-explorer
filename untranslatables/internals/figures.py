"""Figures for the internals run (HANDOFF_internals §9), PNG + SVG, with captions.json.

For the father (no symbols; every title is a finding chosen from the statistic it summarises, falling back to a neutral
description when the numbers do not support the stronger sentence; captions.json records the number):
  F1 what survives the round trip      F2 English keeps more of itself     F3 opening up duende (feature card)
  F4 the model knows more than it translates                               F5 bigger models, same picture?
For the user (technical): T1 spectrum by layer with null band, T2 shared directions vs depth with the R1 English profile,
T3 probe shared share by grammatical feature, T4 rho-bar vs tau*, T5 linear vs MLP R^2, T6 U (N1) vs U (round trip).
A figure whose inputs are missing is skipped (and says so).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from internals.common import MAIN, RESULTS, stage
from sent.figures import CORAL, GREY, GRID, HEAT, INK, MUTED, SANS, SERIF, SURF, TEAL, style

PURPLE = "#5b4fc4"
MODELS_N1 = ["8b", "aya8b", "gemma12b", "nemo12b", "qwen14b", "sarvam2b"]
NAME = {"8b": "Llama 3.1 8B", "aya8b": "Aya 23 8B", "gemma12b": "Gemma 3 12B", "nemo12b": "Mistral Nemo 12B",
        "qwen14b": "Qwen 2.5 14B", "sarvam2b": "Sarvam 1 (2B)"}
CAPTIONS = {}


def J(*p):
    f = RESULTS.joinpath(*p)
    return json.loads(f.read_text()) if f.exists() else None


def save(fig, d, name, title, how, claim, audience):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK, wrap=True)
    fig.text(0.012, 0.012, how, ha="left", va="bottom", fontsize=10.5, color=MUTED, wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}", dpi=160)
    plt.close(fig)
    CAPTIONS[name] = dict(title=title, how_to_read=how, claim=claim, audience=audience, png=f"{name}.png")
    print("wrote", name)


def model_relative(m, pair="es-en"):
    corpus = {"es-en": "Europarl and OPUS-100", "hi-en": "OPUS-100 and Samanantar"}[pair]
    return f"Inside {NAME.get(m, m)}, relative to the human translations in {corpus}."


# ------------------------------------------------------------------ father
def f1_retention(d, m=MAIN):
    S = J(m, "n1_spans_summary.json")
    if not S:
        return
    curves = {**S.get("es-en", {}).get("fig1_retention", {}), **S.get("hi-en", {}).get("fig1_retention", {})}
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(11, 6.4))
    label = {"es_duende:duende": "duende", "es_duende:gusta": "gusta", "es_sobremesa:sobremesa": "sobremesa",
             "hi_jugaad:जुगाड़": "jugaad (जुगाड़)", "es_tren:tren": "tren", "es_casa:hermana": "hermana"}
    hard = {"es_duende:duende", "es_sobremesa:sobremesa", "hi_jugaad:जुगाड़", "es_duende:gusta"}
    hl = {}
    for k, v in curves.items():
        col = CORAL if k in hard else TEAL
        ax.plot(range(11), v["R"], color=col, lw=2.4 if k in hard else 2, alpha=0.95)
        ax.text(10.15, v["R"][10], label.get(k, k), family=SERIF, style="italic", fontsize=12, color=col, va="center")
        if v["h"] < 10:
            ax.plot([v["h"]], [0.5], "o", color=col, ms=7)
        hl[label.get(k, k)] = v["h"]
    ax.axhline(0.5, color=GREY, lw=1, ls="--")
    ax.text(0.1, 0.51, "half the meaning left", color=MUTED, fontsize=10, va="bottom")
    ax.set_xlim(0, 11.8); ax.set_ylim(0, 1.02)
    ax.set_xticks(range(11)); ax.set_xlabel("number of round trips through English and back")
    ax.set_ylabel("share of the word's meaning still there")
    style(ax)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.09, right=0.86)
    du, tr = hl.get("duende"), hl.get("tren")
    fmt = lambda h: "never within 50 trips" if h is None or h >= 50 else f"{h:.1f} trips"
    if du is not None and tr is not None and du < tr:
        title = f"Duende loses half its meaning in {fmt(du)}; tren in {fmt(tr)}"
    else:
        title = "How much of a word survives repeated trips through English"
    save(fig, d, "F1_what_survives_the_round_trip", title,
         "How to read this: each line is one word in its sentence. Each step right sends its meaning into English and back "
         "once more, using the conventional translations in the corpus; the line shows how much is still there. Dots mark "
         "where half is gone. " + model_relative(m) + " Sobremesa stands in for saudade (Portuguese has no maps here).",
         dict(statistic="half-life (round trips) at the peak-sharing layer", half_lives=hl), "father")


def f2_bias(d):
    rows = []
    for m in MODELS_N1:
        for pair in ("es-en", "hi-en"):
            mt = RESULTS / m / f"n1_quick_{pair}.json"
            if not mt.exists():
                continue
            peak = json.loads(mt.read_text())["peak_layer"]
            sp = J(m, f"n1_spectrum_{pair}_L{peak}.json")
            if sp:
                rows.append((m, pair, sp["bias"]["raw"]["E_X"], sp["bias"]["raw"]["E_Y"], sp["bias_heldout"]))
    if not rows:
        return
    hub = J(MAIN, "n1_hub.json")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    labels = []
    for i, (m, pair, ex, ey, ho) in enumerate(rows):
        ax.bar(i - 0.18, ey, 0.34, color=TEAL); ax.bar(i + 0.18, ex, 0.34, color=CORAL)
        labels.append(f"{NAME.get(m, m)}\n{ {'es-en': 'Spanish', 'hi-en': 'Hindi'}[pair] }")
    ax.set_xticks(range(len(rows)), labels, fontsize=9)
    ax.set_ylabel("share of a language's internal variation\nthat the other language predicts")
    ax.set_title("English (teal) vs the other language (coral), conventional translations", fontsize=11, color=MUTED, loc="left")
    style(ax)
    claim = {f"{m} {p}": dict(E_en=ey, E_other=ex, bias=ey - ex, heldout=ho) for m, p, ex, ey, ho in rows}
    if hub:
        ax2 = axs[1]
        L = sorted(hub["layers"], key=int)[0]
        H = hub["layers"][L]
        keys = [k for k in ("es-en", "hi-en", "es-hi") if k in H and "E_Y" in H[k]]
        for i, k in enumerate(keys):
            ax2.bar(i - 0.18, H[k]["E_Y"], 0.34, color=TEAL if k.endswith("en") else PURPLE)
            ax2.bar(i + 0.18, H[k]["E_X"], 0.34, color=CORAL)
        ax2.set_xticks(range(len(keys)), [{"es-en": "Spanish and\nEnglish", "hi-en": "Hindi and\nEnglish",
                                            "es-hi": "Spanish and\nHindi (purple)"}[k] for k in keys], fontsize=9)
        ax2.set_title(f"Same sentences (FLORES), {NAME[MAIN]}", fontsize=11, color=MUTED, loc="left")
        style(ax2)
        claim["flores"] = {k: dict(E_X=H[k]["E_X"], E_Y=H[k]["E_Y"], n_shared=H[k]["n_shared"]) for k in keys}
    else:
        axs[1].axis("off")
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.08, right=0.98, wspace=0.25)
    pos = [ey - ex > 0 for _, _, ex, ey, _ in rows]
    title = ("In every model English keeps more of itself in the shared space than the other language does" if all(pos)
             else "English keeps more of itself in most models" if np.mean(pos) > 0.5
             else "English does not reliably keep more of itself in the shared space")
    save(fig, d, "F2_english_keeps_more_of_itself", title,
         "How to read this: for each model and language pair, the teal bar is how much of English's internal variation can be "
         "predicted from the other language's matching words, the coral bar the reverse. A taller teal bar means English "
         "keeps more of itself when the two are put side by side. Right: the same measure on the same sentences, including "
         "a pair without English. Every number is inside one model, relative to conventional translations.",
         claim, "father")


def f3_card(d, m=MAIN):
    N = J(m, "n2_cards.json")
    if not N:
        return
    cards = {(c["sid"], c["word"]): c for c in N["cards"]}
    pick = [cards.get(("es_duende", "duende")), cards.get(("contrast", "duende"))]
    pick = [c for c in pick if c]
    if not pick:
        return
    layer = sorted(pick[0]["layers"], key=int)[len(pick[0]["layers"]) // 2]
    fig, axs = plt.subplots(1, len(pick), figsize=(7.4 * len(pick), 9.2))
    axs = np.atleast_1d(axs)
    def lab(f):
        n = f["neuronpedia"] or {}
        if n.get("explanation"):
            return n["explanation"][:46]
        if n.get("top_snippets"):
            return "e.g. " + n["top_snippets"][0].replace("\n", " ")[-40:]
        return f"feature {f['feature']} (no description)"
    for ax, c in zip(axs, pick):
        ax.axis("off")
        ax.text(0, 1.0, c["sentence"], family=SERIF, style="italic", fontsize=15, color=INK, va="top", transform=ax.transAxes)
        ax.text(0, 0.95, "standard translation: " + c["translation"], fontsize=11, color=MUTED, va="top", transform=ax.transAxes,
                wrap=True)
        L = c["layers"][layer]
        y = 0.86
        for key, col, head in (("persist", TEAL, "carried into the English"), ("vanish", GREY, "lost in the English"),
                               ("appear", PURPLE, "added by the English")):
            ax.text(0, y, head, fontsize=12.5, fontweight="bold", color=col if col != GREY else MUTED, va="top",
                    transform=ax.transAxes)
            y -= 0.045
            for f in L[key][:6]:
                ax.text(0.02, y, "• " + lab(f), fontsize=10.5, color=INK, va="top", transform=ax.transAxes)
                y -= 0.038
            y -= 0.025
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.03, right=0.98)
    save(fig, d, "F3_opening_up_duende", "What the model's features for 'duende' carry into the English, and what they lose",
         "How to read this: the model's internal features (interpretable units found by Gemma Scope) that fire on 'duende', "
         "sorted by whether they also fire on its English translation. Labels are the descriptions published on Neuronpedia "
         "or, where there is none, an example of text that switches the feature on. Left: the flamenco sense; right: the "
         "goblin sense. " + model_relative(m), dict(statistic="persist / vanish / appear features", layer=layer), "father")


def f4_dissociation(d, m=MAIN):
    p = RESULTS / m / "n3_ablation.jsonl"
    if not p.exists():
        return
    rows = [json.loads(l) for l in open(p)]
    meta_layer = min(r["layer"] for r in rows if r["pair"] == "es-en") if any(r["pair"] == "es-en" for r in rows) else None
    R = [r for r in rows if r["layer"] == meta_layer and "E_mono_sense" in r and r["E_trans"]["private"] is not None
         and r["E_mono_sense"]["private"] is not None]
    if not R:
        return
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axs[0]
    for r in R:
        hot = r["unit"] == "hot"
        ax.scatter(r["E_trans"]["private"], r["E_mono_sense"]["private"], s=70, color=CORAL if hot else TEAL,
                   edgecolor=SURF, zorder=3)
        ax.annotate(r["span"], (r["E_trans"]["private"], r["E_mono_sense"]["private"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=9, family=SERIF, style="italic", color=INK)
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1)
    ax.set_xlabel("how much removing the word's private part\nchanges its English translation")
    ax.set_ylabel("how much it changes the model's\nunderstanding of the word in Spanish")
    style(ax, grid="both")
    I = [json.loads(l) for l in open(RESULTS / m / "n3_interchange.jsonl")] if (RESULTS / m / "n3_interchange.jsonl").exists() else []
    ax2 = axs[1]
    pick = [r for r in I if r["id"] in ("cura", "ti_usted_para") and r["direction"] == "0<-1" and r["layer"] == meta_layer]
    if pick:
        y = 0
        for r in pick:
            for k, col, nm in (("logodds_en", INK, "English translation"), ("logodds_src", PURPLE, "Spanish meaning")):
                dv = r[k]["patch_private"] - r[k]["clean"]
                dw = r[k]["patch_whole"] - r[k]["clean"]
                ax2.barh(y, dw, color=GREY, height=0.35); ax2.barh(y, dv, color=col, height=0.35)
                ax2.text(0, y + 0.3, f"{r['context']} ({nm})", fontsize=9, family=SERIF, style="italic", color=INK)
                y += 1
        ax2.axvline(0, color=MUTED, lw=0.8)
        ax2.set_yticks([])
        ax2.set_xlabel("shift towards the other sense when the private part is swapped in\n(grey: swapping the whole word)")
        style(ax2, grid="x")
    else:
        ax2.axis("off")
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.08, right=0.98, wspace=0.25)
    hot = [r["E_mono_sense"]["private"] - r["E_trans"]["private"] for r in R if r["unit"] == "hot"]
    ctl = [r["E_mono_sense"]["private"] - r["E_trans"]["private"] for r in R if r["unit"] != "hot"]
    from ut.stats import boot_ci
    ch, cc = boot_ci(hot) if hot else None, boot_ci(ctl) if ctl else None
    title = ("For hard words the model understands more than its translation carries" if ch and ch[1] > 0 and
             (cc is None or ch[0] > cc[2]) else "Removing a word's private part does not affect understanding more than translation")
    save(fig, d, "F4_the_model_knows_more_than_it_translates", title,
         "How to read this: left, each dot is a word; dots above the diagonal are words whose private part (the part with no "
         "counterpart in English) matters more for what the model understands in Spanish than for the English it writes. "
         "Coral: hard words; teal: plain words. Right: swapping that private part between two senses of the same word. "
         + model_relative(m), dict(statistic="E_mono_sense - E_trans (private), bootstrap CI", hot=ch, control=cc), "father")


def f5_bigger(d):
    ms = [m for m in ("8b", MAIN, "qwen14b") if (RESULTS / m / "replication_R2" / "curves.json").exists()]
    if not ms:
        return
    fig = plt.figure(figsize=(12, 2.4 * len(ms) + 1.6))
    r_ = fig.canvas.get_renderer()
    claim = {}
    for k, m in enumerate(ms):
        C = json.loads((RESULTS / m / "replication_R2" / "curves.json").read_text())["sentences"].get("es_duende")
        Sn = json.loads((RESULTS / m / "replication_R2" / "sense.json").read_text()) if (RESULTS / m / "replication_R2" / "sense.json").exists() else None
        if not C:
            continue
        y = 1 - (k + 0.55) / (len(ms) + 0.6)
        fig.text(0.02, y + 0.08, NAME.get(m, m), fontsize=11, color=MUTED)
        x = 0.02
        for h in C["spaces"]["rt"]["heat"]:
            lv = h["leftover"]
            t = fig.text(x, y, h["word"], fontsize=24, family=SERIF, style="italic", color=INK, va="center",
                         bbox=dict(boxstyle="round,pad=0.25", fc=HEAT(lv) if lv is not None else SURF,
                                   ec=GREY if lv is None else "none", ls=":" if lv is None else "-"))
            bb = t.get_window_extent(r_).transformed(fig.transFigure.inverted())
            x = bb.x1 + 0.016
        sr = None
        if Sn:
            sp = next((s for s in Sn["spans"] if s["sid"] == "es_duende" and s["span"]["text"] == "duende"), None)
            sr = sp["rungs"].get("B1", {}).get("greedy") if sp else None
        fig.text(x + 0.03, y, f"translation: {C['text_B1']}", fontsize=11, color=INK, va="center")
        fig.text(x + 0.03, y - 0.06, "recovers the intended sense of duende: " +
                 ("n/a" if sr is None else f"{sr:+.2f} (0 = as well as the Spanish)"), fontsize=10, color=MUTED, va="center")
        claim[m] = dict(leftover={h["word"]: h["leftover"] for h in C["spaces"]["rt"]["heat"]}, sense_recovery_duende=sr,
                        translation=C["text_B1"])
    save(fig, d, "F5_bigger_models_same_picture", "Bigger models, same picture? 'Me gusta mucho el duende' in three models",
         "How to read this: the redder a word, the less of it can be rebuilt from the model's standard English translation "
         "(source words copied into the English are hidden first). Under each: how well that translation lets the model "
         "recover the intended (Lorca) sense of duende; 0 means as well as from the Spanish itself.", claim, "father")


# ------------------------------------------------------------------ technical
def t1_spectrum(d, m=MAIN):
    for pair in ("es-en", "hi-en"):
        files = sorted((RESULTS / m).glob(f"n1_spectrum_{pair}_L*.json"), key=lambda p: int(p.stem.split("_L")[-1]))
        if not files:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        cm = plt.get_cmap("viridis")
        for i, f in enumerate(files):
            s = json.loads(f.read_text())
            rho = np.sort(np.asarray(s["rho_heldout"]))[::-1][:400]
            ax.plot(rho, color=cm(i / max(len(files) - 1, 1)), lw=1.3, label=f"L{s['layer']} (r={s['n_shared']})")
            ax.axhline(s["rho_null_heldout_q99"], color=cm(i / max(len(files) - 1, 1)), lw=0.5, ls=":")
        ax.set_xlabel("k (held-out rank)"); ax.set_ylabel(r"held-out $\rho_k$")
        ax.legend(fontsize=7, ncol=3, frameon=False)
        style(ax)
        fig.subplots_adjust(top=0.88, bottom=0.18)
        save(fig, d, f"T1_spectrum_{pair}", f"Held-out canonical correlations by layer ({NAME.get(m, m)}, {pair})",
             "Dotted: 99th percentile of the mismatched-pair null at each layer; r = directions above it.",
             dict(layers=[json.loads(f.read_text())["n_shared"] for f in files]), "user")


def t2_depth(d):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax2 = ax.twinx() if False else None
    claim = {}
    for m in MODELS_N1:
        for pair, ls in (("es-en", "-"), ("hi-en", "--")):
            q = J(m, f"n1_quick_{pair}.json")
            if not q:
                continue
            L = sorted(q["layers"], key=int)
            nl = max(int(l) for l in L)
            ax.plot([int(l) / nl for l in L], [q["layers"][l]["n_shared"] for l in L], ls, lw=1.6, label=f"{NAME.get(m, m)} {pair}")
            claim[f"{m} {pair}"] = dict(peak=q["peak_layer"], n_layers=nl)
        r1 = J(m, "replication_R1.json")
        if r1:
            sh = np.mean([[x[0] for x in v["english_share"]] for v in r1["pairs"].values()], 0)
            claim.setdefault(f"{m} R1", {})["english_share_peak_rel_depth"] = float(np.argmax(sh) / (len(sh) - 1))
    ax.set_xlabel("relative depth (layer / number of layers)"); ax.set_ylabel("shared directions (PCA-256 quick pass)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    style(ax)
    fig.subplots_adjust(top=0.88, bottom=0.16)
    save(fig, d, "T2_shared_vs_depth", "Shared directions against depth, with the R1 English-detour peak",
         "Each line: number of held-out canonical directions above the null at each layer (quick pass, PCA-256). "
         "claim records each model's R1 English-share peak as relative depth.", claim, "user")


def t3_probes(d, m=MAIN):
    P = J(m, "n1_probes.json")
    if not P:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    labels, vals, base = [], [], []
    for pair, v in P.items():
        for f, r in v["features"].items():
            if "shared_share" in r:
                labels.append(f"{pair[:2]}: {f}"); vals.append(r["shared_share"]); base.append(r["random_expectation"])
    ax.bar(range(len(vals)), vals, color=[CORAL if v < b else TEAL for v, b in zip(vals, base)])
    ax.scatter(range(len(vals)), base, marker="_", s=300, color=INK, zorder=3, label="r/d (random)")
    ax.set_xticks(range(len(vals)), labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("probe weight energy on shared directions")
    ax.legend(frameon=False)
    style(ax)
    fig.subplots_adjust(top=0.88, bottom=0.3)
    save(fig, d, "T3_probe_shared_share", f"Where grammatical features live: shared vs private ({NAME.get(m, m)})",
         "Bars: share of the logistic probe's weight energy on the shared canonical directions; tick: r/d. Below the tick "
         "(coral) = more private than a random direction.", dict(zip(labels, zip(vals, base))), "user")


def t4_rhobar(d, m=MAIN):
    p = RESULTS / m / "n2_features.parquet"
    S = J(m, "n2_summary.json")
    if not p.exists() or not S:
        return
    import pandas as pd
    df = pd.read_parquet(p)
    df = df[df.scored & df.rho_bar.notna()]
    ls = sorted(df.layer.unique())
    fig, axs = plt.subplots(1, len(ls), figsize=(5 * len(ls), 5), squeeze=False)
    for ax, l in zip(axs[0], ls):
        x = df[df.layer == l]
        ax.scatter(x.rho_bar, x.tau_star, s=4, alpha=0.3, color=PURPLE)
        sp = S["per_layer"][str(l)]["spearman_rhobar_tau_star"] if str(l) in S["per_layer"] else S["per_layer"][l]["spearman_rhobar_tau_star"]
        ax.set_title(f"SAE layer {l}: Spearman {sp:.2f}" if sp is not None else f"SAE layer {l}", fontsize=11)
        ax.set_xlabel(r"$\bar\rho_f$ (decoder direction in canonical coords)"); ax.set_ylabel(r"$\tau^*_f$")
        style(ax, grid="both")
    fig.subplots_adjust(top=0.82, bottom=0.18)
    save(fig, d, "T4_rhobar_vs_tau", "N1 and N2 at the feature level", "Each dot a scored, non-language-identity feature.",
         {str(l): S["per_layer"].get(str(l), S["per_layer"].get(l, {})).get("spearman_rhobar_tau_star") for l in ls}, "user")


def t5_mlp(d):
    rows = []
    for m in MODELS_N1:
        M = J(m, "n1_mlp.json")
        if M:
            for pair, v in M.items():
                for dirn in ("X_to_Y", "Y_to_X"):
                    if dirn in v:
                        rows.append((f"{NAME.get(m, m)} {pair} {dirn}", v[dirn]["linear_r2"], v[dirn]["mlp_r2"], v[dirn]["recovered"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(rows) + 2))
    for i, (lab, lin, mlp, rec) in enumerate(rows):
        ax.barh(i, mlp, color=PURPLE, height=0.6); ax.barh(i, lin, color=TEAL, height=0.6)
        ax.text(max(lin, mlp) + 0.01, i, f"recovered {rec:.0%}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlabel("held-out R² (PCA-512): teal linear, purple MLP")
    style(ax, grid="x")
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.32)
    save(fig, d, "T5_linear_vs_mlp", "Is linear enough? Held-out R² of linear vs 2-layer MLP maps",
         "recovered = (R²_MLP - R²_lin)/(1 - R²_lin).", {r[0]: dict(linear=r[1], mlp=r[2], recovered=r[3]) for r in rows}, "user")


def t6_internal_vs_behavioural(d, m=MAIN):
    S = J(m, "n1_spans_summary.json")
    p = RESULTS / m / "n1_span_scores.jsonl"
    C = J(m, "replication_R2", "curves.json")
    if not (S and p.exists() and C):
        return
    peak = S["es-en"]["peak_layer"]
    xs, ys, labs, cols = [], [], [], []
    for l in open(p):
        r = json.loads(l)
        if r["unit"] == "sentence" and r["layer"] == (peak if r["lang"] == "es" else S.get("hi-en", {}).get("peak_layer")) \
                and r["sid"] in C["sentences"] and r["U_weighted"] is not None:
            w = C["sentences"][r["sid"]]["spaces"]["rt"]["whole"]
            if w:
                xs.append(w["U"]); ys.append(r["U_weighted"]); labs.append(r["sid"]); cols.append(CORAL if r["category"] != "control" else TEAL)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, c=cols, s=60, edgecolor=SURF)
    for x, y, l in zip(xs, ys, labs):
        ax.annotate(l, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7.5)
    from scipy.stats import spearmanr
    rho = spearmanr(xs, ys)[0] if len(xs) > 3 else None
    ax.set_xlabel("U (sentence round trip, R2, masked)"); ax.set_ylabel("U (N1, token-weighted)")
    style(ax, grid="both")
    fig.subplots_adjust(top=0.88, bottom=0.14)
    save(fig, d, "T6_internal_vs_behavioural", f"Internal (N1) vs behavioural (R2) untranslatability: Spearman {rho:.2f}"
         if rho is not None else "Internal (N1) vs behavioural (R2) untranslatability",
         "es and hi sentences, each at its pair's peak-sharing layer.", dict(spearman=rho, n=len(xs)), "user")


@stage("figures")
def run(cfg):
    d = RESULTS / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for f in (f1_retention, f2_bias, f3_card, f4_dissociation, f5_bigger, t1_spectrum, t2_depth, t3_probes, t4_rhobar,
              t5_mlp, t6_internal_vs_behavioural):
        try:
            f(d)
        except Exception as e:
            print(f"  skipped {f.__name__}: {type(e).__name__}: {e}", flush=True)
            CAPTIONS[f.__name__] = dict(skipped=f"{type(e).__name__}: {e}")
    p = d / "captions.json"
    p.write_text(json.dumps(dict(fonts=dict(sans=SANS, serif=SERIF), **CAPTIONS), ensure_ascii=False, default=float))
    return [p]
