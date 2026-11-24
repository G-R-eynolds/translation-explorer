"""The 13 figures for the father (handoff §9), PNG + SVG, with a finding-as-title and a plain caption each
(written to captions.json for report.py).  A figure whose inputs do not exist yet is skipped.

    python figures.py --model 8b --models 3b 8b qwen7b aya8b    # per-model figures for --model, cross-model for --models

Style: one hue per category everywhere (palette validated with the dataviz validator: reference slots 1-7, worst
adjacent CVD dE 9.1); grey always = leftover / what English can't reach; hatching = context-dependent or
'failed the familiarity check'; big type; direct labels instead of colour-only identity.
"""
import argparse, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ut.common import ROOT, MODELS, load_json, out_dir

CAT = {"untranslatable": "#2a78d6", "ordinary": "#eb6834", "emotion_plain": "#1baf7a", "polysemous": "#eda100",
       "loanword": "#e87ba4", "dialect_known_gloss": "#008300", "dialect_untranslatable": "#6250d6",
       "english_untranslatable": "#2a78d6", "english_ordinary": "#eb6834"}
CAT_LABEL = {"untranslatable": "untranslatable", "ordinary": "ordinary word", "emotion_plain": "plain emotion word",
             "polysemous": "word with several meanings", "loanword": "word English borrowed",
             "dialect_known_gloss": "dialect word, one meaning", "dialect_untranslatable": "dialect, several readings"}
GREY, INK, MUTED, GRID, SURF = "#c9c8c2", "#1f1f1e", "#898781", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold", "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
CAPTIONS = {}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)


def save(fig, d, name, title, how):
    fig.suptitle(title, x=0.01, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    CAPTIONS[name] = dict(finding=title, how_to_read=how, file=str((d / f"{name}.png").relative_to(ROOT / "results")))
    print("wrote", name)


def maybe(path):
    return load_json(path) if path.exists() else None


# ------------------------------------------------------------------ thread 1
def fig_cards(rd, d, rec, fam):
    words = rec["words"]
    order = [w for w in ("pt:saudade", "pt:tristeza") if w in words]
    rest = [w for w, e in words.items() if e["category"] == "untranslatable" and w not in order]
    for name, ids, title in (("fig01_recipe_cards_lead", order, None), ("fig01_recipe_cards_all", order + rest, None)):
        if not ids:
            continue
        fig, axs = plt.subplots(len(ids), 1, figsize=(13, 2.3 * len(ids) + 0.6), squeeze=False)
        for ax, wid in zip(axs[:, 0], ids):
            e = words[wid]; r = e["directions"]["en"]
            x, col, below = 0.0, CAT[e["category"]], 0
            fits = lambda txt, w: w > 0.0085 * len(txt) + 0.01          # does the label fit inside its block?
            for ing in r["ingredients"]:
                core = ing["word"] in r["stable_core"]
                ax.barh(0, ing["share"], left=x, color=col if core else "white", edgecolor=SURF if core else col,
                        lw=2 if core else 1.5, hatch=None if core else "///", height=.6)
                cx = x + ing["share"] / 2
                if fits(ing["word"], ing["share"]):
                    ax.text(cx, 0, ing["word"], ha="center", va="center", fontsize=12, color="white" if core else INK,
                            fontweight="bold", bbox=None if core else dict(fc=SURF, ec="none", pad=1))
                else:                                                   # narrow block: label underneath, staggered
                    ax.text(cx, -0.42 - 0.28 * (below % 2), ing["word"], ha="center", va="top", fontsize=11, color=INK)
                    below += 1
                x += ing["share"]
            ax.barh(0, 1 - x, left=x, color=GREY, height=.6)
            lab = f"leftover {1 - x:.0%}"
            ax.text(x + (1 - x) / 2 if fits(lab, 1 - x) else 1.0, 0 if fits(lab, 1 - x) else -0.42,
                    lab, ha="center" if fits(lab, 1 - x) else "right", va="center" if fits(lab, 1 - x) else "top",
                    fontsize=12, color=INK)
            ax.set_ylim(-1.0, 0.45)
            fl = fam.get(wid, ("?", ""))[0] if fam else "?"
            ax.set_title(f"{e['word']} ({e['language']}): {e['gloss_en']}" + ("   [model did not know it]" if fl == "no" else ""),
                         loc="left", fontsize=13, fontweight="normal", color=INK)
            ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        lead = words[ids[0]]["directions"]["en"]["leftover"]
        if name.endswith("_all"):
            lv = [words[w]["directions"]["en"]["leftover"] for w in ids]
            save(fig, d, name, f"Recipe cards for {len(ids)} words: leftovers range from {min(lv):.0%} to {max(lv):.0%}",
                 "As in the first figure, for every untranslatable: coloured blocks are English ingredients (solid = in "
                 "nearly every context, striped = only in some); grey is what English could not rebuild.")
            continue
        t = (f"Inside this model, '{words[ids[0]]['word']}' leaves {lead:.0%} that five English words can't reach"
             + (f"; '{words[ids[1]]['word']}' leaves {words[ids[1]]['directions']['en']['leftover']:.0%}" if len(ids) > 1 else ""))
        save(fig, d, name, t, "Each bar is one word rebuilt from the English words the model finds closest inside its "
             "'concept space'. Coloured blocks are the English ingredients, widest = most important. Solid blocks appear "
             "in nearly every context the word was read in; striped blocks only in some. Grey is what is left over: the "
             "part of the word that English could not rebuild.")


def fig_curves(rd, d, rec):
    ks = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(10, 6.5)); style(ax)
    nulls = [c for k, v in rec["null_curves"].items() if k.endswith("->en") and not k.startswith("random") for c in v]
    pad = lambda c: (list(c) + [c[-1] if c else 0] * 10)[:10]
    N = np.array([pad(c) for c in nulls])
    ax.fill_between(ks, np.quantile(N, .05, 0), np.quantile(N, .95, 0), color=GREY, alpha=.5, lw=0)
    ax.text(10.1, np.quantile(N, .5, 0)[-1], "chance band\n(random words)", color=MUTED, fontsize=11, va="center")
    med = {}
    for cat in ("untranslatable", "emotion_plain", "ordinary"):
        C = np.array([pad(e["directions"]["en"]["r2_curve"]) for e in rec["words"].values() if e["category"] == cat and "en" in e["directions"]])
        if len(C):
            m = np.median(C, 0); med[cat] = m
            ax.plot(ks, m, color=CAT[cat], lw=2.5, marker="o", ms=8)
            ax.text(10.1, m[-1], CAT_LABEL[cat], color=INK, fontsize=12, va="center")
    ax.set_xlabel("number of English words used"); ax.set_ylabel("share of the word rebuilt")
    ax.set_ylim(0, 1); ax.set_xlim(1, 12)
    u, o = med.get("untranslatable"), med.get("ordinary")
    t = ("Untranslatables need more English words to rebuild than ordinary words" if u is not None and o is not None and u[4] < o[4]
         else "Untranslatables are rebuilt from English about as easily as ordinary words")
    save(fig, d, "fig02_how_many_english_words", t, "Each line follows a typical word of one kind as more English words "
         "are added. Higher means more of the word has been rebuilt. The grey band is what random words of the same "
         "language reach; above it means better than chance.")


def fig_neighbourhood(rd, d, rec, mk):
    try:
        from ut.space import Targets, pool
    except Exception:
        return
    layer = rec["layer"]
    T = Targets(mk)
    words, E, _ = pool(mk, "en", layer)
    ids = [w for w in ("pt:saudade", "de:Sehnsucht", "es:sobremesa") if w in rec["words"]]
    fig, axs = plt.subplots(1, len(ids), figsize=(6.5 * len(ids), 6.5), squeeze=False)
    for ax, wid in zip(axs[0], ids):
        x = T.mean_vector(layer, wid)
        ing = [i["word"] for i in rec["words"][wid]["directions"]["en"]["ingredients"]]
        near = list(np.argsort(-(E @ x))[:25])
        idx = sorted(set(near) | {words.index(w) for w in ing if w in words})
        P = np.vstack([x, E[idx]]); P = P - P.mean(0)
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        Z = P @ Vt[:2].T
        ax.scatter(Z[1:, 0], Z[1:, 1], s=25, color=GREY)
        for (zx, zy), j in zip(Z[1:], idx):
            if words[j] in ing:
                ax.scatter(zx, zy, s=60, color=CAT["ordinary"]); ax.annotate(words[j], (zx, zy), xytext=(4, 4), textcoords="offset points", fontsize=12)
        ax.scatter(*Z[0], s=160, color=CAT[rec["words"][wid]["category"]], marker="*")
        ax.annotate(rec["words"][wid]["word"], Z[0], xytext=(6, -14), textcoords="offset points", fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    save(fig, d, "fig03_neighbourhood", "Each untranslatable sits between its English ingredients rather than on any one of them",
         "A flattened sketch of a small part of the model's concept space (a 2-D projection of many dimensions, so "
         "distances are only roughly right). The star is the word; labelled dots are its English ingredients; grey dots "
         "are other nearby English words.")


def fig_ranking(rd, d, rec, fam):
    ws = [(w, e) for w, e in rec["words"].items() if e["language"] != "en" and e["category"] in CAT_LABEL
          and "en" in e["directions"] and not e["category"].startswith("dialect")]
    ws.sort(key=lambda t: (list(CAT_LABEL).index(t[1]["category"]), -t[1]["directions"]["en"]["leftover"]))
    fig, ax = plt.subplots(figsize=(max(12, .38 * len(ws)), 7)); style(ax)
    nulls = [v for k, vs in rec["nulls"].items() if k.endswith("->en") and not k.startswith("random") for v in vs]
    lo, hi = np.quantile(nulls, [.05, .95])
    ax.axhspan(lo, hi, color=GREY, alpha=.45, lw=0)
    ax.text(len(ws) - .5, hi, "chance band (random words)", ha="right", va="bottom", color=MUTED, fontsize=11)
    for i, (w, e) in enumerate(ws):
        failed = fam and fam.get(w, ("?",))[0] == "no"
        ax.bar(i, e["directions"]["en"]["leftover"], color="white" if failed else CAT[e["category"]],
               edgecolor=CAT[e["category"]], hatch="///" if failed else None, width=.8)
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([e["word"] for _, e in ws], rotation=60, ha="right", fontsize=11)
    ax.set_ylabel("leftover (what English can't reach)")
    for c in dict.fromkeys(e["category"] for _, e in ws):
        xs = [i for i, (_, e) in enumerate(ws) if e["category"] == c]
        ax.text(np.mean(xs), ax.get_ylim()[1] * .98, CAT_LABEL[c], ha="center", va="top", fontsize=12, color=INK)
    med = lambda c: np.median([e["directions"]["en"]["leftover"] for _, e in ws if e["category"] == c] or [np.nan])
    t = ("Untranslatables leave more behind than ordinary words" if med("untranslatable") > med("ordinary")
         else "Untranslatables do not leave more behind than ordinary words")
    save(fig, d, "fig04_leftover_ranking", t, "Each bar is one word; taller means more of it could not be rebuilt "
         "from English. Bars are grouped by kind of word. Striped bars are words the model did not seem to know, so "
         "they should be read with caution. The grey band is what random words reach.")


def fig_patching(rd, d, rec):
    pt = maybe(rd / "patching.json")
    if not pt:
        return
    ids = [w for w in ("pt:saudade", "de:Sehnsucht", "it:magari", "es:sobremesa", "fr:dépaysement", "pt:tristeza") if w in pt["words"]]
    fig, axs = plt.subplots(len(ids), 1, figsize=(11, 2.2 * len(ids) + .5), squeeze=False)
    for ax, wid in zip(axs[:, 0], ids):
        dist = list(pt["words"][wid]["dumas"].items())[:6]
        n = sum(pt["words"][wid]["dumas"].values())
        ax.barh(range(len(dist)), [c / n for _, c in dist], color=CAT[rec["words"][wid]["category"]] if wid in rec["words"] else MUTED)
        ax.set_yticks(range(len(dist))); ax.set_yticklabels([w for w, _ in dist]); ax.invert_yaxis()
        ing = ", ".join(i["word"] for i in rec["words"][wid]["directions"]["en"]["ingredients"][:3]) if wid in rec["words"] else ""
        ax.set_title(f"{rec['words'][wid]['word'] if wid in rec['words'] else wid}   (recipe: {ing})", loc="left", fontsize=13, fontweight="normal")
        style(ax); ax.set_xlim(0, 1)
    save(fig, d, "fig05_what_the_model_would_say", "When the word's inner state is planted in an English prompt, the model says these words",
         "For each word we copy the model's inner representation of it into a prompt asking for an English word, and "
         "let it answer 50 times. Bars show how often each English answer came out. The recipe's top ingredients are "
         "listed for comparison.")


def fig_pivots(rd, d, rec):
    pv = rec.get("pivots") or {}
    ids = [w for w in pv if rec["words"][w]["category"] == "untranslatable"]
    if not ids:
        return
    fig, ax = plt.subplots(figsize=(11, .5 * len(ids) + 1.5)); style(ax)
    for i, w in enumerate(ids):
        for lang, v in pv[w].items():
            ax.scatter(v, i, s=70, color=INK if lang == "en" else MUTED, zorder=3)
            ax.annotate(lang, (v, i), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10, color=INK if lang == "en" else MUTED)
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([rec["words"][w]["word"] for w in ids]); ax.invert_yaxis()
    ax.set_xlabel("leftover when rebuilt from this language")
    rank = np.mean([sorted(pv[w].values()).index(pv[w]["en"]) / max(1, len(pv[w]) - 1) for w in ids])
    t = ("Many untranslatables are harder to rebuild from English than from other languages" if rank > .5
         else "English is not the hardest language to rebuild untranslatables from")
    save(fig, d, "fig06_into_english_vs_others", t, "Each row is one untranslatable; each dot is one language it was "
         "rebuilt from (en = English, es = Spanish, ...). Further right means more is left over. A word can be hard to "
         "carry into one language and easy into another.")


# ------------------------------------------------------------------ English bias (cross-model)
def fig_english_share(d, models):
    fig, ax = plt.subplots(figsize=(11, 6.5)); style(ax)
    found = False
    for m in models:
        ll = maybe(out_dir(m) / "latent_lang.json")
        if not ll:
            continue
        found = True
        curves = np.array([[c[0] for c in p["english_share"]] for p in ll["pairs"].values()]).mean(0)
        x = np.linspace(0, 1, len(curves))
        hl = m == "aya8b"
        ax.plot(x, curves, lw=3 if hl else 2, color="#2a78d6" if hl else (INK if m == "8b" else MUTED))
        ax.text(1.01, curves[-1], MODELS[m]["hf_id"].split("/")[1], fontsize=11, va="center", color=INK)
    if not found:
        plt.close(fig); return
    ax.set_xlabel("depth in the model (0 = input, 1 = output)"); ax.set_ylabel("share of the working answer that is English")
    ax.set_ylim(0, 1)
    save(fig, d, "fig07_does_it_think_in_english", "Midway through, the models' working answer is partly English even when translating French into Spanish",
         "We ask each model to translate single words between two non-English languages and peek at its working "
         "answer layer by layer. Higher means the in-between answer is more English. Aya, built to be less "
         "English-centred, is highlighted in blue.")


def fig_coverage(d, models):
    cov = {m: maybe(out_dir(m) / "coverage.json") for m in models}
    cov = {m: c for m, c in cov.items() if c}
    if not cov:
        return
    fig, axs = plt.subplots(1, len(cov), figsize=(5.2 * len(cov), 5.5), squeeze=False, sharey=True)
    for ax, (m, c) in zip(axs[0], cov.items()):
        langs = list(c["languages"])
        for i, l in enumerate(langs):
            r = c["languages"][l]["50"] if "50" in c["languages"][l] else c["languages"][l][50]
            ax.plot([r["xx_to_en"], r["en_to_xx"]], [i, i], color=GRID, lw=3)
            ax.scatter(r["en_to_xx"], i, color=INK, s=70, zorder=3); ax.scatter(r["xx_to_en"], i, color=MUTED, s=70, zorder=3)
        ax.set_yticks(range(len(langs))); ax.set_yticklabels(langs); ax.set_title(MODELS[m]["hf_id"].split("/")[1], fontsize=13)
        ax.set_xlabel("share of variation covered"); style(ax)
    axs[0, 0].text(0.02, -0.18, "black: English directions covering the other language;  grey: the reverse",
                   transform=axs[0, 0].transAxes, fontsize=11, color=MUTED)
    save(fig, d, "fig08_which_way_does_it_lean", "English's main directions cover other languages better than the reverse",
         "For each language and model: how much of that language's word cloud is captured by English's 50 main "
         "directions (black), and how much of English's cloud by that language's (grey). A black dot to the right of "
         "the grey one means the space is built more around English.")


def fig_distinctions(d, models):
    m = next((m for m in models if (out_dir(m) / "distinctions.json").exists()), None)
    if not m:
        return
    ds = load_json(out_dir(m) / "distinctions.json")
    s = ds["summary"]
    fig, ax = plt.subplots(figsize=(11, 6.5)); style(ax)
    names = list(s)
    ax.bar(range(len(names)), [s[n]["greedy_error"] for n in names],
           color=[CAT["untranslatable"] if s[n]["type"] != "english_keeps" else MUTED for n in names])
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split("_", 1)[1].replace("-", " / ") for n in names], rotation=20, ha="right")
    ax.set_ylabel("share of wrong word choices")
    ex = [n for n in names if s[n]["type"] == "english_lacks" and s[n].get("excess_error_greedy") is not None]
    exc = np.mean([s[n]["excess_error_greedy"] for n in ex]) if ex else 0
    save(fig, d, "fig09_lost_through_english",
         ("Distinctions that English lacks are lost more often" if exc > 0 else "Distinctions English lacks are not lost more often") + f" ({MODELS[m]['hf_id'].split('/')[1]})",
         "Translating between two languages that both make a distinction English does not (savoir/connaître, ser/estar; "
         "blue) versus distinctions English also makes (look/see; grey). Taller bars mean more wrong choices.")


def fig_steering(rd, d):
    st = maybe(rd / "steering_fine.json") or maybe(rd / "steering.json")
    if not st:
        return
    g = st["gammas"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5)); style(a1); style(a2)
    get = lambda dct, gg: next(v for k, v in dct.items() if float(k) == float(gg))     # json keys: '0', '0.0', ...
    lacks = lambda gg: np.nanmean([r["greedy_error"] for sid, r in get(st["distinctions"], gg).items()
                                   if "savoir" in sid or "ser-estar" in sid])
    deg = [get(st["fluency"], gg)["degraded"] for gg in g]
    a1.plot(g, [lacks(gg) for gg in g], color=CAT["untranslatable"], lw=2.5)
    a1.set_xlabel("push toward English (negative = away)"); a1.set_ylabel("wrong choices on English-lacking distinctions")
    from ut.words import by_id, load_words
    cats = {w: e.category for w, e in by_id(load_words()).items()}
    for cat in ("untranslatable", "ordinary", "emotion_plain"):
        ids = [w for w in get(st["leftovers"], 0) if cats.get(w) == cat]
        vals = [np.mean([get(st["leftovers"], gg)[w] for w in ids]) for gg in g]
        a2.plot(g, vals, color=CAT[cat], lw=2.5)
        a2.text(g[-1] + .1, vals[-1], CAT_LABEL[cat], fontsize=11, va="center")
    a2.set_xlabel("push toward English (negative = away)"); a2.set_ylabel("mean leftover")
    for ax in (a1, a2):
        for gg, dd in zip(g, deg):
            if dd:
                ax.axvspan(gg - .5, gg + .5, color=GREY, alpha=.3, lw=0)
    save(fig, d, "fig10_push_away_from_english", "Pushing the model away from English changes what survives translation",
         "We nudge the model's inner state toward English (right) or away from it (left). Left: wrong choices on "
         "distinctions English lacks. Right: how much of the untranslatables is left over. Grey columns mark pushes "
         "strong enough to make the model's output degrade, which should not be trusted.")


# ------------------------------------------------------------------ thread 2
def fig_uncertainty_map(rd, d):
    ig = maybe(rd / "ignorance.json")
    if not ig:
        return
    W = ig["words"]; T = ig["thresholds"]
    fig, ax = plt.subplots(figsize=(11, 8.5)); style(ax)
    for wid, r in W.items():
        ax.scatter(r["ignorance"], r["multiplicity"], s=90, color=CAT[r["category"]], edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(r["word"], (r["ignorance"], r["multiplicity"]), xytext=(5, 4), textcoords="offset points", fontsize=11)
    ax.axvline(T["ignorance_bits"], color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.axhline(T["multiplicity_bits"], color=MUTED, lw=1.2, ls=(0, (4, 3)))
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for x, y, t in ((xl[0], yl[1], "genuinely many readings"), (xl[1], yl[1], "both"),
                    (xl[1], yl[0], "model doesn't know it"), (xl[0], yl[0], "neither")):
        ax.text(x, y, t, ha="left" if x == xl[0] else "right", va="top" if y == yl[1] else "bottom", fontsize=13, color=MUTED)
    for c in dict.fromkeys(r["category"] for r in W.values()):
        ax.scatter([], [], color=CAT[c], label=CAT_LABEL[c], s=80)
    ax.legend(frameon=False, fontsize=11, loc="center right")
    ax.set_xlabel("how much a dictionary note helps (ignorance, bits)"); ax.set_ylabel("spread left after the note (multiplicity, bits)")
    acc = ig["cell_accuracy"]
    save(fig, d, "fig11_uncertainty_map", f"{acc:.0%} of words land in the corner their kind predicts",
         "Each dot is a word. Further right: a short dictionary note changes the model's translations a lot, so it "
         "did not know the word. Higher up: even with the note, its translations still spread over several meanings. "
         "Colours show what kind of word it is.")


def fig_before_after(rd, d):
    ig = maybe(rd / "ignorance.json")
    if not ig:
        return
    prefer = ["rom:bbona sera", "pt:saudade", "it:tempo", "rom:schertri", "pt:colher", "it:magari"]
    ids = [w for w in prefer if w in ig["words"]][:6]
    fig, axs = plt.subplots(len(ids), 1, figsize=(12, 2.3 * len(ids) + .5), squeeze=False)
    ramp = ["#0d366b", "#256abf", "#5598e7", "#86b6ef", "#b7d3f6"]
    for ax, wid in zip(axs[:, 0], ids):
        rows = [r for r in ig["rows"] if r["word_id"] == wid and r["context"] == 0]
        rows.sort(key=lambda r: (r["condition"] != "bare", r["condition"]))
        for y, r in enumerate(rows):
            from collections import Counter
            cnt = Counter(r["labels"]); n = sum(cnt.values()); x = 0
            for j, (c, k) in enumerate(cnt.most_common()):
                ax.barh(y, k / n, left=x, color=ramp[min(j, 4)], edgecolor=SURF, lw=2)
                if k / n > .12:
                    rep = next(s for s, lab in zip(r["samples"], r["labels"]) if lab == c)
                    ax.text(x + k / n / 2, y, rep[:18], ha="center", va="center", fontsize=10, color="white" if j < 2 else INK)
                x += k / n
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["condition"].replace("gloss+", "note + ") for r in rows], fontsize=11)
        ax.set_title(ig["words"][wid]["word"], loc="left", fontsize=13); ax.set_xlim(0, 1); ax.set_xticks([])
        ax.invert_yaxis()
    save(fig, d, "fig12_before_and_after_the_note", "A dictionary note narrows some words' translations and leaves others spread",
         "Each row shows 20 English translations of the word in one sentence, grouped by meaning (each block is one "
         "meaning; wider = more often). 'bare' is the sentence alone; 'note' adds a dictionary note; 'note + sense' "
         "also says which meaning is intended.")


def fig_humans(rd, d):
    ig = maybe(rd / "ignorance.json")
    if not ig or not ig.get("human"):
        return
    H = ig["human"]
    fig, ax = plt.subplots(figsize=(9, 7)); style(ax)
    for wid, h in H.items():
        r = ig["words"][wid]
        ax.scatter(h["H_human"], r["multiplicity"], s=90, color=CAT[r["category"]], zorder=3)
        ax.annotate(r["word"], (h["H_human"], r["multiplicity"]), xytext=(5, 4), textcoords="offset points", fontsize=11)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([0, lim], [0, lim], color=GRID, lw=1.5)
    ax.set_xlabel("spread across human dictionaries and translators (bits)"); ax.set_ylabel("model's spread after the note (bits)")
    save(fig, d, "fig13_model_vs_humans", "Where humans disagree about a word, does the model spread too?",
         "Each dot is a word. Across: how varied the English renderings by dictionaries and published translators are. "
         "Up: how varied the model's translations are once it has a dictionary note. Dots on the diagonal mean the "
         "model is as varied as the humans.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=MODELS)
    ap.add_argument("--models", nargs="*", default=["3b", "8b", "qwen7b", "aya8b"])
    a = ap.parse_args()
    rd = out_dir(a.model)
    d = out_dir(a.model, "figures")
    rec = maybe(rd / "recipes.json")
    fam = None
    if (rd / "familiarity.json").exists():
        from familiarity import status
        fam = status(a.model)
    if rec:
        fig_cards(rd, d, rec, fam); fig_curves(rd, d, rec); fig_ranking(rd, d, rec, fam)
        fig_patching(rd, d, rec); fig_pivots(rd, d, rec)
        try:
            fig_neighbourhood(rd, d, rec, a.model)
        except FileNotFoundError:
            print("fig03 skipped: activation cache not present on this machine")
    cross = ROOT / "results" / "figures"; cross.mkdir(parents=True, exist_ok=True)
    fig_english_share(cross, a.models); fig_coverage(cross, a.models); fig_distinctions(cross, a.models)
    fig_steering(rd, d); fig_uncertainty_map(rd, d); fig_before_after(rd, d); fig_humans(rd, d)
    (d / "captions.json").write_text(json.dumps(CAPTIONS, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
