"""Step 6 (CPU): the six figures for the father (handoff §8), PNG + SVG, plus captions.json.

Every title is a finding; it is chosen from the computed statistic it summarises (see `claim` in captions.json),
and falls back to a neutral description when the numbers do not support the stronger sentence.  A figure whose
inputs do not exist yet is skipped.

    python -m sent.figures --model 3b       # -> results/sent/figures/<model>/
Style follows ../figures_for_dad: Inter for labels, Charter italic for foreign words (DejaVu fallbacks if those
fonts are not installed), grey = what is not carried across, teal = easy, coral = hard.  Category palette checked
with the dataviz validator (light surface #fcfcfb: CVD worst adjacent dE 11.7; amber is below 3:1 contrast, so
every mark is also direct-labelled).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from ut.common import MODELS, describe, load_json, save_json
from sent.data import fig_dir, sent_dir
from sent.ladder import read_jsonl
from sent.prompts import PRACTICE, RUNG_ORDER

INK, MUTED, GRID, SURF, GREY = "#1f1f1e", "#6b6a66", "#e6e5e0", "#fcfcfb", "#d9d6cf"
TEAL, CORAL = "#2f7d6d", "#c4562f"
CAT = {"untranslatable_in_context": "#c4562f", "construction": "#2a78d6", "idiom": "#c98a00",
       "control": "#1a8a72", "dialect": "#6250d6"}
CAT_LABEL = {"untranslatable_in_context": "untranslatable word", "construction": "grammatical construction",
             "idiom": "idiom", "control": "plain sentence", "dialect": "Roman dialect (Belli)"}
SPACE_LABEL = {"rt": "round trip (can the original be rebuilt?)", "cl": "inside the model (concept layer)"}
HEAT = LinearSegmentedColormap.from_list("heat", [SURF, "#f3d2c4", CORAL])


def _font(names, fallback):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), fallback)


SANS = _font(["Inter", "Inter Display"], "DejaVu Sans")
SERIF = _font(["Charter", "Bitstream Charter", "XCharter"], "DejaVu Serif")
plt.rcParams.update({"font.family": SANS, "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "svg.fonttype": "none"})
CAPTIONS = {}
LEAD = ["es_duende", "es_tomar_el_pelo", "it_mi_manchi", "es_tren"]


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.8)
    ax.set_axisbelow(True)


def save(fig, d, name, title, how, claim):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.012, 0.012, how, ha="left", va="bottom", fontsize=11, color=MUTED, wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}", dpi=160)
    plt.close(fig)
    CAPTIONS[name] = dict(title=title, how_to_read=how, claim=claim, png=f"{name}.png")
    print("wrote", name)


def lead_ids(curves):
    have = curves["sentences"]
    ids = [s for s in LEAD if s in have]
    for cat in ("idiom", "construction", "control"):
        if not any(have[s]["record"]["category"] == cat for s in ids):
            ids += [s for s, v in have.items() if v["record"]["category"] == cat][:1]
    return ids[:4]


# ------------------------------------------------------------------ 1. heat map
def fig_heat(d, curves):
    ids = lead_ids(curves)
    fig = plt.figure(figsize=(13, 2.15 * len(ids) + 1.4))
    r = fig.canvas.get_renderer()
    top = 1 - 0.9 / fig.get_figheight()
    rowh = (top - 0.75 / fig.get_figheight()) / len(ids)
    hots = []
    for k, sid in enumerate(ids):
        e = curves["sentences"][sid]
        y = top - k * rowh - 0.25 * rowh
        fig.text(0.02, y + 0.33 * rowh, CAT_LABEL[e["record"]["category"]], fontsize=10.5, color=MUTED)
        rt = {(h["start"], h["end"]): h for h in e["spaces"]["rt"]["heat"]}
        cl = {(h["start"], h["end"]): h for h in e["spaces"]["cl"]["heat"]}
        x = 0.02
        for key, h in rt.items():
            lv = h["leftover"]
            t = fig.text(x, y, h["word"], fontsize=27, family=SERIF, style="italic", color=INK, va="center",
                         bbox=dict(boxstyle="round,pad=0.25", fc=HEAT(lv) if lv is not None else SURF,
                                   ec=GREY if lv is None else "none", ls=":" if lv is None else "-"))
            bb = t.get_window_extent(r).transformed(fig.transFigure.inverted())
            c = cl.get(key, {}).get("leftover")
            if c is not None:                   # grey track = the word's width; dark part = share not carried
                yy = [bb.y0 - 0.02] * 2
                fig.add_artist(plt.Line2D([bb.x0, bb.x1], yy, color=GREY, lw=5, solid_capstyle="butt"))
                fig.add_artist(plt.Line2D([bb.x0, bb.x0 + (bb.x1 - bb.x0) * c], yy, color=INK, lw=5,
                                          solid_capstyle="butt"))
            x = bb.x1 + 0.018
            if lv is not None:
                hots.append((sid, h["word"], lv * h["weight"], c))     # leftover AMOUNT (weight x share)
        fig.text(0.02, y - 0.42 * rowh, "standard translation:  " + e["text_B1"].split("\n")[0], fontsize=13,
                 color=MUTED)
    duende = [h for h in hots if h[0] == "es_duende"]
    best = max(duende, key=lambda h: h[2]) if duende else None
    title = (f"In 'me gusta mucho el duende', the most meaning is lost on '{best[1]}'"
             if best else "Where each sentence resists translation")
    save(fig, d, "fig1_where_a_sentence_resists", title,
         "How to read this: the redder a word's background, the less of it can be rebuilt from the standard English "
         "translation. The bar under a word shows the same thing measured inside the model (all dark = nothing "
         "carried). Dotted boxes: words that carry almost no information of their own.",
         dict(statistic="word leftover amount (information weight x share not carried), round trip, greedy "
              "standard translation", es_duende=duende))


# ------------------------------------------------------------------ 2. distortion vs length
def fig_curves(d, curves, cands):
    ids = lead_ids(curves)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    eps = curves["epsilon"]
    med = {}
    for rung in RUNG_ORDER:
        L = [c["len"] for c in cands if c["rung"] == rung]
        if L:
            med[rung] = np.median(L)
    claim = {}
    # practices: group rungs whose median lengths are within 25% of each other into one label
    groups = []
    for rung, m in sorted(med.items(), key=lambda t: t[1]):
        if groups and m / groups[-1][0] < 1.25:
            groups[-1][1].append(PRACTICE[rung])
        else:
            groups.append([m, [PRACTICE[rung]]])
    for ax, sp in zip(axs, ("rt", "cl")):
        ends = []
        for sid in ids:
            e = curves["sentences"][sid]
            cv = e["spaces"][sp]["whole"]
            if not cv:
                continue
            col = CAT[e["record"]["category"]]
            L = np.r_[cv["L"], cv["L"][-1] * 1.4]
            D = np.r_[cv["D"], cv["D"][-1]]
            ax.step(L, D, where="post", color=col, lw=2.2)
            words = e["record"]["text"].split()
            ends.append([D[-1], col, " ".join(words[:3]) + (" …" if len(words) > 3 else "")])
            claim.setdefault(sp, {})[sid] = dict(U=cv["U"], censored=cv["censored"])
        ends.sort(key=lambda t: t[0])                                   # dodge end labels vertically
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.055)
        xmax = max([c["len"] for c in cands] + [8]) * 1.4
        for yv, col, lab in ends:
            ax.text(xmax * 1.05, yv, lab, fontsize=10.5, family=SERIF, style="italic", color=col, va="center")
        ax.axhline(eps[sp], color=TEAL, lw=1.4, ls="--")
        ax.text(2.1, eps[sp] + 0.01, "gets across", color=TEAL, fontsize=11, va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xlim(2, xmax)
        ax.set_ylim(0, 1)
        ticks = [2 ** k for k in range(1, int(np.log2(xmax)) + 1)]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_title(SPACE_LABEL[sp], fontsize=12.5, color=MUTED, loc="left")
        style(ax)
        ax.set_xlabel("length of the English, in tokens (each step right = twice as long)")
        for m, names in groups:
            ax.axvline(m, color=GRID, lw=.8, zorder=0)
            ax.text(m * 0.97, 0.985, " · ".join(names), rotation=90, fontsize=9.5, color=MUTED, ha="right", va="top",
                    transform=ax.get_xaxis_transform(), bbox=dict(fc=SURF, ec="none", pad=0.6))
    axs[0].set_ylabel("share of the meaning not carried")
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.06, right=0.9, wspace=0.42)
    du = claim.get("rt", {}).get("es_duende")
    ctrl = [v for s, v in claim.get("rt", {}).items() if curves["sentences"][s]["record"]["category"] == "control"]
    if du and ctrl and all(v["U"] == 0 for v in ctrl) and (du["censored"] or du["U"] >= 1):
        title = "A plain sentence gets across at normal length; 'duende' needs a much longer English text"
    else:
        title = "How much English it takes before the meaning gets across"
    save(fig, d, "fig2_how_much_english", title,
         "How to read this: each line is one sentence. Moving right means allowing a longer English text (from a word-"
         "for-word gloss to a full paragraph of commentary); lower means more of the meaning carried. The dashed line "
         "is how close an easy sentence gets with an ordinary translation.", claim)


# ------------------------------------------------------------------ 3. translation card
def fig_card(d, curves, cands, spread, scores, sid="es_duende"):
    if sid not in curves["sentences"]:
        return
    e = curves["sentences"][sid]
    sp = [x for x in spread["spans"] if x["sid"] == sid] if spread else []
    b3 = [s for s in scores if s["sid"] == sid and s["rung"] == "B3" and s["role"] == "cand"]
    best_gloss = min(b3, key=lambda s: s["d_rt"])["text"] if b3 else None
    b7 = next((c["text"] for c in cands if c["cid"] == f"{sid}|B7|-1"), None)
    fig = plt.figure(figsize=(13, 3.2 + 1.5 * len(sp) + (1.3 if best_gloss else 0) + (1.3 if b7 else 0)))
    H = fig.get_figheight()
    y = 1 - 1.0 / H
    fig.text(0.03, y, e["record"]["text"], fontsize=30, family=SERIF, style="italic", color=INK, va="top")
    y -= 0.85 / H
    fig.text(0.03, y, "standard translation:  " + e["text_B1"].split("\n")[0], fontsize=15, color=INK, va="top")
    y -= 0.75 / H
    for s in sp:
        fig.text(0.03, y, s["span"]["text"], fontsize=20, family=SERIF, style="italic", color=CORAL, va="top")
        fig.text(0.03, y - 0.42 / H, f"{s['n_clusters']} sense{'s' * (s['n_clusters'] != 1)} in {s['n_used']} usable translations", fontsize=10,
                 color=MUTED, va="top")
        x = 0.26
        for c in s["clusters"][:5]:
            w = 0.66 * c["share"]
            fig.add_artist(plt.Rectangle((x, y - 0.5 / H), max(w - 0.004, 0.002), 0.42 / H, color=CORAL,
                                         alpha=0.25 + 0.75 * c["share"], transform=fig.transFigure))
            fig.text(x, y - 0.6 / H, f"{c['rep'] or '(left out)'}  {c['share']:.0%}", fontsize=11, color=INK, va="top")
            x += max(w, 0.13)
        y -= 1.5 / H
    for label, txt in (("best short gloss", best_gloss), ("a coined term", b7)):
        if txt:
            fig.text(0.03, y, label, fontsize=11, color=MUTED, va="top")
            fig.text(0.26, y, txt.replace("\n\n", "\n")[:300], fontsize=12.5, color=INK, va="top", wrap=True)
            y -= 1.3 / H
    save(fig, d, "fig3_opening_up_a_sentence", "Opening up a sentence instead of substituting word for word",
         "How to read this: under the standard translation, each hard part is expanded into the different English "
         "senses the model gives it across repeated tries (the bar widths are the shares), followed by the model's best short "
         "gloss and a term it coined. A prototype of a translation that shows its range rather than hiding it.",
         dict(statistic="spread clusters (NLI), lowest-distortion inline gloss, greedy coinage", sid=sid))


# ------------------------------------------------------------------ 4. targeting
def fig_targeting(d, tg):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    claim = {}
    for ax, sp in zip(axs, ("rt", "cl")):
        s = tg["summary"][sp]
        vals = [s["cut_targeted"], s["cut_random"]]
        ax.bar([0, 1], [v[0] for v in vals], color=[CORAL, GREY], width=0.55)
        ax.errorbar([0, 1], [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="none", ecolor=INK, capsize=5, lw=1.2)
        for i, v in enumerate(vals):
            ax.text(i, max(v[2], 0) + 0.005, f"{v[0]:.0%}", ha="center", va="bottom", fontsize=12, color=INK)
        ax.set_xticks([0, 1], ["gloss the words\nthe map marks", "gloss random\nwords"])
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_title(SPACE_LABEL[sp], fontsize=12, color=MUTED, loc="left")
        style(ax)
        claim[sp] = s["targeted_minus_random"]
    axs[0].set_ylabel("distortion removed by the glosses")
    fig.subplots_adjust(top=0.8, bottom=0.25, left=0.1, right=0.97, wspace=0.15)
    # a space "supports" the claim only if the CI excludes 0 AND the effect is at least 1 point of distortion
    # ... and only if the targeted glosses themselves reliably help (a "less harmful" arm is not "helps")
    ok = {sp: claim[sp][1] > 0 and claim[sp][0] >= 0.01 and tg["summary"][sp]["cut_targeted"][1] > 0 for sp in claim}
    both, one = all(ok.values()), any(ok.values())
    title = ("Commentary helps most where the map says" if both else
             "Commentary helps more where the map says, by one of the two measures" if one else
             "Glossing the marked words did not reliably beat glossing random words")
    save(fig, d, "fig4_commentary_where_the_map_says", title,
         "How to read this: starting from the same standard translation, the model added short bracketed glosses "
         "either to the words the heat map marked or to random words (same number of glosses). Taller = more of the "
         "lost meaning recovered. Lines show the 95% range.",
         dict(statistic="targeted minus random cut, bootstrap 95% CI", **claim,
              added_tokens={k: v[0] for k, v in tg["added_length"].items()}))


# ------------------------------------------------------------------ 5. ranking
def fig_ranking(d, curves, sp="rt"):
    items = []
    for sid, e in curves["sentences"].items():
        cv = e["spaces"][sp]["whole"]
        if cv:
            items.append((cv["U"], cv["censored"], e["record"]))
    items.sort(key=lambda t: (t[0] + 100 * t[1]))
    fig, ax = plt.subplots(figsize=(12, 0.36 * len(items) + 2.2))
    for i, (U, cens, rec) in enumerate(items):
        ax.barh(i, max(U, 0.03), color=CAT[rec["category"]], hatch="///" if cens else None,
                edgecolor=SURF if not cens else CAT[rec["category"]], fc=CAT[rec["category"]] if not cens else SURF,
                height=0.72)
        ax.text(-0.08, i, rec["text"], ha="right", va="center", fontsize=10.5, family=SERIF, style="italic", color=INK)
        ax.text(max(U, 0.03) + 0.05, i, (">" if cens else "") + f"{U:.1f}", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks([])
    ax.set_xlabel("doublings of length needed before the meaning gets across")
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT.values()] + \
              [plt.Rectangle((0, 0), 1, 1, fc=SURF, ec=MUTED, hatch="///")]
    ax.legend(handles, list(CAT_LABEL.values()) + ["never got across (lower bound)"], frameon=False, fontsize=10,
              loc="lower right")
    fig.subplots_adjust(left=0.42, right=0.95, top=1 - 1.0 / fig.get_figheight(), bottom=1.6 / fig.get_figheight())
    cats = {}
    for U, cens, rec in items:
        cats.setdefault(rec["category"], []).append(U)
    means = {c: float(np.mean(v)) for c, v in cats.items()}
    # titles state the category averages rather than a rule: individual sentences cross categories
    order = ["control", "untranslatable_in_context", "construction", "idiom", "dialect"]
    short = dict(control="plain", untranslatable_in_context="untranslatable word", construction="construction",
                 idiom="idiom", dialect="dialect")
    parts = [f"{short[c]} {means[c]:.1f}" for c in order if c in means]
    title = "Doublings of length needed, average by kind of sentence: " + ", ".join(parts)
    save(fig, d, "fig5_ranking", title,
         "How to read this: each bar is one sentence; its length is how many times the English had to double in length "
         "(from the standard translation) before the original could be rebuilt as well as an easy sentence's can. "
         "Hatched: it never got there, even with a paragraph of commentary.",
         dict(statistic=f"U ({sp}), mean by category", means=means))


# ------------------------------------------------------------------ 6. agreement
def fig_agreement(d, curves, val):
    fig, ax = plt.subplots(figsize=(9, 8))
    for sid, e in curves["sentences"].items():
        a, b = e["spaces"]["rt"]["whole"], e["spaces"]["cl"]["whole"]
        if not (a and b):
            continue
        col = CAT[e["record"]["category"]]
        hollow = a["censored"] or b["censored"]
        ax.scatter(a["A"], b["A"], s=80, color=SURF if hollow else col, edgecolor=col, lw=2, zorder=3)
        ax.annotate(e["record"]["text"].split()[-1].strip(".,!?") if len(e["record"]["text"]) > 18 else e["record"]["text"],
                    (a["A"], b["A"]), xytext=(5, 4), textcoords="offset points", fontsize=9, family=SERIF,
                    style="italic", color=INK)
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color=GRID, lw=1, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("round trip: average share not carried")
    ax.set_ylabel("inside the model: average share not carried")
    style(ax, grid="both")
    rho = val["agreement"]["A"]["spearman"]
    title = ("The two ways of measuring broadly agree" if rho is not None and rho >= 0.5 else
             "The two ways of measuring only partly agree" if rho is not None and rho >= 0.2 else
             "The two ways of measuring disagree")
    fig.subplots_adjust(top=0.9, bottom=0.16)
    save(fig, d, "fig6_two_measures", title + (f" (rank correlation {rho:.2f})" if rho is not None else ""),
         "How to read this: each dot is a sentence, placed by how much meaning the English loses on average over all "
         "lengths of English, measured two ways. Dots near the diagonal: the two measures agree. Hollow dots never "
         "got across by at least one measure.", dict(statistic="Spearman of A (round trip vs concept layer)", rho=rho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=MODELS)
    a = ap.parse_args()
    src = sent_dir(a.model)
    d = fig_dir(a.model)
    get = lambda n: load_json(src / n) if (src / n).exists() else None
    curves, val, spread, tg = get("curves.json"), get("validation.json"), get("spread.json"), get("targeting.json")
    if curves is None:
        raise SystemExit("run sent.analyze first")
    cands = read_jsonl(src / "candidates.jsonl")
    scores = read_jsonl(src / "scores.jsonl")
    fig_heat(d, curves)
    fig_curves(d, curves, cands)
    fig_card(d, curves, cands, spread, scores)
    if tg:
        fig_targeting(d, tg)
    fig_ranking(d, curves)
    if val:
        fig_agreement(d, curves, val)
    save_json(d / "captions.json", dict(model=describe(a.model), fonts=dict(sans=SANS, serif=SERIF), **CAPTIONS))
    print("captions ->", d / "captions.json")


if __name__ == "__main__":
    main()
