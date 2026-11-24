# strict_pass.py: re-run the C2 probes and Part D with "shared" = held-out rho > 0.71 (more than half kept per round
# trip) instead of "above the shuffled-pair null", which saturated at ~8190 of 8192 dimensions for es on the 70B.
# Run in the same kernel after C2 and D; outputs get a _strict suffix, the original files are untouched.
import torch, threading, contextlib, traceback
STRICT = 0.5 ** 0.5
for _k, _F in C_FITS.items():
    _F["shared_null"] = _F["shared"]
    _F["shared"] = torch.nonzero(_F["rho_h"] > STRICT)[:, 0]
    print("strict shared", _k, len(_F["shared"]), "of", len(_F["rho_h"]))
_save_json_orig, _savefig_orig = save_json, savefig
save_json = lambda name, obj: _save_json_orig(name.replace(".json", "_strict.json"), obj)
savefig = lambda fig, name: _savefig_orig(fig, name + "_strict")
if RUN_C and "es" in C_RES and C_SPANS.get("es"):
    held = torch.load(os.path.join(CACHE, "C_es_held.pt"), weights_only=False); meta = held["meta"]
    l = BEST_LAYER["es"]; F_ = C_FITS[("es", l)]; fit = F_["fit"]
    if any(m["morph"] for m in meta):
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import cross_val_score
        Z = (((held["X"][l].to(CDEV).double() - fit["mx"]) @ fit["Wx"]) @ fit["U"]).float().cpu().numpy()
        Z /= Z.std(0, keepdims=True) + 1e-8
        sh = np.zeros(Z.shape[1], bool); sh[F_["shared"].cpu().numpy()] = True
        def feat(m, key): return dict(kv.split("=") for kv in m["morph"].split("|") if "=" in kv).get(key)
        tasks = {
            "grammatical gender": [(i, feat(m, "Gender")) for i, m in enumerate(meta) if m["pos"] in ("NOUN", "ADJ") and feat(m, "Gender") in ("Masc", "Fem")],
            "number": [(i, feat(m, "Number")) for i, m in enumerate(meta) if m["pos"] in ("NOUN", "ADJ", "VERB") and feat(m, "Number") in ("Sing", "Plur")],
            "subjunctive vs indicative": [(i, feat(m, "Mood")) for i, m in enumerate(meta) if m["pos"] in ("VERB", "AUX") and feat(m, "Mood") in ("Sub", "Ind")],
            "imperfect vs preterite": [(i, feat(m, "Tense")) for i, m in enumerate(meta) if m["pos"] in ("VERB", "AUX") and feat(m, "Mood") == "Ind" and feat(m, "Tense") in ("Imp", "Past")],
            "noun vs verb (sanity)": [(i, m["pos"]) for i, m in enumerate(meta) if m["pos"] in ("NOUN", "VERB")],
        }
        probe = {}
        for name, items in tasks.items():
            labs = [b for _, b in items]
            if len(items) < 60 or min(labs.count(v) for v in set(labs)) < 20: print(f"  {name}: too few examples"); continue
            ii = np.array([i for i, _ in items]); y = np.array([b == sorted(set(labs))[0] for b in labs])
            clf = RidgeClassifier(alpha=float(len(ii)), class_weight="balanced")
            acc = {k: float(cross_val_score(clf, Z[ii][:, cols], y, cv=5, scoring="balanced_accuracy").mean())
                   for k, cols in [("all", np.ones_like(sh)), ("shared only", sh), ("private only", ~sh)] if cols.any()}
            clf.fit(Z[ii], y); wgt = np.ravel(clf.coef_) ** 2      # coef_ is (1, d) or (d,) depending on the sklearn version
            probe[name] = dict(n=len(ii), shared_share=float(wgt[sh].sum() / wgt.sum()), expected=float(sh.mean()), acc=acc)
            print(f"  {name:28s} n={len(ii):5d}  weight on shared directions {probe[name]['shared_share']:.2f} "
                  f"(chance {sh.mean():.2f})  accuracy all/shared/private = " + "/".join(f"{v:.2f}" for v in acc.values()))
        save_json("C_probes_es.json", dict(layer=l, probes=probe))
exec(open("/content/ut/D_mediation.py").read(), globals())
save_json, savefig = _save_json_orig, _savefig_orig
pack()
