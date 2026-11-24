# Reload earlier stages' results, so this stage also works in a fresh kernel (after 00_setup.py).
A = load_json("A_detour.json") if have("A_detour.json") else None
ladder = load_json("B_ladder.json") if have("B_ladder.json") else None
B = load_json("B_roundtrip.json") if have("B_roundtrip.json") else None
summary = load_json("B_summary.json")["per_sentence"] if have("B_summary.json") else {}


if RUN_A and not have("A_detour.json"):
    with Timer("Part A: logit lens"):
        shots = "".join(f'Español: "{e}" - Italiano: "{i}"\n' for e, i in DETOUR_SHOTS)
        rows = []
        for es, it, en in DETOUR_WORDS:
            ti, te, ts = enc(it)["input_ids"][0], enc(en)["input_ids"][0], enc(es)["input_ids"][0]
            if len({ti, te, ts}) < 3: continue           # first tokens must differ
            rows.append((es, it, en, ti, te))
        P_it, P_en = [], []
        for i in range(0, len(rows), 16):
            chunk = rows[i:i + 16]
            seqs = [BOS + enc(shots + f'Español: "{es}" - Italiano: "')["input_ids"] for es, *_ in chunk]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
            for b, s in enumerate(seqs): ids[b, :len(s)] = torch.tensor(s); am[b, :len(s)] = 1
            with torch.no_grad():
                out = BASE(input_ids=ids.to(INPUT_DEV), attention_mask=am.to(INPUT_DEV), output_hidden_states=True)
                last = am.sum(1) - 1
                for l in range(1, N_LAYERS + 1):
                    h = out.hidden_states[l][torch.arange(len(seqs)), last.to(out.hidden_states[l].device)]
                    if l < N_LAYERS: h = FINAL_NORM(h)      # hidden_states[N] already has the final norm
                    p = torch.softmax(LM_HEAD(h.to(LM_HEAD.weight.device)).float(), -1)
                    ti = torch.tensor([r[3] for r in chunk], device=p.device); te = torch.tensor([r[4] for r in chunk], device=p.device)
                    if l == 1: P_it.append([]); P_en.append([])
                    P_it[-1].append(p.gather(1, ti[:, None])[:, 0].cpu().numpy())
                    P_en[-1].append(p.gather(1, te[:, None])[:, 0].cpu().numpy())
            del out
        P_it = np.concatenate([np.stack(x, 1) for x in P_it], 0)     # [words, layers]
        P_en = np.concatenate([np.stack(x, 1) for x in P_en], 0)
        mi, me = P_it.mean(0), P_en.mean(0)
        cut = int(0.9 * N_LAYERS)
        detour_peak = int(np.argmax(me[:cut]) + 1)
        overtake = next((l + 1 for l in range(N_LAYERS) if mi[l] > me[l] and mi[l] > 0.05), None)
        save_json("A_detour.json", dict(words=[r[:3] for r in rows], P_it=P_it, P_en=P_en,
                                        detour_peak_layer=detour_peak, italian_overtakes_at=overtake))
A = load_json("A_detour.json") if have("A_detour.json") else None
if A:
    P_it, P_en = np.array(A["P_it"]), np.array(A["P_en"]); x = np.arange(1, N_LAYERS + 1) / N_LAYERS
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for P, c, lab in [(P_en, HARD, "English word (never asked for)"), (P_it, EASY, "Italian answer")]:
        ax.plot(x, P.mean(0), color=c, lw=2, label=lab)
        ax.fill_between(x, np.percentile(P, 25, 0), np.percentile(P, 75, 0), color=c, alpha=0.15)
    ax.set_xlabel("depth through the model (0 = input, 1 = output)"); ax.set_ylabel("probability")
    ax.set_title(f"Translating Spanish into Italian, the model's working answer passes through English\n({MODEL_ID.split('/')[-1]}, {len(P_it)} words; band = middle half of words)", loc="left")
    ax.legend(frameon=False); savefig(fig, "A_english_detour")
    print(f"English peaks at layer {A['detour_peak_layer']}/{N_LAYERS}; Italian overtakes at {A['italian_overtakes_at']}; "
          f"peak mean P(English) = {P_en.mean(0).max():.3f}")

pack()
