"""Real-GPU live v2 Co-Translation Studio benchmark & recorded session generator.

Exercises the full live CoTranslationSession stack wired to all trained artifacts
(global_axes.npz, multiref_lora.pt, spectral_reft.pt, fork_head.pt) on multi-stanza
poems, untranslatable sentences, and dense literary paragraphs.

Measures:
- Prefill latency, word-step latency, fast glide ms/word, rewind + redo + branch checkout
  (verifying Gemma 3 DynamicSlidingWindowLayer KV crop/clone safety on GPU)
- Tier-1 zero-backward ForkHead(n_dirs=2) compass latency vs Tier-2 gradient compass latency
- Live compass steering responsiveness and global tone/practice slider responsiveness
- Formatting preservation (\n line breaks and \n\n stanza breaks) on multi-stanza poems

Outputs:
- results/explorer/<model>/live_benchmark.json
- results/explorer/<model>/sessions/<text_id>.json  (for multi-model offline showcase)

CLI:
    python -m explorer.bench_live --model g3_12b_pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from explorer.live_engine import LiveTranslationEngine
from ut.common import ROOT


BENCH_TEXTS: list[dict[str, str]] = [
    {
        "id": "duende_flamenco",
        "title": "Camarón's Duende (Flamenco)",
        "lang": "Spanish",
        "text": "Cuando canta Camarón, tiene duende; me gusta mucho el duende.",
    },
    {
        "id": "becquer_liii_2stanzas",
        "title": "Bécquer — Rima LIII (2 stanzas, indented)",
        "lang": "Spanish",
        "text": (
            "Volverán las oscuras golondrinas\n"
            "  en tu balcón sus nidos a colgar,\n"
            "y otra vez con el ala a sus cristales\n"
            "  jugando llamarán.\n"
            "\n"
            "Pero aquellas que el vuelo refrenaban\n"
            "  tu hermosura y mi dicha a contemplar,\n"
            "aquellas que aprendieron nuestros nombres...\n"
            "  ¡ésas... no volverán!"
        ),
    },
    {
        "id": "machado_caminante",
        "title": "Machado — Proverbios y cantares XXIX",
        "lang": "Spanish",
        "text": (
            "Caminante, son tus huellas\n"
            "el camino y nada más;\n"
            "caminante, no hay camino,\n"
            "se hace camino al andar.\n"
            "\n"
            "Al andar se hace camino\n"
            "y al volver la vista atrás\n"
            "se ve la senda que nunca\n"
            "se ha de volver a pisar."
        ),
    },
    {
        "id": "sobremesa_paragraph",
        "title": "Literary Paragraph (Sobremesa & Duende)",
        "lang": "Spanish",
        "text": (
            "La sobremesa del domingo se alargó hasta el anochecer entre risas, café frío "
            "y un silencio cómplice que nadie se atrevía a romper. Cuando alguien puso a "
            "Camarón en el tocadiscos, la habitación entera pareció encogerse: aquello no "
            "era técnica ni adorno, sino un duende áspero que subía desde las plantas de los pies."
        ),
    },
]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms_since(t0: float) -> float:
    _sync_cuda()
    return round((time.perf_counter() - t0) * 1000.0, 2)


def run_benchmark(
    model_key: str,
    out_dir: Path | None = None,
    max_poem_words: int = 56,
    skip_gradient_compass: bool = False,
) -> dict[str, Any]:
    out = out_dir or (ROOT / "results" / "explorer" / model_key)
    out.mkdir(parents=True, exist_ok=True)
    sess_dir = out / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)

    axes_p = out / "phase3" / "global_axes.npz"
    lora_p = out / "phase3" / "multiref_lora.pt"
    reft_p = out / "phase3" / "spectral_reft.pt"
    fh_p = out / "fork_head.pt"

    t_load0 = time.perf_counter()
    eng = LiveTranslationEngine(
        model_key=model_key,
        global_axes_path=axes_p if axes_p.exists() else None,
        lora_path=lora_p if lora_p.exists() else None,
        reft_path=reft_p if reft_p.exists() else None,
        fork_head_path=fh_p if fh_p.exists() else None,
    )
    eng.ensure_loaded()
    load_s = round(time.perf_counter() - t_load0, 2)
    status = eng.status()

    results: list[dict[str, Any]] = []
    try:
        for item in BENCH_TEXTS:
            tid = item["id"]
            print(f"[bench_live:{model_key}] === {tid} ===", flush=True)

            # 1. Start session (prefill + opening frontier)
            _sync_cuda()
            t0 = time.perf_counter()
            st0 = eng.sessions.start(
                source_text=item["text"],
                src_lang=item["lang"],
                tgt_lang="English",
            )
            prefill_ms = _ms_since(t0)
            sid = st0["session_id"]

            # 2. Glide to first fork (up to 18 words)
            _sync_cuda()
            t0 = time.perf_counter()
            glide_events = list(eng.sessions.glide(sid, max_words=18, stop_on_fork=True))
            glide_ms = _ms_since(t0)
            step_evs = [e for e in glide_events if e.get("event") == "step"]
            done_ev = next((e for e in reversed(glide_events) if e.get("event") == "done"), {})
            st_fork = eng.sessions.get_state(sid)
            frontier = st_fork.get("frontier") or {}
            branches = frontier.get("branches") or []
            n_sub_branches = sum(len(b.get("sub_branches") or []) for b in branches)

            # 3. Tier-1 ForkHead compass probe + live steering
            _sync_cuda()
            t0 = time.perf_counter()
            comp_fast = eng.sessions.probe_compass(sid, n_samples=8, lookahead_tokens=10, seed=0)
            compass_fast_ms = _ms_since(t0)

            _sync_cuda()
            t0 = time.perf_counter()
            steer_pos = eng.sessions.steer_compass(sid, x=0.85, y=0.25, max_new=14)
            steer_ms = _ms_since(t0)
            steer_neg = eng.sessions.steer_compass(sid, x=-0.85, y=-0.25, max_new=14)

            # 4. Tier-2 Gradient compass probe (optional, timed once per text)
            compass_grad_ms = None
            compass_grad_mode = None
            if not skip_gradient_compass:
                _sync_cuda()
                t0 = time.perf_counter()
                comp_grad = eng.sessions.probe_compass(
                    sid, n_samples=6, lookahead_tokens=10, seed=1, force_gradient=True
                )
                compass_grad_ms = _ms_since(t0)
                compass_grad_mode = comp_grad.get("source") or comp_grad.get("mode")

            # 5. Branch step -> Rewind -> Redo -> Fork alternate branch -> Checkout
            n_before_branch = st_fork["n_tokens"]
            _sync_cuda()
            t0 = time.perf_counter()
            st_step1 = eng.sessions.step(sid, mode="word", branch_index=0 if branches else None)
            step_word_ms = _ms_since(t0)

            _sync_cuda()
            t0 = time.perf_counter()
            st_rw = eng.sessions.rewind(sid, n_before_branch)
            rewind_ms = _ms_since(t0)

            _sync_cuda()
            t0 = time.perf_counter()
            st_redo = eng.sessions.redo(sid, mode="word")
            redo_ms = _ms_since(t0)
            redo_exact_match = (st_redo["n_tokens"] == st_step1["n_tokens"])

            # Now rewind again and explore alternate branch #1 (if >=2 branches) to record real tree history
            alt_branch_text = None
            if len(branches) >= 2:
                eng.sessions.rewind(sid, n_before_branch)
                st_alt = eng.sessions.step(sid, mode="word", branch_index=1, commit_full_branch=True)
                alt_branch_text = st_alt["committed_text"]
                # Checkout back to main node (root) to continue full translation
                root_id = st_alt["history"][0]["id"]
                eng.sessions.checkout(sid, root_id)

            # 6. Global slider comparison (plain/domesticating vs poetic/foreignising)
            st_sl_plain = eng.sessions.set_sliders(
                sid, {"plain_vs_poetic": -0.8, "foreignising_vs_domesticating": 0.8}
            )
            ghost_plain = (st_sl_plain.get("frontier") or {}).get("ghost", {}).get("text", "")
            st_sl_poetic = eng.sessions.set_sliders(
                sid, {"plain_vs_poetic": 0.8, "foreignising_vs_domesticating": -0.8}
            )
            ghost_poetic = (st_sl_poetic.get("frontier") or {}).get("ghost", {}).get("text", "")
            # Reset sliders to neutral before completing the translation
            eng.sessions.set_sliders(sid, {"plain_vs_poetic": 0.0, "foreignising_vs_domesticating": 0.0})

            # 7. Continue gliding to finish the poem/paragraph (testing long-form formatting & cache stability)
            rem_words = max(8, max_poem_words - len(step_evs))
            list(eng.sessions.glide(sid, max_words=rem_words, stop_on_fork=False))
            st_final = eng.sessions.get_state(sid)

            committed_text = st_final.get("committed_text", "")
            src_newlines = item["text"].count("\n")
            tgt_newlines = committed_text.count("\n")
            src_has_stanza = "\n\n" in item["text"]
            tgt_has_stanza = "\n\n" in committed_text

            # Record standalone session JSON for multi-model offline viewer
            session_record = {
                "session_id": f"{model_key}_{tid}",
                "model_key": model_key,
                "text_id": tid,
                "title": f"{item['title']} ({model_key})",
                "source_text": item["text"],
                "src_lang": item["lang"],
                "tgt_lang": "English",
                "committed_text": committed_text,
                "n_tokens": st_final["n_tokens"],
                "history": st_final["history"],
                "alignment": st_final.get("alignment"),
                "fork_snapshot": {
                    "at_token": n_before_branch,
                    "regime": frontier.get("regime"),
                    "entropy": frontier.get("entropy"),
                    "fork_score": frontier.get("fork_score"),
                    "branches": [
                        {
                            "first_word": b.get("first_word_text"),
                            "preview": b.get("preview_text"),
                            "prob": b.get("prob"),
                            "distinctive": b.get("distinctive"),
                            "sub_branches": [
                                {"word": sb.get("first_word_text"), "prob": sb.get("prob")}
                                for sb in (b.get("sub_branches") or [])
                            ],
                        }
                        for b in branches
                    ],
                },
                "compass_sample": {
                    "source": comp_fast.get("source") or comp_fast.get("mode"),
                    "fast_ms": compass_fast_ms,
                    "grad_ms": compass_grad_ms,
                    "steer_pos": steer_pos.get("preview_text"),
                    "steer_neg": steer_neg.get("preview_text"),
                },
                "slider_comparison": {
                    "plain_domesticating": ghost_plain,
                    "poetic_foreignising": ghost_poetic,
                },
            }
            (sess_dir / f"{tid}.json").write_text(
                json.dumps(session_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            rec = {
                "id": tid,
                "prefill_ms": prefill_ms,
                "glide_words": len(step_evs),
                "glide_ms": glide_ms,
                "glide_ms_per_word": round(glide_ms / max(len(step_evs), 1), 2),
                "glide_stopped_reason": done_ev.get("stopped_reason") or done_ev.get("reason"),
                "fork_regime": frontier.get("regime"),
                "fork_entropy": frontier.get("entropy"),
                "fork_score": frontier.get("fork_score"),
                "n_branches": len(branches),
                "n_sub_branches": n_sub_branches,
                "compass_fast_mode": comp_fast.get("source") or comp_fast.get("mode"),
                "compass_fast_ms": compass_fast_ms,
                "compass_grad_mode": compass_grad_mode,
                "compass_grad_ms": compass_grad_ms,
                "steer_ms": steer_ms,
                "steer_distinct": (steer_pos.get("preview_text", "").strip() != steer_neg.get("preview_text", "").strip()),
                "step_word_ms": step_word_ms,
                "rewind_ms": rewind_ms,
                "redo_ms": redo_ms,
                "redo_exact_match": redo_exact_match,
                "n_history_nodes": len(st_final["history"]),
                "slider_shifted": (ghost_plain.strip() != ghost_poetic.strip()),
                "src_newlines": src_newlines,
                "tgt_newlines": tgt_newlines,
                "src_has_stanza": src_has_stanza,
                "tgt_has_stanza": tgt_has_stanza,
                "final_n_tokens": st_final["n_tokens"],
                "final_text_preview": committed_text[:280],
                "alt_branch_preview": (alt_branch_text or "")[:180],
            }
            results.append(rec)
            print(
                f"  prefill={prefill_ms}ms glide={len(step_evs)}w ({rec['glide_ms_per_word']}ms/w, {rec['glide_stopped_reason']}) "
                f"compass_fast={compass_fast_ms}ms({rec['compass_fast_mode']}) grad={compass_grad_ms}ms "
                f"step={step_word_ms}ms rewind={rewind_ms}ms redo_ok={redo_exact_match} "
                f"nl={tgt_newlines}/{src_newlines}",
                flush=True,
            )
            eng.sessions.delete(sid)
    finally:
        eng.close()

    vram_peak_gb = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    report = {
        "model_key": model_key,
        "gpu": gpu_name,
        "vram_peak_gb": vram_peak_gb,
        "load_seconds": load_s,
        "artifacts": {
            "global_axes": axes_p.exists(),
            "lora": lora_p.exists(),
            "reft": reft_p.exists(),
            "fork_head": fh_p.exists(),
            "status": status,
        },
        "summary": {
            "mean_prefill_ms": round(sum(r["prefill_ms"] for r in results) / len(results), 2),
            "mean_step_word_ms": round(sum(r["step_word_ms"] for r in results) / len(results), 2),
            "mean_glide_ms_per_word": round(sum(r["glide_ms_per_word"] for r in results) / len(results), 2),
            "mean_rewind_ms": round(sum(r["rewind_ms"] for r in results) / len(results), 2),
            "mean_compass_fast_ms": round(sum(r["compass_fast_ms"] for r in results) / len(results), 2),
            "mean_steer_ms": round(sum(r["steer_ms"] for r in results) / len(results), 2),
            "all_redo_exact": all(r["redo_exact_match"] for r in results),
            "steer_distinct_count": sum(1 for r in results if r["steer_distinct"]),
            "slider_shifted_count": sum(1 for r in results if r["slider_shifted"]),
            "total_sub_branches": sum(r["n_sub_branches"] for r in results),
        },
        "texts": results,
    }
    bench_path = out / "live_benchmark.json"
    bench_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench_live:{model_key}] Saved {bench_path} (peak VRAM {vram_peak_gb} GB)", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark v2 live Co-Translation Studio on GPU and record offline sessions.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-poem-words", type=int, default=56)
    ap.add_argument("--skip-gradient-compass", action="store_true")
    args = ap.parse_args(argv)
    run_benchmark(
        model_key=args.model,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        max_poem_words=args.max_poem_words,
        skip_gradient_compass=args.skip_gradient_compass,
    )


if __name__ == "__main__":
    main()
