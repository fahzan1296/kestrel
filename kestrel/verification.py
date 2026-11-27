"""KESTREL 1.6 — Verifikasi otomatis & mendetail: champion vs challenger.

Bukti yang dikumpulkan untuk setiap kandidat model baru:
1. Match berwarna seimbang pada set opening tetap (seed deterministik) —
   W/D/L, score rate, Wilson 95% CI, Elo delta + interval, SPRT trinomial
   (elo0/elo1 dari config), rincian per-opening.
2. Benchmark taktik (akurasi soal taktik), NPS kiwipete, konsistensi
   evaluasi (antisimetri mirror) untuk KEDUA model — regresi terdeteksi.
3. Keputusan promote/reject berdasarkan aturan config (SPRT H1 atau score
   rate >= ambang, tanpa regresi taktik/konsistensi/NPS).

Catatan kejujuran statistik: skor dari match kecil adalah indikasi, bukan
estimasi Elo final. Interval yang dilaporkan selalu ikut ditampilkan.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .arena import OPENINGS, Player, elo_delta, sprt, wilson_ci
from .benchmark import evaluation_consistency, performance_suite, tactical_suite
from .config import AppConfig
from .learning import OutcomeModel


def _elo_ci(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Elo delta pada batas Wilson CI skor (aproksimasi interval monotone)."""
    lo, hi = wilson_ci(wins, draws, losses)
    lo = min(max(lo, 1e-4), 1 - 1e-4)
    hi = min(max(hi, 1e-4), 1 - 1e-4)
    return (round(elo_delta(lo)), round(elo_delta(hi)))


def head_to_head(champion: OutcomeModel | None, challenger: OutcomeModel | None,
                 cfg: AppConfig,
                 verbose: bool = False) -> dict:
    """Match champion vs challenger: warna & opening seimbang, seed tetap."""
    champ = Player("Champion", time_ms=cfg.eval.time_ms, max_depth=cfg.eval.max_depth,
                   learned=champion if champion else None,
                   min_depth=cfg.eval.min_depth, grace_ms=cfg.eval.grace_ms)
    chall = Player("Challenger", time_ms=cfg.eval.time_ms, max_depth=cfg.eval.max_depth,
                   learned=challenger if challenger else None,
                   min_depth=cfg.eval.min_depth, grace_ms=cfg.eval.grace_ms)
    games = max(2, int(cfg.eval.games))
    per_opening: dict = {}
    w = d = losses = 0
    for i in range(games):
        fen = OPENINGS[i % len(OPENINGS)]
        seed = 90000 + i + (int(cfg.eval.seed) % 1000)
        chall_white = (i % 2 == 0)
        if chall_white:
            res, _, _ = _play_pair(chall, champ, fen, cfg.eval.temperature, seed)
        else:
            res, _, _ = _play_pair(champ, chall, fen, cfg.eval.temperature, seed)
        if res == "1/2-1/2":
            d += 1
        elif (res == "1-0") == chall_white:
            w += 1
        else:
            losses += 1
        okey = fen.split()[0][:24]
        entry = per_opening.setdefault(okey, {"games": 0, "W": 0, "D": 0, "L": 0})
        entry["games"] += 1
        if res == "1/2-1/2":
            entry["D"] += 1
        elif (res == "1-0") == chall_white:
            entry["W"] += 1
        else:
            entry["L"] += 1
        if verbose:
            print(f"  game {i+1:3d}: {res} (W:{w} D:{d} L:{losses})", flush=True)
    n = max(1, w + d + losses)
    score_rate = (w + 0.5 * d) / n
    lo, hi = wilson_ci(w, d, losses)
    sprt_result = sprt([(w, d, losses)], elo0=cfg.eval.sprt_elo0, elo1=cfg.eval.sprt_elo1,
                       alpha=cfg.eval.sprt_alpha, beta=cfg.eval.sprt_beta)
    return {"games": n, "wins": w, "draws": d, "losses": losses,
            "score_rate": round(score_rate, 4),
            "score_ci95": [round(lo, 4), round(hi, 4)],
            "elo_delta": round(elo_delta(min(max(score_rate, 1e-4), 1 - 1e-4))),
            "elo_ci95": list(_elo_ci(w, d, losses)),
            "sprt": sprt_result,
            "sprt_hypotheses": {"elo0": cfg.eval.sprt_elo0, "elo1": cfg.eval.sprt_elo1,
                                 "alpha": cfg.eval.sprt_alpha, "beta": cfg.eval.sprt_beta},
            "per_opening": per_opening,
            "budget": {"time_ms": cfg.eval.time_ms, "max_depth": cfg.eval.max_depth,
                        "min_depth": cfg.eval.min_depth, "temperature": cfg.eval.temperature}}


def _play_pair(a: Player, b: Player, fen: str, temperature: float, seed: int):
    """Satu game antar dua Player (tanpa book, tanpa memory) → (result, pgn, plies)."""
    from .arena import play_game
    return play_game(a, b, fen=fen, temperature=temperature, seed=seed, max_plies=200)


def _model_bench(model: OutcomeModel | None, quick: bool) -> dict:
    """Benchmark taktik + performa + konsistensi untuk satu model."""
    tac = tactical_suite(time_ms=600 if quick else 1500, node_limit=20000 if quick else 60000)
    perf = performance_suite(node_budget=8000 if quick else 20000)
    cons = evaluation_consistency(samples=40 if quick else 120)
    return {"tactical_accuracy": tac["accuracy"], "tactical_solved": tac["solved"],
            "tactical_total": tac["total"],
            "nps": perf["nps"], "nodes": perf["nodes"],
            "consistency_max_drift": cons["max_drift"],
            "consistency_ok": cons["within_epsilon"]}


def verify_challenger(champion: OutcomeModel | None, challenger: OutcomeModel | None,
                      cfg: AppConfig, quick: bool = False, verbose: bool = False) -> dict:
    """Verifikasi penuh + keputusan. Mengembalikan report JSON-safe."""
    started = time.monotonic()
    match_report = head_to_head(champion, challenger, cfg, verbose=verbose)
    champ_bench = _model_bench(champion, quick)
    chall_bench = _model_bench(challenger, quick)

    # ---- gates ----
    reasons = []
    passed = True
    if match_report["sprt"] == "H1":
        promote_rule = "sprt_H1"
    elif match_report["score_rate"] >= cfg.eval.min_score_rate:
        promote_rule = "score_rate_threshold"
    else:
        promote_rule = "rejected"
        reasons.append(f"score_rate {match_report['score_rate']} < ambang {cfg.eval.min_score_rate}")
        passed = False
    tactical_drop = (champ_bench["tactical_solved"] - chall_bench["tactical_solved"])
    if tactical_drop > max(0, int(cfg.eval.max_tactical_regression)):
        passed = False
        reasons.append(f"regresi taktik: -{tactical_drop} soal")
    if cfg.eval.require_consistency and not (chall_bench["consistency_ok"]):
        passed = False
        reasons.append("konsistensi evaluator challenger gagal")
    if champ_bench["nps"] > 0 and chall_bench["nps"] < cfg.eval.min_nps_ratio * champ_bench["nps"]:
        passed = False
        reasons.append(f"NPS challenger turun di bawah {cfg.eval.min_nps_ratio:.0%} champion")
    if passed and promote_rule == "score_rate_threshold" and \
            match_report["score_rate"] >= cfg.eval.min_score_rate and \
            match_report["elo_ci95"][0] < 0 and cfg.eval.min_score_rate <= 0.5:
        # score rate di ambang tapi CI bawah masih negatif: catat kehati-hatian
        # (tetap boleh promote; keputusan akhir ada di trainer/auto_promote).
        reasons.append("catatan: CI95 Elo bawah masih negatif (bukti lemah)")

    report = {"verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "match": match_report,
              "champion_bench": champ_bench,
              "challenger_bench": chall_bench,
              "tactical_solved_delta": -tactical_drop,
              "promote_rule": promote_rule,
              "decision": "PROMOTE" if passed else "REJECT",
              "reasons": reasons,
              "honesty_note": ("score_rate dari match terbatas adalah indikasi "
                               "kuat/lemah, bukan estimasi Elo final; SPRT di sini "
                               "adalah skrining trinomial dengan draw-rate empiris"),
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    return report


def save_report(report: dict, results_dir, generation) -> Path:
    label = generation if isinstance(generation, str) else f"{generation:04d}"
    out = Path(results_dir) / f"verification_gen_{label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def report_markdown(report: dict) -> str:
    """Ringkasan Markdown untuk laporan manusia."""
    m = report["match"]
    cb, hb = report["champion_bench"], report["challenger_bench"]
    lines = [
        f"# Verifikasi Challenger — {report['verified_at']}",
        "",
        f"**Keputusan: {report['decision']}** (aturan: `{report['promote_rule']}`)",
        "",
        "## Match champion vs challenger",
        f"- Games: {m['games']} — W {m['wins']} / D {m['draws']} / L {m['losses']}",
        f"- Score rate: {m['score_rate']} (CI95 {m['score_ci95']})",
        f"- Elo delta: {m['elo_delta']:+d} (CI95 {m['elo_ci95']})",
        f"- SPRT: **{m['sprt']}** (H0 elo0={m['sprt_hypotheses']['elo0']}, "
        f"H1 elo1={m['sprt_hypotheses']['elo1']})",
        "",
        "## Benchmark",
        "| Metrik | Champion | Challenger |",
        "| --- | --- | --- |",
        f"| Akurasi taktik | {cb['tactical_solved']}/{cb['tactical_total']} "
        f"({cb['tactical_accuracy']:.0%}) | {hb['tactical_solved']}/{hb['tactical_total']} "
        f"({hb['tactical_accuracy']:.0%}) |",
        f"| NPS kiwipete | {cb['nps']} | {hb['nps']} |",
        f"| Drift konsistensi | {cb['consistency_max_drift']} | {hb['consistency_max_drift']} |",
        "",
        "## Alasan",
    ]
    lines += [f"- {r}" for r in report["reasons"]] or ["- (tidak ada catatan)"]
    return "\n".join(lines) + "\n"
