"""KESTREL 2.0 — TRAINER UNIFIED: satu pipeline, satu perintah.

Menggabungkan self-play, knowledge, dan training TinyNNUE menjadi SATU alur
otomatis per generasi (semua budget mengikuti satu config):

    [generasi N]
    1. SELFPLAY   — batch game proses-paralel dengan champion saat ini
                    (+ langkah book dari knowledge base).
    2. KNOWLEDGE  — perbarui opening book & pengalaman posisi dari korpus.
    3. NNUE       — latih net challenger: distilasi skor search (komentar PGN
                    self-play) + hasil game, warm start dari champion.
    4. CANDIDATE  — simpan checkpoint net gen_N.
    5. VERIFY     — match champion vs challenger (warna seimbang, SPRT, Elo
                    CI, per-opening) + benchmark taktik/NPS/konsistensi.
    6. PROMOTE    — champion baru hanya bila lolos gate; kalau gagal, champion
                    lama tetap (rollback implisit) dan kandidat diarsipkan.

Setiap generasi menghasilkan laporan JSON; seluruh run menghasilkan
``train_report_<ts>.json`` + log per generasi ``training_log.jsonl``. Kegagalan
di satu tahap tidak menghentikan run kecuali selfplay gagal total.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config, validate_config
from .knowledge import KnowledgeBase
from .nnue import TinyNNUE
from .verification import report_markdown, save_report, verify_challenger
from .versioning import CheckpointStore


def _json_safe(value: Any) -> Any:
    """Normalisasi nilai laporan (termasuk skalar NumPy) agar JSON-native."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            normalized = item()
            if normalized is not value:
                return _json_safe(normalized)
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


class QuickOverrides:
    """Budget smoke-test agar seluruh pipeline bisa berjalan dalam ~2 menit."""

    @staticmethod
    def apply(cfg: AppConfig) -> AppConfig:
        cfg = cfg.with_overrides("selfplay", games_per_batch=2, time_ms=100,
                                 max_depth=6, min_depth=3, grace_ms=200,
                                 max_plies=60, concurrency=2, use_book=True)
        cfg = cfg.with_overrides("nnue", epochs=2, batch_size=64,
                                 max_samples_per_game=40)
        cfg = cfg.with_overrides("eval", games=2, time_ms=100, max_depth=6,
                                 min_depth=3, grace_ms=200, min_score_rate=0.40)
        cfg = cfg.with_overrides("knowledge", book_min_games=1)
        return cfg


def _count_pgn_games(path: Path) -> int:
    """Hitung game dalam PGN dengan pemindaian header yang murah.

    dipakai saat ``reuse_pgn`` agar statistik selfplay dilaporkan dari file
    nyata, bukan angka karangan.
    """
    import re
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if not text.strip():
        return 0
    n = len(re.findall(r'^\[Event\s+"', text, flags=re.MULTILINE))
    return max(n, 1 if text.strip() else 0)


def run_training(cfg: AppConfig | None = None, *, generations: int | None = None,
                 games: int | None = None, quick: bool = False,
                 verify: bool = True, stop_flag=None, progress=None,
                 reuse_pgn: bool = False, corpus_games: int | None = None,
                 corpus_source: str = "cross_engine") -> dict:
    """Jalankan loop training unified. Blokir sampai selesai/batal.

    Return: report ringkas seluruh run (dict JSON-safe).
    """
    if cfg is None:
        cfg = load_config()
    if quick:
        cfg = QuickOverrides.apply(cfg)
    if generations is not None:
        cfg = cfg.with_overrides("train", generations=int(generations))
    if games is not None:
        cfg = cfg.with_overrides("selfplay", games_per_batch=int(games))
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Konfigurasi tidak valid: " + "; ".join(errors))

    from .selfplay import run_selfplay

    results = cfg.results_path
    results.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(results, keep_last=cfg.train.keep_last_checkpoints)
    kb = (KnowledgeBase(cfg.knowledge_db_path, cfg.knowledge.book_max_plies,
                        cfg.knowledge.book_min_games,
                        cfg.knowledge.book_max_moves_per_position)
          if cfg.knowledge.enabled else None)
    pgn_path = cfg.pgn_path

    def emit(msg, **info):
        line = f"[trainer] {msg}"
        if info:
            line += " " + " ".join(f"{k}={v}" for k, v in info.items())
        print(line, flush=True)
        if progress:
            progress({"message": str(msg), **info})

    run_report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "quick": quick, "corpus_source": corpus_source if reuse_pgn else "selfplay",
                  "generations_requested": cfg.train.generations,
                  "generations": [], "config": {
                      "selfplay": cfg.selfplay.__dict__,
                      "eval": cfg.eval.__dict__,
                      "nnue": dict(cfg.nnue.__dict__)}}
    ok_all = True
    stopped_early = False

    for gen_index in range(cfg.train.generations):
        if stop_flag is not None and stop_flag.is_set():
            emit("dihentikan oleh pengguna", generation=gen_index)
            # hentian oleh pengguna bukan kegagalan pipeline — generasi yang
            # sudah selesai tetap dilaporkan status STOPPED.
            stopped_early = True
            break
        gen_t0 = time.monotonic()
        meta = store.champion_meta()
        champion_gen = meta.get("generation", 0)
        attempt = gen_index + 1
        # Label kandidat unik: lineage champion + nomor percobaan loop. Saat
        # kandidat DITOLAK champion tidak naik generasi, jadi nomor champion
        # saja tidak cukup dan file akan saling menimpa.
        gen_label = f"{champion_gen:04d}_try{attempt:02d}"
        gen_number = f"{champion_gen + 1} (try {attempt})"
        gen_summary = {"generation": champion_gen + 1, "attempt": attempt,
                       "label": gen_label, "stages": {}}
        champion = store.load_champion() if store.has_trained_champion() else None
        emit(f"=== GENERASI {champion_gen + 1} (percobaan {attempt}) ===",
             champion_gen=champion_gen)

        # ---------- 1. SELFPLAY ----------
        # Spawn worker multiprocessing bisa gagal sesaat (umum di VPS kecil /
        # Termux dengan CPU & RAM terbatas) — coba maksimal 2 kali sebelum
        # menyatakan generasi gagal.
        sp: Any = {}
        if reuse_pgn:
            # statistik korpus dihitung dari file PGN nyata, bukan angka karangan
            counted = _count_pgn_games(pgn_path) if pgn_path.exists() else 0
            sp = {"games_requested": corpus_games or counted
                  or cfg.selfplay.games_per_batch,
                  "games_finished": counted or (corpus_games or 0), "failed": 0,
                  "source": corpus_source}
            emit("korpus cross-engine dipakai", games=sp["games_finished"],
                 source=corpus_source)
        for sp_try in ((1,) if reuse_pgn else (1, 2)):
            try:
                if reuse_pgn:
                    break
                sp = run_selfplay(
                    games=cfg.selfplay.games_per_batch, concurrency=cfg.selfplay.concurrency,
                    time_ms=cfg.selfplay.time_ms, max_depth=cfg.selfplay.max_depth,
                    max_plies=cfg.selfplay.max_plies, temperature=cfg.selfplay.temperature,
                    min_depth=cfg.selfplay.min_depth, grace_ms=cfg.selfplay.grace_ms,
                    net_path=str(store.champion_model_path) if store.has_trained_champion() else None,
                    book_path=str(kb.db_path) if kb and cfg.selfplay.use_book else None,
                    pgn_path=pgn_path, db_path=cfg.db_path,
                    book_max_plies=cfg.knowledge.book_max_plies,
                    book_min_games=cfg.knowledge.book_min_games,
                    book_max_moves=cfg.knowledge.book_max_moves_per_position,
                    seed_base=777000 + champion_gen * 997 + attempt + (sp_try - 1) * 131,
                    progress=lambda info: emit(
                        "selfplay", **{k: info[k] for k in ("completed", "total", "result")
                                       if k in info}))
            except Exception as exc:
                sp = {"games_requested": cfg.selfplay.games_per_batch,
                      "games_finished": 0, "failed": cfg.selfplay.games_per_batch,
                      "error": f"{type(exc).__name__}: {exc}"}
            if int(sp.get("games_finished", 0) or 0) > 0:
                break
            if sp_try == 1:
                emit("selfplay tanpa game selesai (spawn flaky?); mencoba ulang",
                     error=sp.get("error", "no finished games"))
        gen_summary["stages"]["selfplay"] = sp
        gen_summary["selfplay_ok"] = int(sp.get("games_finished", 0) or 0) > 0
        if int(sp.get("games_finished", 0) or 0) == 0:
            ok_all = False
            emit("selfplay gagal total setelah retry; generasi dilewati",
                 generation=gen_number, error=sp.get("error"))
            run_report["generations"].append({**gen_summary, "status": "selfplay_failed"})
            continue

        # ---------- 2. KNOWLEDGE ----------
        if kb is not None:
            kbsum = kb.learn_from_pgn([pgn_path])
            gen_summary["stages"]["knowledge"] = kbsum
            emit("knowledge diperbarui", book_moves=kbsum["book_moves_upserted"],
                 experience=kbsum["experience_positions"])

        # ---------- 3. NNUE (distilasi + outcome) ----------
        from .nnue import load_default
        from .nnue_train import train_from_pgn
        candidate: TinyNNUE | None = champion
        try:
            # Warm start dari champion; tanpa champion, mulai dari baseline
            # bawaan (terlatih) — trainer tidak pernah mulai dari net acak.
            base_net = champion if champion is not None else load_default()
            candidate, nnue_report = train_from_pgn(
                [pgn_path], base=base_net,
                warm_start=cfg.nnue.warm_start,
                epochs=cfg.nnue.epochs, lr=cfg.nnue.lr,
                batch_size=cfg.nnue.batch_size, patience=cfg.nnue.patience,
                validation_fraction=cfg.nnue.validation_fraction,
                seed=cfg.nnue.seed, lr_decay=cfg.nnue.lr_decay,
                distill_weight=cfg.nnue.distill_weight,
                outcome_scale=cfg.nnue.outcome_scale_cp,
                distill_clip=cfg.nnue.distill_clip_cp,
                max_samples_per_game=cfg.nnue.max_samples_per_game,
                teacher_fallback=cfg.nnue.teacher_fallback)
            gen_summary["stages"]["nnue"] = {
                "samples": nnue_report["samples"],
                "labels_from_comments": nnue_report["corpus"]["labels_from_comments"],
                "labels_from_teacher": nnue_report["corpus"]["labels_from_teacher"],
                "best_epoch": nnue_report["best_epoch"],
                "validation_mse_cp2": nnue_report["best_validation_mse_cp2"],
                "validation_mode": nnue_report["validation_mode"],
                "warm_start": nnue_report["warm_start"],
                "params": nnue_report["params"]}
            emit("nnue training selesai", samples=nnue_report["samples"],
                 best_epoch=nnue_report["best_epoch"],
                 val_mse_cp2=nnue_report["best_validation_mse_cp2"])
        except Exception as exc:
            gen_summary["stages"]["nnue"] = {"error": f"{type(exc).__name__}: {exc}"}
            emit("nnue training gagal; kandidat = champion tanpa bobot baru", error=str(exc))

        # ---------- 4. CANDIDATE ----------
        # Jika training NNUE gagal, kandidat tetap champion/base net — bukan None.
        if candidate is None:
            candidate = load_default()
        cand_path = store.save_candidate(
            candidate, gen_label,
            metadata=_json_safe({"nnue": gen_summary["stages"].get("nnue"),
                                 "selfplay": {k: sp.get(k) for k in ("games_finished",
                                                                     "draw_rate", "avg_plies")}}))
        gen_summary["candidate"] = str(cand_path)

        # ---------- 5. VERIFY + 6. PROMOTE ----------
        promoted = False
        if verify and cfg.train.auto_promote:
            ver = verify_challenger(champion, candidate, cfg, quick=quick,
                                    verbose=False)
            ver["generation"] = champion_gen + 1
            ver["attempt"] = attempt
            ver["candidate"] = str(cand_path)
            ver_path = save_report(ver, results, gen_label)
            (results / f"verification_gen_{gen_label}.md").write_text(
                report_markdown(ver), encoding="utf-8")
            gen_summary["stages"]["verification"] = {
                "decision": ver["decision"], "promote_rule": ver["promote_rule"],
                "score_rate": ver["match"]["score_rate"],
                "elo_delta": ver["match"]["elo_delta"], "sprt": ver["match"]["sprt"],
                "reasons": ver["reasons"], "report": str(ver_path)}
            m = ver["match"]
            emit("verifikasi", decision=ver["decision"],
                 score=m["score_rate"], elo=m["elo_delta"], sprt=m["sprt"])
            if ver["decision"] == "PROMOTE":
                store.promote(cand_path, f"gen {champion_gen + 1} try {attempt}: {ver['promote_rule']}", ver)
                promoted = True
                gen_summary["promoted"] = True
                emit("champion baru dipromosikan", generation=champion_gen + 1)
            else:
                gen_summary["promoted"] = False
                emit("kandidat ditolak; champion dipertahankan", reasons="; ".join(ver["reasons"]))
        elif not verify:
            # Mode tanpa verifikasi (CLI --no-verify): promote langsung.
            if cfg.train.auto_promote:
                store.promote(cand_path, f"gen {champion_gen + 1} try {attempt}: no-verify mode", None)
                promoted = True
                gen_summary["promoted"] = True
        else:  # verify aktif tapi auto_promote False (mode manual)
            gen_summary["promoted"] = False

        gen_summary["status"] = "promoted" if promoted else "kept_champion"
        gen_summary["elapsed_seconds"] = round(time.monotonic() - gen_t0, 1)
        run_report["generations"].append(gen_summary)
        _append_log(results / "training_log.jsonl", _json_safe(gen_summary))
        emit(f"generasi {champion_gen + 1} percobaan {attempt} selesai "
             f"dalam {gen_summary['elapsed_seconds']}s",
             status=gen_summary["status"])

    # ---------- final report ----------
    promoted_count = sum(1 for g in run_report["generations"] if g.get("promoted"))
    if stopped_early:
        run_report["overall"] = ("STOPPED" if run_report["generations"]
                                 else "FAIL")
    else:
        run_report["overall"] = ("PASS" if (ok_all and run_report["generations"])
                                 else "FAIL")
    run_report["stopped_early"] = stopped_early
    run_report["promoted_count"] = promoted_count
    run_report["final_generation"] = store.champion_meta().get("generation", 0)
    run_report["knowledge_stats"] = kb.stats() if kb else None
    run_report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = results / f"train_report_{int(time.time())}.json"
    out.write_text(json.dumps(_json_safe(run_report), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    run_report["report_path"] = str(out)
    emit("training selesai", promoted=promoted_count,
         generation=run_report["final_generation"], report=str(out))
    return run_report


def _append_log(path: Path, summary: dict):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rep = run_training()
    print(json.dumps({k: rep[k] for k in ("overall", "promoted_count",
                                          "final_generation", "report_path")},
                     indent=2))
