"""KESTREL 1.6 — Runner self-play paralel (sinkron) untuk trainer & CLI.

Berbeda dari dashboard (async + event live), runner ini dipanggil dari
``kestrel.trainer`` / CLI dan BLOCKING: memainkan N game dalam process pool,
menulis PGN + SQLite, dan mengembalikan statistik batch. Worker memakai
``dashboard_worker.run_game_worker`` (satu implementasi game, dua pemakai).

Argumen ``net_path``/``book_path`` membuat generasi trainer berjalan
dengan model champion + knowledge base saat itu tanpa menyentuh file live.

Protokol sinkronisasi: setiap worker mengirim event ``worker_done`` sebagai
item TERAKHIR di queue. Parent hanya mengakhiri drain setelah semua sentinel
diterima — menjamin feeder thread tiap worker sudah flush queue, mencegah
deadlock multiprocessing.Queue saat worker exit dengan pipe penuh.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue as queue_mod
import sqlite3
import time
from pathlib import Path
from typing import Any

from .dashboard_worker import run_game_worker


def run_selfplay(games: int, concurrency: int, time_ms: int, max_depth: int,
                 max_plies: int = 160, temperature: float = 0.25,
                 min_depth: int = 0, grace_ms: int = 600,
                 net_path: str | None = None, book_path: str | Path | None = None,
                 pgn_path: str | Path | None = None, db_path: str | Path | None = None,
                 seed_base: int = 10000, progress=None,
                 stop_flag=None, book_max_plies: int | None = None,
                 book_min_games: int | None = None,
                 book_max_moves: int | None = None) -> dict:
    """Mainkan ``games`` game (spawn proses), tulis hasil, kembalikan statistik.

    ``progress(info)`` dipanggil dengan dict ringkas per game selesai.
    ``stop_flag`` — objek dengan ``.is_set()`` untuk pembatalan.
    """
    games = max(1, int(games))
    concurrency = max(1, min(int(concurrency), games))
    ctx = mp.get_context("spawn")
    event_q: mp.Queue = ctx.Queue()
    all_gids = [10000 + i for i in range(games)]
    pending = [(gid, seed_base + i) for i, gid in enumerate(all_gids)]
    active: dict[int, Any] = {}
    results: list[dict] = []
    failures: list[dict] = []
    sentinels: set[int] = set()
    started = time.monotonic()

    while pending or active or len(sentinels) < games:
        if stop_flag is not None and stop_flag.is_set():
            for proc in active.values():
                if proc.is_alive():
                    proc.terminate()
            for proc in active.values():
                proc.join(timeout=2)
            active.clear()
            pending.clear()
            break
        # 1.7 bugfix: worker yang mati tanpa sentinel (SIGKILL/OOM) tidak
        # membuat loop menunggu selamanya — tandai hilang dan keluar.
        if not pending and not active and len(sentinels) < games:
            for gid in all_gids:
                if gid not in sentinels:
                    sentinels.add(gid)
            break
        while pending and len(active) < concurrency:
            gid, seed = pending.pop(0)
            proc = ctx.Process(target=_worker_entry,
                               args=(gid, seed, time_ms, max_depth, max_plies,
                                     event_q, net_path, book_path, min_depth,
                                     grace_ms, temperature, book_max_plies,
                                     book_min_games, book_max_moves))
            try:
                proc.start()
            except Exception:
                # 1.7 bugfix: kegagalan start dicatat terpisah, tidak lagi
                # dihitung sebagai game selesai.
                failures.append({"game_id": gid, "result": "*", "plies": 0,
                                 "error": "worker_start_failed"})
                sentinels.add(gid)
                continue
            active[gid] = proc
        # Kumpulkan event (non-blocking drain per siklus).
        try:
            while True:
                ev = event_q.get(timeout=0.05)
                if not isinstance(ev, dict):
                    continue
                typ = ev.get("type")
                if typ == "game_finished":
                    results.append(ev)
                    if progress:
                        progress({"status": "game_finished",
                                  "completed": len(results), "total": games,
                                  "result": ev.get("result"), "plies": ev.get("plies")})
                elif typ == "worker_failed":
                    failures.append(ev)
                elif typ == "worker_done":
                    game_id = ev.get("game_id")
                    if isinstance(game_id, int):
                        sentinels.add(game_id)
        except queue_mod.Empty:
            pass
        for gid, proc in list(active.items()):
            if not proc.is_alive():
                proc.join()
                active.pop(gid)
        time.sleep(0.02)

    # Semua sentinel diterima → feeder tiap worker sudah flush; drain sisa.
    while True:
        try:
            ev = event_q.get(timeout=0.2)
            if isinstance(ev, dict) and ev.get("type") == "game_finished":
                results.append(ev)
        except queue_mod.Empty:
            break

    elapsed = time.monotonic() - started
    wins = sum(1 for r in results if r.get("result") == "1-0")
    losses = sum(1 for r in results if r.get("result") == "0-1")
    draws = sum(1 for r in results if r.get("result") == "1/2-1/2")
    # 1.7 bugfix: failed kini dihitung dari game yang TIDAK menghasilkan hasil
    # nyata (start-gagal / worker mati / dibatalkan), bukan dari stub.
    failed = games - len(results)
    avg_plies = (sum(r.get("plies", 0) for r in results) / max(1, len(results)))
    # persist
    if pgn_path is not None:
        _persist(results, Path(pgn_path), Path(db_path) if db_path else None)
    return {"games_requested": games, "games_finished": len(results),
            "failed": failed, "failure_count": len(failures),
            "wins": wins, "draws": draws, "losses": losses,
            "draw_rate": round(draws / max(1, len(results)), 3),
            "avg_plies": round(avg_plies, 1),
            "elapsed_seconds": round(elapsed, 1),
            "concurrency": concurrency, "time_ms": time_ms,
            "max_depth": max_depth, "min_depth": min_depth}


def _worker_entry(gid, seed, time_ms, max_depth, max_plies, event_q,
                  net_path, book_path, min_depth, grace_ms, temperature,
                  book_max_plies=None, book_min_games=None, book_max_moves=None):
    try:
        run_game_worker(gid, seed, time_ms, max_depth, max_plies, event_q,
                        batch_id=None, net_path=net_path, book_path=book_path,
                        min_depth=min_depth, grace_ms=grace_ms,
                        temperature=temperature, book_max_plies=book_max_plies,
                        book_min_games=book_min_games,
                        book_max_moves=book_max_moves)
    except Exception as exc:
        import traceback
        with contextlib.suppress(Exception):
            event_q.put({"type": "worker_failed", "game_id": gid,
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=6)})
    finally:
        # Sentinel TERAKHIR: parent menunggu ini sebelum berhenti membaca.
        with contextlib.suppress(Exception):
            event_q.put({"type": "worker_done", "game_id": gid})


def _persist(results, pgn_path: Path, db_path: Path | None):
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    with pgn_path.open("a", encoding="utf-8") as f:
        for r in results:
            pgn = r.get("pgn")
            if pgn:
                f.write(pgn + "\n\n")
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(db_path, timeout=30)
        db.execute("CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "white TEXT, black TEXT, result TEXT, plies INTEGER, pgn TEXT, "
                   "created REAL, seed INTEGER)")
        for r in results:
            db.execute("INSERT INTO games(white,black,result,plies,pgn,created,seed) "
                       "VALUES(?,?,?,?,?,?,?)",
                       (f"Kestrel #{r.get('game_id')}", f"Kestrel #{r.get('game_id')}",
                        r.get("result"), r.get("plies", 0), r.get("pgn", ""),
                        time.time(), r.get("game_id", 0)))
        db.commit()
        db.close()
