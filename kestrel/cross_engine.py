"""Cross-engine training games: Kestrel vs Stockfish.

Unlike ``selfplay.run_selfplay`` both sides are different engines. The PGN is
written to the same corpus consumed by knowledge/learning stages, and a live
board snapshot is emitted after every ply so a CLI run can be watched from the
dashboard through the shared train-status file.

1.7 upgrade — sistem cross dikembangkan:
- **Game paralel**: ``parallel > 1`` menjalankan beberapa game secara
  bersamaan pada proses terpisah (fork), masing-masing dengan Stockfish
  subprocess sendiri; PGN ditulis berurutan per gelombang agar deterministik.
- **Adaptive skill**: Skill Level Stockfish naik/turun otomatis mengikuti
  score_rate Kestrel pada jendela game terakhir (``skill_window``) sehingga
  korpus tetap kompetitif, tidak terlalu mudah atau mustahil.
- **Per-opening breakdown** + telemetry Kestrel (rata-rata NPS, depth,
  panjang game, pemakaian book) untuk pemantauan kualitas korpus.
- **Komentar book move tidak lagi menulis "0/d0 n0"** yang mempolusi
  target REDA/experience (komentar book non-numerik seperti arena.py).
- Game worker yang gagal (mis. Stockfish mati) dilaporkan sebagai game
  gagal, bukan menghentikan seluruh batch.
"""
from __future__ import annotations

import contextlib
import random
import time
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from .arena import OPENINGS, Player
from .nnue import TinyNNUE, load_default
from .stockfish import StockfishOpponent

_WORKER: dict[str, Any] = {}   # state per proses worker (event stop, dsb.)


def _snapshot(game_id: int, board: chess.Board, moves: list[dict[str, str]],
              white: str, black: str, *, result: str | None = None,
              kestrel_white: bool = True, plies: int = 0,
              score_cp: int | None = None) -> dict[str, Any]:
    return {"game_id": game_id, "fen": board.fen(), "moves": list(moves[-40:]),
            "white": white, "black": black, "kestrel_white": kestrel_white,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "plies": plies, "result": result, "score_cp": score_cp,
            "updated_at": time.time()}


def _result_of(board: chess.Board) -> tuple[str, str]:
    """(hasil, alasan terminasi) untuk posisi yang sudah selesai."""
    outcome = board.outcome(claim_draw=True)
    return (outcome.result() if outcome else "1/2-1/2",
            outcome.termination.name if outcome else "Draw")


def _load_models(net_path: str | None, book_path: str | None):
    net: TinyNNUE | None = None
    if net_path and Path(net_path).exists():
        with contextlib.suppress(ValueError, KeyError, TypeError, OSError):
            net = TinyNNUE.load(Path(net_path))
    if net is None:
        net = load_default()
    book = None
    if book_path and Path(book_path).exists():
        with contextlib.suppress(Exception):
            from .knowledge import KnowledgeBase
            book = KnowledgeBase(book_path)
    return net, book


def _worker_init(stop_event) -> None:
    _WORKER["stop"] = stop_event


def _play_cross_game(task: dict[str, Any]) -> dict[str, Any]:
    """Mainkan SATU game cross-engine (dipanggil di proses worker fork).

    Semua state dibuat lokal proses ini; hasil berupa dict JSON-safe berisi
    teks PGN + telemetry, sehingga yang dikembalikan lintas proses hanya
    data murni.
    """
    stop = _WORKER.get("stop")
    index = int(task["index"])
    out: dict[str, Any] = {"index": index, "ok": False}
    try:
        game_id = index + 1
        net, book = _load_models(task.get("net_path"),
                                task.get("book_path"))
        rng = random.Random(task["seed"])
        kestrel_white = bool(task["kestrel_white"])
        skill = int(task["skill"])
        board = chess.Board(task["opening"])
        pgn = chess.pgn.Game()
        if board.fen() != chess.STARTING_FEN:
            pgn.setup(board)
        sf_name = "Stockfish"
        pgn.headers["Event"] = "Kestrel Cross-Engine Training"
        pgn.headers["Site"] = "localhost"
        pgn.headers["Date"] = time.strftime("%Y.%m.%d")
        pgn.headers["Round"] = str(game_id)
        pgn.headers["White"] = "Kestrel" if kestrel_white else sf_name
        pgn.headers["Black"] = sf_name if kestrel_white else "Kestrel"
        node = pgn
        mover_white = kestrel_white
        moves: list[dict[str, str]] = []
        forfeit: str | None = None
        nps_sum = 0
        nps_n = 0
        depth_sum = 0
        book_moves = 0

        kestrel = Player("Kestrel", net=net, book=book,
                         time_ms=task["time_ms"], max_depth=task["max_depth"],
                         min_depth=task["min_depth"], grace_ms=task["grace_ms"],
                         seed=task["seed"])
        with StockfishOpponent(path=task.get("stockfish_path"),
                               skill_level=skill,
                               time_limit=task["stockfish_time"]) as sf:
            sf_name = sf.version or "Stockfish"
            pgn.headers["White"] = "Kestrel" if kestrel_white else sf_name
            pgn.headers["Black"] = sf_name if kestrel_white else "Kestrel"
            while not board.is_game_over(claim_draw=True) \
                    and len(moves) < task["max_plies"]:
                if stop is not None and stop.is_set():
                    forfeit = "*"
                    break
                kestrel_turn = (board.turn == chess.WHITE) == kestrel_white
                if kestrel_turn:
                    # Simpan warna pemain SEBELUM push: komentar PGN dinormalisasi
                    # White-relative (konsisten dengan arena.play_game) agar
                    # distilasi REDA/pembandingan swing punya satu perspektif.
                    mover_white = board.turn == chess.WHITE
                    move, info = kestrel.play(
                        board, 0.25 if len(moves) < 12 else 0.0, rng)
                else:
                    try:
                        move, info = sf.play(board), None
                    except Exception:
                        # Stockfish mati/error di tengah game: anggap forfeit,
                        # jangan bunuh seluruh batch.
                        move, info = None, None
                if move is None:
                    forfeit = "0-1" if board.turn == chess.WHITE else "1-0"
                    break
                san = board.san(move)
                node = node.add_variation(move)
                if info is not None:
                    if info.get("book"):
                        # Komentar book non-numerik: tidak mempolusi target
                        # distilasi REDA/experience (bugfix 1.7).
                        book_moves += 1
                        node.comment = (f"book {info['book_san']} "
                                        f"({info['book_games']}g,"
                                        f"{info['book_score']:.2f})")
                    else:
                        white_score = (info.get("score_cp", 0) if mover_white
                                       else -info.get("score_cp", 0))
                        node.comment = (f"{white_score}"
                                        f"/d{info.get('depth', 0)}"
                                        f" n{info.get('nodes', 0)}")
                        nps_sum += int(info.get("nps", 0) or 0)
                        nps_n += 1
                        depth_sum += int(info.get("depth", 0) or 0)
                board.push(move)
                moves.append({"uci": move.uci(), "san": san})

        if forfeit == "*":
            result = "*"
            pgn.headers["Termination"] = "Stopped by user"
        elif forfeit is not None:
            result = forfeit
            pgn.headers["Termination"] = "Engine forfeit: no move returned"
        elif board.is_game_over(claim_draw=True):
            result, term = _result_of(board)
            pgn.headers["Termination"] = term
        else:
            result = "1/2-1/2"
            pgn.headers["Termination"] = "Max plies reached; adjudicated as draw"
        pgn.headers["Result"] = result
        out.update({"ok": True, "pgn": str(pgn), "result": result,
                    "kestrel_white": kestrel_white, "plies": len(moves),
                    "opening_idx": int(task["opening_idx"]),
                    "skill": skill,
                    "avg_kestrel_nps": (nps_sum / nps_n) if nps_n else 0,
                    "avg_depth": (depth_sum / nps_n) if nps_n else 0,
                    "book_moves": book_moves})
    except Exception as exc:  # game gagal: laporkan, batch tetap jalan
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _adapt_skill(skill: int, results: list[float],
                 cfg: dict[str, Any]) -> tuple[int, bool]:
    """Skill naik/turun 1 langkah bila score_rate jendela menembus ambang."""
    if not cfg.get("adaptive_skill") or not results:
        return skill, False
    window = results[-int(cfg.get("skill_window", 4)):]
    rate = sum(window) / len(window)
    lo, hi = int(cfg.get("skill_min", 0)), int(cfg.get("skill_max", 12))
    up = float(cfg.get("skill_up_score", 0.62))
    down = float(cfg.get("skill_down_score", 0.32))
    if rate >= up and skill < hi:
        return skill + 1, True
    if rate <= down and skill > lo:
        return skill - 1, True
    return skill, False


def run_cross_engine_games(*, games: int, time_ms: int, max_plies: int,
                           pgn_path: str | Path, skill: int = 5,
                           max_depth: int = 64,
                           stockfish_path: str | None = None,
                           net_path: str | Path | None = None,
                           book_path: str | Path | None = None,
                           stockfish_time: float = 0.2,
                           min_depth: int = 0, grace_ms: int = 600,
                           seed: int = 20261126, progress=None,
                           board_callback=None, stop_flag=None,
                           parallel: int = 1,
                           skill_min: int = 0, skill_max: int = 12,
                           adaptive_skill: bool = True,
                           skill_up_score: float = 0.62,
                           skill_down_score: float = 0.32,
                           skill_window: int = 4) -> dict[str, Any]:
    """Play ``games`` Kestrel-vs-Stockfish games and append them to ``pgn_path``.

    ``parallel > 1`` menjalankan game dalam gelombang proses terpisah.
    ``progress`` menerima dict ringkas setelah setiap game selesai.
    ``board_callback`` dipertahankan untuk kompatibilitas pemantauan.
    """
    if int(games) < 1:
        raise ValueError("games must be positive")
    parallel = max(1, int(parallel))
    pgn_path = Path(pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    skill_start = int(skill)
    skill = int(skill)
    wins = draws = losses = 0
    failed = 0
    finished = 0
    started = time.monotonic()
    plies_sum = 0
    nps_samples: list[float] = []
    depth_samples: list[float] = []
    per_opening: dict[str, dict[str, int]] = {}
    skill_history: list[dict[str, Any]] = []
    window: list[float] = []
    adapt_cfg = {"adaptive_skill": adaptive_skill, "skill_window": skill_window,
                 "skill_min": skill_min, "skill_max": skill_max,
                 "skill_up_score": skill_up_score,
                 "skill_down_score": skill_down_score}

    def task_for(index: int) -> dict[str, Any]:
        opening_idx = index % len(OPENINGS)
        return {"index": index, "opening": OPENINGS[opening_idx],
                "opening_idx": opening_idx,
                "kestrel_white": index % 2 == 0, "skill": skill,
                "time_ms": time_ms, "max_depth": max_depth,
                "max_plies": max_plies, "min_depth": min_depth,
                "grace_ms": grace_ms, "seed": seed,
                "stockfish_path": stockfish_path,
                "stockfish_time": stockfish_time,
                "net_path": net_path and str(net_path),
                "book_path": book_path and str(book_path)}

    def tally(out: dict[str, Any]) -> None:
        nonlocal wins, draws, losses, failed, finished, plies_sum
        result = out.get("result")
        kw = bool(out.get("kestrel_white", True))
        if out.get("ok"):
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == kw:
                wins += 1
            else:
                losses += 1
            finished += 1
            plies_sum += int(out.get("plies", 0) or 0)
            if out.get("avg_kestrel_nps"):
                nps_samples.append(float(out["avg_kestrel_nps"]))
            if out.get("avg_depth"):
                depth_samples.append(float(out["avg_depth"]))
            fen = OPENINGS[int(out.get("opening_idx", 0)) % len(OPENINGS)]
            po = per_opening.setdefault(fen, {"games": 0, "W": 0, "D": 0, "L": 0})
            po["games"] += 1
            if result == "1/2-1/2":
                po["D"] += 1
            elif (result == "1-0") == kw:
                po["W"] += 1
            else:
                po["L"] += 1
            window.append(1.0 if result == "1-0" == kw
                          else (0.5 if result == "1/2-1/2" else 0.0))
        else:
            failed += 1

    if parallel == 1:
        # ---------- mode sekuensial ----------
        for index in range(int(games)):
            if stop_flag is not None and stop_flag.is_set():
                break
            out = _play_cross_game(task_for(index))
            if out.get("ok"):
                with pgn_path.open("a", encoding="utf-8") as fh:
                    fh.write(out["pgn"] + "\n\n")
            tally(out)
            if progress is not None:
                progress({"stage": "cross_engine", "game_id": index + 1,
                          "completed": finished, "total": int(games),
                          "failed": failed, "skill": skill,
                          "result": out.get("result"),
                          "plies": out.get("plies", 0)})
            new_skill, changed = _adapt_skill(skill, window, adapt_cfg)
            if changed:
                skill = new_skill
                skill_history.append({"after_game": index + 1, "skill": skill})
    else:
        # ---------- mode paralel: gelombang proses terpisah ----------
        methods = get_all_start_methods()
        start_method = "fork" if "fork" in methods else "spawn"
        ctx = get_context(start_method)
        stop_event = ctx.Event()
        if stop_flag is not None:
            import threading

            def _relay(flag=stop_flag, ev=stop_event):
                while not flag.is_set():
                    if ev.is_set():
                        break
                    time.sleep(0.2)
                ev.set()

            threading.Thread(target=_relay, daemon=True).start()
        tasks = [task_for(i) for i in range(int(games))]
        with ctx.Pool(processes=min(parallel, int(games)),
                      initializer=_worker_init, initargs=(stop_event,)) as pool:
            done = 0
            wave = max(1, parallel)
            while done < len(tasks):
                if stop_event.is_set():
                    break
                batch = tasks[done:done + wave]
                outs = pool.map(_play_cross_game, batch)
                for out in sorted(outs, key=lambda o: o["index"]):
                    if out.get("ok"):
                        with pgn_path.open("a", encoding="utf-8") as fh:
                            fh.write(out["pgn"] + "\n\n")
                    tally(out)
                    done += 1
                    if progress is not None:
                        progress({"stage": "cross_engine",
                                  "game_id": out["index"] + 1,
                                  "completed": finished, "total": int(games),
                                  "failed": failed, "skill": skill,
                                  "result": out.get("result"),
                                  "plies": out.get("plies", 0)})
                if done < len(tasks):
                    new_skill, changed = _adapt_skill(skill, window, adapt_cfg)
                    if changed:
                        skill = new_skill
                        skill_history.append({"after_game": finished,
                                              "skill": skill})
                        # skill baru berlaku untuk gelombang berikutnya
                        for t in tasks[done:]:
                            t["skill"] = skill

    n = max(wins + draws + losses, 1)
    return {"games": finished, "failed": failed, "opponent": "Stockfish",
            "skill_start": skill_start, "skill": skill,
            "skill_history": skill_history,
            "adaptive_skill": bool(adaptive_skill),
            "parallel": parallel,
            "kestrel": {"W": wins, "D": draws, "L": losses},
            "score_rate": (wins + 0.5 * draws) / n,
            "per_opening": per_opening,
            "avg_plies": (plies_sum / finished) if finished else 0,
            "avg_kestrel_nps": (sum(nps_samples) / len(nps_samples))
                               if nps_samples else 0,
            "avg_kestrel_depth": (sum(depth_samples) / len(depth_samples))
                                 if depth_samples else 0,
            "pgn": str(pgn_path), "provisional": finished < 30,
            "elapsed_seconds": round(time.monotonic() - started, 2)}


def run_cross_engine_training(cfg, *, run_id: str, skill: int = 5,
                              stockfish_path: str | None = None,
                              stockfish_time_ms: int | None = None,
                              games: int | None = None,
                              time_ms: int | None = None,
                              max_depth: int | None = None,
                              max_plies: int | None = None,
                              min_depth: int | None = None,
                              grace_ms: int | None = None,
                              generations: int | None = None,
                              parallel: int | None = None,
                              adaptive_skill: bool | None = None,
                              quick: bool = False, verify: bool = True,
                              stop_flag=None, progress=None,
                              board_callback=None) -> dict[str, Any]:
    """Play cross-engine games, train from their PGN, publish shared state.

    ``run_id`` must come from ``train_status.start``. Nilai ``None`` diambil
    dari section ``[cross]`` config.toml (1.7) dengan fallback ke [selfplay].
    """
    from .train_status import finish, update  # pyright: ignore[reportMissingImports]
    from .trainer import QuickOverrides, run_training
    from .versioning import CheckpointStore

    work = QuickOverrides.apply(cfg) if quick else cfg
    xc = getattr(work, "cross", None)
    if games is None:
        games = xc.games_per_batch if xc else work.selfplay.games_per_batch
    if time_ms is None:
        time_ms = xc.time_ms if xc else work.selfplay.time_ms
    if max_depth is None:
        max_depth = xc.max_depth if xc else work.selfplay.max_depth
    if max_plies is None:
        max_plies = xc.max_plies if xc else work.selfplay.max_plies
    if min_depth is None:
        min_depth = xc.min_depth if xc else work.selfplay.min_depth
    if grace_ms is None:
        grace_ms = xc.grace_ms if xc else work.selfplay.grace_ms
    if stockfish_time_ms is None:
        stockfish_time_ms = xc.stockfish_time_ms if xc else 200
    if parallel is None:
        parallel = xc.parallel if xc else 1
    if adaptive_skill is None:
        adaptive_skill = xc.adaptive_skill if xc else True
    assert stockfish_time_ms is not None
    assert parallel is not None
    assert adaptive_skill is not None
    if skill == 5 and xc is not None:
        skill = xc.skill
    skill_min = xc.skill_min if xc else 0
    skill_max = xc.skill_max if xc else 12
    skill_up = xc.skill_up_score if xc else 0.62
    skill_down = xc.skill_down_score if xc else 0.32
    skill_window = xc.skill_window if xc else 4
    use_book = xc.use_book if xc else work.selfplay.use_book

    overrides = {"games_per_batch": games, "time_ms": time_ms,
                 "max_depth": max_depth, "max_plies": max_plies,
                 "min_depth": min_depth, "grace_ms": grace_ms}
    if any(v is not None for v in overrides.values()):
        work = work.with_overrides("selfplay", **overrides)
    if generations is not None:
        work = work.with_overrides("train", generations=int(generations))
    games = work.selfplay.games_per_batch
    store = CheckpointStore(work.results_path)

    def publish(message: str, *, event: str = "train_update", **fields: Any) -> None:
        detail = {k: v for k, v in fields.items() if v is not None}
        live = detail.pop("live_game", None)
        prog = detail.pop("progress", None)
        update(work.results_path, run_id=run_id, message=message, event=event,
               progress=prog, live_game=live, **detail)
        if prog is not None and progress is not None:
            progress({**prog, "message": message})
        if live is not None and board_callback is not None:
            board_callback(live)

    try:
        cross = run_cross_engine_games(
            games=games, time_ms=work.selfplay.time_ms,
            max_depth=work.selfplay.max_depth,
            max_plies=work.selfplay.max_plies, pgn_path=work.pgn_path,
            skill=skill, stockfish_path=stockfish_path,
            net_path=(str(store.champion_model_path)
                          if store.champion_model_path.exists() else None),
            book_path=(str(work.knowledge_db_path) if use_book else None),
            stockfish_time=stockfish_time_ms / 1000.0,
            min_depth=work.selfplay.min_depth, grace_ms=work.selfplay.grace_ms,
            stop_flag=stop_flag, parallel=parallel,
            skill_min=skill_min, skill_max=skill_max,
            adaptive_skill=adaptive_skill,
            skill_up_score=skill_up, skill_down_score=skill_down,
            skill_window=skill_window,
            progress=lambda info: publish("cross-engine", **info),
            board_callback=lambda board: publish(
                "live_game", event="game_board", live_game=board))
        publish("cross-engine games selesai", event="cross_engine_done",
                progress={"stage": "cross_engine", "completed": cross["games"],
                          "total": games, "score_rate": cross["score_rate"]},
                result=cross)
        report = run_training(
            work, quick=quick, verify=verify, stop_flag=stop_flag,
            progress=lambda info: publish(
                str(dict(info).pop("message", "training")), **dict(info)),
            reuse_pgn=True, corpus_games=cross["games"],
            corpus_source="cross_engine")
        result = {"overall": report.get("overall"), "corpus": cross,
                  "promoted_count": report.get("promoted_count"),
                  "final_generation": report.get("final_generation"),
                  "report_path": report.get("report_path"),
                  "settings": {"mode": "cross_engine", "games": games,
                               "time_ms": work.selfplay.time_ms,
                               "skill": skill, "parallel": parallel,
                               "adaptive_skill": adaptive_skill,
                               "stockfish_time_ms": stockfish_time_ms,
                               "generations": work.train.generations,
                               "quick": quick, "verify": verify,
                               "results_path": str(work.results_path)}}
        status = str(report.get("overall", "FAIL")).lower()
        finish(work.results_path, run_id=run_id, status=status, result=result)
        return result
    except Exception as exc:
        finish(work.results_path, run_id=run_id, status="failed",
               error=f"{type(exc).__name__}: {exc}")
        raise
