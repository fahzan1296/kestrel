"""Process-isolated live self-play worker. Events are JSON-safe dicts."""
from __future__ import annotations

import contextlib
import os
import random
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import chess
import chess.pgn

from .arena import OPENINGS, Player
from .learning import OutcomeModel


def _search_info(info: dict[str, Any], emit: Callable[..., None], is_white: bool) -> None:
    """Kirim info pencarian ke dashboard dengan orientasi sisi yang benar."""
    if info.get("book"):
        return
    score = info["score_cp"] if is_white else -info["score_cp"]
    emit("search_info", side="white" if is_white else "black",
         score_white_cp=score, **info)


def run_game_worker(game_id: int, seed: int, time_ms: int, max_depth: int,
                   max_plies: int, event_queue, start_fen: str | None = None,
                   batch_id: int | None = None, learned_path: str | None = None,
                   book_path: str | None = None, min_depth: int = 0,
                   grace_ms: int = 600, temperature: float = 0.25,
                   book_max_plies: int | None = None,
                   book_min_games: int | None = None,
                   book_max_moves: int | None = None,
                   hard_ext_ms: int = 0, hard_ext_max: int = 0):
    def emit(kind, **payload):
        event_queue.put({"type": kind, "game_id": game_id, "batch_id": batch_id,
                         "timestamp": time.time(), "worker_pid": os.getpid(), **payload})
    # Deterministic per-game randomized opening selection avoids repeating the
    # same opening mechanically every len(OPENINGS) games (e.g. games 3 and 8).
    opening_rng = random.Random(seed ^ 0x5EED5EED)
    opening_index = opening_rng.randrange(len(OPENINGS)) if start_fen is None else None
    board = chess.Board(start_fen) if start_fen else chess.Board(OPENINGS[opening_index or 0])
    game = chess.pgn.Game()
    if board.fen() != chess.STARTING_FEN:
        game.setup(board)
    game.headers["Event"] = "Kestrel Live Dashboard Selfplay"
    game.headers["Site"] = "localhost"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_id)
    game.headers["White"] = f"Kestrel W #{game_id}"
    game.headers["Black"] = f"Kestrel B #{game_id}"
    node = game
    rng = random.Random(seed)
    # Do not feed an untrained random MLP into evaluation: it adds arbitrary
    # position/color noise. The handcrafted evaluator is the clean baseline.
    net = None
    learned = OutcomeModel()
    from pathlib import Path as _Path
    model_path = (_Path(learned_path) if learned_path else
                  _Path(__file__).resolve().parent.parent / "results" / "learned_eval.json")
    if model_path.exists():
        with contextlib.suppress(ValueError, KeyError, TypeError):
            learned.load(model_path)
    # 1.6: buku opening statistik opsional (pengetahuan hasil self-play).
    # 1.7: batas book dari config [knowledge] tidak lagi diabaikan
    # (bugfix: worker membangun KnowledgeBase dengan default semua).
    book = None
    if book_path and _Path(book_path).exists():
        try:
            from .knowledge import KnowledgeBase
            kwargs: dict[str, Any] = {}
            if book_max_plies is not None:
                kwargs["book_max_plies"] = int(book_max_plies)
            if book_min_games is not None:
                kwargs["book_min_games"] = int(book_min_games)
            if book_max_moves is not None:
                kwargs["book_max_moves_per_position"] = int(book_max_moves)
            book = KnowledgeBase(book_path, **kwargs)
        except Exception:
            book = None
    worker_kwargs: dict[str, Any] = {
        "time_ms": time_ms, "max_depth": max_depth, "learned": learned,
        "min_depth": min_depth, "grace_ms": grace_ms, "book": book,
        # 1.8: default 0/0 — selfplay korpus tetap deterministik; pemainan
        # nyata mengaktifkan ekstensi waktu lewat config.
        "hard_ext_ms": int(hard_ext_ms), "hard_ext_max": int(hard_ext_max),
    }
    white = Player("Kestrel", net=net, **worker_kwargs)
    black = Player("Kestrel", net=net, **worker_kwargs)
    plies = 0
    game_start_time = time.monotonic()
    forfeit_result = None
    emit("game_started", fen=board.fen(), opening_index=opening_index,
         white=game.headers["White"], black=game.headers["Black"])
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        color = board.turn
        player = white if color == chess.WHITE else black
        callback = partial(_search_info, emit=emit, is_white=color)
        mv, info = player.play(board, temperature if plies < 12 else 0.0, rng,
                               search_callback=callback)
        if mv is None:
            if next(iter(board.legal_moves), None) is not None:
                forfeit_result = "0-1" if color == chess.WHITE else "1-0"
            break
        san = board.san(mv)
        board.push(mv)
        node = node.add_variation(mv)
        if info.get("book"):
            # Komentar book non-numerik (bugfix 1.7): nilai 0/d0 tidak lagi
            # ditulis sehingga target distilasi REDA tidak terpolusi.
            node.comment = (f"book {info['book_san']} "
                            f"({info['book_games']}g,{info['book_score']:.2f})")
        else:
            # Komentar dinormalisasi White-relative (konsisten arena.play_game).
            white_score = (info["score_cp"] if color == chess.WHITE
                           else -info["score_cp"])
            node.comment = f"{white_score}/d{info['depth']} n{info['nodes']}"
        plies += 1
        white_score = (info["score_cp"] if color == chess.WHITE
                       else -info["score_cp"])
        emit("move", ply=plies, uci=mv.uci(), san=san, fen=board.fen(), turn="white" if board.turn else "black",
             score_cp=white_score, score_stm_cp=info["score_cp"], depth=info["depth"], nodes=info["nodes"], nps=info["nps"],
             game_seconds=round(time.monotonic()-game_start_time, 2))
    if forfeit_result is not None:
        result = forfeit_result
        game.headers["Termination"] = "Engine forfeit: no move returned"
    elif board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "1/2-1/2"
        game.headers["Termination"] = outcome.termination.name if outcome else "Draw"
    else:
        result = "1/2-1/2"
        game.headers["Termination"] = "Max plies reached; adjudicated as draw"
    game.headers["Result"] = result
    emit("game_finished", result=result, plies=plies, pgn=str(game), final_fen=board.fen(),
         duration_seconds=round(time.monotonic() - game_start_time, 3), opening_index=opening_index)
    return {"game_id": game_id, "result": result, "plies": plies, "pgn": str(game), "fen": board.fen()}
