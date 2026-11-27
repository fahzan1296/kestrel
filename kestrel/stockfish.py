"""KESTREL — Stockfish UCI sparring (Section 10 PRD).

Wrapper proses UCI dengan graceful fallback bila binary tidak tersedia.
"""
from __future__ import annotations

import os
import shutil

import chess
import chess.engine


def find_stockfish(path: str | None = None) -> str | None:
    candidates = [path] if path else []
    candidates += [shutil.which("stockfish"), shutil.which("stockfish.exe")]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


class StockfishOpponent:
    """Opponent eksternal terkontrol (threads/hash/skill/limit configurable)."""

    def __init__(self, path=None, skill_level=5, time_limit=0.2, depth=None,
                 threads=1, hash_mb=128):
        self.binary = find_stockfish(path)
        self.skill = skill_level
        self.limit = chess.engine.Limit(time=time_limit, depth=depth)
        self.threads, self.hash_mb = threads, hash_mb
        self.engine = None
        self.version = None

    def __enter__(self):
        if not self.binary:
            raise RuntimeError(
                "Stockfish tidak ditemukan. Install (apt install stockfish) "
                "atau set path eksplisit.")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.binary)
        self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb,
                               "Skill Level": self.skill})
        self.version = str(self.engine.id.get("name", "unknown"))
        return self

    def __exit__(self, *exc):
        if self.engine:
            self.engine.quit()

    def play(self, board: chess.Board):
        if not self.engine:
            raise RuntimeError("Engine belum start (gunakan context manager).")
        return self.engine.play(board, self.limit).move


def sparring_match(kestrel_player, games: int = 4, path=None, skill=5,
                   time_limit=0.2):
    """Mainkan N game Kestrel vs Stockfish, balanced warna. Return dict."""
    wins = draws = losses = 0
    with StockfishOpponent(path=path, skill_level=skill,
                           time_limit=time_limit) as sf:
        for i in range(games):
            board = chess.Board()
            kestrel_is_white = (i % 2 == 0)
            forfeit = None
            while not board.is_game_over(claim_draw=True):
                if board.turn == chess.WHITE:
                    mv = (kestrel_player.play(board)[0] if kestrel_is_white
                          else sf.play(board))
                else:
                    mv = (sf.play(board) if kestrel_is_white
                          else kestrel_player.play(board)[0])
                if mv is None:
                    forfeit = "0-1" if board.turn == chess.WHITE else "1-0"
                    break
                board.push(mv)
            res = forfeit or board.result(claim_draw=True)
            if res == "1/2-1/2":
                draws += 1
            elif (res == "1-0") == kestrel_is_white:
                wins += 1
            else:
                losses += 1
    n = max(wins + draws + losses, 1)
    return {"games": n, "opponent": "Stockfish",
            "kestrel": {"W": wins, "D": draws, "L": losses},
            "score_rate": (wins + 0.5 * draws) / n, "provisional": n < 100}
