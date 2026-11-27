"""KESTREL — Adaptive Memory (Section 18, 32-34) + Persistence (SQLite).

Memory BUKAN training neural; menyimpan observasi & koreksi terbatas,
dengan decay eksponensial, confidence, dan eviksi berbasis retention.
"""
from __future__ import annotations
import math
import json
import sqlite3
import time
import chess
from .core import zobrist, material, phase


class AdaptiveMemory:
    def __init__(self, db_path="kestrel.db", lam_age=0.01, lam_c=0.15,
                 a_max=40, capacity=5000):
        self.db = sqlite3.connect(db_path)
        self._migrate()
        self.lam_age = lam_age
        self.lam_c = lam_c
        self.a_max = a_max      # |A(m,s)| <= A_max (bounded influence)
        self.capacity = capacity

    def _migrate(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT, context TEXT, move TEXT,
            outcome REAL, eval_cp INTEGER, confidence REAL,
            sample_count INTEGER, created REAL, active INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            white TEXT, black TEXT, result TEXT, plies INTEGER,
            pgn TEXT, created REAL, seed INTEGER);
        """)
        self.db.commit()

    # ---------------- store / retrieve ----------------
    def remember(self, board: chess.Board, move, outcome, eval_cp):
        ctx = json.dumps({"phase": round(phase(board), 2),
                          "material": material(board)})
        self.db.execute(
            "INSERT INTO memory(signature,context,move,outcome,eval_cp,"
            "confidence,sample_count,created) VALUES(?,?,?,?,?,0.5,1,?)",
            (str(zobrist(board)), ctx, str(move), outcome, eval_cp, time.time()))
        self.db.commit()
        self.evict()

    def retrieve(self, board: chess.Board, top_k: int = 3):
        """Retrieve berbasis relevance R = Sim * C * F (Section 34)."""
        sig = str(zobrist(board))
        rows = self.db.execute(
            "SELECT signature,move,outcome,confidence,created,active "
            "FROM memory WHERE active=1").fetchall()
        scored = []
        now = time.time()
        for r in rows:
            if r[0] == sig:
                sim = 1.0  # exact signature match
            else:
                sim = 0.3  # context-level similarity (diperluas via Ctx vector)
            fresh = math.exp(-self.lam_age * (now - r[4]) / 86400.0)
            conf = 1 - math.exp(-self.lam_c * r[3])
            scored.append((sim * conf * fresh, r[1], r[2]))
        scored.sort(reverse=True)
        return scored[:top_k]

    def influence(self, board: chess.Board):
        """A(m,s) = lambda_memory * U(m), bounded oleh A_max."""
        total = 0.0
        for util, _, _ in self.retrieve(board):
            total += util
        return min(total * 10.0, self.a_max)

    def activate(self, entry_id: int):
        """Aktivasi manual — tidak pernah otomatis (Prinsip keamanan)."""
        self.db.execute("UPDATE memory SET active=1 WHERE id=?", (entry_id,))
        self.db.commit()

    def disable(self, entry_id: int):
        self.db.execute("UPDATE memory SET active=0 WHERE id=?", (entry_id,))
        self.db.commit()

    def evict(self):
        """Evict retention terendah bila kapasitas terlampaui (Section 32)."""
        n = self.db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        if n <= self.capacity:
            return
        rows = self.db.execute(
            "SELECT id, confidence, created FROM memory").fetchall()
        now = time.time()
        rows.sort(key=lambda r: r[1] * math.exp(-self.lam_age * (now - r[2]) / 86400.0))
        for rid, _, _ in rows[:n - self.capacity]:
            self.db.execute("DELETE FROM memory WHERE id=?", (rid,))
        self.db.commit()

    def save_game(self, white, black, result, plies, pgn, seed):
        self.db.execute(
            "INSERT INTO games(white,black,result,plies,pgn,created,seed) "
            "VALUES(?,?,?,?,?,?,?)",
            (white, black, result, plies, pgn, time.time(), seed))
        self.db.commit()

    def stats(self):
        mem = self.db.execute("SELECT COUNT(*), AVG(confidence) FROM memory").fetchone()
        games = self.db.execute(
            "SELECT white, COUNT(*), SUM(CASE WHEN result='1-0' THEN 1 WHEN "
            "result='1/2-1/2' THEN 0.5 ELSE 0 END) FROM games GROUP BY white"
        ).fetchall()
        return {"memory_entries": mem[0], "avg_confidence": mem[1], "games_by_white": games}
