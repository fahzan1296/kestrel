"""KESTREL 1.6 — Knowledge Base: pengetahuan nyata hasil self-play.

Tiga lapisan pengetahuan yang dipersistenkan dan benar-benar dipakai saat
bermain / berlatih:

1. **Opening book statistik** (`book_move`): frekuensi + skor (White-relative)
   setiap langkah pada N ply pertama dari seluruh korpus self-play. Langkah
   dipilih hanya bila dukungannya >= ``book_min_games`` — bukan memory book
   buta, melainkan agregat pengalaman yang bisa diaudit.
2. **Pengalaman posisi** (`experience`): statistik evaluasi rata-rata per
   posisi (kunci Zobrist) — korpus target distilasi untuk REDA di generasi
   berikutnya sehingga verifikasi mendalam tidak selalu diulang dari nol.
3. **Ringkasan pelajaran** (`lessons`): tren per generasi (opening terbaik
   per warna, draw rate, depth rata-rata) untuk laporan bukti.

Semua angka dihitung dari PGN nyata; tidak ada nilai yang dikarang.
"""
from __future__ import annotations
import json
import math
import sqlite3
import time
from pathlib import Path
import chess
import chess.pgn
from .core import zobrist, phase


class KnowledgeBase:
    def __init__(self, db_path, book_max_plies: int = 10, book_min_games: int = 2,
                 book_max_moves: int = 3):
        self.db_path = str(db_path)
        self.book_max_plies = max(1, min(int(book_max_plies), 40))
        self.book_min_games = max(1, int(book_min_games))
        self.book_max_moves = max(1, int(book_max_moves))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS book (
            key TEXT, ply INTEGER, uci TEXT, san TEXT,
            games INTEGER, white_wins INTEGER, draws INTEGER, black_wins INTEGER,
            PRIMARY KEY (key, ply, uci));
        CREATE TABLE IF NOT EXISTS experience (
            key TEXT PRIMARY KEY, count INTEGER, eval_sum REAL,
            phase REAL, updated REAL);
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, kind TEXT,
            payload TEXT);
        """)
        self.db.commit()

    # ---------------- build from PGN ----------------
    def learn_from_pgn(self, pgn_paths, max_plies: int | None = None) -> dict:
        """Perbarui book + experience dari korpus PGN (inkremental, dua pass).

        Pass 1: agregasi langkah pembuka (ply <= max_plies) dengan hasil game.
        Pass 2: statistik evaluasi per posisi dari komentar PGN (telemetry).
        """
        max_plies = self.book_max_plies if max_plies is None else int(max_plies)
        started = time.time()
        game_count = 0
        positions_scanned = 0
        book_rows: dict = {}
        exp_rows: dict = {}
        for path in pgn_paths:
            path = Path(path)
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open(encoding="utf-8", errors="replace") as fh:
                while game := chess.pgn.read_game(fh):
                    result = game.headers.get("Result", "1/2-1/2")
                    white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)
                    game_count += 1
                    board = game.board()
                    ply = 0
                    prev_key = None
                    prev_eval = None
                    for node in game.mainline():
                        move = node.move
                        if move is None or not board.is_valid():
                            break
                        key = str(zobrist(board))
                        # --- pass 1: book (posisi sebelum langkah) ---
                        if ply < max_plies:
                            row = book_rows.setdefault((key, ply, move.uci()),
                                                       [0, 0.0, board.san(move)])
                            row[0] += 1
                            row[1] += white_score
                        # --- pass 2: experience (posisi + komentar eval) ---
                        positions_scanned += 1
                        if prev_key is not None and prev_eval is not None:
                            exp = exp_rows.setdefault(prev_key, [0, 0.0])
                            exp[0] += 1
                            exp[1] += prev_eval
                        comment = (node.comment or "").split()
                        prev_eval = None
                        if comment:
                            try:
                                prev_eval = float(int(comment[0].split("/")[0]))
                            except (ValueError, IndexError):
                                prev_eval = None
                        prev_key = key
                        board.push(move)
                        ply += 1
        # tulis book
        book_new = 0
        for (key, ply, uci), (games, score_sum, san) in book_rows.items():
            # Simpan agregat skor (white-relative) langsung; W/D/L persentil
            # tidak dipisah untuk menghemat ruang — skor cukup untuk pemilihan.
            # 1.7 bugfix: kolom dead "white_wins int(round(...))" dihapus.
            self.db.execute(
                "INSERT INTO book(key,ply,uci,san,games,white_wins,draws,black_wins) "
                "VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(key,ply,uci) DO UPDATE SET games=games+excluded.games, "
                "white_wins=white_wins+excluded.white_wins",
                (key, ply, uci, san, games, score_sum, 0, 0))
            book_new += 1
        exp_new = 0
        for key, (count, eval_sum) in exp_rows.items():
            self.db.execute(
                "INSERT INTO experience(key,count,eval_sum,phase,updated) "
                "VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "count=count+excluded.count, eval_sum=eval_sum+excluded.eval_sum, "
                "updated=excluded.updated",
                (key, count, eval_sum, 0.0, time.time()))
            exp_new += 1
        self.db.commit()
        summary = {"games_processed": game_count,
                   "positions_scanned": positions_scanned,
                   "book_moves_upserted": book_new,
                   "experience_positions": exp_new,
                   "book_max_plies": max_plies,
                   "elapsed_seconds": round(time.time() - started, 2)}
        self.add_lesson("corpus_learned", summary)
        return summary

    # ---------------- book selection ----------------
    def book_move(self, board: chess.Board, rng=None) -> tuple | None:
        """Langkah book terbaik untuk posisi (atau None).

        Pemilihan deterministik bila ``rng`` None; dengan rng, dipilih secara
        proporsional terhadap skor^2 di antara langkah yang memenuhi ambang
        dukungan — memberi keragaman opening tanpa langkah lemah.
        """
        if board.ply() >= self.book_max_plies:
            return None
        key = str(zobrist(board))
        rows = self.db.execute(
            "SELECT uci, san, games, white_wins FROM book "
            "WHERE key=? AND ply=? AND games>=? "
            "ORDER BY (white_wins * 1.0) / games DESC",
            (key, board.ply(), self.book_min_games)).fetchall()
        if not rows:
            return None
        rows = rows[:self.book_max_moves]
        # Skor disimpan White-relative; konversi ke perspektif pemain.
        def my_score(games, white_score):
            score = white_score / games
            return score if board.turn == chess.WHITE else 1.0 - score
        scored = [(uci, san, games, my_score(games, ws)) for uci, san, games, ws in rows]
        legal = {m.uci(): m for m in board.legal_moves}
        scored = [row for row in scored if row[0] in legal]
        if not scored:
            return None
        if rng is None:
            return scored[0]  # (uci, san, games, score) terbaik
        weights = [max(row[3], 0.05) ** 2 for row in scored]
        pick = rng.choices(scored, weights=weights, k=1)[0]
        return pick

    # ---------------- experience stats ----------------
    def experience_lookup(self, board: chess.Board) -> dict | None:
        key = str(zobrist(board))
        row = self.db.execute(
            "SELECT count, eval_sum FROM experience WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        count, eval_sum = row
        return {"count": count, "avg_eval_cp": eval_sum / max(1, count)}

    # ---------------- lessons / reporting ----------------
    def add_lesson(self, kind: str, payload: dict):
        self.db.execute("INSERT INTO lessons(created,kind,payload) VALUES(?,?,?)",
                        (time.time(), kind, json.dumps(payload, ensure_ascii=False)))
        self.db.commit()

    def lessons(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        if kind:
            rows = self.db.execute(
                "SELECT created,kind,payload FROM lessons WHERE kind=? "
                "ORDER BY id DESC LIMIT ?", (kind, limit)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT created,kind,payload FROM lessons ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        out = []
        for created, k, payload in rows:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload
            out.append({"created": created, "kind": k, "payload": data})
        return out

    def top_book_lines(self, limit: int = 10) -> list[dict]:
        """Garasi opening paling sering & paling menang di korpus (laporan)."""
        rows = self.db.execute(
            "SELECT ply, uci, san, games, white_wins FROM book "
            "WHERE games >= ? ORDER BY games DESC, (white_wins * 1.0) / games "
            "DESC LIMIT ?",
            (self.book_min_games, limit)).fetchall()
        return [{"ply": p, "uci": u, "san": s, "games": g,
                 "white_score": round(w / max(1, g), 3)}
                for p, u, s, g, w in rows]

    def stats(self) -> dict:
        book = self.db.execute("SELECT COUNT(*), SUM(games) FROM book").fetchone()
        exp = self.db.execute("SELECT COUNT(*), SUM(count) FROM experience").fetchone()
        les = self.db.execute("SELECT COUNT(*) FROM lessons").fetchone()
        return {"book_moves": book[0] or 0, "book_total_games": book[1] or 0,
                "experience_positions": exp[0] or 0,
                "experience_samples": exp[1] or 0,
                "lessons": les[0] or 0,
                "book_max_plies": self.book_max_plies,
                "book_min_games": self.book_min_games}

    def clear_book(self):
        self.db.execute("DELETE FROM book")
        self.db.commit()
