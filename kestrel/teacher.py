"""KESTREL 2.0 — Teacher: evaluator klasik (handcrafted) untuk distilasi.

Evaluator klasik 1.x (material + PST tapered + mobility + struktur pion +
bishop pair + rook file + king safety + tempo) TIDAK lagi menjadi jalur
evaluasi runtime — runtime memakai TinyNNUE (kestrel.nnue). Fungsi-fungsi di
modul ini hanya punya dua pemakai:

1. TEACHER distilasi: ``teacher_score()`` / ``TeacherEvaluator`` memberi
   label target saat bootstrap (`kestrel nnue distill`) dan fallback label
   untuk posisi tanpa komentar skor di PGN korpus.
2. BASELINE pembanding: ``TeacherEvaluator`` mengimplementasikan antarmuka
   evaluator yang sama dengan TinyNNUE, sehingga match "klasik vs AI"
   bisa dijalankan secara head-to-head yang jujur.

Semua nilai di sini identik bit-per-bit dengan ``handcrafted()`` 1.8
(fusi satu-pass bitboard, diverifikasi antisimetri mirror).
"""
from __future__ import annotations

import chess

from .core import PIECE_VALUES, phase

# ---------------- Piece-Square Tables (mg/eg), White perspective, a1=0 --------
PST_MG = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0],
    chess.KNIGHT: [
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0],
    chess.QUEEN: [
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
         0,  0,  5,  5,  5,  5,  0, -5,
       -10,  5,  5,  5,  5,  5,  0,-10,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20],
    chess.KING: [
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -20,-30,-30,-40,-40,-30,-30,-20,
       -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20],
}
PST_EG = {k: list(v) for k, v in PST_MG.items()}
PST_EG[chess.KING] = [
       -50,-30,-30,-30,-30,-30,-30,-50,
       -30,-20,-20,-20,-20,-20,-20,-30,
       -30,-20,-10,  0,  0,-10,-20,-30,
       -30,-20,-10,  0,  0,-10,-20,-30,
       -30,-20,-10,  0,  0,-10,-20,-30,
       -30,-20,-10,  0,  0,-10,-20,-30,
       -30,-20,-20,-20,-20,-20,-20,-30,
       -50,-30,-30,-30,-30,-30,-30,-50]

# ---------------- pra-hitungan mask bitboard ----------------
_FILES = [0x0101010101010101 << f for f in range(8)]
_ALL = (1 << 64) - 1
_ABOVE = [_ALL & ~((1 << (8 * (r + 1))) - 1) if r < 7 else 0 for r in range(8)]
_BELOW = [(1 << (8 * r)) - 1 for r in range(8)]
_PASS_ZONE_W = [0] * 64
_PASS_ZONE_B = [0] * 64
for _sq in range(64):
    _f, _r = _sq & 7, _sq >> 3
    _fmask = 0
    for _df in (-1, 0, 1):
        if 0 <= _f + _df <= 7:
            _fmask |= _FILES[_f + _df]
    _PASS_ZONE_W[_sq] = _fmask & _ABOVE[_r]
    _PASS_ZONE_B[_sq] = _fmask & _BELOW[_r]

_BB_KNIGHT_ATT = chess.BB_KNIGHT_ATTACKS
_BB_DIAG_M = chess.BB_DIAG_MASKS
_BB_DIAG = chess.BB_DIAG_ATTACKS
_BB_RANK_M = chess.BB_RANK_MASKS
_BB_RANK = chess.BB_RANK_ATTACKS
_BB_FILE_M = chess.BB_FILE_MASKS
_BB_FILE = chess.BB_FILE_ATTACKS
_BB_PAWN_ATT = chess.BB_PAWN_ATTACKS

_PST_TUPLES = tuple((pt, PIECE_VALUES.get(pt, 0), PST_MG[pt], PST_EG[pt])
                    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                               chess.ROOK, chess.QUEEN, chess.KING))
_PST_TABLE_CACHE: dict = {}


def _tapered_table(pt: int, p: int) -> list:
    key = (pt, p)
    t = _PST_TABLE_CACHE.get(key)
    if t is None:
        mg = PST_MG[pt]
        eg = PST_EG[pt]
        ph = 1.0 if p >= 24 else p / 24
        one_minus = 1.0 - ph
        t = [int(mg[s] * ph + eg[s] * one_minus) for s in range(64)]
        _PST_TABLE_CACHE[key] = t
    return t


_PASSER_BONUS = [int(8 * (p ** 1.3)) for p in range(9)]


def static_eval(board: chess.Board, *, in_check: bool | None = None,
                has_moves: bool | None = None, terminal: bool = True) -> int:
    """H(s) klasik, White-relative (centipawn) — fusi satu pass per (pt, warna).

    Identik dengan ``handcrafted()`` 1.8 (nilai & semantik hint sama).
    """
    if terminal or has_moves is None:
        if in_check is None:
            in_check = board.is_check()
        if in_check:
            if board.is_checkmate():
                return -100000 if board.turn == chess.WHITE else 100000
        elif has_moves is not True:
            if board.is_stalemate():
                return 0
        if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
            return 0
        clock = board.halfmove_clock
        if clock >= 150 and board.is_seventyfive_moves():
            return 0
        if clock >= 8 and board.is_fivefold_repetition():
            return 0
    tempo = 10 if board.turn == chess.WHITE else -10
    p = (board.knights.bit_count() + board.bishops.bit_count()
         + 2 * board.rooks.bit_count() + 4 * board.queens.bit_count())
    if p > 24:
        p = 24
    white_occ = board.occupied_co[chess.WHITE]
    black_occ = board.occupied_co[chess.BLACK]
    white_pawns = board.pawns & white_occ
    black_pawns = board.pawns & black_occ
    w_pf = [(white_pawns & _FILES[f]).bit_count() for f in range(8)]
    b_pf = [(black_pawns & _FILES[f]).bit_count() for f in range(8)]
    w_files = 0
    for f in range(8):
        if w_pf[f]:
            w_files |= 1 << f
    b_files = 0
    for f in range(8):
        if b_pf[f]:
            b_files |= 1 << f
    tables = tuple(_tapered_table(pt, p) for pt, _, _, _ in _PST_TUPLES)
    mat = pst_e = mob_e = pawn_e = bonus_e = king_e = 0
    tw = tables[0]
    att_w = _BB_PAWN_ATT[chess.WHITE]
    att_b = _BB_PAWN_ATT[chess.BLACK]
    m = white_pawns
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat += 100
        pst_e += tw[sq]
        mob_e += 2 * att_w[sq].bit_count()
        if not (black_pawns & _PASS_ZONE_W[sq]):
            pawn_e += _PASSER_BONUS[sq >> 3]
    m = black_pawns
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat -= 100
        pst_e -= tw[sq ^ 56]
        mob_e -= 2 * att_b[sq].bit_count()
        if not (white_pawns & _PASS_ZONE_B[sq]):
            pawn_e -= _PASSER_BONUS[7 - (sq >> 3)]
    for f in range(8):
        c = w_pf[f]
        if c:
            pawn_e -= 12 * (c - 1)
            if (f == 0 or not w_pf[f - 1]) and (f == 7 or not w_pf[f + 1]):
                pawn_e -= 14 * c
        c = b_pf[f]
        if c:
            pawn_e += 12 * (c - 1)
            if (f == 0 or not b_pf[f - 1]) and (f == 7 or not b_pf[f + 1]):
                pawn_e += 14 * c
    tk = tables[1]
    m = board.knights & white_occ
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat += 320
        pst_e += tk[sq]
        mob_e += 4 * _BB_KNIGHT_ATT[sq].bit_count()
    m = board.knights & black_occ
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat -= 320
        pst_e -= tk[sq ^ 56]
        mob_e -= 4 * _BB_KNIGHT_ATT[sq].bit_count()
    tb = tables[2]
    white_bishops = board.bishops & white_occ
    black_bishops = board.bishops & black_occ
    m = white_bishops
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat += 335
        pst_e += tb[sq]
        mob_e += 4 * _BB_DIAG[sq][_BB_DIAG_M[sq] & board.occupied].bit_count()
    m = black_bishops
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat -= 335
        pst_e -= tb[sq ^ 56]
        mob_e -= 4 * _BB_DIAG[sq][_BB_DIAG_M[sq] & board.occupied].bit_count()
    if white_bishops.bit_count() >= 2:
        bonus_e += 30
    if black_bishops.bit_count() >= 2:
        bonus_e -= 30
    tr = tables[3]
    occupied = board.occupied
    m = board.rooks & white_occ
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat += 500
        pst_e += tr[sq]
        mob_e += 2 * (_BB_RANK[sq][_BB_RANK_M[sq] & occupied].bit_count()
                      + _BB_FILE[sq][_BB_FILE_M[sq] & occupied].bit_count())
        fbit = 1 << (sq & 7)
        if not (w_files & fbit):
            bonus_e += 16 if not (b_files & fbit) else 8
    m = board.rooks & black_occ
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat -= 500
        pst_e -= tr[sq ^ 56]
        mob_e -= 2 * (_BB_RANK[sq][_BB_RANK_M[sq] & occupied].bit_count()
                      + _BB_FILE[sq][_BB_FILE_M[sq] & occupied].bit_count())
        fbit = 1 << (sq & 7)
        if not (b_files & fbit):
            bonus_e -= 16 if not (w_files & fbit) else 8
    tq = tables[4]
    m = board.queens & white_occ
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat += 900
        pst_e += tq[sq]
        mob_e += (_BB_DIAG[sq][_BB_DIAG_M[sq] & occupied].bit_count()
                  + _BB_RANK[sq][_BB_RANK_M[sq] & occupied].bit_count()
                  + _BB_FILE[sq][_BB_FILE_M[sq] & occupied].bit_count())
    m = board.queens & black_occ
    while m:
        sq = (m & -m).bit_length() - 1
        m &= m - 1
        mat -= 900
        pst_e -= tq[sq ^ 56]
        mob_e -= (_BB_DIAG[sq][_BB_DIAG_M[sq] & occupied].bit_count()
                  + _BB_RANK[sq][_BB_RANK_M[sq] & occupied].bit_count()
                  + _BB_FILE[sq][_BB_FILE_M[sq] & occupied].bit_count())
    tkg = tables[5]
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        ksq = board.king(color)
        if ksq is None:
            continue
        if color == chess.WHITE:
            pst_e += tkg[ksq]
            own_pawns = white_pawns
        else:
            pst_e -= tkg[ksq ^ 56]
            own_pawns = black_pawns
        kf = ksq & 7
        shield = 0
        for f in (kf - 1, kf, kf + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, 1 if color == chess.WHITE else 6)
                if (own_pawns >> sq) & 1:
                    shield += 1
        king_e += 8 * shield * sign
        king_e -= 6 * board.attackers_mask(not color, ksq).bit_count() * sign
    return mat + pst_e + 2 * mob_e + pawn_e + bonus_e + king_e + tempo


def teacher_qsearch(board: chess.Board, depth: int = 8,
                    alpha: int = -10 ** 9, beta: int = 10 ** 9,
                    node_budget: int = 4000) -> int:
    """Qsearch-lite teacher (captures + promosi saja, stand-pat static_eval).

    Memberi label yang TAKTIS lebih tajam daripada static_eval murni: nilai
    sudah memperhitungkan penyelesaian rantai tukar-menukar. Dipakai sebagai
    target distilasi bootstrap.

    Konvensi tanda: ``static_eval`` WHITE-relative (termasuk cabang mate,
    semantik 1.8) — negamax butuh STM-relative, jadi stand-pat dan daun
    dikalikan tanda giliran.
    """
    sign = 1 if board.turn == chess.WHITE else -1
    if depth <= 0 or node_budget <= 0:
        return static_eval(board) * sign
    stand = static_eval(board) * sign
    if stand >= beta:
        return stand
    alpha = max(alpha, stand)
    best = stand
    captures = [mv for mv in board.generate_legal_captures()]
    # urutkan MVV-LVA kasar
    captures.sort(key=lambda mv: -(
        PIECE_VALUES.get(board.piece_type_at(mv.to_square), 0) * 10
        - PIECE_VALUES.get(board.piece_type_at(mv.from_square), 0)))
    for mv in captures:
        board.push(mv)
        score = -teacher_qsearch(board, depth - 1, -beta, -alpha,
                                 node_budget - 1)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


class TeacherEvaluator:
    """Adapter evaluator klasik dengan antarmuka TinyNNUE (score_white dll).

    Murni untuk distilasi & match pembanding "klasik vs AI" — bukan jalur
    evaluasi runtime engine 2.0. Mengimplementasikan antarmuka inkremental
    yang sama (refresh/delta) supaya bisa dipakai langsung oleh Searcher.
    """

    kind = "teacher"

    def params(self) -> int:
        return 0  # tanpa parameter terlatih (rule-based)

    def refresh(self, board: chess.Board):
        import numpy as np
        z = np.zeros(128, np.float32)
        return z, z.copy()

    def delta(self, removed, added):
        import numpy as np
        z = np.zeros(128, np.float32)
        return z, z.copy()

    def raw(self, board: chess.Board, accs=None) -> float:
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        return float(static_eval(board) * sign)

    def stm_score(self, board: chess.Board, accs=None) -> int:
        sign = 1 if board.turn == chess.WHITE else -1
        return static_eval(board) * sign

    def white_score(self, board: chess.Board, accs=None) -> int:
        # static_eval WHITE-relative utuh (termasuk cabang mate).
        return static_eval(board)
