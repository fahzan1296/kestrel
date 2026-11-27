"""KESTREL — Chess Core (Python).

Menggunakan python-chess sebagai authority legal state (Prinsip #11:
"Legal chess state is the source of truth"). State s=(B,p,c,e,h,r,z)
direpresentasikan oleh chess.Board; Zobrist hashing via chess.polyglot.
"""
from __future__ import annotations
import chess
import chess.polyglot

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 335,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}
PHASE_WEIGHTS = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
    chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0,
}
PHASE_MAX = 24


def material(board: chess.Board) -> int:
    """M(s) = sum v(q)[n(q,W)-n(q,B)], White-relative (bitboard-fast).

    Jumlah bidak putih = mask total − mask hitam, jadi satu popcount per
    jenis bidak, tanpa membangun objek SquareSet. Nilai identik dengan 1.5.
    """
    black = board.occupied_co[chess.BLACK]
    m = 100 * (board.pawns.bit_count() - 2 * (board.pawns & black).bit_count())
    m += 320 * (board.knights.bit_count() - 2 * (board.knights & black).bit_count())
    m += 335 * (board.bishops.bit_count() - 2 * (board.bishops & black).bit_count())
    m += 500 * (board.rooks.bit_count() - 2 * (board.rooks & black).bit_count())
    m += 900 * (board.queens.bit_count() - 2 * (board.queens & black).bit_count())
    return m


def phase(board: chess.Board) -> float:
    """phi(s) = min(1, Phi/24); 1 = full middlegame (bitboard-fast).

    Jumlah putih + hitam = jumlah total mask, jadi cukup satu popcount.
    """
    p = (1 * board.knights.bit_count()
         + 1 * board.bishops.bit_count()
         + 2 * board.rooks.bit_count()
         + 4 * board.queens.bit_count())
    return min(1.0, p / PHASE_MAX)


def is_terminal(board: chess.Board):
    """Return U(s) in {-1,0,+1} dari perspektif side-to-move, atau None."""
    if board.is_checkmate():
        return -1  # side to move dibuat mati
    if board.is_stalemate() or board.is_insufficient_material() \
       or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0
    return None


def zobrist(board: chess.Board) -> int:
    return chess.polyglot.zobrist_hash(board)


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    n = 0
    for mv in board.legal_moves:
        board.push(mv)
        n += perft(board, depth - 1)
        board.pop()
    return n
