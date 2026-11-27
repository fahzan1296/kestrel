"""KESTREL — Tiny Evaluator (1.7: fused single-pass handcrafted term).

E(s) = E_material + E_PST + E_mobility + E_pawn + E_king + E_tempo
     + alpha*H(s) blended dengan N_theta(s) (MLP numpy, 768->128->64->1).

Parameter MLP dihitung & dilaporkan (batas < 1.000.000).

1.6 upgrade: fungsi evaluasi panas (pst, mobility, pawn_structure,
king_safety, piece_bonuses) membaca bitboard python-chess secara langsung
(popcount pada `board.pieces_mask`) alih-alih membangun objek ``SquareSet``
untuk setiap bidak. Semantik dan nilai keluaran identik dengan 1.5
(truncation ``int()`` per term PST dan antisimetri mirror tetap eksak).

1.7 upgrade (performa murni, nilai keluaran identik bit-per-bit):
- ``mobility`` mengakses tabel attack diag/rank/file python-chess langsung
  per jenis bidak (tipe bidak sudah diketahui dari loop) — tanpa dispatch
  ``board.attacks_mask`` per bidak.
- ``material`` + ``pst`` dilebur jadi satu pass per jenis bidak (12 loop
  alih-alih 17), dan ``pawn_structure`` + ``piece_bonuses`` berbagi satu
  perhitungan file-pion (``_pawn_and_bonuses``).
- Blok terminal (mate/stalemate/draw) di ``handcrafted`` menerima hint dari
  caller search (``in_check``/``has_moves``/``terminal``) supaya daun qsearch
  tidak membayar movegen penuh hanya untuk mengonfirmasi "bukan stalemate";
  gate ketat berbasis halfmove_clock/popcount menjaga hasil identik.
"""
from __future__ import annotations
import math
import chess
import numpy as np
from .core import PIECE_VALUES, material, phase

# ---------------- Piece-Square Tables (mg/eg), White perspective, a1=0 --------
# Nilai ringkas klasik (sederhana, tunable per spec).
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

# ---------------- Pra-hitungan mask bitboard (1.6 performa) ------------------
# Mask FILE f: bit f, f+8, f+16, ... (kolom vertikal), bukan mask rank.
_FILES = [0x0101010101010101 << f for f in range(8)]
_ALL = (1 << 64) - 1
# Semua rank STRIKT di atas/bawah rank r (bukan satu rank saja).
_ABOVE = [_ALL & ~((1 << (8 * (r + 1))) - 1) if r < 7 else 0 for r in range(8)]
_BELOW = [(1 << (8 * r)) - 1 for r in range(8)]
# Zona pion-lolos per petak: pion lawan pada file pion atau file tetangga,
# pada rank lebih maju dari pion, menghalangi pion tersebut.
_PASS_ZONE_W = [0] * 64  # untuk pion putih: lawan di rank > r
_PASS_ZONE_B = [0] * 64  # untuk pion hitam: lawan di rank < r
for _sq in range(64):
    _f, _r = _sq & 7, _sq >> 3
    _fmask = 0
    for _df in (-1, 0, 1):
        if 0 <= _f + _df <= 7:
            _fmask |= _FILES[_f + _df]
    _PASS_ZONE_W[_sq] = _fmask & _ABOVE[_r]
    _PASS_ZONE_B[_sq] = _fmask & _BELOW[_r]

_MOB_WEIGHTS = ((chess.PAWN, 2), (chess.KNIGHT, 4), (chess.BISHOP, 4),
                (chess.ROOK, 2), (chess.QUEEN, 1))

# ---------------- Tabel attack python-chess (1.7: akses langsung) ------------
_BB_PAWN_ATT = chess.BB_PAWN_ATTACKS          # [color][sq] -> list bitboard
_BB_KNIGHT_ATT = chess.BB_KNIGHT_ATTACKS      # [sq] -> list bitboard
_BB_DIAG_M = chess.BB_DIAG_MASKS
_BB_DIAG = chess.BB_DIAG_ATTACKS              # [sq][mask-occupied] -> dict
_BB_RANK_M = chess.BB_RANK_MASKS
_BB_RANK = chess.BB_RANK_ATTACKS
_BB_FILE_M = chess.BB_FILE_MASKS
_BB_FILE = chess.BB_FILE_ATTACKS

# Tuple terurut (pt, nilai_material, mg, eg) untuk loop fusi 1.7.
_PST_TUPLES = tuple((pt, PIECE_VALUES.get(pt, 0), PST_MG[pt], PST_EG[pt])
                    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                               chess.ROOK, chess.QUEEN, chess.KING))

# 1.7: tabel PST taper pra-hitung per (pt, fase-integer). ``phase()`` selalu
# mengembalikan min(1, p/24) dengan p integer (hasil popcount), jadi hanya
# ada <= 26 nilai ph berbeda; tabel dihitung dengan rumus yang sama persis
# (int(mg[sq]*ph + eg[sq]*(1-ph))) sehingga nilai identik bit-per-bit.
_PST_TABLE_CACHE: dict = {}


def _tapered_table(pt: int, p: int) -> list:
    """Tabel PST int 64-kotak untuk (jenis bidak, fase integer p)."""
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


def mirror(sq: int) -> int:
    return sq ^ 56


def pst(board: chess.Board, ph: float | None = None) -> int:
    """E_PST(s) tapered; Black pakai mirror vertikal (bitboard-fast)."""
    if ph is None:
        ph = phase(board)
    e = 0
    for pt, table_mg in PST_MG.items():
        table_eg = PST_EG[pt]
        m = board.pieces_mask(pt, chess.WHITE)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            e += int(table_mg[sq] * ph + table_eg[sq] * (1 - ph))
        m = board.pieces_mask(pt, chess.BLACK)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            msq = sq ^ 56
            e -= int(table_mg[msq] * ph + table_eg[msq] * (1 - ph))
    return e


def mobility(board: chess.Board) -> int:
    """E_mobility = sum beta_q sign(c) Mob(q,c,s) (pseudo-legal, hemat CPU).

    King dikecualikan dari iterasi karena bobotnya 0 (hasil identik, lebih
    cepat). Attack dihitung langsung dari tabel python-chess per jenis
    bidak — nilai identik dengan ``board.attacks_mask`` tanpa biaya
    dispatch per bidak.
    """
    occ = board.occupied
    score = 0
    for pt, wgt in _MOB_WEIGHTS:
        w = 0
        b = 0
        if pt == chess.PAWN:
            att_w = _BB_PAWN_ATT[chess.WHITE]
            att_b = _BB_PAWN_ATT[chess.BLACK]
            m = board.pawns & board.occupied_co[chess.WHITE]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                w += wgt * att_w[sq].bit_count()
            m = board.pawns & board.occupied_co[chess.BLACK]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                b += wgt * att_b[sq].bit_count()
        elif pt == chess.KNIGHT:
            m = board.knights & board.occupied_co[chess.WHITE]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                w += wgt * _BB_KNIGHT_ATT[sq].bit_count()
            m = board.knights & board.occupied_co[chess.BLACK]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                b += wgt * _BB_KNIGHT_ATT[sq].bit_count()
        elif pt == chess.BISHOP:
            m = board.bishops & board.occupied_co[chess.WHITE]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                w += wgt * _BB_DIAG[sq][_BB_DIAG_M[sq] & occ].bit_count()
            m = board.bishops & board.occupied_co[chess.BLACK]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                b += wgt * _BB_DIAG[sq][_BB_DIAG_M[sq] & occ].bit_count()
        elif pt == chess.ROOK:
            m = board.rooks & board.occupied_co[chess.WHITE]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                w += wgt * (_BB_RANK[sq][_BB_RANK_M[sq] & occ].bit_count()
                            + _BB_FILE[sq][_BB_FILE_M[sq] & occ].bit_count())
            m = board.rooks & board.occupied_co[chess.BLACK]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                b += wgt * (_BB_RANK[sq][_BB_RANK_M[sq] & occ].bit_count()
                            + _BB_FILE[sq][_BB_FILE_M[sq] & occ].bit_count())
        else:  # queen: diag + rank + file
            m = board.queens & board.occupied_co[chess.WHITE]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                w += wgt * (_BB_DIAG[sq][_BB_DIAG_M[sq] & occ].bit_count()
                            + _BB_RANK[sq][_BB_RANK_M[sq] & occ].bit_count()
                            + _BB_FILE[sq][_BB_FILE_M[sq] & occ].bit_count())
            m = board.queens & board.occupied_co[chess.BLACK]
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                b += wgt * (_BB_DIAG[sq][_BB_DIAG_M[sq] & occ].bit_count()
                            + _BB_RANK[sq][_BB_RANK_M[sq] & occ].bit_count()
                            + _BB_FILE[sq][_BB_FILE_M[sq] & occ].bit_count())
        score += w - b
    return score


_PASSER_BONUS = [int(8 * (p ** 1.3)) for p in range(9)]


def pawn_structure(board: chess.Board) -> int:
    """Doubled, isolated, passed (rank^gamma) per Section 6."""
    lam_d, lam_i = 12, 14
    e = 0
    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        own = white_pawns if color == chess.WHITE else black_pawns
        enemy = black_pawns if color == chess.WHITE else white_pawns
        per_file = [0] * 8
        m = own
        while m:
            per_file[(m & -m).bit_length() - 1 & 7] += 1
            m &= m - 1
        for f in range(8):
            count = per_file[f]
            if count:
                e += -lam_d * (count - 1) * sign
                if (f == 0 or not per_file[f - 1]) and (f == 7 or not per_file[f + 1]):
                    e += -lam_i * count * sign
        zone = _PASS_ZONE_W if color == chess.WHITE else _PASS_ZONE_B
        m = own
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            if not (enemy & zone[sq]):
                prog = (sq >> 3) if color == chess.WHITE else 7 - (sq >> 3)
                e += _PASSER_BONUS[prog] * sign
    return e


def king_safety(board: chess.Board) -> int:
    """E_shield - danger(exposure) sederhana per Section 7."""
    lam_shield, lam_danger = 8, 6
    e = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        ksq = board.king(color)
        if ksq is None:
            continue
        kf = ksq & 7
        own_pawns = board.pawns & board.occupied_co[color]
        shield = 0
        for f in (kf - 1, kf, kf + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, 1 if color == chess.WHITE else 6)
                if (own_pawns >> sq) & 1:
                    shield += 1
        e += lam_shield * shield * sign
        e -= lam_danger * board.attackers_mask(not color, ksq).bit_count() * sign
    return e


def piece_bonuses(board: chess.Board) -> int:
    """Bishop pair and rook placement bonuses (White-relative).

    Both terms are computed per color with opposite signs, so the mirror
    antisymmetry of the handcrafted evaluator stays exact.
    """
    white_pawn_files = 0
    m = board.pawns & board.occupied_co[chess.WHITE]
    while m:
        white_pawn_files |= 1 << ((m & -m).bit_length() - 1 & 7)
        m &= m - 1
    black_pawn_files = 0
    m = board.pawns & board.occupied_co[chess.BLACK]
    while m:
        black_pawn_files |= 1 << ((m & -m).bit_length() - 1 & 7)
        m &= m - 1
    e = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        # Bishop pair: two bishops cover both complex colors.
        if board.pieces_mask(chess.BISHOP, color).bit_count() >= 2:
            e += 30 * sign
        # Rook on open file (no pawns at all) / semi-open file (own pawns only).
        own_files = white_pawn_files if color == chess.WHITE else black_pawn_files
        enemy_files = black_pawn_files if color == chess.WHITE else white_pawn_files
        m = board.pieces_mask(chess.ROOK, color)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            fbit = 1 << (sq & 7)
            if not (own_files & fbit):
                e += (16 if not (enemy_files & fbit) else 8) * sign
    return e


def _pawn_and_bonuses(board: chess.Board) -> int:
    """pawn_structure + piece_bonuses dalam satu pass (nilai identik).

    File-pion per warna dihitung sekali (1.7.1: via popcount mask file, tanpa
    loop per pion) dan dipakai bersama oleh istilah doubled/isolated, passed,
    dan open/semi-open rook.
    """
    lam_d, lam_i = 12, 14
    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    w_per_file = [(white_pawns & _FILES[f]).bit_count() for f in range(8)]
    b_per_file = [(black_pawns & _FILES[f]).bit_count() for f in range(8)]
    white_files = 0
    for f in range(8):
        if w_per_file[f]:
            white_files |= 1 << f
    black_files = 0
    for f in range(8):
        if b_per_file[f]:
            black_files |= 1 << f
    e = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        per_file = w_per_file if color == chess.WHITE else b_per_file
        own_files = white_files if color == chess.WHITE else black_files
        enemy_files = black_files if color == chess.WHITE else white_files
        for f in range(8):
            count = per_file[f]
            if count:
                e += -lam_d * (count - 1) * sign
                if (f == 0 or not per_file[f - 1]) and (f == 7 or not per_file[f + 1]):
                    e += -lam_i * count * sign
        if color == chess.WHITE:
            zone, own, enemy = _PASS_ZONE_W, white_pawns, black_pawns
        else:
            zone, own, enemy = _PASS_ZONE_B, black_pawns, white_pawns
        m = own
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            if not (enemy & zone[sq]):
                prog = (sq >> 3) if color == chess.WHITE else 7 - (sq >> 3)
                e += _PASSER_BONUS[prog] * sign
        if board.pieces_mask(chess.BISHOP, color).bit_count() >= 2:
            e += 30 * sign
        m = board.pieces_mask(chess.ROOK, color)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            fbit = 1 << (sq & 7)
            if not (own_files & fbit):
                e += (16 if not (enemy_files & fbit) else 8) * sign
    return e


def handcrafted(board: chess.Board, *, in_check: bool | None = None,
                has_moves: bool | None = None, terminal: bool = True) -> int:
    """H(s), White-relative (centipawn).

    Hint opsional dari caller yang SUDAH menghitung informasi tersebut
    (jalur search), supaya predikat mahal tidak dibayar dua kali. Tanpa
    hint, perilaku dan nilai identik dengan 1.5/1.6:
    - ``in_check``: hasil ``board.is_check()`` bila sudah diketahui.
    - ``has_moves``: True bila caller sudah memastikan ada langkah legal
      (stalemate/mate mustahil), False bila dipastikan tidak ada, None =
      belum diketahui (pemeriksaan lengkap dijalankan).
    - ``terminal``: False bila caller sudah menangani draw/terminal di
      pintu masuk node (repetisi, 50-langkah, material tak cukup).
    """
    if terminal or has_moves is None:
        if in_check is None:
            in_check = board.is_check()
        if in_check:
            # Predikat mahal (generate langkah) hanya dibayar pada daun skak.
            if board.is_checkmate():
                # White-relative score: the side to move has been checkmated.
                return -100000 if board.turn == chess.WHITE else 100000
        elif has_moves is not True:
            if board.is_stalemate():
                return 0
        # Gate ketat yang menjaga hasil identik: posisi material-tak-cukup
        # python-chess selamanya punya <= 4 bidak total; 75-langkah butuh
        # clock >= 150; lima-kali repetisi butuh >= 8 ply sejak langkah
        # ireversibel terakhir.
        if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
            return 0
        clock = board.halfmove_clock
        if clock >= 150 and board.is_seventyfive_moves():
            return 0
        if clock >= 8 and board.is_fivefold_repetition():
            return 0
    tempo = 10 if board.turn == chess.WHITE else -10
    # Fusi material+PST: satu pass per jenis bidak (1.7), truncation int()
    # per bidak tetap eksak seperti 1.5/1.6. Fase dihitung dari popcount
    # (integer) sehingga tabel taper pra-hitung bisa dipakai (1.7.1).
    p = (board.knights.bit_count() + board.bishops.bit_count()
         + 2 * board.rooks.bit_count() + 4 * board.queens.bit_count())
    if p > 24:
        p = 24
    mat = 0
    pst_e = 0
    for pt, val, _mg, _eg in _PST_TUPLES:
        table = _tapered_table(pt, p)
        m = board.pieces_mask(pt, chess.WHITE)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            if val:
                mat += val
            pst_e += table[sq]
        m = board.pieces_mask(pt, chess.BLACK)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            msq = sq ^ 56
            if val:
                mat -= val
            pst_e -= table[msq]
    return (mat + pst_e + mobility(board) * 2 + _pawn_and_bonuses(board)
            + king_safety(board) + tempo)


# ---------------- Neural evaluator (numpy MLP) -------------------------------
def count_params(aux_dim: int = 8) -> int:
    d = 768 + aux_dim
    return (d * 128 + 128) + (128 * 64 + 64) + (64 * 1 + 1)


class TinyMLP:
    """Input 768+aux -> 128 -> 64 -> 1. ReLU. FP32. CPU-only."""

    def __init__(self, aux_dim: int = 8, seed: int = 42):
        if int(aux_dim) != aux_dim or aux_dim < 0:
            raise ValueError("aux_dim must be a non-negative integer")
        self.aux_dim = int(aux_dim)
        rng = np.random.default_rng(seed)
        d = 768 + self.aux_dim
        self.w1 = (rng.standard_normal((d, 128)) * math.sqrt(2 / d)).astype(np.float32)
        self.b1 = np.zeros(128, np.float32)
        self.w2 = (rng.standard_normal((128, 64)) * math.sqrt(2 / 128)).astype(np.float32)
        self.b2 = np.zeros(64, np.float32)
        # A random output head injects arbitrary, color-biased noise into chess
        # evaluation. Start neutral; a trained/checkpointed head can replace it.
        self.w3 = np.zeros((64, 1), np.float32)
        self.b3 = np.zeros(1, np.float32)

    def params(self) -> int:
        return count_params(self.aux_dim)

    def encode(self, board: chess.Board) -> np.ndarray:
        """12 piece planes x 64 + aux features, clipped ke [-1,1]."""
        x = np.zeros(768 + self.aux_dim, np.float32)
        for pt in range(1, 7):
            for sq in chess.SquareSet(board.pieces(pt, chess.WHITE)):
                x[(pt - 1) * 64 + sq] = 1.0
            for sq in chess.SquareSet(board.pieces(pt, chess.BLACK)):
                x[384 + (pt - 1) * 64 + sq] = 1.0
        aux = np.asarray([
            1.0 if board.turn == chess.WHITE else -1.0,
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            min(board.halfmove_clock, 100) / 100.0,
            phase(board) * 2 - 1,
            float(board.is_check()),
        ], dtype=np.float32)
        n_aux = min(self.aux_dim, len(aux))
        x[768:768 + n_aux] = aux[:n_aux]
        return x

    def forward(self, board: chess.Board) -> float:
        x = self.encode(board)
        h1 = np.maximum(x @ self.w1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0)
        out = float((h2 @ self.w3 + self.b3)[0])
        # Kalibrasi + bound per Section 9
        return float(np.clip(320.0 * math.tanh(out), -1000, 1000))


ALPHA = 0.85  # alpha tinggi utk model belum tervalidasi (Section 9)


def evaluate(board: chess.Board, net: TinyMLP | None = None, learned=None, *,
             in_check: bool | None = None, has_moves: bool | None = None,
             terminal: bool = True) -> int:
    """E(s) blended, White-relative, centipawn.

    Hint ``in_check``/``has_moves``/``terminal`` diteruskan ke handcrafted;
    lihat docstring di sana. Tanpa hint, hasil identik dengan 1.6.
    """
    h = handcrafted(board, in_check=in_check, has_moves=has_moves,
                    terminal=terminal)
    # Dashboard workers run without a TinyMLP; keep the learned residual active
    # in that mode instead of bypassing the model via this early return.
    base = h if net is None else int(ALPHA * h + (1 - ALPHA) * net.forward(board))
    if learned is not None:
        # Learned model is deliberately a small residual until validated by matches.
        learned_score = learned.score(board) if hasattr(learned, "score") else learned.position_score(board)
        return int(base + max(-220, min(220, learned_score * 120.0)))
    return base


def stm_eval(board: chess.Board, net: TinyMLP | None = None, learned=None) -> int:
    """V(s) = sign(p) E(s), perspektif side-to-move."""
    e = evaluate(board, net, learned)
    return e if board.turn == chess.WHITE else -e


def uncertainty(board: chess.Board, net: TinyMLP | None = None) -> float:
    """U(s) = |H - N| dinormalisasi; C(s)=1/(1+U)."""
    if net is None:
        return 0.0
    d = abs(handcrafted(board) - net.forward(board))
    u = min(d / 400.0, 1.0)
    return 1.0 / (1.0 + u)
