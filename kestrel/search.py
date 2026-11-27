"""KESTREL — Search Engine.

Negamax + alpha-beta, iterative deepening, transposition table (Zobrist),
quiescence with capture-only generation + delta/SEE pruning, MVV-LVA /
killers / butterfly history / continuation history / capture history
ordering, PVS, adaptive null-move, LMR adaptif (sejarah), late-move pruning,
futility pruning, internal iterative reduction, check extensions (bounded
per path), mate-distance pruning, aspiration windows with progressive
widening, per-node single move generation, bounded white-relative evaluation
cache, dan ekstensi waktu adaptif untuk posisi sulit.

1.8 upgrade (kekuatan + akurasi, tradeoff kecil yang disengaja):
- CONTINUATION HISTORY (1-ply): skor quiet move menggabungkan butterfly
  history dengan riwayat pasangan (langkah sebelumnya, langkah ini).
- CAPTURE HISTORY: reorder antar capture berdasar riwayat cutoff per
  (victim, from, to), bukan hanya MVV-LVA statis.
- Penalti history dua arah: quiet/capture yang dicoba di node yang sama
  tetap gagal meng-cutoff menerima bonus NEGATIF (gravity), bukan hanya
  bonus positif untuk langkah yang cutoff.
- IIR (internal iterative reduction): node dalam tanpa hash move disearch
  satu ply lebih dangkal — memangkas pencarian buta di subtree tanpa info.
- LMR adaptif sejarah: langkah quiet ber-sejarah buruk direduksi +1 ply,
  sejarah sangat baik -1 ply.
- FUTILITY pruning parent (depth <= 2): quiet move dilewati bila evaluasi
  statis + margin tidak mungkin mencapai alpha.
- Null-move R adaptif kedalaman: R=2 (<6), R=3 (6-9), R=4 (>=10).
- Delta pruning qsearch memakai aproksimasi SEE (victim - attacker).
- EKSTENSI WAKTU POSISI SULIT: bila budget waktu habis sebelum iterasi
  selesai pada posisi sulit, ATAU best move root tidak stabil / skor
  berayun besar antar iterasi, budget diperpanjang ``hard_ext_ms``
  (default 5000 ms) maksimal ``hard_ext_max`` kali per langkah — hanya
  untuk budget waktu (budget node tidak pernah dilonggar), dengan
  ambang budget minimum ``hard_ext_min_budget_ms``; alasan selalu
  dilaporkan di info.

1.7 upgrade (performa, logika search tetap):
- Zobrist INKREMENTAL eksak-polyglot, TT flat-array, move ordering
  satu-pass, draw gate ketat, tabel LMR pra-hitung, null-move
  ``_heavy_material`` via popcount bitboard.
"""

from __future__ import annotations

import math
import operator
import time
from typing import NamedTuple

import chess
from chess.polyglot import zobrist_hash

from .core import PIECE_VALUES
from .eval import evaluate
from .nnue import TinyNNUE

INF = 10 ** 9
MATE = 100000

EXACT, LOWER, UPPER = 0, 1, 2
_U64 = (1 << 64) - 1
_TT_CLOCK_MIX = 0x9E3779B97F4A7C15
_INF_EVAL_CACHE_MAX = 1 << 18
_EVAL_CACHE_MAX = _INF_EVAL_CACHE_MAX
# 1.7: cache daftar langkah (legal/capture) per kunci Zobrist — daftar
# langkah murni fungsi posisi (turn/castling/ep termasuk dalam kunci),
# sehingga caching tidak mengubah perilaku, hanya memangkas movegen yang
# berulang pada transposisi.
_MOVES_CACHE_MAX = 1 << 15


class TTEntry(NamedTuple):
    """Entri transposition table (kompat 1.6; entri flat 1.7 adalah tuple)."""
    key: int
    depth: int
    score: int
    bound: int
    best: "chess.Move | None"
    age: int

_ITEM0 = operator.itemgetter(0)


def _tt_key(board: chess.Board) -> int:
    """Polyglot position key salted with the half-move clock.

    The fifty-move counter affects the value of a position but is not part of
    the Polyglot hash. Mixing it in prevents a score searched at clock 0 from
    being reused blindly at clock 99.
    """
    clock = min(max(int(board.halfmove_clock), 0), 150)
    return (zobrist_hash(board) ^ ((clock * _TT_CLOCK_MIX) & _U64)) & _U64


def _score_to_tt(score: int, ply: int) -> int:
    """Normalize mate-distance scores so transpositions at different plies compare correctly."""
    if score >= MATE - 1000:
        return score + ply
    if score <= -MATE + 1000:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score >= MATE - 1000:
        return score - ply
    if score <= -MATE + 1000:
        return score + ply
    return score


# ---------------- Zobrist inkremental (eksak polyglot, 1.7) ------------------
# Tabel acak yang SAMA dengan chess.polyglot.POLYGLOT_RANDOM_ARRAY, dikonversi
# ke list int biasa agar XOR-nya cepat (tanpa numpy scalar di jalur panas).
_PA = [int(v) for v in chess.polyglot.POLYGLOT_RANDOM_ARRAY]
_TURN_W = _PA[780]  # term giliran: XOR saat WHITE to move (semantik polyglot)
_BB_H1 = chess.BB_H1
_BB_A1 = chess.BB_A1
_BB_A8 = chess.BB_A8
_BB_H8 = chess.BB_H8

# Term castling untuk 16 kombinasi hak (bit a1/h1/a8/h8 pada
# ``board.castling_rights`` yang sudah clean).
_CASTLE_TABLE = {}
for _wk in (0, _BB_H1):
    for _wq in (0, _BB_A1):
        for _bk in (0, _BB_H8):
            for _bq in (0, _BB_A8):
                _t = 0
                if _wk:
                    _t ^= _PA[768]
                if _wq:
                    _t ^= _PA[769]
                if _bk:
                    _t ^= _PA[770]
                if _bq:
                    _t ^= _PA[771]
                _CASTLE_TABLE[_wk | _wq | _bk | _bq] = _t


def _castle_term(rights: int) -> int:
    """Term zobrist untuk mask hak castling (fallback utk mask tak dikenal)."""
    t = _CASTLE_TABLE.get(rights)
    if t is None:
        t = 0
        if rights & _BB_H1:
            t ^= _PA[768]
        if rights & _BB_A1:
            t ^= _PA[769]
        if rights & _BB_H8:
            t ^= _PA[770]
        if rights & _BB_A8:
            t ^= _PA[771]
    return t


# Posisi pion penangkap en-passant per kotak ep (semantik hash_ep_square
# polyglot: pion lawan pada file tetangga tepat di bawah/di atas kotak ep).
_EP_PAWN_POS_W = [0] * 64  # pion PUTIH yang bisa menangkap ep di kotak tsb
_EP_PAWN_POS_B = [0] * 64  # pion HITAM yang bisa menangkap ep di kotak tsb
for _sq in range(64):
    _bb = 1 << _sq
    _down = _bb >> 8                      # chess.shift_down
    _up = (_bb << 8) & chess.BB_ALL       # chess.shift_up
    # Semantik python-chess: shift_right = (bb << 1) & ~FILE_A,
    # shift_left = (bb >> 1) & ~FILE_H (mask menahan wrap a<->h).
    _EP_PAWN_POS_W[_sq] = (((_down << 1) & ~chess.BB_FILE_A & chess.BB_ALL)
                           | ((_down >> 1) & ~chess.BB_FILE_H & chess.BB_ALL))
    _EP_PAWN_POS_B[_sq] = (((_up << 1) & ~chess.BB_FILE_A & chess.BB_ALL)
                           | ((_up >> 1) & ~chess.BB_FILE_H & chess.BB_ALL))

# Rok shuffle: kotak tujuan raja -> (rook_asal, rook_tujuan), catur standar.
_ROOK_SHUFFLE = {6: (7, 5), 2: (0, 3), 62: (63, 61), 58: (56, 59)}
# Urutan piece promosi (python-chess tidak mengekspor konstanta ini).
_PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
# Tabel attack python-chess untuk generator capture bitboard (1.7).
_BB_KNIGHT_ATT = chess.BB_KNIGHT_ATTACKS
_BB_DIAG_M = chess.BB_DIAG_MASKS
_BB_DIAG = chess.BB_DIAG_ATTACKS
_BB_RANK_M = chess.BB_RANK_MASKS
_BB_RANK = chess.BB_RANK_ATTACKS
_BB_FILE_M = chess.BB_FILE_MASKS
_BB_FILE = chess.BB_FILE_ATTACKS

# Tabel LMR pra-hitung: _LMR[depth][idx]; rumus identik dengan 1.6.
_LMR = [[0] * 128 for _ in range(128)]
for _d in range(3, 128):
    _max_reduction = min(3, _d - 2)
    for _i in range(3, 128):
        _LMR[_d][_i] = min(max(int(math.log(_d) * math.log(_i) / 2.2), 1),
                           _max_reduction)


def _ep_term(board: chess.Board) -> int:
    """Term zobrist en-passant; replikasi eksak ``hash_ep_square`` polyglot."""
    ep = board.ep_square
    if ep:
        turn = board.turn
        mask = _EP_PAWN_POS_W[ep] if turn == chess.WHITE else _EP_PAWN_POS_B[ep]
        if (board.pawns & board.occupied_co[turn]) & mask:
            return _PA[772 + (ep & 7)]
    return 0


def _heavy_material(board: chess.Board) -> bool:
    """Null-move hanya saat material non-pawn cukup (anti-zugzwang)."""
    pts = (320 * board.knights.bit_count() + 335 * board.bishops.bit_count()
           + 500 * board.rooks.bit_count() + 900 * board.queens.bit_count())
    return pts >= 1300


class Searcher:
    def __init__(self, net: TinyNNUE | None = None, node_limit: int = 0,
                 time_ms: int = 1000, max_depth: int = 64, tt_size: int = 1 << 18,
                 use_null_move: bool = True, use_lmr: bool = True, on_info=None,
                 use_lmp: bool = True, check_extension: bool = True,
                 min_depth: int = 0, grace_ms: int = 1000,
                 hard_ext_ms: int = 5000, hard_ext_max: int = 1,
                 hard_ext_min_budget_ms: int = 500,
                 use_iir: bool = True, use_futility: bool = True,
                 use_cont_history: bool = True):
        if int(node_limit) != node_limit or node_limit < 0:
            raise ValueError("node_limit must be a non-negative integer")
        if int(time_ms) != time_ms or time_ms < 1:
            raise ValueError("time_ms must be a positive integer")
        if int(max_depth) != max_depth or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if int(tt_size) != tt_size or tt_size < 1:
            raise ValueError("tt_size must be a positive integer")
        if int(min_depth) != min_depth or min_depth < 0:
            raise ValueError("min_depth must be a non-negative integer")
        if int(grace_ms) != grace_ms or grace_ms < 0:
            raise ValueError("grace_ms must be a non-negative integer")
        if min_depth > max_depth:
            raise ValueError("min_depth cannot exceed max_depth")
        if int(hard_ext_ms) != hard_ext_ms or hard_ext_ms < 0:
            raise ValueError("hard_ext_ms must be a non-negative integer")
        if int(hard_ext_max) != hard_ext_max or hard_ext_max < 0:
            raise ValueError("hard_ext_max must be a non-negative integer")
        if (int(hard_ext_min_budget_ms) != hard_ext_min_budget_ms
                or hard_ext_min_budget_ms < 0):
            raise ValueError("hard_ext_min_budget_ms must be a non-negative integer")
        self.net = net
        self.node_limit = int(node_limit)
        self.time_ms = int(time_ms)
        self.max_depth = int(max_depth)
        # 1.6: jaminan kedalaman — selama completed depth masih di bawah
        # min_depth, iterasi berikutnya boleh memakai anggaran grace time
        # tambahan (batas atas: time_ms + grace_ms). Batas node tidak
        # dilonggar. min_depth=0 menonaktifkan fitur ini.
        self.min_depth = int(min_depth)
        self.grace_ms = int(grace_ms)
        # 1.8: ekstensi waktu posisi sulit — hanya untuk budget waktu.
        self.hard_ext_ms = int(hard_ext_ms)
        self.hard_ext_max = int(hard_ext_max)
        self.hard_ext_min_budget_ms = int(hard_ext_min_budget_ms)
        # 1.8: saklar fitur kekuatan (semua default ON, kompatibel mundur
        # secara API — perilaku search 1.7 berubah by design).
        self.use_iir = use_iir
        self.use_futility = use_futility
        self.use_cont_history = use_cont_history
        # 1.7: TT flat-array (ukuran dibulatkan ke bawah ke pangkat dua).
        size = max(1024, int(tt_size))
        if size & (size - 1):
            size = 1 << (size.bit_length() - 1)
        self.tt_size = size
        self._tt_mask = size - 1
        self.tt: list = [None] * size  # entri: (key, depth, score, bound, best, age)
        self.tt_hits = self.tt_misses = self.cutoffs = self.qnodes = 0
        self.killers: list = [[None, None] for _ in range(128)]
        # 1.8: kunci history int (turn<<12 | frm<<6 | to) — dict int lebih
        # cepat daripada tuple. Kunci continuation: prev_frm<<18 |
        # prev_to<<12 | frm<<6 | to. Kunci capture: victim<<12 | frm<<6 | to.
        self.history: dict = {}
        self.cont_history: dict = {}
        self.cap_history: dict = {}
        self.nodes = 0
        self.age = 0
        self.use_null_move = use_null_move
        self.use_lmr = use_lmr
        self.use_lmp = use_lmp
        self.check_extension = check_extension
        self.on_info = on_info
        self.start = 0.0
        self.deadline = time.monotonic() + max(0.001, self.time_ms / 1000.0)
        self.stop = False
        self._time_check_nodes = 0
        self._time_expired = False
        self.stop_reason: str | None = None
        self.soft_time_ms = max(1, int(time_ms * 0.72))
        self._hard_deadline = self.deadline
        self._grace_deadline = self.deadline
        # 1.8: state ekstensi waktu (direset tiap best_move).
        self._ext_count = 0
        self._ext_ms_total = 0
        self._ext_reasons: list = []
        # Bounded eval cache (white-relative). Keyed by the incremental
        # Zobrist key; positions with a high half-move clock bypass it
        # because draw-value rules become path/clock dependent there.
        self._eval_cache: dict[int, int] = {}
        # 1.7: cache daftar langkah legal & capture (murni fungsi posisi).
        self._moves_cache: dict[int, list] = {}
        self._caps_cache: dict[int, list] = {}
        self._check_cache: dict[int, bool] = {}
        # 1.7: state Zobrist inkremental.
        self._hb: chess.Board | None = None   # board yang ter-sinkron
        self._zkey = 0                        # hash posisi saat ini (eksak polyglot)
        self._zstack: list[int] = []          # hash sebelum tiap push milik kita
        self._zlen = -1                       # len(board.move_stack) yang ter-sinkron
        self._pre_castle = 0                  # hak castling clean di root
        self._first_push = True
        # 2.0: akumulator TinyNNUE inkremental — dipelihara bersama stack
        # Zobrist (referensi array disimpan di stack, array TIDAK pernah
        # dimutasi in-place sehingga restore referensi selalu aman).
        self._acc_w = None                     # acc perspektif putih (numpy 128)
        self._acc_b = None                     # acc perspektif hitam
        self._acc_stack: list = []

    # ---------------- sinkronisasi & push/pop ber-hash (1.7) ----------------
    def _sync_hash(self, board: chess.Board) -> None:
        """Pastikan ``self._zkey`` == zobrist_hash(board), dikelola inkremental.

        Cek murah (identitas objek + panjang move stack); rebuild penuh hanya
        saat board berbeda atau ada push/pop eksternal (mis. pemanggilan
        langsung search/qsearch dari test).
        """
        if self._hb is not board or len(board.move_stack) != self._zlen:
            self._hb = board
            self._zkey = zobrist_hash(board)
            self._zstack = []
            self._pre_castle = board.clean_castling_rights()
            self._first_push = True
            self._zlen = len(board.move_stack)
            # 2.0: akumulator ikut di-rebuild saat board diganti/diubah
            # eksternal (root baru, push/pop manual dari test).
            if self.net is not None:
                self._acc_w, self._acc_b = self.net.refresh(board)
            else:
                self._acc_w = self._acc_b = None
            self._acc_stack = []

    def _push(self, board: chess.Board, mv: chess.Move) -> None:
        """board.push(mv) + update Zobrist inkremental (eksak polyglot)."""
        self._zstack.append(self._zkey)
        key = self._zkey
        turn = board.turn
        # Buang term turn/castling/ep milik posisi SEBELUM push.
        if turn == chess.WHITE:
            key ^= _TURN_W
        if self._first_push:
            # Root board bisa punya attr castling_rights yang belum clean;
            # kunci awal dibangun dari versi clean (sama dengan zobrist_hash).
            self._first_push = False
            rights = self._pre_castle
        else:
            rights = board.castling_rights
        key ^= _castle_term(rights)
        key ^= _ep_term(board)
        to_sq = mv.to_square
        from_sq = mv.from_square
        promo = mv.promotion
        # PENTING: pivot di hash_board polyglot mengikuti urutan
        # ``enumerate(board.occupied_co)`` = [BLACK, WHITE] (chess.WHITE ==
        # True == 1), jadi base putih = 128*(pt-1) + 64 dan base hitam =
        # 128*(pt-1). Terverifikasi via brute-force indeks (193 utk Nb1).
        if turn == chess.WHITE:
            us_base, them_base = 64, 0
        else:
            us_base, them_base = 0, 64
        pt = board.piece_type_at(from_sq)
        base = 128 * (pt - 1)
        key ^= _PA[base + us_base + from_sq]              # pindahkan bidak keluar
        # 2.0: kumpulkan perubahan bidak untuk delta akumulator TinyNNUE.
        removed: list = [(pt, turn == chess.WHITE, from_sq)]
        added: list = []
        if board.occupied_co[not turn] >> to_sq & 1:      # capture biasa
            vt = board.piece_type_at(to_sq)
            key ^= _PA[128 * (vt - 1) + them_base + to_sq]
            removed.append((vt, turn != chess.WHITE, to_sq))
        elif (pt == chess.PAWN and to_sq == board.ep_square
              and (to_sq - from_sq == 7 or to_sq - from_sq == 9
                   or from_sq - to_sq == 7 or from_sq - to_sq == 9)):
            # en passant: pion lawan yang ditangkap ada di belakang kotak ep
            cap_sq = to_sq - 8 if turn == chess.WHITE else to_sq + 8
            key ^= _PA[base + them_base + cap_sq]
            removed.append((chess.PAWN, turn != chess.WHITE, cap_sq))
        if promo:
            key ^= _PA[128 * (promo - 1) + us_base + to_sq]
            added.append((promo, turn == chess.WHITE, to_sq))
        else:
            key ^= _PA[base + us_base + to_sq]
            added.append((pt, turn == chess.WHITE, to_sq))
        if pt == chess.KING:
            rs = _ROOK_SHUFFLE.get(to_sq)
            if rs is not None and (to_sq - from_sq == 2 or from_sq - to_sq == 2):
                rbase = 128 * (chess.ROOK - 1) + us_base
                key ^= _PA[rbase + rs[0]] ^ _PA[rbase + rs[1]]
                rook_color = turn == chess.WHITE
                removed.append((chess.ROOK, rook_color, rs[0]))
                added.append((chess.ROOK, rook_color, rs[1]))
        board.push(mv)
        # Term baru: giliran (flip), castling pasca-push (selalu clean), ep.
        if board.turn == chess.WHITE:
            key ^= _TURN_W
        key ^= _castle_term(board.castling_rights)
        key ^= _ep_term(board)
        self._zkey = key & _U64
        self._zlen += 1
        # 2.0: delta akumulator TinyNNUE (setelah push; array baru, tanpa
        # mutasi in-place agar referensi di stack tetap valid).
        if self.net is not None:
            dw, db = self.net.delta(removed, added)
            self._acc_stack.append((self._acc_w, self._acc_b))
            self._acc_w = self._acc_w + dw
            self._acc_b = self._acc_b + db

    def _push_null(self, board: chess.Board) -> None:
        """Null move: hanya flip giliran + clear ep (castling tak berubah)."""
        self._zstack.append(self._zkey)
        key = self._zkey
        if board.turn == chess.WHITE:
            key ^= _TURN_W
        key ^= _ep_term(board)
        board.push(chess.Move.null())
        if board.turn == chess.WHITE:
            key ^= _TURN_W
        self._zkey = key & _U64
        self._zlen += 1
        if self.net is not None:
            # Tidak ada perubahan bidak, tapi stack harus selaras dgn _pop.
            self._acc_stack.append((self._acc_w, self._acc_b))

    def _pop(self, board: chess.Board) -> None:
        """board.pop() + pulihkan Zobrist & akumulator dari stack (selalu tepat)."""
        board.pop()
        self._zkey = self._zstack.pop()
        self._zlen -= 1
        if self.net is not None:
            self._acc_w, self._acc_b = self._acc_stack.pop()

    def _is_check(self, board: chess.Board) -> bool:
        """board.is_check() dengan cache per kunci Zobrist (1.7)."""
        key = self._zkey
        hit = self._check_cache.get(key)
        if hit is None:
            hit = board.is_check()
            if len(self._check_cache) < _MOVES_CACHE_MAX:
                self._check_cache[key] = hit
        return hit

    # ---------------- budget control ----------------
    def _budget_exceeded(self, force_clock: bool = False) -> bool:
        """Cheap node-budget check with amortized monotonic-clock polling."""
        if self.stop or self._time_expired:
            return True
        if self.node_limit and self.nodes >= self.node_limit:
            self.stop = True
            self.stop_reason = "node_limit"
            return True
        # Poll the clock every 64 nodes to reduce timer overhead in Python hot paths.
        if force_clock or (self.nodes - self._time_check_nodes) >= 64:
            self._time_check_nodes = self.nodes
            if time.monotonic() >= self.deadline:
                self._time_expired = True
                self.stop = True
                self.stop_reason = "time_limit"
                return True
        return False

    # ---------------- transposition table (flat, 1.7) ----------------
    def _probe(self, key: int):
        e = self.tt[key & self._tt_mask]
        if e is not None and e[0] == key:
            return e
        return None

    def _store_tt(self, key: int, depth: int, score: int, bound: int,
                  best_move: chess.Move | None, ply: int) -> None:
        """Depth/age-aware replacement dalam slot langsung (O(1), tanpa scan)."""
        idx = key & self._tt_mask
        old = self.tt[idx]
        if old is None or old[5] != self.age:
            self.tt[idx] = (key, depth, _score_to_tt(score, ply), bound,
                            best_move, self.age)
        elif depth >= old[1] or (bound == EXACT and old[3] != EXACT):
            # An exact score only displaces another exact entry when it is at
            # least as deep; a deeper bound entry must not be downgraded.
            self.tt[idx] = (key, depth, _score_to_tt(score, ply), bound,
                            best_move, self.age)

    # ---------------- evaluation (cached) ----------------
    def _eval_white(self, board: chess.Board, *, in_check=None,
                    has_moves=None, terminal=True) -> int:
        """White-relative evaluation with a bounded transposition-aware cache."""
        if board.halfmove_clock < 90:
            key = self._zkey
            hit = self._eval_cache.get(key)
            if hit is not None:
                return hit
        else:
            key = None
        value = int(evaluate(board, self.net,
                             accs=(self._acc_w, self._acc_b)
                             if self.net is not None else None,
                             in_check=in_check, has_moves=has_moves,
                             terminal=terminal))
        if key is not None and len(self._eval_cache) < _EVAL_CACHE_MAX:
            self._eval_cache[key] = value
        return value

    def _stm(self, board: chess.Board, *, in_check=None,
             has_moves=None, terminal=True) -> int:
        """Side-to-move-relative cached evaluation."""
        value = self._eval_white(board, in_check=in_check, has_moves=has_moves,
                                 terminal=terminal)
        return value if board.turn == chess.WHITE else -value

    def _extract_pv(self, board: chess.Board, first_move: chess.Move | None = None,
                    max_len=24):
        """Build a legal principal variation from TT; always restore board state."""
        pv = []
        seen = set()
        b = board.copy(stack=False)
        if first_move is not None and first_move in b.legal_moves:
            pv.append(first_move.uci())
            b.push(first_move)
        for _ in range(max_len - len(pv)):
            key = _tt_key(b)
            if key in seen:
                break
            seen.add(key)
            e = self._probe(key)
            move = e[4] if e is not None else None
            if move is None or move not in b.legal_moves:
                break
            pv.append(move.uci())
            b.push(move)
        return pv

    # ---------------- move ordering ----------------
    def _score_moves(self, board: chess.Board, ply: int, tt_move, moves,
                     prev_key: int = 0):
        """Skor MVV-LVA / TT / killer / history + flag capture/promo sekali jalan.

        1.8: kunci history INT (turn<<12|frm<<6|to) dan dua sumber skor baru:
        - continuation history 1-ply (pasangan langkah sebelumnya→ini),
          hanya untuk quiet move dan hanya bila ``prev_key`` > 0;
        - capture history per (victim, frm, to) untuk reorder antar capture.
        Skor quiet di-clamp di bawah ambang MVV-LVA terendah agar invariant
        "capture selalu di atas quiet" tetap terjaga.

        Mengembalikan list ``(score, move, capture, promotion, victim)``
        terurut menurun; sort stabel. ``gives_check`` sengaja TIDAK
        dikonsultasikan di sini (predikat python-chess termahal).
        """
        killers = self.killers[ply & 127]
        history = self.history
        cap_history = self.cap_history
        cont = self.cont_history if (self.use_cont_history and prev_key) else None
        turn = board.turn
        hbase = 4096 if turn else 0  # turn<<12 (WHITE=1, BLACK=0)
        ep = board.ep_square
        them_co = board.occupied_co[not turn]
        pawns = board.pawns
        out = []
        append = out.append
        for mv in moves:
            to = mv.to_square
            frm = mv.from_square
            promo = mv.promotion
            cap = bool(them_co >> to & 1)
            if not cap and ep is not None and to == ep and pawns >> frm & 1 \
                    and (to - frm == 7 or to - frm == 9
                         or frm - to == 7 or frm - to == 9):
                cap = True  # en passant
            if tt_move is not None and mv == tt_move:
                s = 10_000_000
                victim = 0
            elif cap:
                victim = board.piece_type_at(to) or chess.PAWN
                attacker = board.piece_type_at(frm) or chess.PAWN
                s = (1_000_000 + PIECE_VALUES[victim] * 10
                     - PIECE_VALUES[attacker]
                     + min(cap_history.get((victim << 12) | (frm << 6) | to, 0),
                           500_000))
            elif promo:
                s = 900_000
                victim = 0
            elif mv in killers:
                s = 700_000
                victim = 0
            else:
                hk = hbase | (frm << 6) | to
                s = history.get(hk, 0)
                if cont is not None:
                    s += 2 * cont.get(prev_key | (frm << 6) | to, 0)
                if s > 990_000:
                    s = 990_000  # invariant: quiet < capture MVV-LVA terendah
                victim = 0
            append((s, mv, cap, promo, victim))
        out.sort(key=_ITEM0, reverse=True)
        return out

    def _order(self, board: chess.Board, ply: int, tt_move, moves):
        """MVV-LVA / TT / killer / history ordering (kompat 1.6; dipakai root)."""
        return [t[1] for t in self._score_moves(board, ply, tt_move, moves)]

    def _capture_moves(self, board: chess.Board) -> list:
        """Legal captures plus non-capture promotions, without a full movegen.

        1.7: hasil di-cache per kunci Zobrist (daftar murni fungsi posisi).
        Pseudo-captures dibangun langsung dari bitboard (tabel attack sama
        dengan python-chess), lalu difilter legalitas — tanpa biaya frame
        generator python-chess. Promosi non-capture dibangun langsung dari
        pion di rank terakhir. Caller tidak boleh memutasi list hasil.
        """
        key = self._zkey
        cached = self._caps_cache.get(key)
        if cached is not None:
            return cached
        turn = board.turn
        us_co = board.occupied_co[turn]
        them_co = board.occupied_co[not turn]
        them = them_co
        occupied = board.occupied
        is_legal = board.is_into_check  # pseudo-legal dijamin konstruksi
        Move = chess.Move
        moves = []
        append = moves.append
        # Pion: capture diagonal (+ promosi-capture), en passant.
        pawns = board.pawns & us_co
        if turn == chess.WHITE:
            caps_w = chess.BB_PAWN_ATTACKS[chess.WHITE]
            while pawns:
                sq = (pawns & -pawns).bit_length() - 1
                pawns &= pawns - 1
                att = caps_w[sq] & them
                while att:
                    to = (att & -att).bit_length() - 1
                    att &= att - 1
                    if to >> 3 == 7:
                        for promo in _PROMO_PIECES:
                            mv = Move(sq, to, promo)
                            if not is_legal(mv):
                                append(mv)
                    else:
                        mv = Move(sq, to)
                        if not is_legal(mv):
                            append(mv)
        else:
            caps_b = chess.BB_PAWN_ATTACKS[chess.BLACK]
            while pawns:
                sq = (pawns & -pawns).bit_length() - 1
                pawns &= pawns - 1
                att = caps_b[sq] & them
                while att:
                    to = (att & -att).bit_length() - 1
                    att &= att - 1
                    if to >> 3 == 0:
                        for promo in _PROMO_PIECES:
                            mv = Move(sq, to, promo)
                            if not is_legal(mv):
                                append(mv)
                    else:
                        mv = Move(sq, to)
                        if not is_legal(mv):
                            append(mv)
        ep = board.ep_square
        if ep is not None:
            pawns = board.pawns & us_co & (
                _EP_PAWN_POS_W[ep] if turn == chess.WHITE else _EP_PAWN_POS_B[ep])
            while pawns:
                sq = (pawns & -pawns).bit_length() - 1
                pawns &= pawns - 1
                mv = Move(sq, ep)
                if not is_legal(mv):
                    append(mv)
        # Bidak lain: knight/bishop/rook/queen/king — attack & bidak lawan.
        m = board.knights & us_co
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            att = _BB_KNIGHT_ATT[sq] & them
            while att:
                to = (att & -att).bit_length() - 1
                att &= att - 1
                mv = Move(sq, to)
                if not is_legal(mv):
                    append(mv)
        m = board.bishops & us_co
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            att = _BB_DIAG[sq][_BB_DIAG_M[sq] & occupied] & them
            while att:
                to = (att & -att).bit_length() - 1
                att &= att - 1
                mv = Move(sq, to)
                if not is_legal(mv):
                    append(mv)
        m = board.rooks & us_co
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            att = ((_BB_RANK[sq][_BB_RANK_M[sq] & occupied]
                    | _BB_FILE[sq][_BB_FILE_M[sq] & occupied]) & them)
            while att:
                to = (att & -att).bit_length() - 1
                att &= att - 1
                mv = Move(sq, to)
                if not is_legal(mv):
                    append(mv)
        m = board.queens & us_co
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            att = ((_BB_DIAG[sq][_BB_DIAG_M[sq] & occupied]
                    | _BB_RANK[sq][_BB_RANK_M[sq] & occupied]
                    | _BB_FILE[sq][_BB_FILE_M[sq] & occupied]) & them)
            while att:
                to = (att & -att).bit_length() - 1
                att &= att - 1
                mv = Move(sq, to)
                if not is_legal(mv):
                    append(mv)
        m = board.kings & us_co
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            att = chess.BB_KING_ATTACKS[sq] & them
            while att:
                to = (att & -att).bit_length() - 1
                att &= att - 1
                mv = Move(sq, to)
                if not is_legal(mv):
                    append(mv)
        # Promosi non-capture: pion di rank terakhir yang dorongannya bebas.
        pawn_rank = chess.BB_RANK_7 if turn == chess.WHITE else chess.BB_RANK_2
        pawns = board.pawns & us_co & pawn_rank
        if pawns:
            while pawns:
                sq = (pawns & -pawns).bit_length() - 1
                pawns &= pawns - 1
                to = sq + 8 if turn == chess.WHITE else sq - 8
                if occupied >> to & 1:
                    continue  # kotak tujuan terisi: hanya promosi-capture
                for promo in _PROMO_PIECES:
                    mv = Move(sq, to, promo)
                    if not is_legal(mv):
                        append(mv)
        if len(self._caps_cache) < _MOVES_CACHE_MAX:
            self._caps_cache[key] = moves
        return moves

    # ---------------- quiescence ----------------
    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int = 0) -> int:
        self.nodes += 1
        self.qnodes += 1
        self._sync_hash(board)
        if self._budget_exceeded():
            return self._stm(board)
        # Cheap draw detection, ber-gate ketat (1.7): repetition(3) butuh
        # >= 4 ply reversible, klaim 50-langkah aktif sejak clock >= 99,
        # material tak cukup pasti <= 4 bidak total.
        clock = board.halfmove_clock
        if ((clock >= 4 and board.is_repetition(3))
                or (clock >= 99 and board.can_claim_fifty_moves())
                or (board.occupied.bit_count() <= 4
                    and board.is_insufficient_material())):
            return 0
        in_check = self._is_check(board)
        if in_check:
            # Stand-pat is illegal in check: search every evasion.
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE - ply)
            best = -INF
            scored = self._score_moves(board, ply, None, moves)
        else:
            # Urutan 1.6 dipertahankan (stand-pat cutoff dulu, capture-gen
            # belakangan) — cutoff stand-pat adalah kasus dominan qsearch;
            # in_check diketahui False sehingga handcrafted tak mengulanginya.
            stand = self._stm(board, in_check=False)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            moves = self._capture_moves(board)
            if not moves:
                if next(iter(board.legal_moves), None) is None:
                    return 0  # stalemate
                return alpha
            best = stand
            scored = self._score_moves(board, ply, None, moves)
        for s0, mv, cap, promo, victim in scored:
            if self.stop:
                return self._stm(board)
            # Delta pruning dua-tahap (1.8): tahap murah dulu (1.7), lalu
            # aproksimasi SEE — biaya attacker (piece_type_at) hanya dibayar
            # untuk capture yang sudah melewati tahap murah tapi tetap
            # berpotensi di-prune. Margin 200 menutup taktik termurah.
            if not in_check and cap and not promo:
                pv_val = PIECE_VALUES[victim]
                if stand + pv_val + 200 < alpha:
                    att_val = PIECE_VALUES[board.piece_type_at(mv.from_square)
                                           or chess.PAWN]
                    if stand + pv_val - att_val + 200 < alpha:
                        continue
            self._push(board, mv)
            sc = -self.qsearch(board, -beta, -alpha, ply + 1)
            self._pop(board)
            if self.stop:
                return self._stm(board)
            if sc > best:
                best = sc
            if sc >= beta:
                return sc
            if sc > alpha:
                alpha = sc
        return best

    # ---------------- main negamax ----------------
    def search(self, board: chess.Board, depth: int, alpha: int, beta: int,
               ply: int = 0, can_null: bool = True, ext_left: int = 24,
               prev_key: int = 0) -> int:
        # Depth-zero nodes are counted by qsearch, not twice (once here and
        # again in qsearch). This keeps node budgets, NPS and qnode_ratio honest.
        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply)
        self.nodes += 1
        self._sync_hash(board)
        if ply > 0 and self._budget_exceeded():
            return self._stm(board)
        # Mate-distance pruning: no line can beat a mate already found faster.
        lo = -(MATE - ply)
        hi = MATE - ply - 1
        if alpha < lo:
            alpha = lo
        if beta > hi:
            beta = hi
        if alpha >= beta:
            return alpha
        # Cheap draw detection first (no move generation), ber-gate ketat (1.7).
        clock = board.halfmove_clock
        rep2 = bool(clock >= 2 and board.is_repetition(2))
        if ((rep2 and clock >= 4 and board.is_repetition(3))
                or (clock >= 99 and board.can_claim_fifty_moves())
                or (board.occupied.bit_count() <= 4
                    and board.is_insufficient_material())):
            return 0

        key = self._zkey ^ (((clock if clock < 150 else 150) * _TT_CLOCK_MIX) & _U64)
        tt_move = None
        e = self._probe(key)
        if e is not None:
            self.tt_hits += 1
            # A repeated node has path-dependent draw value. Keep the hash move
            # for ordering, but do not consume its stored score/bounds.
            if e[1] >= depth and not rep2:
                tt_score = _score_from_tt(e[2], ply)
                eb = e[3]
                if eb == EXACT:
                    return tt_score
                if eb == LOWER and tt_score >= beta:
                    return tt_score
                if eb == UPPER and tt_score <= alpha:
                    return tt_score
            tt_move = e[4]
        else:
            self.tt_misses += 1

        in_check = self._is_check(board)
        # IIR (1.8): node tanpa hash move pada depth sedang disearch satu ply
        # lebih dangkal — informasi TT kosong menandakan subtree yang tidak
        # pernah dijelajahi serupa, jadi pencarian penuh di sini boros.
        if (self.use_iir and tt_move is None and not in_check and depth >= 4
                and ply > 0):
            depth -= 1
        # Null-move pruning, guarded utk zugzwang/endgame. Reduction adaptif
        # kedalaman (1.8): R=2 <6, R=3 6-9, R=4 >=10.
        if (self.use_null_move and can_null and not in_check and ply > 0
                and depth >= 3 and _heavy_material(board)):
            r = 2 if depth < 6 else (3 if depth < 10 else 4)
            self._push_null(board)
            sc = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False,
                              ext_left, prev_key)
            self._pop(board)
            if self.stop:
                return self._stm(board)
            if sc >= beta:
                return beta

        # Single move generation per node: terminal detection and ordering both
        # consume this list (the old flow generated legal moves twice).
        # 1.7: daftar legal di-cache per kunci Zobrist (murni fungsi posisi).
        moves = self._moves_cache.get(self._zkey)
        if moves is None:
            moves = list(board.legal_moves)
            if len(self._moves_cache) < _MOVES_CACHE_MAX:
                self._moves_cache[self._zkey] = moves
        if not moves:
            return -(MATE - ply) if in_check else 0
        scored = self._score_moves(board, ply, tt_move, moves, prev_key)

        # Futility pruning setup (1.8): pada depth <= 2, quiet move yang
        # bahkan dengan margin tidak bisa mencapai alpha dilewati. Posisi
        # node ini sudah lolos gate draw di pintu masuk dan punya langkah
        # legal, sehingga evaluasi terminal=False bernilai sama dengan 1.7.
        fut_base = None
        fut_margin = 0
        if (self.use_futility and depth <= 2 and not in_check
                and -(MATE - 1000) < alpha < MATE - 1000):
            fut_base = self._stm(board, terminal=False)
            fut_margin = 60 + 110 * depth

        # Check extension: search one ply deeper while in check, bounded by the
        # per-path extension budget so perpetual-check lines must terminate.
        ext = 1 if (self.check_extension and in_check and ext_left > 0) else 0

        best, best_mv, bound = -INF, None, UPPER
        first = True
        idx = 0
        quiet_count = 0
        lmp_limit = 4 + 3 * depth
        push = self._push
        pop = self._pop
        lmr_row = _LMR[depth] if depth < 128 else _LMR[127]
        hbase = 4096 if board.turn else 0
        history = self.history
        cont_history = self.cont_history
        cap_history = self.cap_history
        # tried_quiets: kunci butterfly & continuation tiap quiet yang sudah
        # dicoba; tried_caps: kunci capture history. Dipakai untuk penalti
        # dua arah saat cutoff (gravity history 1.8).
        tried_quiets: list = []
        tried_caps: list = []
        for s0, mv, cap, promo, victim in scored:
            if self.stop:
                return self._stm(board)
            frm = mv.from_square
            to = mv.to_square
            is_quiet = not cap and not promo
            # Late-move pruning: at shallow depth skip excess quiet moves once a
            # real best move exists. Checks and captures are never pruned.
            if (self.use_lmp and is_quiet and not first and quiet_count >= lmp_limit
                    and depth <= 3 and not in_check and alpha > -(MATE - 1000)):
                continue
            # Futility pruning (1.8): quiet non-pertama yang statis + margin
            # tetap di bawah alpha dilewati (tidak pernah utk move pertama).
            if (fut_base is not None and not first and is_quiet
                    and fut_base + fut_margin <= alpha):
                continue
            # History tried-lists: hanya langkah yang benar-benar disearch
            # (skip LMP/futility tidak boleh kena penalti cutoff). Kunci
            # butterfly int hanya relevan utk quiet (capture pakai kunci lain).
            hk = 0
            if is_quiet:
                hk = hbase | (frm << 6) | to
                tried_quiets.append(hk)
            elif not promo:
                tried_caps.append((victim << 12) | (frm << 6) | to)
            do_reduce = False
            lmr_adjust = 0
            # gives_check is computed lazily, only when reduction is already
            # justified by every other criterion.
            if (self.use_lmr and not first and idx >= 3 and depth >= 3
                    and not in_check and is_quiet and not board.gives_check(mv)):
                do_reduce = True
                # LMR adaptif sejarah (1.8): sejarah buruk → reduksi +1,
                # sejarah sangat baik → reduksi -1.
                hval = history.get(hk, 0)
                if hval <= -40_000:
                    lmr_adjust = 1
                elif hval >= 400_000 and lmr_row[idx if idx < 128 else 127] > 1:
                    lmr_adjust = -1
            push(board, mv)
            child_key = (frm << 18) | (to << 12)
            if do_reduce:
                r = lmr_row[idx if idx < 128 else 127] + lmr_adjust
                if r < 1:
                    r = 1
                reduce_sc = -self.search(board, depth - 1 - r, -alpha - 1, -alpha,
                                         ply + 1, True, ext_left, child_key)
                if reduce_sc > alpha:
                    # PVS re-search penuh (do_reduce hanya utk langkah non-pertama)
                    sc = -self.search(board, depth - 1 + ext, -alpha - 1, -alpha,
                                      ply + 1, True, ext_left - ext, child_key)
                    if alpha < sc < beta:
                        sc = -self.search(board, depth - 1 + ext, -beta, -alpha,
                                          ply + 1, True, ext_left - ext, child_key)
                else:
                    sc = reduce_sc
            elif first:
                # PVS: full window utk move pertama
                sc = -self.search(board, depth - 1 + ext, -beta, -alpha, ply + 1,
                                  True, ext_left - ext, child_key)
            else:
                # PVS: null window utk sisanya, re-search saat fail high
                sc = -self.search(board, depth - 1 + ext, -alpha - 1, -alpha,
                                  ply + 1, True, ext_left - ext, child_key)
                if alpha < sc < beta:
                    sc = -self.search(board, depth - 1 + ext, -beta, -alpha,
                                      ply + 1, True, ext_left - ext, child_key)
            pop(board)
            if self.stop:
                return self._stm(board)
            idx += 1
            if sc > best:
                best, best_mv = sc, mv
            if sc > alpha:
                alpha = sc
                bound = EXACT
            first = False
            if is_quiet:
                quiet_count += 1
            if alpha >= beta:
                bonus = depth * depth
                if is_quiet:
                    kl = self.killers[ply & 127]
                    kl[1] = kl[0]
                    kl[0] = mv
                    # Butterfly + continuation history, bounded (self-play
                    # panjang), dengan penalti utk quiet yang gagal cutoff.
                    v = history.get(hk, 0) + bonus
                    history[hk] = 1_000_000 if v > 1_000_000 else v
                    if prev_key:
                        ck = prev_key | (frm << 6) | to
                        v = cont_history.get(ck, 0) + bonus
                        cont_history[ck] = 600_000 if v > 600_000 else v
                    for bk in tried_quiets:
                        if bk == hk:
                            continue
                        v = history.get(bk, 0) - bonus
                        history[bk] = v if v > -1_000_000 else -1_000_000
                        if prev_key:
                            # Rekonstruksi kunci continuation dari kunci
                            # butterfly: frm = (bk>>6)&63, to = bk&63.
                            ck2 = prev_key | (((bk >> 6) & 63) << 6) | (bk & 63)
                            v = cont_history.get(ck2, 0) - bonus
                            cont_history[ck2] = v if v > -600_000 else -600_000
                else:
                    ckk = (victim << 12) | (frm << 6) | to
                    v = cap_history.get(ckk, 0) + bonus
                    cap_history[ckk] = 1_000_000 if v > 1_000_000 else v
                    for tc in tried_caps:
                        if tc == ckk:
                            continue
                        v = cap_history.get(tc, 0) - bonus
                        cap_history[tc] = v if v > -1_000_000 else -1_000_000
                bound = LOWER
                self.cutoffs += 1
                break
        # Do not persist a score whose value depends on an earlier occurrence
        # of this exact position in the current path.
        if best_mv is not None and not self._budget_exceeded() and not rep2:
            self._store_tt(key, depth, best, bound, best_mv, ply)
        return best

    def _root_iteration(self, board: chess.Board, depth: int, alpha: int, beta: int,
                        previous_best=None, prev_scores=None):
        """PVS root pass; returns (best move, best score, completed, per-move scores).

        Per-move scores feed the next iteration's ordering, keeping the previous
        best moves near the front of the list.
        """
        moves = self._moves_cache.get(self._zkey)
        if moves is None:
            moves = list(board.legal_moves)
            if len(self._moves_cache) < _MOVES_CACHE_MAX:
                self._moves_cache[self._zkey] = moves
        moves = self._order(board, 0, previous_best, moves)
        if prev_scores:
            # Stable sort: equal scores keep the TT/MVV-LVA ordering above.
            moves.sort(key=lambda m: prev_scores.get(m.uci(), 0), reverse=True)
        best_move, best_score = None, -INF
        move_scores: dict = {}
        completed = True
        first = True
        for mv in moves:
            if self.stop or self._budget_exceeded(force_clock=True):
                completed = False
                break
            self._push(board, mv)
            child_key = (mv.from_square << 18) | (mv.to_square << 12)
            if first:
                score = -self.search(board, depth - 1, -beta, -alpha, 1,
                                     True, 24, child_key)
            else:
                score = -self.search(board, depth - 1, -alpha - 1, -alpha, 1,
                                     True, 24, child_key)
                if alpha < score < beta and not self._budget_exceeded():
                    score = -self.search(board, depth - 1, -beta, -alpha, 1,
                                         True, 24, child_key)
            self._pop(board)
            if self.stop or self._budget_exceeded(force_clock=True):
                completed = False
                break
            move_scores[mv.uci()] = score
            if score > best_score:
                best_score, best_move = score, mv
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
            first = False
        return best_move, best_score, completed, move_scores

    # ---------------- ekstensi waktu posisi sulit (1.8) ----------------
    def _can_extend(self) -> bool:
        """Ekstensi hanya untuk budget waktu (budget node tak pernah dilonggar)."""
        return (self.node_limit == 0
                and self._ext_count < self.hard_ext_max
                and self.hard_ext_ms > 0
                and self.time_ms >= self.hard_ext_min_budget_ms)

    def _extend_budget(self, reason: str) -> None:
        """Perpanjang deadline sekali sesuai ``hard_ext_ms`` dan reset stop."""
        self._ext_count += 1
        self._ext_reasons.append(reason)
        self._ext_ms_total += self.hard_ext_ms
        ext_s = self.hard_ext_ms / 1000.0
        self.deadline += ext_s
        self._hard_deadline += ext_s
        if self._time_expired and time.monotonic() < self.deadline:
            self._time_expired = False
            self.stop = False
            self.stop_reason = None

    # ---------------- iterative deepening ----------------
    def best_move(self, board: chess.Board):
        """Return (move, info). Hanya iterasi lengkap yang dipublikasikan."""
        self.start = time.monotonic()
        self.deadline = self.start + max(0.001, self.time_ms / 1000.0)
        self._hard_deadline = self.deadline
        self._grace_deadline = self.start + (self.time_ms + self.grace_ms) / 1000.0
        self._time_check_nodes = -64
        self._time_expired = False
        self.stop_reason = None
        self.nodes = 0
        self.tt_hits = self.tt_misses = self.cutoffs = self.qnodes = 0
        self.stop = False
        self.age += 1
        # 1.8: reset state ekstensi waktu posisi sulit.
        self._ext_count = 0
        self._ext_ms_total = 0
        self._ext_reasons = []
        # Fresh, bounded evaluation cache per search: within one search,
        # transpositions and qsearch leaves repeat massively.
        self._eval_cache = {}
        # 1.7: refresh juga cache daftar langkah (posisi baru bisa memakai
        # entri lama dengan aman, tapi di-refresh agar memori terkendali).
        self._moves_cache = {}
        self._caps_cache = {}
        self._check_cache = {}
        # 1.7: sinkronkan Zobrist inkremental dengan root board.
        self._sync_hash(board)
        # Keep a legal emergency move even if the budget expires before depth 1
        # completes. Never publish a partially searched iteration as completed.
        best = next(iter(board.legal_moves), None)
        best_score = self._stm(board)
        depth_reached = 0
        prev_scores: dict = {}
        # 1.8: pelacak stabilitas root antar iterasi (trigger ekstensi).
        last_best_uci = None
        last_score = None
        for d in range(1, self.max_depth + 1):
            # 1.6 min-depth guarantee: selama depth yang sudah selesai masih di
            # bawah min_depth, iterasi berikutnya memakai anggaran grace. Begitu
            # min_depth tercapai, deadline kembali ke batas keras. Flag stop
            # yang berasal dari kedaluwarsa waktu di-reset saat deadline
            # diperpanjang; batas node tidak pernah dilonggar.
            if depth_reached < self.min_depth:
                if self._grace_deadline > self.deadline:
                    self.deadline = self._grace_deadline
                if self._time_expired and time.monotonic() < self.deadline:
                    self._time_expired = False
                    self.stop = False
                    self.stop_reason = None
            else:
                self.deadline = self._hard_deadline
            # Aspiration windows from the previous completed iteration reduce work
            # when scores are stable. A fail-low/high widens progressively before
            # falling back to a full-window pass.
            widths = (48, 192, INF) if d >= 3 and depth_reached else (INF,)
            iter_best = None
            iter_score = best_score
            move_scores: dict = {}
            completed = False
            for width in widths:
                a = max(-INF, best_score - width) if width < INF else -INF
                b = min(INF, best_score + width) if width < INF else INF
                iter_best, iter_score, completed, move_scores = self._root_iteration(
                    board, d, a, b, best, prev_scores)
                if not completed:
                    break
                if width == INF or (iter_score > a and iter_score < b):
                    break
                # fail low/high: widen and retry this depth
            if iter_best is None or not completed:
                # Iterasi terpotong oleh waktu: ulangi depth yang sama selama
                # jaminan min_depth belum terpenuhi dan grace masih tersisa;
                # bila grace habis dan posisi masih terlalu sulit (depth
                # jaminan tidak tercapai), pakai ekstensi waktu posisi sulit
                # (1.8) — paling banyak hard_ext_max kali per langkah.
                if self.stop_reason == "time_limit":
                    if (depth_reached < self.min_depth
                            and time.monotonic() < self._grace_deadline):
                        continue
                    if depth_reached < self.min_depth and self._can_extend():
                        self._extend_budget("hard_position_timeout")
                        continue
                break
            best, best_score, depth_reached = iter_best, iter_score, d
            # remember this iteration's root move scores for next-depth ordering
            prev_scores = move_scores
            # 1.8: trigger ekstensi via stabilitas root — best move berubah
            # antar iterasi atau skor berayun besar menandakan posisi sulit;
            # bila sisa budget tinggal < 40%, perpanjang sebelum iterasi
            # berikutnya agar keputusan tidak diambil pada pencarian dangkal.
            unstable = (d >= 4 and last_best_uci is not None
                        and iter_best.uci() != last_best_uci)
            swing = (d >= 4 and last_score is not None
                     and abs(iter_score - last_score) >= 150)
            last_best_uci = iter_best.uci()
            last_score = iter_score
            if (unstable or swing) and self._can_extend():
                remaining_ms = (self.deadline - time.monotonic()) * 1000.0
                if remaining_ms < 0.4 * self.time_ms:
                    self._extend_budget("root_instability" if unstable
                                        else "score_swing")
            if self._budget_exceeded(force_clock=True):
                break
            if self.on_info is not None:
                elapsed = max(time.monotonic() - self.start, 1e-6)
                pv = self._extract_pv(board, iter_best)
                self.on_info({"depth": d, "score_cp": iter_score, "nodes": self.nodes,
                             "nps": int(self.nodes / elapsed), "bestmove": iter_best.uci() if iter_best else None,
                             "pv": pv, "elapsed_ms": int(elapsed * 1000), "tt_hits": self.tt_hits,
                             "tt_misses": self.tt_misses, "tt_hit_rate": self.tt_hits / max(1, self.tt_hits + self.tt_misses),
                             "cutoffs": self.cutoffs, "qnodes": self.qnodes,
                             "qnode_ratio": self.qnodes / max(1, self.nodes)})
        elapsed = max(time.monotonic() - self.start, 1e-6)
        return best, {"depth": depth_reached, "score_cp": best_score,
                      "nodes": self.nodes, "pv": self._extract_pv(board, best),
                      "nps": int(self.nodes / elapsed),
                      "tt_hits": self.tt_hits, "tt_misses": self.tt_misses,
                      "tt_hit_rate": self.tt_hits / max(1, self.tt_hits + self.tt_misses),
                      "cutoffs": self.cutoffs, "qnodes": self.qnodes,
                      "qnode_ratio": self.qnodes / max(1, self.nodes),
                      "elapsed_ms": int(elapsed * 1000), "timed_out": self._time_expired,
                      "node_limited": self.stop_reason == "node_limit",
                      "stop_reason": self.stop_reason,
                      "completed_depth": depth_reached,
                      "min_depth_met": (depth_reached >= self.min_depth
                                        if self.min_depth else None),
                      "time_extended_ms": self._ext_ms_total,
                      "extensions": self._ext_count,
                      "extension_reasons": list(self._ext_reasons)}
