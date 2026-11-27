"""KESTREL 2.0 — Evaluator AI (TinyNNUE sepenuhnya).

E(s) = TinyNNUE(s) — jaringan NNUE-lite terlatih (kestrel.nnue), tanpa blend
handcrafted, tanpa residual linear, tanpa REDA. Evaluator klasik hanya
dipertahankan sebagai teacher distilasi (kestrel.teacher) dan pembanding
benchmark, bukan jalur runtime.

Gate TERMINAL (aturan permainan, bukan evaluasi) tetap dijalankan di depan
supaya search benar: skakmat, pat, material tak cukup, 75-langkah, lima-kali
repetisi. Semua keputusan posisional/taktik lain kini dari jaringan.

Antisimetri mirror dijamin eksak BY KONSTRUKSI (perspektif STM pada net):
eval(mirror(s)) == -eval(s), drift 0.
"""
from __future__ import annotations

import chess

from .nnue import TinyNNUE, load_default

__all__ = ["evaluate", "stm_eval", "default_net"]


def default_net() -> TinyNNUE:
    """Net baseline terlatih yang dibundel (dipakai bila caller tanpa net)."""
    return load_default()


def evaluate(board: chess.Board, net: TinyNNUE | None = None, *,
             accs: tuple | None = None, in_check: bool | None = None,
             has_moves: bool | None = None, terminal: bool = True) -> int:
    """E(s) White-relative, centipawn, dari TinyNNUE.

    Hint ``in_check``/``has_moves``/``terminal`` dari caller search mencegah
    predikat mahal dibayar dua kali; ``accs`` = akumulator inkremental
    (acc_putih, acc_hitam) yang dipelihara Searcher (None = hitung dari nol).
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
    if net is None:
        net = default_net()
    return net.white_score(board, accs=accs)


def stm_eval(board: chess.Board, net: TinyNNUE | None = None, *,
             accs: tuple | None = None) -> int:
    """V(s) = sign(p) E(s), perspektif side-to-move."""
    if net is None:
        net = default_net()
    return (net.stm_score(board, accs=accs) if board.turn == chess.WHITE
            else -net.stm_score(board, accs=accs))
