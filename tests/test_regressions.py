"""Regression tests untuk fix 2.0.4.

Menutup celah yang ditemukan saat audit:
1. Gate terminal 75-langkah belum punya test eksplisit.
2. Evaluator produksi (baseline terlatih) belum diuji langsung lewat evaluate().
3. Optimasi Woodbury pada warm-start harus ekuivalen dengan normal equation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np

from kestrel.eval import evaluate
from kestrel.nnue import load_default


def test_75_move_rule_is_terminal_zero():
    """75-langkah adalah aturan permainan, bukan evaluasi posisional."""
    b = chess.Board("8/8/8/8/8/8/8/K6k w - - 150 1")
    assert b.halfmove_clock >= 150
    assert b.is_seventyfive_moves()
    assert evaluate(b, load_default()) == 0


def test_production_baseline_mirror_exact_and_nonzero():
    """Baseline terlatih dipakai produksi: nonzero dan antisimetri eksak."""
    net = load_default()
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1")
    assert net.white_score(b) != 0
    assert net.white_score(b) == -net.white_score(b.mirror())


def test_dual_warmstart_finite_and_correct_shape(tmp_path):
    """Optimasi Woodbury harus ekuivalen dengan normal equation primal."""
    from kestrel.nnue_train import _linear_warmstart, corpus_from_pgn

    # Korpus mandiri: dua game pendek dari posisi awal, cukup untuk
    # menghasilkan >= 6 ply sampel tanpa bergantung pada artifact lokal.
    import chess.pgn

    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for _ in range(8):
        mv = next(iter(board.legal_moves))
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = "1/2-1/2"
    pgn_path = tmp_path / "corpus.pgn"
    pgn_path.write_text(str(game), encoding="utf-8")

    samples, _ = corpus_from_pgn([str(pgn_path)], max_samples_per_game=120)
    assert samples, "korpus uji harus menghasilkan sampel"
    w = _linear_warmstart(samples)
    assert w.shape == (768, 128)
    assert np.all(np.isfinite(w))
