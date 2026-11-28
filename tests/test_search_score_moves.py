import json
from pathlib import Path

import chess

from kestrel.nnue import load_default
from kestrel.search import Searcher

CASES = (
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None, (), 0, False),
    ("r1bq1rk1/ppp2ppp/2n5/3np3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 9", 0, (1, 2), 0, True),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", None, (0, 1), 0x1234, True),
    ("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1", None, (), 0x2345, True),
    ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", 1, (0,), 0, True),
    ("rnbqkbnr/ppp1p1pp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", None, (), 0, False),
    ("r3k2r/pppb1ppp/2n5/3np3/8/2NP1N2/PPP2PPP/R3K2R w KQkq - 0 10", 0, (1,), 0x3456, True),
    ("8/P6k/8/8/8/8/7K/8 w - - 0 1", None, (0,), 0, True),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", None, (), 0, False),
)


def _setup(fen, tt_idx, killer_idxs, prev_key, fill):
    board = chess.Board(fen)
    searcher = Searcher(net=load_default(), time_ms=10_000)
    moves = list(board.legal_moves)
    searcher.killers[0] = [moves[i] for i in killer_idxs[:2]]
    while len(searcher.killers[0]) < 2:
        searcher.killers[0].append(None)
    searcher.history.clear()
    searcher.cap_history.clear()
    searcher.cont_history.clear()
    if fill:
        for mv in moves:
            frm, to = mv.from_square, mv.to_square
            searcher.history[(board.turn << 12) | (frm << 6) | to] = \
                (frm * 7 + to * 3) % 500_000
            if prev_key:
                searcher.cont_history[prev_key | (frm << 6) | to] = \
                    (frm * 11 + to * 5) % 500_000
            if board.is_capture(mv):
                victim = board.piece_type_at(to) or chess.PAWN
                searcher.cap_history[(victim << 12) | (frm << 6) | to] = \
                    (victim * 1000 + to) % 500_000
    tt = moves[tt_idx] if tt_idx is not None else None
    return board, searcher, moves, tt, prev_key


def test_score_moves_golden_is_exact_and_input_is_not_mutated():
    """Semua komponen ordering harus tetap identik setelah optimasi."""
    golden = json.loads(
        Path(__file__).with_name("score_moves_golden.json").read_text(encoding="utf-8")
    )
    assert len(golden) == len(CASES)
    for expected, case in zip(golden, CASES, strict=True):
        board, searcher, moves, tt, prev_key = _setup(*case)
        original = list(moves)
        scored = searcher._score_moves(board, 0, tt, moves, prev_key)
        actual = [[sc, mv.uci(), bool(cap), promo, victim]
                  for sc, mv, cap, promo, victim in scored]
        assert actual == expected["scored"]
        assert moves == original
