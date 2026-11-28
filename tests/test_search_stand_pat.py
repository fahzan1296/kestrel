import chess

from kestrel.search import Searcher


def test_stand_pat_skips_redundant_terminal_checks(monkeypatch):
    searcher = Searcher(net=None, node_limit=1)
    seen = {}

    def fake_stm(board, **kwargs):
        seen.update(kwargs)
        return 7

    monkeypatch.setattr(searcher, "_stm", fake_stm)
    assert searcher._stand_pat(chess.Board()) == 7
    assert seen["terminal"] is False
    assert seen["has_moves"] is True
    assert seen["in_check"] is False


def test_stand_pat_matches_regular_eval_on_live_positions():
    """Flag stand-pat tidak boleh mengubah skor posisi non-terminal."""
    searcher = Searcher(net=None, node_limit=1)
    fens = [
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        assert searcher._stand_pat(board) == searcher._stm(board)
