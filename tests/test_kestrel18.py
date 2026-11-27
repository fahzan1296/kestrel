"""Test fitur 1.8: ekstensi waktu posisi sulit, fitur kekuatan search,
skema fitur learning v2 + migrasi, warm start/class balance/lr decay,
dan identitas eval fusi. (run: python -m pytest -q tests/test_kestrel18.py)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa: E402
import pytest  # noqa: E402

from kestrel.search import Searcher  # noqa: E402

# Posisi middlegame padat — pencarian butuh waktu, cocok memicu timeout.
HARD_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

TACTICAL = [
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
    ("k7/8/2K5/8/8/8/8/1Q6 w - - 0 1", "b1b7"),
    ("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1", "c1c8"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     "h5f7"),
]


def _mate_found(fen, want):
    b = chess.Board(fen)
    s = Searcher(time_ms=1000, max_depth=64)
    mv, info = s.best_move(b)
    return mv is not None and mv.uci() == want


# ---------------- ekstensi waktu posisi sulit ----------------

def test_hard_extension_triggers_on_timeout():
    """Budget kecil + grace kurang + min_depth tinggi → ekstensi dipakai."""
    s = Searcher(time_ms=40, max_depth=32, min_depth=5, grace_ms=60,
                 hard_ext_ms=700, hard_ext_max=1, hard_ext_min_budget_ms=0)
    b = chess.Board(HARD_FEN)
    mv, info = s.best_move(b)
    assert info["extensions"] >= 1
    assert "hard_position_timeout" in info["extension_reasons"]
    assert info["time_extended_ms"] >= 700
    assert mv is not None and mv in chess.Board(HARD_FEN).legal_moves
    # Ekstensi memberi waktu nyata: elapsed jelas melewati budget dasar+grace.
    assert info["elapsed_ms"] >= 100


def test_hard_extension_disabled_with_node_budget():
    """Budget node tidak pernah dilonggar oleh ekstensi waktu."""
    s = Searcher(node_limit=3000, time_ms=600000, max_depth=32, min_depth=6,
                 grace_ms=10, hard_ext_ms=700, hard_ext_max=1,
                 hard_ext_min_budget_ms=0)
    mv, info = s.best_move(chess.Board(HARD_FEN))
    assert info["extensions"] == 0
    assert info["extension_reasons"] == []
    assert info["node_limited"]


def test_hard_extension_respects_max_and_min_budget():
    s = Searcher(time_ms=40, max_depth=32, min_depth=6, grace_ms=10,
                 hard_ext_ms=0, hard_ext_max=1, hard_ext_min_budget_ms=0)
    _, info = s.best_move(chess.Board(HARD_FEN))
    assert info["extensions"] == 0  # hard_ext_ms=0 menonaktifkan
    # _can_extend: ambang budget minimum menyaring budget kecil.
    s2 = Searcher(time_ms=40, max_depth=8, min_depth=0,
                  hard_ext_ms=5000, hard_ext_max=1, hard_ext_min_budget_ms=500)
    assert not s2._can_extend()
    s3 = Searcher(time_ms=1000, max_depth=8, min_depth=0,
                  hard_ext_ms=5000, hard_ext_max=2, hard_ext_min_budget_ms=500)
    assert s3._can_extend()
    s3._extend_budget("root_instability")
    assert s3._can_extend()
    s3._extend_budget("score_swing")
    assert not s3._can_extend()  # hard_ext_max=2 tercapai
    assert s3._ext_reasons == ["root_instability", "score_swing"]
    assert s3._ext_ms_total == 10000


def test_searcher_validates_hard_ext_params():
    with pytest.raises(ValueError):
        Searcher(time_ms=100, hard_ext_ms=-1)
    with pytest.raises(ValueError):
        Searcher(time_ms=100, hard_ext_max=-2)
    with pytest.raises(ValueError):
        Searcher(time_ms=100, hard_ext_min_budget_ms=-1)


# ---------------- fitur kekuatan search ----------------

def test_tactical_mates_with_all_features_on():
    for fen, want in TACTICAL:
        assert _mate_found(fen, want), f"mate gagal: {fen} -> {want}"


def test_tactical_mates_with_features_disabled():
    """Fitur baru boleh dimatikan satu per satu tanpa merusak taktik dasar."""
    for fen, want in TACTICAL[:2]:
        b = chess.Board(fen)
        s = Searcher(time_ms=1000, max_depth=64, use_iir=False,
                     use_futility=False, use_cont_history=False)
        mv, _ = s.best_move(b)
        assert mv is not None and mv.uci() == want


def test_history_bounds_and_int_keys():
    """History int & bounded: ±1M butterfly, ±600k continuation, ±1M capture."""
    s = Searcher(time_ms=120, max_depth=8, hard_ext_ms=0)
    b = chess.Board(HARD_FEN)
    for _ in range(3):
        mv, info = s.best_move(b.copy())
    for v in s.history.values():
        assert -1_000_000 <= v <= 1_000_000
    for v in s.cont_history.values():
        assert -600_000 <= v <= 600_000
    for v in s.cap_history.values():
        assert -1_000_000 <= v <= 1_000_000
    # Semua kunci harus int (bukan tuple 1.6/1.7).
    assert all(isinstance(k, int) for k in s.history)


def test_cont_history_changes_ordering_after_cutoffs():
    """Setelah cutoff berulang, langkah bersejarah baik naik peringkat."""
    s = Searcher(time_ms=300, max_depth=8, hard_ext_ms=0)
    b = chess.Board(chess.STARTING_FEN)
    s.best_move(b)
    s.best_move(b.copy())
    # History putih harus terisi non-nol setelah dua pencarian.
    assert any(v > 0 for k, v in s.history.items() if (k >> 12) == 1)


# ---------------- learning v2 ----------------




def test_config_hard_ext_and_learning_fields():
    from kestrel.config import AppConfig, validate_config
    cfg = AppConfig()
    assert cfg.search.hard_ext_ms == 5000 and cfg.search.hard_ext_max == 1
    assert cfg.search.hard_ext_min_budget_ms == 500
    assert cfg.lichess.hard_ext_ms == 5000
    assert cfg.nnue.warm_start and cfg.nnue.teacher_fallback
    assert 0 < cfg.nnue.lr_decay <= 1.0
    assert validate_config(cfg) == []
    bad = AppConfig()
    bad.nnue.lr_decay = 1.5
    assert validate_config(bad)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_config_nnue_fields():
    """[nnue] 2.0: field distilasi/outcome tervalidasi; section lama dipetakan."""
    from kestrel.config import load_config, validate_config
    cfg = load_config()
    assert cfg.nnue.warm_start is True
    assert 0.0 <= cfg.nnue.distill_weight <= 1.0
    assert cfg.nnue.outcome_scale_cp > 0
    assert validate_config(cfg) == []
    bad = cfg.with_overrides("nnue", distill_weight=1.5)
    assert any("distill_weight" in e for e in validate_config(bad))
