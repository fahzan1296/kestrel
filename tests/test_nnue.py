"""KESTREL 2.0 — Test suite TinyNNUE: arsitektur, kuantisasi, antisimetri,
akumulator inkremental, dan trainer distilasi (menggantikan test_learning.py).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
import numpy as np
import pytest  # pyright: ignore[reportMissingImports]

from kestrel.nnue import TinyNNUE, load_default
from kestrel.nnue_train import corpus_from_pgn, train_nnue
from kestrel.search import Searcher


def _net_with_head(seed: int = 3) -> TinyNNUE:
    """Net acak dengan kepala output kecil (untuk uji jalur evaluasi nyata)."""
    net = TinyNNUE()
    rng = np.random.default_rng(seed)
    net.w3 = (rng.standard_normal((32, 1)) * 0.05).astype(np.float32)
    net.b3 = np.zeros(1, np.float32)
    return net


def _random_position(rng: np.random.Generator, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            break
        b.push(rng.choice(moves))
    return b


def test_param_count_in_budget():
    net = TinyNNUE()
    assert net.params() == 106_689
    assert net.params() < 1_000_000


def test_zero_head_neutral_and_invalid_shape_rejected():
    net = TinyNNUE()
    assert float(net.raw(chess.Board())) == 0.0
    with pytest.raises(ValueError):
        TinyNNUE(np.zeros((10, 128), np.float32))


def test_mirror_antisymmetry_exact():
    net = _net_with_head()
    rng = np.random.default_rng(5)
    for _ in range(40):
        b = _random_position(rng, int(rng.integers(1, 50)))
        if b.is_valid() and not b.is_game_over():
            assert net.white_score(b) == -net.white_score(b.mirror())


def test_save_load_quantized_roundtrip():
    net = _net_with_head()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "net.json"
        net.save(p, {"note": "unit-test"})
        data = p.read_text()
        assert '"kind": "tinynnue"' in data
        loaded = TinyNNUE.load(p)
    assert loaded.params() == net.params()
    rng = np.random.default_rng(9)
    max_drift = 0
    for _ in range(40):
        b = _random_position(rng, int(rng.integers(1, 60)))
        if b.is_valid() and not b.is_game_over():
            max_drift = max(max_drift, abs(loaded.white_score(b)
                                           - net.white_score(b)))
    # kuantisasi int8 per-tensor: drift evaluasi harus kecil (<= 12 cp)
    assert max_drift <= 12, f"drift kuantisasi {max_drift} cp"


def test_load_rejects_foreign_or_broken_checkpoint(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"kind": "other", "version": 3}')
    with pytest.raises(ValueError):
        TinyNNUE.load(p)
    p2 = tmp_path / "bad2.json"
    p2.write_text('{"kind": "tinynnue", "version": 1}')
    with pytest.raises(ValueError):
        TinyNNUE.load(p2)


def test_search_incremental_accumulator_matches_scratch():
    """Akumulator inkremental di Searcher == hitung dari nol, termasuk
    castling, en passant, promosi, dan null move."""
    net = _net_with_head(4)
    rng = np.random.default_rng(21)
    s = Searcher(net=net, time_ms=10_000, max_depth=4)
    b = chess.Board()
    s._sync_hash(b)
    checked = 0
    for i in range(300):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            b = chess.Board()
            s._sync_hash(b)
            continue
        mv = rng.choice(moves)
        s._push(b, mv)
        aw, ab = net.refresh(b)
        assert float(np.max(np.abs(s._acc_w - aw))) < 1e-3
        assert float(np.max(np.abs(s._acc_b - ab))) < 1e-3
        checked += 1
        if i % 4 == 3:
            s._pop(b)
        if i % 7 == 5 and not b.is_check():
            s._push_null(b)
            s._pop(b)
    assert checked > 200


def test_search_finds_mate1_with_net():
    net = _net_with_head(6)
    s = Searcher(net=net, time_ms=4000, node_limit=120000)
    mv, info = s.best_move(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"))
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    b.push(mv)
    assert b.is_checkmate()
    assert info["completed_depth"] >= 2


TINY_PGN = (
    '[Event "A"]\n[Result "1-0"]\n\n1. e4 { 34/d7 n123 } e5 { -12/d7 n150 } '
    '2. Qh5 { 20/d7 n90 } Nc6 { -8/d7 n99 } 3. Bc4 { 30/d8 n210 } Nf6 '
    '{ -5/d8 n180 } 4. Qxf7# { 100000/d9 n300 } 1-0\n\n'
    '[Event "B"]\n[Result "0-1"]\n\n1. e4 { 30/d7 n100 } e5 { -10/d7 n120 } '
    '2. Nf3 { 15/d7 n110 } Nc6 { -9/d7 n140 } 3. Bb5 { 18/d8 n160 } '
    'Nf6 { -6/d8 n150 } 1/2-1/2 * 0-1\n\n'
    '[Event "C"]\n[Result "1/2-1/2"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 '
    '4. Bg5 Be7 1/2-1/2\n'
)


def test_corpus_from_pgn_parses_comments_and_outcomes(tmp_path):
    p = tmp_path / "tiny.pgn"
    p.write_text(TINY_PGN)
    samples, stats = corpus_from_pgn([str(p)], min_ply=2)
    assert stats["games_read"] == 3
    assert stats["samples"] > 10
    # mayoritas label dari komentar skor; game C (tanpa komentar) dari teacher
    assert stats["labels_from_comments"] > 0
    assert stats["labels_from_teacher"] > 0
    # mate target (100000) harus ter-clip ke +1200 * alpha + outcome
    targets = [s["target_cp"] for s in samples]
    assert max(targets) <= 1200.0 * 0.75 + 350.0 * 0.25 + 1.0
    assert min(targets) >= -1200.0 * 0.75 - 350.0 * 0.25 - 1.0


def test_train_nnue_learns_teacher_signal():
    """Dari korpus posisi acak ber-label teacher, net mempelajari sinyalnya.

    Target = evaluasi statis teacher (rentang luas, dominasi material). Net
    harus mencapai korelasi tinggi — membuktikan pipeline encode->forward->
    backward benar-benar belajar, bukan sekadar menghitung.
    """
    from kestrel.nnue_train import _plane_lists, _predict, _encode_batch
    from kestrel.nnue import OUT_SCALE, _aux
    from kestrel.teacher import static_eval
    rng = np.random.default_rng(13)
    samples = []
    seen = set()
    while len(samples) < 300:
        b = _random_position(rng, int(rng.integers(2, 90)))
        if not b.is_valid() or b.is_game_over():
            continue
        import chess.polyglot
        key = chess.polyglot.zobrist_hash(b)
        if key in seen:
            continue
        seen.add(key)
        t = float(static_eval(b, terminal=False))
        if abs(t) > 1200:
            continue
        iw, ib = _plane_lists(b)
        samples.append({"plane_w": iw, "plane_b": ib, "aux": _aux(b),
                        "stm_white": b.turn == chess.WHITE,
                        "target_cp": t, "game_id": len(samples) // 30})
    net, report = train_nnue(samples, base=None, warm_start=False,
                             epochs=10, lr=0.1, batch_size=32, seed=7,
                             validation_fraction=0.2)
    assert report["params"] == 106_689
    best = report["best_validation_mse_cp2"]
    first = report["history"][0]["validation_mse_cp2"]
    assert best < first, f"MSE validasi tidak turun: {first} -> {best}"
    preds = _predict(net, samples) * OUT_SCALE
    targets = np.asarray([s["target_cp"] for s in samples])
    corr = float(np.corrcoef(preds, targets)[0, 1])
    assert corr > 0.5, f"korelasi prediksi-target rendah: {corr}"


def test_default_baseline_is_trained():
    """Checkpoint baseline bawaan HARUS terlatih (kepala bukan nol)."""
    net = load_default()
    assert net.metadata.get("trained_by", "").startswith("scripts/bootstrap")
    b = chess.Board()
    assert float(net.raw(b)) != 0.0
    # baseline mengikuti antisimetri eksak juga
    rng = np.random.default_rng(2)
    for _ in range(15):
        b2 = _random_position(rng, int(rng.integers(1, 40)))
        if b2.is_valid() and not b2.is_game_over():
            assert net.white_score(b2) == -net.white_score(b2.mirror())
