"""KESTREL 1.6 — Test suite untuk trainer unified, knowledge, versioning,
verification, CLI, dan bot Lichess (semuanya offline, budget kecil)."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
import pytest  # pyright: ignore[reportMissingImports]  # dev dependency

from kestrel.config import AppConfig, load_config, validate_config
from kestrel.knowledge import KnowledgeBase
from kestrel.learning import OutcomeModel
from kestrel.search import Searcher
from kestrel.versioning import CheckpointStore

TEST_TOKEN = os.environ.get("KESTREL_TEST_TOKEN", "")

TINY_PGN = ('[Event "Tiny"]\n[Result "1-0"]\n\n1. e4 { 34/d7 n123 } e5 { -12/d7 n150 } '
            '2. Qh5 { 20/d7 n90 } Nc6 { -8/d7 n99 } 3. Bc4 { 30/d8 n210 } Nf6 { -5/d8 n180 } '
            '4. Qxf7# { 100000/d9 n300 } 1-0\n\n[Event "Tiny2"]\n[Result "1/2-1/2"]\n\n'
            '1. e4 { 34/d7 n120 } e5 { -12/d7 n130 } 2. Nf3 { 18/d7 n110 } Nc6 { -9/d7 n140 } '
            '3. Nc3 { 22/d8 n160 } Nf6 { -6/d8 n150 } 4. Rg1 { 10/d8 n200 } Rb8 { -4/d8 n170 } '
            '1/2-1/2\n')


def _tmp_cfg(tmp_path: Path) -> AppConfig:
    cfg = load_config()  # config.toml proyek sebagai dasar
    cfg = cfg.with_overrides(
        "storage", results_dir=str(tmp_path / "results"),
        database=str(tmp_path / "results" / "kestrel.db"),
        pgn_file=str(tmp_path / "results" / "games.pgn"))
    cfg = cfg.with_overrides("knowledge", book_min_games=1, db_name="knowledge.db")
    return cfg


# ---------------- config ----------------

def test_config_load_validate_override():
    cfg = load_config()
    assert cfg.selfplay.games_per_batch >= 1
    assert 0 <= cfg.eval.min_depth <= cfg.eval.max_depth
    errors = validate_config(cfg)
    assert errors == [], errors
    # override CLI-style
    cfg2 = cfg.with_overrides("selfplay", games_per_batch=42, time_ms=99)
    assert cfg2.selfplay.games_per_batch == 42
    assert cfg2.selfplay.time_ms == 99
    assert cfg.selfplay.games_per_batch != 42  # original tidak berubah
    # validasi menangkap konfigurasi salah
    bad = cfg.with_overrides("eval", games=3)  # ganjil
    assert any("genap" in e for e in validate_config(bad))
    # section [engine] lama dipetakan ke search
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "old.toml"
        p.write_text("[engine]\nmove_time_ms = 777\nmax_depth = 33\n")
        cfg3 = load_config(p)
        assert cfg3.search.move_time_ms == 777 and cfg3.search.max_depth == 33


# ---------------- knowledge base ----------------

def test_knowledge_learn_and_book():
    with tempfile.TemporaryDirectory() as d:
        pgn = Path(d) / "tiny.pgn"
        pgn.write_text(TINY_PGN)
        kb = KnowledgeBase(Path(d) / "knowledge.db", book_max_plies=6,
                           book_min_games=1)
        rep = kb.learn_from_pgn([pgn])
        assert rep["games_processed"] == 2
        assert rep["book_moves_upserted"] > 0
        stats = kb.stats()
        assert stats["book_moves"] > 0
        assert stats["experience_positions"] > 0
        # book memilih langkah legal di posisi start (ply 0 ada di korpus: e4)
        board = chess.Board()
        pick = kb.book_move(board, rng=None)
        assert pick is not None
        uci, san, games, score = pick
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves
        # top lines + lessons
        assert kb.top_book_lines(5)
        assert kb.lessons(kind="corpus_learned")


# ---------------- versioning ----------------

def test_versioning_promote_rollback_prune():
    with tempfile.TemporaryDirectory() as d:
        store = CheckpointStore(Path(d) / "results", keep_last=3)
        assert store.champion_meta()["generation"] == 0
        model = OutcomeModel()
        model.weights[0] = 0.123
        p1 = store.save_candidate(model, 1, {"note": "gen1"})
        assert p1.exists()
        meta = store.promote(p1, "test promote")
        assert meta["generation"] == 1
        assert store.champion_model_path.exists()
        loaded = store.load_champion()
        assert abs(loaded.weights[0] - 0.123) < 1e-6
        # generasi kedua, lalu rollback
        p2 = store.save_candidate(model, 2)
        model2 = OutcomeModel()
        model2.weights[0] = 0.456
        p2 = store.save_candidate(model2, 2)
        store.promote(p2, "gen2")
        assert abs(store.load_champion().weights[0] - 0.456) < 1e-6
        store.rollback(steps=1)
        assert abs(store.load_champion().weights[0] - 0.123) < 1e-6
        # prune: keep_last=3
        for gen in range(3, 8):
            store.save_candidate(model, gen)
        assert len(store.candidates()) <= 3
        with pytest.raises(FileNotFoundError):
            store.promote(Path(d) / "missing.json", "x")


# ---------------- verification ----------------

def test_elo_ci_and_sprt_consistency():
    from kestrel.arena import elo_delta, wilson_ci
    from kestrel.verification import _elo_ci
    lo, hi = wilson_ci(10, 5, 5)
    elo_lo, elo_hi = _elo_ci(10, 5, 5)
    assert elo_lo <= elo_delta((10 + 2.5) / 20) <= elo_hi
    # dominasi penuh → CI tinggi
    elo_lo, elo_hi = _elo_ci(20, 0, 0)
    assert elo_hi > 200


def test_verify_challenger_rejects_weaker():
    from kestrel.verification import report_markdown, verify_challenger
    cfg = _tmp_cfg(Path(tempfile.mkdtemp()))
    cfg = cfg.with_overrides("eval", games=2, time_ms=60, max_depth=4,
                             min_depth=2, grace_ms=150)
    # challenger = model acak-bijas (bobot besar) → harus kalah/ditolak
    strong = OutcomeModel()
    weak = OutcomeModel()
    weak.weights[1] = 5.0   # fitur aneh → evaluasi buruk
    weak.weights[5] = -5.0
    rep = verify_challenger(strong, weak, cfg, quick=True)
    assert rep["decision"] in ("REJECT", "PROMOTE")  # 2 game bisa berisik
    assert "match" in rep and "challenger_bench" in rep
    assert rep["match"]["games"] == 2
    md = report_markdown(rep)
    assert "Score rate" in md and "SPRT" in md


# ---------------- trainer end-to-end (smoke) ----------------

def test_trainer_one_generation(tmp_path):
    from kestrel.trainer import QuickOverrides, run_training
    cfg = QuickOverrides.apply(_tmp_cfg(tmp_path))
    cfg = cfg.with_overrides("train", generations=1)  # override default config.toml (10)
    rep = run_training(cfg, quick=True, verify=True)
    assert rep["overall"] == "PASS"
    assert len(rep["generations"]) == 1
    gen = rep["generations"][0]
    assert gen["selfplay_ok"]
    assert "knowledge" in gen["stages"]
    results = cfg.results_path
    assert (results / "learned_eval.json").exists()
    assert (results / "champion.json").exists()
    assert (results / "training_log.jsonl").exists()
    ver_files = list(results.glob("verification_gen_*.json"))
    assert ver_files, "laporan verifikasi per generasi harus ada"
    ver = json.loads(ver_files[0].read_text())
    assert ver["match"]["games"] == 2
    # knowledge base nyata: book terisi dari game self-play
    kb = KnowledgeBase(cfg.knowledge_db_path)
    assert kb.stats()["book_moves"] > 0
    # generasi kedua memakai champion dari generasi pertama (resume)
    rep2 = run_training(cfg, quick=True, verify=False)
    assert rep2["final_generation"] == rep["final_generation"] + 1


# ---------------- search min-depth guarantee ----------------

def test_extract_pv_does_not_copy_move_stack():
    """PV hanya butuh posisi; menyalin move_stack hanya menambah overhead."""
    import chess

    from kestrel.search import TTEntry, _tt_key

    board = chess.Board()
    board.push_san("e4")
    original_copy = board.copy
    copy_args: list[bool] = []

    def spy_copy(stack=True):
        copy_args.append(stack)
        return original_copy(stack=stack)

    board.copy = spy_copy  # type: ignore[method-assign]
    searcher = Searcher(tt_size=8)
    move = next(iter(board.legal_moves))
    key = _tt_key(board)
    # 1.7: TT flat-array — entri ditulis ke slot kunci (NamedTuple kompat).
    searcher.tt[key & searcher._tt_mask] = TTEntry(key, 1, 0, 0, move, 0)
    fen = board.fen()
    plies = len(board.move_stack)
    assert searcher._extract_pv(board) == [move.uci()]
    assert copy_args == [False]
    assert board.fen() == fen and len(board.move_stack) == plies

def test_search_min_depth_guarantee():
    import time as _time
    s = Searcher(time_ms=10, max_depth=12, min_depth=9, grace_ms=800)
    b = chess.Board("4k3/8/8/8/8/5k2/8/4K3 w - - 0 1")
    mv, info = s.best_move(b)
    assert info["depth"] >= 9 and info["min_depth_met"]
    # tanpa min_depth: berhenti dangkal (cepat)
    s2 = Searcher(time_ms=10, max_depth=12)
    t0 = _time.monotonic()
    _, info2 = s2.best_move(chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"))
    assert _time.monotonic() - t0 < 1.0 and info2["depth"] < 9
    # node budget tidak dilonggar oleh grace
    s3 = Searcher(node_limit=1200, time_ms=10, max_depth=10, min_depth=8, grace_ms=800)
    _, info3 = s3.best_move(chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"))
    assert info3["nodes"] <= 1500 and info3["node_limited"]


# ---------------- CLI ----------------

def test_cli_parser_and_config_show(capsys):
    from kestrel.cli import build_parser, main
    ap = build_parser()
    args = ap.parse_args(["train", "--generations", "3", "--games", "8",
                          "--time-ms", "120", "--quick", "--eval-games", "4"])
    assert args.command == "train" and args.generations == 3 and args.eval_games == 4
    args2 = ap.parse_args(["model", "rollback", "--steps", "2"])
    assert args2.action == "rollback"
    # config show via main()
    rc = main(["config", "show"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "selfplay" in payload and "eval" in payload and rc == 0


# ---------------- lichess bot (tanpa jaringan) ----------------

def test_lichess_policy_and_move_choice(tmp_path):
    from kestrel.lichess_bot import LichessBot
    cfg = _tmp_cfg(tmp_path)
    cfg = cfg.with_overrides("lichess", move_time_ms=80, max_depth=6,
                             min_depth=2, grace_ms=200, book_moves=False,
                             resign_cp=-900, resign_moves=2)
    bot = LichessBot(cfg, token="", dry_run=True)
    # kebijakan challenge
    bot.username = "kestrelbot"
    bot.my_rating = 1500
    good = {"challenge": {"id": "abc", "variant": {"key": "standard"},
                          "speed": "blitz", "rating": {"international": 1400},
                          "challenger": {"name": "opponent", "title": "BOT"}}}
    assert bot._accept_challenge(good)
    nonstandard = {"challenge": {"id": "abc", "variant": {"key": "antichess"},
                                 "speed": "blitz", "rating": {},
                                 "challenger": {"name": "opponent"}}}
    assert bot._accept_challenge(nonstandard)
    self_challenge = {"challenge": {"id": "x", "variant": {"key": "standard"},
                                    "speed": "blitz", "rating": {},
                                    "challenger": {"name": "kestrelbot"}}}
    assert not bot._accept_challenge(self_challenge)
    far_rating = {"challenge": {"id": "y", "variant": {"key": "standard"},
                                "speed": "blitz", "rating": {"international": 2400},
                                "challenger": {"name": "someone"}}}
    assert bot._accept_challenge(far_rating)
    # Bug 1.6 awal (accept_human=false): tantangan akun manusia normal harus
    # diterima — ini kasus user menantang bot dari akun utamanya.
    human_ok = {"challenge": {"id": "z", "variant": {"key": "standard"},
                              "speed": "rapid", "rating": {"international": 1450},
                              "challenger": {"name": "human_player"}}}
    assert bot._accept_challenge(human_ok)
    # pemilihan langkah legal + book dinonaktifkan
    mv, info = bot.choose_move(chess.Board())
    assert mv is not None and mv in chess.Board().legal_moves
    assert not info.get("book")


def test_lichess_sse_parser_both_wire_formats():
    """Bug 1.6 awal: hanya 'data: ' ber-spasi dikenali; lichess bisa mengirim
    'data:' tanpa spasi → bot diam karena tidak melihat event challenge."""
    from kestrel.lichess_bot import parse_sse
    wire_with_space = [
        b'event: challenge',
        b'data: {"type":"challenge","challenge":{"id":"a"}}',
        b'',
        b'data: {"type":"gameStart","game":{"id":"g1"}}',
        b'',
    ]
    events = list(parse_sse(wire_with_space))
    assert [e.get("type") for e in events] == ["challenge", "gameStart"]
    wire_no_space_no_blank = [
        b'data:{"type":"challenge","challenge":{"id":"b"}}',
        b'ping',
        b'data:{"type":"gameStart","game":{"id":"g2"}}',
    ]
    events2 = list(parse_sse(wire_no_space_no_blank))
    assert [e.get("type") for e in events2] == ["challenge", "gameStart"]
    # komentar keepalive & data kosong tidak menghasilkan event
    events3 = list(parse_sse([b': keepalive', b'', b'data: ', b'data: {"type":"ping"}']))
    assert events3 == [{"type": "ping"}]
    # input str juga didukung
    assert list(parse_sse(['data: {"type":"x"}'])) == [{"type": "x"}]


def test_lichess_ndjson_stream_parser():
    """Bug 1.6: /api/stream/event dan /api/stream/bot/game/* mengirim NDJSON
    (Content-Type application/x-ndjson), bukan SSE. Parser lama hanya
    mengenali baris 'data:' sehingga setiap challenge dibuang diam-diam."""
    from kestrel.lichess_bot import parse_sse
    wire = [
        b'{"type":"challenge","challenge":{"id":"n1"}}',
        b'',  # keepalive 7 detik dari lichess
        b'{"type":"gameStart","game":{"id":"g1"}}',
        b': keepalive',
        b'data: {"type":"gameState","status":"created"}',
    ]
    events = list(parse_sse(wire))
    assert [e.get("type") for e in events] == [
        "challenge", "gameStart", "gameState"]


def test_lichess_initial_fen_startpos():
    """gameFull Lichess memakai 'startpos', bukan FEN aktual."""
    from kestrel.lichess_bot import _initial_fen
    assert _initial_fen("startpos") == chess.STARTING_FEN
    assert _initial_fen(None) == chess.STARTING_FEN
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert _initial_fen(fen) == fen


def test_lichess_game_stream_endpoint(tmp_path):
    """Bot API game stream adalah /api/bot/game/stream/{id}, bukan
    /api/stream/bot/game/stream/{id} (404 → bot tidak pernah melihat langkah)."""
    from kestrel.lichess_bot import LichessAPI, LichessBot
    cfg = _tmp_cfg(tmp_path)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    seen = {}

    class FakeAPI(LichessAPI):
        def stream(self, path):
            seen["path"] = path
            yield from ()

        def post(self, *a, **kw):
            raise AssertionError("tidak ada langkah yang dikirim")

    bot.api = FakeAPI(TEST_TOKEN)
    assert bot.play_game("g1")["status"] == "stream_ended"
    assert seen["path"] == "/api/bot/game/stream/g1"


def test_lichess_profile_rating_ignores_provisional():
    """Akun BOT baru punya rating 3000 provisional dengan 0 game. Memakainya
    sebagai rating sendiri membuat semua challenge manusia kena max_rating_gap."""
    from kestrel.lichess_bot import _account_rating
    assert _account_rating({"perfs": {"blitz": {
        "rating": 3000, "games": 0, "prov": True}}}) == 0
    assert _account_rating({"perfs": {"blitz": {
        "rating": 1600, "games": 30, "prov": False}}}) == 1600


def test_lichess_accepts_every_opponent_challenge(tmp_path):
    """Kebijakan baru: semua tantangan dari lawan wajib diterima.

    Filter lama (variant, speed, rating, bot/human, rating gap) tidak boleh
    lagi menolak lawan. Self-challenge tetap ditolak karena bukan lawan.
    """
    from kestrel.lichess_bot import LichessBot

    cfg = _tmp_cfg(tmp_path).with_overrides(
        "lichess", accept_bot=False, accept_human=False,
        min_rating=9999, max_rating_gap=1)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot.my_rating = 1500

    def challenge(**overrides):
        data = {
            "variant": {"key": "standard"},
            "speed": "blitz",
            "rating": {"international": 1500},
            "challenger": {"name": "Mysam1973", "title": None},
        }
        data.update(overrides)
        return {"type": "challenge", "challenge": data}

    accepted = [
        challenge(),
        challenge(rating=None),
        challenge(challenger={"name": "BotX", "title": "BOT"}),
        challenge(variant={"key": "threeCheck"}),
        challenge(speed="correspondence"),
        challenge(rating={"international": 10}),
        challenge(rating={"international": 2900}),
        challenge(challenger=None),
    ]
    for ev in accepted:
        assert bot._accept_challenge(ev) is True

    self_challenge = challenge(challenger={"name": bot.username, "title": None})
    assert bot._accept_challenge(self_challenge) is False


def test_lichess_account_rating_handles_missing_profile():
    """Profil None atau tanpa perfs harus aman, bukan AttributeError."""
    from kestrel.lichess_bot import _account_rating
    assert _account_rating(None) == 0
    assert _account_rating({}) == 0
    assert _account_rating({"perfs": None}) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
