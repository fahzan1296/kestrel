from __future__ import annotations

import contextlib
import os

TEST_TOKEN = os.environ.get("KESTREL_TEST_TOKEN", "")



def test_lichess_config_new_fields_and_cli_override():
    from kestrel.cli import build_parser
    from kestrel.config import load_config

    cfg = load_config()
    assert cfg.lichess.concurrency == 2
    assert cfg.lichess.auto_seek is True
    assert 1 <= cfg.lichess.seek_pool_size <= 512

    args = build_parser().parse_args(["lichess", "--concurrency", "4"])
    assert args.concurrency == 4
    cfg2 = cfg.with_overrides("lichess", concurrency=args.concurrency)
    assert cfg2.lichess.concurrency == 4


def test_choose_move_uses_explicit_player():
    import chess

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    sentinel = object()
    called = {}

    class FakePlayer:
        def play(self, board, time, rng):
            called["player"] = sentinel
            return next(iter(board.legal_moves)), {}

    mv, _ = bot.choose_move(chess.Board(), rng=None, player=FakePlayer())
    assert mv is not None
    assert called["player"] is sentinel
    assert bot._player is None


def test_reserve_game_respects_concurrency():
    from concurrent.futures import Future

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", concurrency=1, max_games=0)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot._games["g1"] = Future()
    assert bot._reserve_game("c2") is False


def test_reserve_game_is_idempotent_for_started_game():
    """gameStart bisa mendahului reservasi seek; jangan double counting."""
    from concurrent.futures import Future

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", concurrency=1, max_games=5)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot._games["g1"] = Future()
    assert bot._reserve_game("g1") is True
    assert "g1" not in bot._reservations

def test_reserve_game_respects_max_games():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", max_games=1)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    assert bot._reserve_game("a") is True
    assert bot._reserve_game("b") is False
    bot._release_game("a")
    assert bot._reserve_game("b") is True


def test_run_ignores_outgoing_challenge_without_declining(monkeypatch, capsys):
    """Challenge kiriman sendiri tidak boleh kita tolak/dibatalkan."""
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    class StopLoop(Exception):
        pass

    cfg = AppConfig().with_overrides(
        "lichess", username="Azchess-1", auto_seek=False,
        learn_from_games=False, poll_seconds=0.01)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=False)

    def fake_get(path, **params):
        return (
            b'{"username":"Azchess-1","title":"BOT",'
            b'"perfs":{"blitz":{"rating":1500,"games":10}}}'
        )

    def fake_stream(path):
        yield {
            "type": "challenge",
            "challenge": {
                "id": "OUT1",
                "direction": "out",
                "challenger": {"name": "Azchess-1"},
                "destUser": {"name": "Opponent"},
            },
        }
        raise StopLoop

    posted: list[str] = []
    monkeypatch.setattr(bot, "_get", fake_get)
    monkeypatch.setattr(bot.api, "stream", fake_stream)
    monkeypatch.setattr(
        bot, "_post", lambda path, params="", data=None: posted.append(path) or b"{}")
    with contextlib.suppress(StopLoop):
        bot.run()
    out = capsys.readouterr().out
    assert not any("/decline" in path or "/accept" in path for path in posted)
    assert "kiriman sendiri" not in out

def test_result_header_uses_winner_and_resign_summary():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    game_full = {"white": {"name": "A"}, "black": {"name": "B"}}
    assert bot._result_header(game_full, {"winner": "black"},
                              {"my_color": "white"}) == "0-1"
    assert bot._result_header(game_full, {},
                              {"status": "resign_sent", "my_color": "white"}) == "0-1"
    assert bot._result_header(game_full, {"status": "draw"},
                              {"my_color": "white"}) == "1/2-1/2"


def test_submit_game_uses_executor():
    from concurrent.futures import ThreadPoolExecutor

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", max_games=2)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot._executor = ThreadPoolExecutor(max_workers=2)
    bot.play_game = lambda game_id, player=None: {"status": "ok"}
    assert bot._submit_game("g1") is True
    assert bot._submit_game("g2") is True
    assert bot._submit_game("g3") is False
    bot._executor.shutdown(wait=True)


def test_pgn_candidates_exclude_corpus_and_dedup(tmp_path):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    bot.pgn_dir = tmp_path
    for name in ["a.pgn", "b.pgn", ".learn_corpus.pgn", "a.pgn.tmp"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [p.name for p in bot._pgn_candidates()] == ["a.pgn", "b.pgn"]


def test_write_learning_corpus_is_valid_pgn(tmp_path):
    import chess.pgn

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    p1 = tmp_path / "a.pgn"
    p2 = tmp_path / "b.pgn"
    p1.write_text('[Result "1-0"]\n\n1. e4 e5 *\n\n', encoding="utf-8")
    p2.write_text('[Result "0-1"]\n\n1. d4 d5 *\n\n', encoding="utf-8")
    target = bot._write_learning_corpus([p1, p2], tmp_path / ".learn_corpus.pgn")
    games = []
    with target.open(encoding="utf-8") as f:
        while game := chess.pgn.read_game(f):
            games.append(game)
    assert len(games) == 2


def test_learn_cycle_skips_small_corpus(tmp_path):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    bot.pgn_dir = tmp_path
    (tmp_path / "a.pgn").write_text('[Result "1-0"]\n\n1. e4 *\n', encoding="utf-8")
    rep = bot._learn_cycle()
    assert rep["ran"] is False


def test_maybe_schedule_learning_requires_idle(monkeypatch):
    from concurrent.futures import Future

    from kestrel import lichess_bot as module
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", learn_every_games=1)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot._finished_since_learn = 1
    started = []

    class FakeThread:
        def __init__(self, target, daemon=False, name=None):
            started.append(target)

        def start(self):
            return None

    monkeypatch.setattr(module.threading, "Thread", FakeThread)
    bot._games["active"] = Future()
    bot._maybe_schedule_learning()
    assert started == []
    bot._games.clear()
    bot._maybe_schedule_learning()
    assert len(started) == 1


def test_learn_thread_is_joined_before_finish(monkeypatch):
    """run selector harus bisa menunggu worker learning selesai."""
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides(
        "lichess", learn_from_games=True, learn_every_games=1)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot._finished_since_learn = 1
    cycles: list[int] = []
    monkeypatch.setattr(
        bot, "_learn_cycle", lambda: cycles.append(1) or {"ran": True})
    bot._maybe_schedule_learning()
    bot._join_learn_thread(timeout=5)
    assert cycles == [1]

def test_parse_online_bots_filters_self_and_invalid():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", username="Azchess-1")
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    raw = (b'{"username":"bot-a"}\n\n{"username":"Azchess-1"}\n'
           b'{"username":"bad name"}\n')
    assert bot._parse_online_bots(raw) == ["bot-a"]
    assert bot._choose_seek_target(["bot-a", "Azchess-1", "bad name"]) == "bot-a"


def test_seek_once_http_403_disables_without_raise():
    import urllib.error
    from email.message import Message

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", username="Azchess-1")
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)

    def boom(*a, **k):
        raise urllib.error.HTTPError("url", 403, "forbidden", Message(), None)

    bot._get = boom
    bot._choose_seek_target = lambda names: "bot-a"
    rep = bot._seek_once()
    assert rep["status"] == "disabled"
    assert bot._auto_seek_enabled is False


def test_closed_challenge_releases_reservation():
    """Challenge batal/ditolak harus membebaskan slot seek."""
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", concurrency=2, max_games=5)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    for event_type in ("challengeCanceled", "challengeDeclined"):
        assert bot._reserve_game("C1") is True
        ev = {"type": event_type, "challenge": {"id": "C1"}}
        assert bot._release_closed_challenge(ev) is True
        assert "C1" not in bot._reservations

def test_seek_once_keeps_game_that_started_before_reservation(monkeypatch):
    """Challenge yang sudah jadi game tidak boleh dibatalkan seek."""
    from concurrent.futures import Future

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides(
        "lichess", username="Azchess-1", concurrency=1, max_games=5,
        auto_seek=True)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)

    def fake_get(path, **params):
        return b'{"username":"opponent"}\n'

    def fake_post(path, params="", data=None):
        posted.append(path)
        # Simulasikan gameStart yang masuk sebelum reservasi dicatat.
        if path == "/api/challenge/opponent":
            bot._games["CH1"] = Future()
        return b'{"id":"CH1"}'

    posted: list[str] = []
    monkeypatch.setattr(bot, "_get", fake_get)
    monkeypatch.setattr(bot, "_choose_seek_target", lambda names: "opponent")
    monkeypatch.setattr(bot, "_post", fake_post)
    rep = bot._seek_once()
    assert rep["status"] == "sent"
    assert not any("/cancel" in path for path in posted)

def test_choose_seek_target_honors_cooldown():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides(
        "lichess", username="Azchess-1", seek_cooldown_games=2)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot._seek_last_game["bot-a"] = bot.games_played + 2
    assert bot._choose_seek_target(["bot-a", "bot-b"]) == "bot-b"


def test_seek_once_http_429_is_rate_limited():
    import urllib.error
    from email.message import Message

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)

    def boom(*a, **k):
        raise urllib.error.HTTPError("url", 429, "too many", Message(), None)

    bot._get = boom
    assert bot._seek_once()["status"] == "rate_limited"


def test_seek_loop_stops_after_one_cycle(monkeypatch):
    import threading

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", seek_interval_seconds=0.01)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    stop = threading.Event()
    calls = []

    def once():
        calls.append(1)
        stop.set()
        return {"status": "sent"}

    monkeypatch.setattr(bot, "_seek_once", once)
    bot._seek_loop(stop)
    assert len(calls) == 1


def test_run_summary_keys_exist_offline():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    summary = bot._summary()
    for key in ("games_started", "games_finished", "learn_cycles",
                "promoted", "seek_sent", "seek_failed"):
        assert key in summary

def test_game_worker_exception_is_logged(capsys):
    """Future gagal harus terlihat di log tanpa mengubah status game."""
    from concurrent.futures import Future

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", learn_from_games=False)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    future = Future()
    bot._games["ERR1"] = future
    future.set_exception(RuntimeError("boom"))
    bot._on_game_done(future)
    out = capsys.readouterr().out
    assert "ERR1" in out
    assert "boom" in out

def test_play_game_closes_book_connection(tmp_path, monkeypatch):
    """Player per game membuka SQLite; koneksi harus ditutup usai game."""
    import sqlite3
    from typing import Any, cast

    from kestrel.arena import Player
    from kestrel.config import AppConfig
    from kestrel.knowledge import KnowledgeBase
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", move_time_ms=100, max_depth=2)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot.username = "Azchess-1"
    book = KnowledgeBase(tmp_path / "book.db")
    player = Player("closer", time_ms=100, max_depth=2, book=book)

    def fake_stream(path):
        yield {
            "type": "gameFull",
            "id": "CLOSE1",
            "white": {"name": "Azchess-1"},
            "black": {"name": "Opponent"},
            "initialFen": "startpos",
            "state": {"moves": "", "status": "started", "winner": None},
        }

    monkeypatch.setattr(bot.api, "stream", fake_stream)
    cast(Any, bot)._post = lambda path, params="", data=None: b"{}"
    bot.play_game("CLOSE1", player=player)
    try:
        book.db.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("koneksi book belum ditutup")

def test_log_summary_emits_parseable_json(capsys):
    """Ringkasan akhir harus bisa dibaca mesin, bukan teks bebas."""
    import json

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    bot.games_played = 3
    bot._games_finished = 2
    bot._seek_sent = 4
    bot._log_summary()
    out = capsys.readouterr().out
    payload = json.loads(out.split("[lichess] summary ", 1)[1].strip())
    assert payload == {
        "games_started": 3, "games_finished": 2, "learn_cycles": 0,
        "promoted": 0, "seek_sent": 4, "seek_failed": 0,
    }

def test_submit_game_builds_player_inside_worker_thread(tmp_path, monkeypatch):
    """Book SQLite harus dipakai di thread yang sama saat dibuat."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any, cast

    from kestrel.arena import Player
    from kestrel.config import AppConfig
    from kestrel.knowledge import KnowledgeBase
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", concurrency=1, max_games=2)
    bot = LichessBot(cfg, token=TEST_TOKEN, dry_run=True)
    bot.username = "Azchess-1"
    main_thread = threading.get_ident()
    maker_threads = []

    def make_player():
        maker_threads.append(threading.get_ident())
        book = KnowledgeBase(tmp_path / "book.db")
        return Player("worker", time_ms=100, max_depth=2, book=book)

    cast(Any, bot)._make_player = make_player

    def fake_stream(path):
        yield {
            "type": "gameFull",
            "id": "THREAD1",
            "white": {"name": "Azchess-1"},
            "black": {"name": "Opponent"},
            "initialFen": "startpos",
            "state": {"moves": "", "status": "started", "winner": None},
        }

    monkeypatch.setattr(bot.api, "stream", fake_stream)
    posted: list[str] = []
    cast(Any, bot)._post = lambda path, params="", data=None: posted.append(path) or b"{}"
    executor = ThreadPoolExecutor(max_workers=1)
    cast(Any, bot)._executor = executor
    assert bot._submit_game("THREAD1") is True
    future = bot._games["THREAD1"]
    future.result(timeout=10)
    executor.shutdown(wait=True)
    assert maker_threads and main_thread not in maker_threads
    assert any("/move/" in path for path in posted)


def test_claim_draw_400_falls_back_to_plain_move():
    """HTTP 400 saat claimDraw harus diretry sebagai langkah biasa."""
    import chess

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)
    board = chess.Board()
    board.can_claim_draw = lambda: True  # type: ignore[method-assign]

    def fake_choose_move(board, rng=None, player=None):
        return chess.Move.from_uci("e2e4"), {"score_cp": -500, "depth": 8, "nodes": 100}

    posted: list[tuple[str, str]] = []

    def fake_post(path, params="", data=None):
        posted.append((path, params))
        if params == "claimDraw=true":
            raise _http_error(400)
        return b"{}"

    bot.choose_move = fake_choose_move  # type: ignore[method-assign]
    bot._post = fake_post  # type: ignore[method-assign]
    outcome = bot._take_turn(
        "G400", board, None, {"n": 0}, player=object(), my_color=True)

    assert outcome is None
    assert posted == [
        ("/api/bot/game/G400/move/e2e4", "claimDraw=true"),
        ("/api/bot/game/G400/move/e2e4", ""),
    ]


def test_plain_move_http_400_returns_stale_summary():
    """Langkah ditolak Lichess (game sudah usang) bukan worker error."""
    import chess

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)

    def fake_choose_move(board, rng=None, player=None):
        return chess.Move.from_uci("e2e4"), {"score_cp": 0, "depth": 8, "nodes": 100}

    def fake_post(path, params="", data=None):
        raise _http_error(400)

    bot.choose_move = fake_choose_move  # type: ignore[method-assign]
    bot._post = fake_post  # type: ignore[method-assign]
    outcome = bot._take_turn(
        "G400B", chess.Board(), None, {"n": 0}, player=object(), my_color=True)

    assert outcome is not None
    assert outcome["status"] == "move_rejected_400"


def test_game_stream_timeout_is_handled_without_worker_error(monkeypatch):
    """Timeout stream harus ditutup sebagai stream_ended, bukan exception."""
    import chess

    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token=TEST_TOKEN, dry_run=True)

    def fake_stream(path):
        yield {
            "type": "gameFull",
            "id": "TOUT1",
            "white": {"name": "Azchess-1"},
            "black": {"name": "Opponent"},
            "initialFen": "startpos",
            "state": {"moves": "", "status": "started", "winner": None},
        }
        raise TimeoutError("read timed out")

    def fake_choose_move(board, rng=None, player=None):
        return chess.Move.from_uci("e2e4"), {"score_cp": 0, "depth": 8, "nodes": 100}

    monkeypatch.setattr(bot.api, "stream", fake_stream)
    bot.choose_move = fake_choose_move  # type: ignore[method-assign]
    bot._post = lambda path, params="", data=None: b"{}"  # type: ignore[method-assign]

    summary = bot.play_game("TOUT1", player=object())
    assert summary["status"] == "stream_ended"


def _http_error(code):
    from email.message import Message
    import urllib.error

    return urllib.error.HTTPError("https://lichess.org", code, "bad request", Message(), None)
