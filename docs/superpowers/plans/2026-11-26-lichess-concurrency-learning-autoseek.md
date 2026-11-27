# Lichess concurrency, learning, dan auto-seek Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Lichess bot play multiple opponents concurrently, learn from its own Lichess PGNs without endangering the champion, and actively seek games from online bots.

**Architecture:** Keep the synchronous `urllib` event loop and add `ThreadPoolExecutor` for per-game workers plus daemon threads for seek/learning. Serialize all non-stream REST calls behind one lock, isolate each game with its own `Player`, and put learning behind a full verification gate before any promotion.

**Tech Stack:** Python 3.11, `urllib`, `threading`, `concurrent.futures`, `python-chess`, pytest, Ruff, py_compile.

**Spec:** `docs/superpowers/specs/2026-11-26-lichess-concurrency-learning-autoseek-design.md`

## Global Constraints

- Python 3.11+, python-chess, NumPy. Tidak ada dependensi async baru.
- Host: aarch64 Linux, 8 vCPU, ~5.7 GiB RAM, sekitar 2 GiB tersedia; hindari build/test berat.
- Jangan menyentuh atau menampilkan token Lichess di `config.toml`.
- Lichess API: satu request non-stream pada satu waktu.
- Bot API hanya challenge games; tidak ada pool/tournament.
- Semua username dari API divalidasi `^[A-Za-z0-9_-]{1,50}$` sebelum dipakai sebagai URL path.
- `GET /api/bot/online?nb=N`: `nb` 1..512.
- `train_pgn` hanya memakai PGN dengan `Result` `1-0`, `0-1`, atau `1/2-1/2`.
- Verifikasi champion memakai `cfg.eval.games` penuh; tidak menurunkan budget.
- Verifikasi ringan saja: Ruff, `py_compile`, targeted pytest, LSP error-only.

## Review Focus

- Event stream idle/putus tapi auto-seek tetap jalan → test `_seek_loop` memanggil seek setelah `Event.wait`, bukan event stream.
- Dua `gameStart` bersamaan tidak memakai `Searcher` yang sama → test `choose_move(board, player=...)` memakai player eksplisit dan tidak mengisi `self._player`.
- `max_games` terlampaui karena challenge diterima sebelum gameStart → test `_reserve_game` menolak saat budget penuh.
- PGN resign/stream_ended ber-label `*` sehingga training skip → test `_result_header` dan `_save_pgn` untuk `resign_sent`.
- Token tanpa `challenge:write` membuat auto-seek crash → test HTTP 401/403 menonaktifkan auto-seek tanpa exception.

---

### Task 1: Konfigurasi dan CLI concurrency/seek/learning

**Files:**
- Modify: `kestrel/config.py:112-135`
- Modify: `config.toml:107-128`
- Modify: `kestrel/cli.py:298-312,396-405`
- Test: `tests/test_lichess_extensions.py`

**Interfaces:**
- Consumes: `AppConfig.with_overrides(section, **kwargs)`.
- Produces: `LichessConfig` fields `auto_seek: bool`, `seek_interval_seconds: float`, `seek_pool_size: int`, `seek_cooldown_games: int`, `seek_rated: bool`, `seek_clock_seconds: int`, `seek_increment_seconds: int`, `seek_color: str`, `learn_from_games: bool`, `learn_every_games: int`, `learn_min_games: int`, `learn_max_games: int`.
- Produces: `python -m kestrel.cli lichess --concurrency N` menimpa `lichess.concurrency`.

- [ ] **Step 1: Write failing tests**

Tambah test:
```python
def test_lichess_config_new_fields_and_cli_override():
    from kestrel.cli import build_parser, cmd_lichess
    from kestrel.config import AppConfig, load_config

    cfg = load_config()
    assert cfg.lichess.concurrency == 2
    assert cfg.lichess.auto_seek is True
    assert 1 <= cfg.lichess.seek_pool_size <= 512

    args = build_parser().parse_args(["lichess", "--concurrency", "4"])
    assert args.concurrency == 4
    cfg2 = cfg.with_overrides("lichess", concurrency=args.concurrency)
    assert cfg2.lichess.concurrency == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py::test_lichess_config_new_fields_and_cli_override -q`
Expected: FAIL karena field/flag belum ada.

- [ ] **Step 3: Implement config fields and CLI override**

Tambahkan field ke `LichessConfig` dengan default sesuai spec; isi nilai konkret di `[lichess] `config.toml tanpa mengubah baris token. Di parser `lichess`, tambah `p.add_argument("--concurrency", type=int)`. Di `cmd_lichess`, bila `args.concurrency` tidak None, panggil `cfg = cfg.with_overrides("lichess", concurrency=args.concurrency)`.

- [ ] **Step 4: Run test to verify it passes**

Run: perintah Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kestrel/config.py config.toml kestrel/cli.py tests/test_lichess_extensions.py
git commit -m "feat(lichess): add concurrency, seek, and learning config"
```

---

### Task 2: API lock, player isolation, dan reservation game

**Files:**
- Modify: `kestrel/lichess_bot.py:112-200,317-390`
- Test: `tests/test_lichess_extensions.py`

**Interfaces:**
- Consumes: `LichessBot.__init__(cfg, token=None, dry_run=False)`.
- Produces: `LichessBot._get(path, **params) -> bytes`, `_post(path, params="", data=None) -> bytes`, kedu serialize via `self._api_lock`.
- Produces: `LichessBot._reserve_game(key: str) -> bool` and `_release_game(key: str) -> None`.
- Produces: `LichessBot.choose_move(board, rng=None, player=None)`.
- Produces: `LichessBot.play_game(game_id, player=None)`.

- [ ] **Step 1: Write failing tests**

```python
def test_choose_move_uses_explicit_player(tmp_path):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot
    import chess

    cfg = AppConfig()
    bot = LichessBot(cfg, token="x", dry_run=True)
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

def test_reserve_game_respects_max_games():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", max_games=1)
    bot = LichessBot(cfg, token="x", dry_run=True)
    assert bot._reserve_game("a") is True
    assert bot._reserve_game("b") is False
    bot._release_game("a")
    assert bot._reserve_game("b") is True
```

`FakePlayer` hanya perlu `play`; Ruff boleh minta `__init__` tidak dipakai.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py -q -k "explicit_player or reserve_game"`
Expected: FAIL because signatures/helper belum ada.

- [ ] **Step 3: Implement locks and signatures**

Tambah `threading`, `concurrent.futures`, dan tipe `Callable` ke imports. Di `__init__`, buat `self._api_lock = threading.RLock()`, `self._games_lock = threading.Lock()`, `self._learn_lock = threading.Lock()`, `self._games: dict[str, Future] = {}`, `self._reservations: dict[str, float] = {}`, counter hasil, dan `self._executor: ThreadPoolExecutor | None = None`. Implement `_get`, `_post`, `_reserve_game`, `_release_game` dengan lock. Ubah `choose_move`, `_take_turn`, `play_game` menerima `player` eksplisit; live path tidak lagi menulis `self._player`.

- [ ] **Step 4: Run test to verify it passes**

Run: perintah Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kestrel/lichess_bot.py tests/test_lichess_extensions.py
git commit -m "feat(lichess): serialize API calls and isolate game players"
```

---

### Task 3: Dispatch gameStart tanpa blocking dan PGN result benar

**Files:**
- Modify: `kestrel/lichess_bot.py:202-260,268-320,373-390`
- Test: `tests/test_lichess_extensions.py`

**Interfaces:**
- Consumes: `_reserve_game`, `_release_game`, `play_game(game_id, player=None)`.
- Produces: `LichessBot._result_header(game_full, state, summary) -> str`.
- Produces: `LichessBot._submit_game(game_id) -> bool`.
- Produces: `LichessBot._on_game_done(future)` callback membersihkan `self._games` dan counter.

- [ ] **Step 1: Write failing tests**

```python
def test_result_header_uses_winner_and_resign_summary():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    bot = LichessBot(AppConfig(), token="x", dry_run=True)
    game_full = {"white": {"name": "A"}, "black": {"name": "B"}}
    assert bot._result_header(game_full, {"winner": "black"}, {"my_color": "white"}) == "0-1"
    assert bot._result_header(game_full, {}, {"status": "resign_sent", "my_color": "white"}) == "0-1"
    assert bot._result_header(game_full, {"status": "draw"}, {"my_color": "white"}) == "1/2-1/2"

def test_submit_game_uses_executor(tmp_path):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot
    from concurrent.futures import ThreadPoolExecutor

    cfg = AppConfig().with_overrides("lichess", max_games=2)
    bot = LichessBot(cfg, token="x", dry_run=True)
    bot._executor = ThreadPoolExecutor(max_workers=2)
    bot.play_game = lambda game_id, player=None: "ok"
    assert bot._submit_game("g1") is True
    assert bot._submit_game("g2") is True
    assert bot._submit_game("g3") is False
    bot._executor.shutdown(wait=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py -q -k "result_header or submit_game"`
Expected: FAIL.

- [ ] **Step 3: Implement result header and submit path**

Implement `_result_header`: normalisasi `winner` (`white`, `black`, `1-0`, `0-1`), status draw (`draw`, `stalemate`, `agreement`, `timeout`/`insufficient` bila tidak ada winner), dan fallback summary `resign_sent` = lawan menang. `_save_pgn` memakai helper ini. Di `_take_turn`, summary resign menyertakan `winner` lawan. `play_game` memanggil `_save_pgn` di semua jalur akhir dengan `last_state` terakhir. `run()` membuat executor `max_workers=max(1, cfg.lichess.concurrency)`; branch `gameStart` memanggil `_submit_game(gid)`; `finally` shutdown executor.

- [ ] **Step 4: Run test to verify it passes**

Run: perintah Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kestrel/lichess_bot.py tests/test_lichess_extensions.py
git commit -m "feat(lichess): dispatch games concurrently and label PGN results"
```

---

### Task 4: Learning cycle dari PGN Lichess

**Files:**
- Modify: `kestrel/lichess_bot.py` (tambah helper learning)
- Test: `tests/test_lichess_extensions.py`

**Interfaces:**
- Consumes: `cfg.learning`, `cfg.reda`, `cfg.eval`, `cfg.train`, `CheckpointStore`, `KnowledgeBase`, `train_pgn`, `reda_adapt_model`, `verify_challenger`.
- Produces: `LichessBot._pgn_candidates() -> list[Path]`.
- Produces: `LichessBot._write_learning_corpus(paths, target) -> Path`.
- Produces: `LichessBot._learn_cycle() -> dict`.
- Produces: `LichessBot._maybe_schedule_learning() -> None`.

- [ ] **Step 1: Write failing tests**

```python
def test_pgn_candidates_exclude_corpus_and_dedup(tmp_path):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig()
    bot = LichessBot(cfg, token="x", dry_run=True)
    bot.pgn_dir = tmp_path
    for name in ["a.pgn", "b.pgn", ".learn_corpus.pgn", "a.pgn.tmp"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [p.name for p in bot._pgn_candidates()] == ["a.pgn", "b.pgn"]

def test_write_learning_corpus_is_valid_pgn(tmp_path):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot
    import chess.pgn

    cfg = AppConfig()
    bot = LichessBot(cfg, token="x", dry_run=True)
    p1 = tmp_path / "a.pgn"; p2 = tmp_path / "b.pgn"
    p1.write_text('[Result "1-0"]\n\n1. e4 e5 *\n\n', encoding="utf-8")
    p2.write_text('[Result "0-1"]\n\n1. d4 d5 *\n\n', encoding="utf-8")
    target = bot._write_learning_corpus([p1, p2], tmp_path / ".learn_corpus.pgn")
    games = []
    with target.open(encoding="utf-8") as f:
        while game := chess.pgn.read_game(f):
            games.append(game)
    assert len(games) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py -q -k "pgn_candidates or learning_corpus"`
Expected: FAIL.

- [ ] **Step 3: Implement learning helpers**

`_pgn_candidates` mengembalikan `*.pgn` terurut, abaikan nama diawali `.`, dan dedup by stem. `_write_learning_corpus` menulis gabungan isi PGN ke path target via file `.tmp` lalu `os.replace`. `_learn_cycle` mengambil maks `learn_max_games` PGN terakhir; bila < `learn_min_games`, return `{"ran": False}`; else jalankan knowledge ingest (jika enabled), `train_pgn`, merge, REDA (jika enabled), `verify_challenger(..., quick=True)`, save candidate, dan promote hanya bila `decision == PROMOTE` dan `cfg.train.auto_promote`. `_maybe_schedule_learning` hanya jalan bila `learn_from_games`, `learn_every_games > 0`, tidak ada game aktif, dan lock available.

- [ ] **Step 4: Run test to verify it passes**

Run: perintah Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kestrel/lichess_bot.py tests/test_lichess_extensions.py
git commit -m "feat(lichess): learn from completed Lichess PGNs"
```

---

### Task 5: Auto-seek dari bot online

**Files:**
- Modify: `kestrel/lichess_bot.py` (helper seek + loop)
- Test: `tests/test_lichess_extensions.py`

**Interfaces:**
- Consumes: `_get`, `_post`, `_reserve_game`, `_release_game`, `parse_sse`.
- Produces: `VALID_SEEK_USERNAME` (`re.Pattern`) and `LichessBot._parse_online_bots(raw: bytes) -> list[str]`.
- Produces: `LichessBot._choose_seek_target(names: list[str]) -> str | None`.
- Produces: `LichessBot._seek_once() -> dict`.
- Produces: `LichessBot._seek_loop(stop: threading.Event) -> None`.

- [ ] **Step 1: Write failing tests**

```python
def test_parse_online_bots_filters_self_and_invalid():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", username="Azchess-1")
    bot = LichessBot(cfg, token="x", dry_run=True)
    raw = b'{"username":"bot-a"}\n\n{"username":"Azchess-1"}\n{"username":"bad name"}\n'
    assert bot._parse_online_bots(raw) == ["bot-a"]
    assert bot._choose_seek_target(["bot-a", "Azchess-1", "bad name"]) == "bot-a"

def test_seek_once_http_403_disables_without_raise():
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot
    import urllib.error

    cfg = AppConfig().with_overrides("lichess", username="Azchess-1")
    bot = LichessBot(cfg, token="x", dry_run=True)
    def boom(*a, **k):
        raise urllib.error.HTTPError("url", 403, "forbidden", {}, None)
    bot._get = boom
    bot._choose_seek_target = lambda names: "bot-a"
    rep = bot._seek_once()
    assert rep["status"] == "disabled"
    assert bot._auto_seek_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py -q -k "online_bots or 403_disables"`
Expected: FAIL.

- [ ] **Step 3: Implement seek helper and loop**

Tambah `import re`, `urllib.parse.urlencode`. `_parse_online_bots` memakai `parse_sse(raw.splitlines())`, ambil `username` valid. `_choose_seek_target` filter username sendiri dan cooldown berdasarkan `self._seek_last_game[username]`. `_seek_once` cek slot, ambil `/api/bot/online` via `_get(nb=...)`, pilih target, POST `/api/challenge/{username}` dengan params `clock.limit`, `clock.increment`, `rated`, `variant=standard`, `color`; catat challenge id ke `_reservations`; 401/403 set `_auto_seek_enabled = False`; 429 return `{"status": "rate_limited"}`; error lain return `{"status": "error"}`. `_seek_loop` menunggu dengan `stop.wait(interval)` lalu `_seek_once`; mulai/stop dari `run()` dengan daemon thread.

- [ ] **Step 4: Run test to verify it passes**

Run: perintah Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kestrel/lichess_bot.py tests/test_lichess_extensions.py
git commit -m "feat(lichess): seek games from online bots"
```

---

### Task 6: Integrasi run loop dan verifikasi ringan akhir

**Files:**
- Modify: `kestrel/lichess_bot.py`
- Modify: `docs/superpowers/specs/2026-11-26-lichess-concurrency-learning-autoseek-design.md`
- Test: `tests/test_lichess_extensions.py`

**Interfaces:**
- Consumes: semua helper dari Task 1-5.
- Produces: `LichessBot.run()` mengembalikan ringkasan `games_started`, `games_finished`, `learn_cycles`, `promoted`, `seek_sent`, `seek_failed`.

- [ ] **Step 1: Write failing test**

```python
def test_run_summary_keys_exist_offline(monkeypatch):
    from kestrel.config import AppConfig
    from kestrel.lichess_bot import LichessBot

    cfg = AppConfig().with_overrides("lichess", max_games=0)
    bot = LichessBot(cfg, token="x", dry_run=True)
    summary = bot._summary()
    for key in ("games_started", "games_finished", "learn_cycles", "promoted", "seek_sent", "seek_failed"):
        assert key in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py::test_run_summary_keys_exist_offline -q`
Expected: FAIL.

- [ ] **Step 3: Wire run loop**

`run()` membuat executor, thread seek, dan thread learning sesuai config; branch challenge memakai `_reserve_game` sebelum accept; gameStart memakai `_submit_game`; `finally` set stop event, join seek thread, shutdown executor, tunggu future selesai, dan kembalikan `self._summary()`. Update doc status dari `usulan` menjadi `disetujui` dan tanggal eksekusi.

- [ ] **Step 4: Run light verification**

Run:
```bash
python3 -m py_compile kestrel/lichess_bot.py kestrel/config.py kestrel/cli.py tests/test_lichess_extensions.py
PYTHONPATH=/tmp/kestrel-tools python3 -m ruff check --config /root/.pi/npm/node_modules/pi-lens/config/ruff/core.toml kestrel/lichess_bot.py kestrel/config.py kestrel/cli.py tests/test_lichess_extensions.py
PYTHONPATH=/tmp/kestrel-pytest python3 -m pytest tests/test_lichess_extensions.py tests/test_unified.py -q -k "lichess or config_new_fields or pgn or seek or reserve or submit"
```
Expected: compile OK, Ruff `All checks passed!`, targeted tests PASS.

- [ ] **Step 5: LSP error check**

Run lens diagnostics error-only pada 4 file yang diubah. Expected: 0 error.

- [ ] **Step 6: Commit**

```bash
git add kestrel/lichess_bot.py docs/superpowers/specs/2026-11-26-lichess-concurrency-learning-autoseek-design.md
git commit -m "feat(lichess): integrate concurrent play, learning, and auto-seek"
```

## Self-review checklist

- [x] Spec coverage: D-1..D-7 and all three user asks have tasks.
- [x] Step scan: each step has one verifiable output.
- [x] Type consistency: helper names/signatures consistent across tasks.
- [x] Review Focus: each risk has a test in the owning task.
- [x] Proportion: plan is shorter than a full transcript; no function bodies except test examples.
