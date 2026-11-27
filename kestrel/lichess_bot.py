"""KESTREL 1.6 — Bot Lichess: mainkan engine Kestrel di lichess.org.

Fitur jadi "AI Lichess": koneksi ke Bot API lichess (REST + stream SSE via
``urllib``, tanpa dependensi tambahan), menerima challenge sesuai aturan
config, bermain dengan Searcher/Player Kestrel (champion model + knowledge
book + jaminan min-depth), resign/claim draw sesuai ambang, dan mencatat
PGN setiap game ke ``results/lichess/``.

Persiapan akun (sekali):
1. Buat akun lichess khusus bot dan aktifkan email.
2. Buat token OAuth dengan scope ``bot:play`` di
   https://lichess.org/account/oauth/token
3. ``python -m kestrel.cli lichess --token LICH_TOKEN`` — pada panggilan
   pertama bot menawarkan upgrade akun ke tipe BOT (wajib oleh API untuk
   memakai /api/bot/*).

Catatan jujur: modul ini klien API minimal yang mengikuti alur resmi
lichess-bot. Game parallel memakai ``ThreadPoolExecutor`` dan Player per game;
semua REST non-stream diserialisasi sesuai panduan Lichess.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path

import chess
import chess.pgn

from .arena import Player
from .config import AppConfig
from .versioning import CheckpointStore

BASE = "https://lichess.org"
VALID_SEEK_USERNAME = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


class LichessAPI:
    """Klien REST + SSE minimal."""

    def __init__(self, token: str):
        self.token = token

    @staticmethod
    def _url(path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("path API harus diawali '/'")
        return f"{BASE}{path}"

    def request(self, method: str, path: str, params: str = "",
                data: bytes | None = None, timeout: int = 15) -> bytes:
        url = self._url(path) + (f"?{params}" if params else "")
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310  # URL tervalidasi _url
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("User-Agent", "kestrel-bot/1.6")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # host tetap lichess.org
            return resp.read()

    def get(self, path: str, **params) -> bytes:
        query = "&".join(f"{k}={v}" for k, v in params.items() if v)
        return self.request("GET", path, params=query)

    def post(self, path: str, params: str = "", data: bytes | None = None) -> bytes:
        return self.request("POST", path, params=params, data=data or b"")

    def stream(self, path: str):
        """Yield event SSE satu per satu (dict JSON), reconnect aman dipanggil
        ulang oleh pemanggil."""
        req = urllib.request.Request(self._url(path))  # noqa: S310  # URL tervalidasi _url
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("User-Agent", "kestrel-bot/1.6")
        resp = urllib.request.urlopen(req, timeout=20)  # noqa: S310  # host tetap lichess.org
        try:
            yield from parse_sse(resp)
        finally:
            resp.close()


def parse_sse(lines):
    """Parser stream lichess yang toleran terhadap NDJSON dan SSE.

    Endpoints Bot API (``/api/stream/event`` dan ``/api/bot/game/stream/...``)
    mengirim NDJSON: satu objek JSON per baris, dengan baris kosong sebagai
    keepalive. Format SSE (``data: {json}`` / ``data:{json}``) tetap diterima
    untuk kompatibilitas dengan wire lama.
    """
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line or line.startswith(":"):
            continue  # keepalive / komentar SSE
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
        elif line.startswith("{"):
            payload = line  # NDJSON dari Bot API lichess
        else:
            continue  # event: / ping / baris non-JSON
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _initial_fen(value: str | None) -> str:
    """Lichess mengirim ``startpos`` untuk posisi awal, bukan FEN penuh."""
    if not value or value == "startpos":
        return chess.STARTING_FEN
    return value


def _user_name(payload: dict, key: str) -> str:
    """Ambil nama user dari objek gameFull yang bisa berupa None/dict."""
    value = payload.get(key)
    if not isinstance(value, dict):
        return "?"
    return str(value.get("name") or "?")


def _account_rating(profile: dict | None) -> int:
    """Rating sendiri untuk kebijakan challenge.

    Akun BOT baru memiliki rating 3000 *provisional* dengan 0 game. Itu
    bukan kekuatan nyata dan membuat semua tantangan manusia tertolak oleh
    ``max_rating_gap``. Perlakukan sebagai "belum diketahui" (0).
    """
    perfs = (profile or {}).get("perfs") or {}
    blitz = perfs.get("blitz") or {}
    if blitz.get("prov") or not blitz.get("games"):
        return 0
    return int(blitz.get("rating") or 0)


class LichessBot:
    def __init__(self, cfg: AppConfig, token: str | None = None, dry_run: bool = False):
        self.cfg = cfg
        self.lic = cfg.lichess
        self.token = token or os.environ.get("KESTREL_LICHESS_TOKEN", "")
        self.api = LichessAPI(self.token)
        self.dry_run = bool(dry_run)
        self.games_played = 0
        self.my_rating = 0
        self._player = None
        self.pgn_dir = cfg.results_path / self.lic.pgn_dir
        self.pgn_dir.mkdir(parents=True, exist_ok=True)
        self.username = self.lic.username or ""
        self._api_lock = threading.RLock()
        self._games_lock = threading.RLock()
        self._learn_lock = threading.Lock()
        self._games: dict[str, Future] = {}
        self._reservations: dict[str, float] = {}
        self._games_finished = 0
        self._learn_cycles = 0
        self._promoted = 0
        self._seek_sent = 0
        self._seek_failed = 0
        self._executor: ThreadPoolExecutor | None = None
        self._auto_seek_enabled = bool(self.lic.auto_seek)
        self._seek_last_game: dict[str, int] = {}
        self._learn_state_path = self.pgn_dir / ".learn_state.json"
        self._learned_pgn_ids: set[str] = set()
        try:
            raw = json.loads(self._learn_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            raw = None
        if isinstance(raw, list):
            self._learned_pgn_ids = {str(x) for x in raw}
        self._finished_since_learn = 0
        self._learn_thread: threading.Thread | None = None

    # ---------------- thread-safe REST helpers ----------------
    def _get(self, path: str, **params) -> bytes:
        """GET yang diserialisasi; Docs Lichess: satu request pada satu waktu."""
        with self._api_lock:
            return self.api.get(path, **params)

    def _post(self, path: str, params: str = "", data: bytes | None = None) -> bytes:
        """POST yang diserialisasi; user-agent sudah ditambahkan LichessAPI."""
        with self._api_lock:
            return self.api.post(path, params=params, data=data)

    # ---------------- game budget ----------------
    def _reserve_game(self, key: str) -> bool:
        """Reserve slot untuk challenge/game; False bila budget habis."""
        if not key:
            return False
        with self._games_lock:
            # gameStart dapat mendahului reservasi seek. Game yang sudah
            # aktif/telah direservasi tidak boleh dihitung dua kali.
            if key in self._games or key in self._reservations:
                return True
            active = sum(1 for fut in self._games.values() if not fut.done())
            cap = max(1, int(self.lic.concurrency))
            if active + len(self._reservations) >= cap:
                return False
            limit = int(self.lic.max_games)
            if limit and self.games_played + len(self._reservations) >= limit:
                return False
            self._reservations[key] = time.monotonic()
            return True

    def _release_game(self, key: str) -> None:
        with self._games_lock:
            self._reservations.pop(key, None)

    def _release_closed_challenge(self, ev: dict) -> bool:
        """Lepaskan slot untuk challenge yang batal/ditolak lawan."""
        if ev.get("type") not in ("challengeCanceled", "challengeDeclined"):
            return False
        cid = str((ev.get("challenge") or {}).get("id") or "")
        if not cid:
            return False
        with self._games_lock:
            if cid not in self._reservations:
                return False
            self._reservations.pop(cid, None)
        print(f"[lichess] challenge {cid} ditutup; slot dilepas", flush=True)
        return True

    # ---------------- game dispatch ----------------
    def _submit_game(self, game_id: str) -> bool:
        """Kirim game ke worker pool; False bila slot tidak tersedia."""
        with self._games_lock:
            if game_id in self._games:
                return False
            reserved = game_id in self._reservations
            if not reserved and not self._reserve_game(game_id):
                return False
            self._reservations.pop(game_id, None)
            self.games_played += 1
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, int(self.lic.concurrency)),
                    thread_name_prefix="lichess-game")
            # Player (dan koneksi book SQLite-nya) dibuat di worker thread
            # supaya dipakai di thread yang sama saat pembuatannya.
            future = self._executor.submit(self.play_game, game_id)
            future.add_done_callback(self._on_game_done)
            self._games[game_id] = future
            return True

    def _on_game_done(self, future) -> None:
        with self._games_lock:
            game_id = next((gid for gid, fut in self._games.items() if fut is future), None)
            if game_id is not None:
                self._games.pop(game_id, None)
                self._reservations.pop(game_id, None)
            self._games_finished += 1
        try:
            err = future.exception()
        except CancelledError:
            err = None
        if err is not None:
            label = game_id or "?"
            print(
                f"[lichess] game {label} worker error: "
                f"{type(err).__name__}: {err}",
                flush=True,
            )
        # 1.7 bugfix: increment counter belajar dilakukan DI DALAM lock yang
        # sama dengan reset di _maybe_schedule_learning agar game yang selesai
        # bersamaan tidak saling menghilangkan pemicu siklus pembelajaran.
        with self._games_lock:
            self._finished_since_learn += 1
        self._maybe_schedule_learning()

    # ---------------- learning from completed Lichess games ----------------
    def _pgn_candidates(self) -> list[Path]:
        """PGN selesai yang belum/kembali dipakai; korpus internal dilewati."""
        out: list[Path] = []
        seen: set[str] = set()
        for path in sorted(self.pgn_dir.glob("*.pgn")):
            if path.name.startswith(".") or path.stem in seen:
                continue
            seen.add(path.stem)
            out.append(path)
        return out

    def _write_learning_corpus(self, paths: list[Path], target: Path) -> Path:
        """Gabungkan PGN ke satu file secara atomik."""
        target.parent.mkdir(parents=True, exist_ok=True)
        chunks = [p.read_text(encoding="utf-8", errors="replace").strip()
                  for p in paths]
        body = "\n\n".join(chunk for chunk in chunks if chunk) + "\n"
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, target)
        return target

    def _learn_cycle(self) -> dict:
        """Latih/evaluasi kandidat dari PGN Lichess; champion aman sampai verifikasi."""
        if not self.lic.learn_from_games:
            return {"ran": False, "reason": "disabled"}
        fresh = [p for p in self._pgn_candidates()
                 if p.stem not in self._learned_pgn_ids]
        minimum = max(0, int(self.lic.learn_min_games))
        if len(fresh) < minimum:
            return {"ran": False, "reason": "insufficient_games",
                    "games": len(fresh), "need": minimum}
        limit = int(self.lic.learn_max_games)
        if limit > 0:
            fresh = fresh[-limit:]
        corpus = self._write_learning_corpus(
            fresh, self.pgn_dir / ".learn_corpus.pgn")
        try:
            from .knowledge import KnowledgeBase
            from .learning import train_pgn
            from .reda_pipeline import reda_adapt_model
            from .trainer import _champion_candidate_merge
            from .verification import save_report, verify_challenger

            if self.cfg.knowledge.enabled:
                book = KnowledgeBase(
                    self.cfg.knowledge_db_path,
                    self.cfg.knowledge.book_max_plies,
                    self.cfg.knowledge.book_min_games,
                    self.cfg.knowledge.book_max_moves_per_position)
                try:
                    book.learn_from_pgn(fresh)
                finally:
                    book.db.close()
            trained, training_report = train_pgn(
                fresh,
                epochs=self.cfg.learning.epochs,
                lr=self.cfg.learning.lr,
                seed=self.cfg.learning.seed,
                max_samples_per_game=self.cfg.learning.max_samples_per_game,
                validation_fraction=self.cfg.learning.validation_fraction,
                batch_size=self.cfg.learning.batch_size,
                patience=self.cfg.learning.patience)
            store = CheckpointStore(self.cfg.results_path)
            champion = store.load_champion()
            candidate = _champion_candidate_merge(champion, trained)
            reda_report = {"ran": False}
            if self.cfg.reda.enabled:
                candidate, reda_report = reda_adapt_model(
                    corpus, candidate,
                    limit=self.cfg.reda.max_candidate_positions_per_batch,
                    depth=self.cfg.reda.verification_depth,
                    time_ms=self.cfg.reda.verification_time_ms,
                    max_active=self.cfg.reda.max_active_features,
                    max_abs_weight=self.cfg.reda.max_abs_residual_weight,
                    learning_rate=self.cfg.reda.learning_rate,
                    l2=self.cfg.reda.l2,
                    min_improvement=self.cfg.reda.min_validation_improvement,
                    steps=self.cfg.reda.verification_steps,
                    memory_cap=self.cfg.reda.max_tracked_positions)
            report = verify_challenger(champion, candidate, self.cfg, quick=True)
            label = f"lichess_{int(time.time())}"
            report_path = save_report(report, self.cfg.results_path, label)
            candidate_path = store.save_candidate(
                candidate, label,
                metadata={"source": "lichess_pgn", "games": len(fresh),
                          "decision": report.get("decision"),
                          "training": training_report,
                          "reda": reda_report})
            promoted = False
            if report.get("decision") == "PROMOTE" and self.cfg.train.auto_promote:
                store.promote(candidate_path, "lichess_pgn_verified", report)
                promoted = True
                self._promoted += 1
        except Exception as exc:  # noqa: BLE001  # background cycle tak boleh matikan bot
            print(f"[lichess] learn cycle gagal: {exc}", flush=True)
            return {"ran": False, "reason": "error", "error": str(exc)}
        self._learned_pgn_ids.update(p.stem for p in fresh)
        state_tmp = self._learn_state_path.with_name(
            self._learn_state_path.name + ".tmp")
        with contextlib.suppress(OSError):
            state_tmp.write_text(
                json.dumps(sorted(self._learned_pgn_ids)), encoding="utf-8")
            os.replace(state_tmp, self._learn_state_path)
        self._learn_cycles += 1
        return {"ran": True, "games": len(fresh),
                "decision": report.get("decision"), "promoted": promoted,
                "report": str(report_path), "candidate": str(candidate_path)}

    def _maybe_schedule_learning(self) -> None:
        if not self.lic.learn_from_games:
            return
        every = int(self.lic.learn_every_games)
        if every < 1:
            return
        with self._games_lock:
            if any(not fut.done() for fut in self._games.values()):
                return
            if self._finished_since_learn < every:
                return
            if not self._learn_lock.acquire(blocking=False):
                return
            self._finished_since_learn = 0

        def worker() -> None:
            try:
                self._learn_cycle()
            except Exception as exc:  # noqa: BLE001  # worker background
                print(f"[lichess] learn worker error: {exc}", flush=True)
            finally:
                self._learn_lock.release()

        thread = threading.Thread(
            target=worker, daemon=True, name="lichess-learn")
        self._learn_thread = thread
        thread.start()

    def _join_learn_thread(self, timeout: float = 5.0) -> None:
        """Tunggu siklus learning terakhir sebelum proses selesai."""
        thread = self._learn_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # ---------------- auto-seek dari bot online ----------------
    def _parse_online_bots(self, raw: bytes) -> list[str]:
        """Ekstrak username valid dari NDJSON `/api/bot/online`."""
        out: list[str] = []
        seen: set[str] = set()
        me = self.username.strip().lower()
        for ev in parse_sse(raw.splitlines()):
            name = str(ev.get("username") or "").strip()
            key = name.lower()
            if not VALID_SEEK_USERNAME.fullmatch(name) or key == me or key in seen:
                continue
            seen.add(key)
            out.append(name)
        return out

    def _choose_seek_target(self, names: list[str]) -> str | None:
        """Pilih lawan acak yang tidak sedang dalam cooldown."""
        cooldown = max(0, int(self.lic.seek_cooldown_games))
        me = self.username.strip().lower()
        ready = [name for name in names
                 if VALID_SEEK_USERNAME.fullmatch(name)
                 and name.lower() != me
                 and (self.games_played - self._seek_last_game.get(name.lower(), -cooldown))
                 >= cooldown]
        return random.choice(ready) if ready else None

    def _seek_once(self) -> dict:
        """Satu siklus: daftar bot online -> kirim satu challenge santai."""
        if not self._auto_seek_enabled:
            return {"status": "disabled"}
        with self._games_lock:
            active = sum(1 for fut in self._games.values() if not fut.done())
            limit = int(self.lic.max_games)
            if limit and self.games_played + active + len(self._reservations) >= limit:
                return {"status": "no_slot"}
            if active >= max(1, int(self.lic.concurrency)):
                return {"status": "busy"}
            cutoff = time.monotonic() - max(60.0, 2.0 * float(self.lic.seek_interval_seconds))
            for key, started in list(self._reservations.items()):
                if started < cutoff:
                    self._reservations.pop(key, None)
        try:
            raw = self._get("/api/bot/online",
                            nb=max(1, min(512, int(self.lic.seek_pool_size))))
        except urllib.error.HTTPError as exc:
            return self._seek_http_error(exc)
        except urllib.error.URLError as exc:
            self._seek_failed += 1
            return {"status": "error", "reason": type(exc).__name__}
        target = self._choose_seek_target(self._parse_online_bots(raw))
        if target is None:
            return {"status": "no_target"}
        params = urllib.parse.urlencode({
            "clock.limit": max(1, int(self.lic.seek_clock_seconds)),
            "clock.increment": max(0, int(self.lic.seek_increment_seconds)),
            "rated": "true" if self.lic.seek_rated else "false",
            "variant": "standard",
            "color": self.lic.seek_color if self.lic.seek_color in ("white", "black") else "random",
        })
        try:
            response = self._post(f"/api/challenge/{target}", params=params)
        except urllib.error.HTTPError as exc:
            return self._seek_http_error(exc)
        except urllib.error.URLError as exc:
            self._seek_failed += 1
            return {"status": "error", "reason": type(exc).__name__}
        try:
            payload = json.loads(response) if response else {}
        except ValueError:
            payload = {}
        challenge_id = str(payload.get("id") or "")
        if challenge_id and not self._reserve_game(challenge_id):
            with contextlib.suppress(urllib.error.HTTPError, urllib.error.URLError):
                self._post(f"/api/challenge/{challenge_id}/cancel")
            self._seek_failed += 1
            return {"status": "no_slot", "target": target}
        self._seek_last_game[target.lower()] = self.games_played
        self._seek_sent += 1
        print(f"[lichess] seek -> {target}", flush=True)
        return {"status": "sent", "target": target, "id": challenge_id}

    def _seek_http_error(self, exc: urllib.error.HTTPError) -> dict:
        self._seek_failed += 1
        if exc.code in (401, 403):
            self._auto_seek_enabled = False
            print("[lichess] auto-seek disabled (scope)", flush=True)
            return {"status": "disabled", "code": exc.code}
        if exc.code == 429:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            return {"status": "rate_limited", "code": exc.code,
                    "retry_after": retry_after}
        return {"status": "error", "code": exc.code}

    def _seek_loop(self, stop: threading.Event) -> None:
        retry = 0
        while not stop.is_set():
            interval = max(1.0, float(self.lic.seek_interval_seconds))
            if stop.wait(interval):
                break
            if not self._auto_seek_enabled:
                return
            report = self._seek_once()
            if report.get("status") == "rate_limited":
                retry += 1
                stop.wait(min(60.0, 2.0 ** retry))
                continue
            retry = 0

    # ---------------- player setup ----------------
    def _make_player(self) -> Player:
        store = CheckpointStore(self.cfg.results_path)
        learned = None
        if store.champion_model_path.exists():
            try:
                from .learning import OutcomeModel
                learned = OutcomeModel().load(store.champion_model_path)
            except (ValueError, KeyError, TypeError):
                learned = None
        book = None
        if self.lic.book_moves and self.cfg.knowledge_db_path.exists():
            try:
                from .knowledge import KnowledgeBase
                book = KnowledgeBase(self.cfg.knowledge_db_path,
                                     self.cfg.knowledge.book_max_plies,
                                     max(1, self.cfg.knowledge.book_min_games))
            except Exception:
                book = None
        return Player("Kestrel-Lichess", time_ms=self.lic.move_time_ms,
                      max_depth=self.lic.max_depth, learned=learned, book=book,
                      min_depth=self.lic.min_depth, grace_ms=self.lic.grace_ms)

    # ---------------- move decision (testable tanpa jaringan) ----------------
    def choose_move(self, board: chess.Board, rng=None, player=None):
        """Kembalikan (move, info). Player eksplisit dipakai game parallel."""
        if player is None:
            if self._player is None:
                self._player = self._make_player()
            player = self._player
        mv, info = player.play(board, 0.0, rng or random.Random(7))
        return mv, info

    def _eval_for_me(self, board: chess.Board, info: dict) -> int:
        """Skor terakhir dari perspektif kita (cp, side-to-move based)."""
        score_stm = info.get("score_cp", 0)
        return score_stm  # setelah push, hasil pencarian relatif sisi yang gerak

    # ---------------- PGN result ----------------
    def _result_header(self, game_full: dict | None, state: dict | None,
                       summary: dict) -> str:
        """Normalisasi hasil Lichess ke header PGN Standar."""
        del game_full  # disimpan untuk stabilitas interface helper
        state = state or {}
        summary = summary or {}
        winner = state.get("winner") or summary.get("winner")
        if isinstance(winner, str):
            norm = winner.strip().lower()
            if norm in ("white", "w", "1-0"):
                return "1-0"
            if norm in ("black", "b", "0-1"):
                return "0-1"
        status = str(state.get("status") or summary.get("status") or "")
        if status in ("draw", "stalemate", "agreement", "timeout",
                      "insufficientmaterial"):
            return "1/2-1/2"
        if status in ("resign_sent", "resign", "outoftime", "cheat", "mate"):
            my_color = str(summary.get("my_color") or "white")
            opponent = "black" if my_color == "white" else "white"
            winner_side = opponent if status in ("resign_sent", "resign") else my_color
            return "1-0" if winner_side == "white" else "0-1"
        return "*"

    # ---------------- challenge policy ----------------
    def _is_outgoing_challenge(self, ev: dict) -> bool:
        """Challenge keluar (dari auto-seek) bukan tantangan yang harus dijawab."""
        ch = ev.get("challenge") or {}
        return str(ch.get("direction") or "").lower() == "out"

    def _accept_challenge(self, ev: dict) -> bool:
        """Terima semua tantangan dari lawan.

        Kebijakan 1.6: tidak ada filter variant, speed, rating, bot/human,
        atau rating gap. Satu-satunya penolakan adalah self-challenge karena
        itu bukan lawan.
        """
        ch = ev.get("challenge") or {}
        challenger = ch.get("challenger") or {}
        return challenger.get("name") != self.username

    # ---------------- game loop ----------------
    def play_game(self, game_id: str, player=None) -> dict:
        """Jalankan satu game; tutup koneksi book milik player di akhir."""
        active = player if player is not None else self._make_player()
        book = getattr(active, "book", None)
        try:
            return self._play_game_impl(game_id, active)
        finally:
            db = getattr(book, "db", None)
            if db is not None:
                with contextlib.suppress(sqlite3.Error):
                    db.close()

    def _play_game_impl(self, game_id: str, player) -> dict:
        """Ikuti satu game sampai selesai; mengembalikan ringkasan."""
        board = None
        my_color = chess.WHITE
        game_full = None
        moves_seen = ""
        last_state: dict = {}
        resign_streak_state = {"n": 0}
        rng = random.Random(int(time.time()) & 0xFFFF)
        for ev in self.api.stream(f"/api/bot/game/stream/{game_id}"):
            ev_type = ev.get("type")
            if ev_type == "gameFull":
                game_full = ev
                last_state = ev.get("state") or {}
                initial = _initial_fen(ev.get("initialFen"))
                my_color = (chess.WHITE if ev.get("white", {}).get("name") == self.username
                            else chess.BLACK)
                board = chess.Board(initial)
                moves_seen = ev.get("state", {}).get("moves", "")
                for u in moves_seen.split():
                    with contextlib.suppress(ValueError):
                        board.push(chess.Move.from_uci(u))
                # Bisa saja challenge sudah jadi game dan giliran kita saat
                # gameFull tiba — jangan menunggu gameState berikutnya.
                if board.is_game_over(claim_draw=True):
                    continue
                if board.turn == my_color:
                    outcome = self._take_turn(game_id, board, rng, resign_streak_state,
                                              player=player, my_color=my_color)
                    if outcome is not None:
                        self._save_pgn(game_full, last_state, outcome)
                        return outcome
            elif ev_type == "gameState":
                state = ev
                last_state = state
                status = state.get("status")
                if status in ("mate", "draw", "outoftime", "resign", "stalemate",
                              "aborted", "cheat", "timeout"):
                    summary = {"game_id": game_id, "status": status,
                               "my_color": "white" if my_color else "black"}
                    self._save_pgn(game_full, state, summary)
                    return summary
                moves_now = state.get("moves", "")
                if moves_now == moves_seen or board is None:
                    continue
                moves_seen = moves_now
                board = chess.Board(_initial_fen(
                    game_full.get("initialFen") if game_full else None))
                for u in moves_seen.split():
                    with contextlib.suppress(ValueError):
                        board.push(chess.Move.from_uci(u))
                if board.is_game_over(claim_draw=True):
                    continue
                if board.turn != my_color:
                    continue
                outcome = self._take_turn(game_id, board, rng, resign_streak_state,
                                          player=player, my_color=my_color)
                if outcome is not None:
                    self._save_pgn(game_full, last_state, outcome)
                    return outcome
        summary = {"game_id": game_id, "status": "stream_ended",
                   "my_color": "white" if my_color else "black"}
        self._save_pgn(game_full, last_state, summary)
        return summary

    def _take_turn(self, game_id: str, board: chess.Board, rng,
                   resign_streak_state, player=None,
                   my_color: bool = chess.WHITE) -> dict | None:
        """Hitung & kirim satu langkah kita. resign_streak_state = dict ['n'].
        Return dict ringkasan bila bot resign (game selesai dari sisi kita)."""
        mv, info = self.choose_move(board, rng, player=player)
        if mv is None:
            return None
        score = self._eval_for_me(board, info)
        if board.can_claim_draw() and score < self.lic.offer_draw_cp:
            self._post(f"/api/bot/game/{game_id}/move/{mv.uci()}",
                       params="claimDraw=true")
            print(f"[lichess] {game_id}: {mv.uci()} + claim draw", flush=True)
            return None
        if score <= self.lic.resign_cp:
            resign_streak_state["n"] += 1
        else:
            resign_streak_state["n"] = 0
        if resign_streak_state["n"] >= max(1, self.lic.resign_moves):
            self._post(f"/api/bot/game/{game_id}/resign")
            opponent = "black" if my_color else "white"
            return {"game_id": game_id, "status": "resign_sent", "winner": opponent}
        # Budget waktu sederhana: hormati remaining time.
        self._post(f"/api/bot/game/{game_id}/move/{mv.uci()}")
        print(f"[lichess] {game_id}: memainkan {mv.uci()} "
              f"(d{info.get('depth')}, {info.get('nodes')}n)", flush=True)
        return None

    def _save_pgn(self, game_full: dict | None, state: dict, summary: dict):
        try:
            if not game_full:
                return
            white = _user_name(game_full, "white")
            black = _user_name(game_full, "black")
            game = chess.pgn.Game()
            game.headers["Event"] = f"Lichess {game_full.get('id', '')}"
            game.headers["White"] = white
            game.headers["Black"] = black
            game.headers["Result"] = self._result_header(game_full, state, summary)
            node = game
            board = chess.Board(_initial_fen(game_full.get("initialFen")))
            for u in state.get("moves", "").split():
                try:
                    mv = chess.Move.from_uci(u)
                    node = node.add_variation(mv)
                    board.push(mv)
                except ValueError:
                    break
            if board.is_checkmate():
                winner = "white" if board.turn == chess.BLACK else "black"
                game.headers["Result"] = "1-0" if winner == "white" else "0-1"
            elif board.is_game_over(claim_draw=True):
                game.headers["Result"] = "1/2-1/2"
            with (self.pgn_dir / f"{summary['game_id']}.pgn").open("w", encoding="utf-8") as f:
                f.write(str(game) + "\n")
        except Exception as exc:
            print(f"[lichess] gagal menyimpan PGN: {exc}", flush=True)

    # ---------------- main loop ----------------
    def _summary(self) -> dict:
        return {"games_started": self.games_played,
                "games_finished": self._games_finished,
                "learn_cycles": self._learn_cycles,
                "promoted": self._promoted,
                "seek_sent": self._seek_sent,
                "seek_failed": self._seek_failed}

    def _log_summary(self) -> dict:
        """Cetak satu baris JSON agar ringkasan mudah diparsing."""
        summary = self._summary()
        print(f"[lichess] summary {json.dumps(summary, sort_keys=True)}",
              flush=True)
        return summary

    def run(self):
        """Loop event utama; dry_run memainkan game simulasi lokal."""
        if self.dry_run:
            return self._dry_run()
        stop = threading.Event()
        seek_thread: threading.Thread | None = None
        try:
            try:
                profile = json.loads(self._get("/api/account"))
            except (ValueError, urllib.error.URLError) as exc:
                print(f"[lichess] gagal membaca profil akun: {exc}", flush=True)
                return {**self._log_summary(), "error": "profile_unavailable"}
            self.username = profile.get("username", self.username)
            self.my_rating = _account_rating(profile)
            if profile.get("title") != "BOT":
                try:
                    self._post("/api/bot/account/upgrade")
                    print("[lichess] akun di-upgrade ke BOT; login ulang dengan token "
                          "baru bila diminta laman lichess.", flush=True)
                except urllib.error.HTTPError as exc:
                    print(f"[lichess] upgrade gagal ({exc.code}) — jika akun sudah "
                          "BOT, abaikan.", flush=True)
            print(f"[lichess] berjalan sebagai {self.username}; menunggu challenge "
                  f"(Ctrl+C untuk berhenti)", flush=True)
            print(f"[lichess] policy: terima semua tantangan lawan (self-challenge "
                  f"ditolak); concurrency={max(1, int(self.lic.concurrency))}; "
                  f"auto_seek={self._auto_seek_enabled}; "
                  f"learn={self.lic.learn_from_games}", flush=True)
            if self._auto_seek_enabled:
                seek_thread = threading.Thread(
                    target=self._seek_loop, args=(stop,), daemon=True,
                    name="lichess-seek")
                seek_thread.start()
            while not stop.is_set():
                if self.lic.max_games and self.games_played >= self.lic.max_games:
                    print(f"[lichess] batas {self.lic.max_games} game tercapai",
                          flush=True)
                    break
                try:
                    for ev in self.api.stream("/api/stream/event"):
                        if stop.is_set():
                            break
                        ev_type = ev.get("type")
                        if self._release_closed_challenge(ev):
                            continue
                        if ev_type == "challenge":
                            ch = ev.get("challenge", {})
                            ch_name = (ch.get("challenger") or {}).get("name", "?")
                            ch_speed = ch.get("speed", "?")
                            ch_rated = ch.get("rated", False)
                            cid = str(ch.get("id") or "")
                            if self._is_outgoing_challenge(ev):
                                continue
                            accepted = cid and self._accept_challenge(ev)
                            if accepted and self._reserve_game(cid):
                                try:
                                    self._post(f"/api/challenge/{cid}/accept")
                                except urllib.error.HTTPError:
                                    self._release_game(cid)
                                    raise
                                print(f"[lichess] challenge {cid} dari {ch_name} "
                                      f"({ch_speed}, rated={ch_rated}) DITERIMA",
                                      flush=True)
                            else:
                                reason = ("self-challenge" if not accepted
                                          else "budget penuh")
                                print(f"[lichess] challenge {cid} dari {ch_name} "
                                      f"DITOLAK ({reason})", flush=True)
                                with contextlib.suppress(urllib.error.HTTPError):
                                    self._post(f"/api/challenge/{cid}/decline",
                                               params="reason=general")
                        elif ev_type == "gameStart":
                            gid = str((ev.get("game") or {}).get("id") or "")
                            if gid and self._submit_game(gid):
                                print(f"[lichess] game {gid} dimulai", flush=True)
                            elif gid:
                                print(f"[lichess] game {gid} tidak punya slot",
                                      flush=True)
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:
                        print("[lichess] rate-limited; tidur 60s", flush=True)
                        time.sleep(60)
                    else:
                        print(f"[lichess] HTTP {exc.code}; tidur "
                              f"{self.lic.poll_seconds}s", flush=True)
                        time.sleep(self.lic.poll_seconds)
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    print(f"[lichess] koneksi terputus ({type(exc).__name__}); "
                          f"reconnect dalam {self.lic.poll_seconds}s", flush=True)
                    time.sleep(self.lic.poll_seconds)
        except KeyboardInterrupt:
            print("\n[lichess] dihentikan", flush=True)
        finally:
            stop.set()
            if seek_thread is not None:
                seek_thread.join(timeout=5)
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._join_learn_thread(timeout=30.0)
        return self._log_summary()

    # ---------------- dry run (tanpa jaringan) ----------------
    def _dry_run(self) -> dict:
        """Simulasi: mainkan 1 game lokal vs random, cetak keputusan bot."""
        print("[lichess] DRY-RUN: simulasi keputusan tanpa jaringan", flush=True)
        self._player = self._make_player()
        board = chess.Board()
        rng = random.Random(11)
        opponent = random.Random(23)
        book_used = 0
        while not board.is_game_over(claim_draw=True) and board.ply() < 60:
            if board.turn == chess.WHITE:
                mv, info = self.choose_move(board, rng)
                if mv is None:
                    break
                if info.get("book"):
                    book_used += 1
                print(f"  ply {board.ply():2d} (kita, putih): {mv.uci():6s} "
                      f"d{info.get('depth')} book={bool(info.get('book'))}", flush=True)
            else:
                mv = opponent.choice(list(board.legal_moves))
                print(f"  ply {board.ply():2d} (lawan, hitam): {mv.uci():6s}", flush=True)
            board.push(mv)
        result = board.outcome(claim_draw=True)
        summary = {"dry_run": True, "plies": board.ply(),
                   "result": result.result() if result else "*",
                   "book_moves": book_used,
                   "max_depth_cfg": self.lic.max_depth,
                   "min_depth_cfg": self.lic.min_depth}
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv=None) -> int:
    """Entri langsung: python -m kestrel.lichess_bot --token XXX [--dry-run]"""
    import argparse

    from .config import load_config
    ap = argparse.ArgumentParser(prog="kestrel.lichess_bot")
    ap.add_argument("--token")
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-games", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.max_games:
        cfg = cfg.with_overrides("lichess", max_games=args.max_games)
    token = args.token or cfg.lichess.token or os.environ.get("KESTREL_LICHESS_TOKEN", "")
    if not token and not args.dry_run:
        print("Token belum diisi: --token, config.toml [lichess].token, atau "
              "env KESTREL_LICHESS_TOKEN")
        return 2
    bot = LichessBot(cfg, token=token, dry_run=args.dry_run)
    try:
        bot.run()
        return 0
    except KeyboardInterrupt:
        print("\n[lichess] dihentikan")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
