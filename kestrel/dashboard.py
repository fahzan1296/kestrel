"""Optional localhost real-time dashboard and process-parallel self-play controller."""
from __future__ import annotations  # noqa: I001

import contextlib
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import tomllib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .train_status import finish as train_finish  # pyright: ignore[reportMissingImports]
from .train_status import read_log as train_read_log  # pyright: ignore[reportMissingImports]
from .train_status import read_state as train_read_state  # pyright: ignore[reportMissingImports]
from .train_status import start as train_start  # pyright: ignore[reportMissingImports]
from .train_status import update as train_update  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"
try:
    with CONFIG_PATH.open("rb") as _cfg_file:
        _CONFIG = tomllib.load(_cfg_file)
except (OSError, tomllib.TOMLDecodeError):
    _CONFIG = {}
_SELFPLAY_CONFIG = _CONFIG.get("selfplay", {})
RESULTS = ROOT / "results"
STATIC = Path(__file__).resolve().parent / "web"
RESULTS.mkdir(exist_ok=True)
app = FastAPI(title="Kestrel Live Monitor", version="2.0.0")

LOCK = threading.RLock()
STATE: dict[str, Any] = {
    "status": "idle", "batch": {"requested": 0, "completed": 0, "failed": 0, "concurrency": 1},
    "games": {}, "events": [], "totals": {"nodes": 0, "moves": 0, "nps": 0, "depth": 0, "depth_sum": 0, "wins": 0, "draws": 0, "losses": 0, "plies": 0},
    "started_at": None, "finished_at": None, "last_error": None,
}
EVENT_LIMIT = 5000
EVENT_QUEUE = None
CTX = mp.get_context("spawn")
ACTIVE: dict[int, mp.Process] = {}
STOP_REQUESTED = False
COORDINATOR: threading.Thread | None = None
NEXT_GAME_ID = 1
# --- 2.0: trainer unified (selfplay+knowledge+nnue+verify+promote) ---
_TRAIN_LOCK = threading.RLock()
TRAIN_STATE: dict[str, Any] = {"status": "idle", "run_id": None,
                               "last_message": None, "progress": {},
                               "updated_at": None, "result": None, "error": None}
TRAIN_THREAD: threading.Thread | None = None
TRAIN_STOP = threading.Event()


def _current_train_snapshot() -> dict[str, Any]:
    """Merge in-memory and on-disk training state (CLI or dashboard run)."""
    with _TRAIN_LOCK:
        snap = dict(TRAIN_STATE)
    external = train_read_state(RESULTS)
    if external and (snap.get("status") in (None, "idle")
                     or external.get("run_id") == snap.get("run_id")):
        snap.update(external)
    snap["log"] = train_read_log(RESULTS, limit=200)
    return _json_safe(snap)

def _json_safe(value: Any) -> Any:
    """Normalize dashboard snapshots to JSON-native values (incl. NumPy scalars)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    # NumPy scalar types (e.g. numpy.bool_) expose item() returning a Python scalar.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            normalized = item()
            if normalized is not value:
                return _json_safe(normalized)
        except (TypeError, ValueError):
            pass
    # Enum-like values and paths are safe as readable strings for API snapshots.
    if hasattr(value, "value"):
        with contextlib.suppress(Exception):
            return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


_STORAGE_PATHS: tuple | None = None


def _storage_paths() -> tuple:
    """Path PGN/SQLite dari config (bugfix 1.7 #14) dengan fallback RESULTS.

    Dashboard sebelumnya meng-hardcode results/games.pgn + kestrel.db sehingga
    divergensi dari storage yang dikonfigurasi trainer/CLI.
    """
    global _STORAGE_PATHS
    if _STORAGE_PATHS is None:
        try:
            from .config import load_config
            cfg = load_config()
            _STORAGE_PATHS = (cfg.pgn_path, cfg.db_path)
        except Exception:
            _STORAGE_PATHS = (RESULTS / "games.pgn", RESULTS / "kestrel.db")
    return _STORAGE_PATHS


def _persist_game(event):
    """Single-writer persistence, called only by the collector thread."""
    pgn_path, db_path = _storage_paths()
    pgn = event.get("pgn", "")
    try:
        if pgn:
            with open(pgn_path, "a", encoding="utf-8") as f:
                f.write(pgn + "\n\n")
        import sqlite3
        db = sqlite3.connect(db_path, timeout=15)
        db.execute("CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY AUTOINCREMENT, white TEXT, black TEXT, result TEXT, plies INTEGER, pgn TEXT, created REAL, seed INTEGER)")
        db.execute("INSERT INTO games(white,black,result,plies,pgn,created,seed) VALUES(?,?,?,?,?,?,?)",
                  (f"Kestrel W #{event['game_id']}", f"Kestrel B #{event['game_id']}", event.get("result"), event.get("plies",0), pgn, event.get("timestamp",time.time()), event.get("game_id",0)))
        db.commit()
        db.close()
    except Exception as e:
        with LOCK:
            STATE["last_error"] = f"Persistence: {e}"


def _collector(q):
    while True:
        try:
            ev = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if ev is None:
            return
        typ, gid = ev.get("type"), str(ev.get("game_id"))
        with LOCK:
            # Terminated workers can leave already-enqueued events in the shared
            # queue. A batch token prevents those stale events contaminating the
            # next dashboard run. Completed PGNs are still persisted below.
            same_batch = ev.get("batch_id") == STATE.get("batch", {}).get("batch_id")
            if same_batch:
                STATE["events"].append(ev)
                if len(STATE["events"]) > EVENT_LIMIT:
                    STATE["events"] = STATE["events"][-EVENT_LIMIT:]
                game = STATE["games"].setdefault(
                    gid, {"id": int(gid), "status": "queued", "moves": [], "metrics": {}})
                if typ == "game_started":
                    game.update(status="running", initial_fen=ev.get("fen"),
                                started_at=ev.get("timestamp"), white=ev.get("white"),
                                black=ev.get("black"), worker_pid=ev.get("worker_pid"),
                                opening_index=ev.get("opening_index"))
                elif typ == "search_info":
                    game["metrics"].update(ev)
                    STATE["totals"]["nps"] = ev.get("nps", STATE["totals"]["nps"])
                    STATE["totals"]["depth"] = ev.get("depth", STATE["totals"]["depth"])
                elif typ == "move":
                    game["moves"].append(ev)
                    game.update(fen=ev.get("fen"), current_turn=ev.get("turn"),
                                last_move=ev.get("san"))
                    STATE["totals"]["moves"] += 1
                    STATE["totals"]["plies"] += 1
                    STATE["totals"]["depth_sum"] += ev.get("depth", 0)
                    STATE["totals"]["nodes"] += ev.get("nodes", 0)
                    STATE["totals"]["nps"] = ev.get("nps", STATE["totals"]["nps"])
                    STATE["totals"]["depth"] = ev.get("depth", STATE["totals"]["depth"])
                elif typ == "game_finished":
                    if game.get("status") != "finished":
                        game.update(status="finished", result=ev.get("result"),
                                    plies=ev.get("plies"), final_fen=ev.get("final_fen"),
                                    finished_at=ev.get("timestamp"),
                                    duration_seconds=ev.get("duration_seconds"),
                                    opening_index=ev.get("opening_index"))
                        STATE["batch"]["completed"] += 1
                        if ev.get("result") == "1-0":
                            STATE["totals"]["wins"] += 1
                        elif ev.get("result") == "0-1":
                            STATE["totals"]["losses"] += 1
                        else:
                            STATE["totals"]["draws"] += 1
                elif typ == "worker_failed" and game.get("status") not in ("finished", "failed"):
                    game.update(status="failed", error=ev.get("error", "Worker failed"),
                                finished_at=ev.get("timestamp"))
                    STATE["batch"]["failed"] = STATE["batch"].get("failed", 0) + 1
                    STATE["last_error"] = f"Game {gid}: {game['error']}"
        if typ == "game_finished":
            _persist_game(ev)



def _run_batch(games: int, concurrency: int, time_ms: int, max_depth: int, max_plies: int):
    global STOP_REQUESTED, ACTIVE, NEXT_GAME_ID
    q = EVENT_QUEUE
    next_id = NEXT_GAME_ID
    NEXT_GAME_ID += games
    pending = list(range(next_id, next_id + games))
    active: dict[int, Any] = {}
    with LOCK:
        STATE.update(status="running", started_at=time.time(), finished_at=None, last_error=None)
        STATE["games"] = {
            str(gid): {"id": gid, "status": "queued", "moves": [], "metrics": {}}
            for gid in pending
        }
        STATE["events"] = []
        STATE["totals"] = {"nodes": 0, "moves": 0, "nps": 0, "depth": 0,
                           "depth_sum": 0, "wins": 0, "draws": 0, "losses": 0, "plies": 0}
        STATE["batch"] = {"requested": games, "completed": 0, "failed": 0,
                          "concurrency": concurrency, "time_ms": time_ms,
                          "max_depth": max_depth, "max_plies": max_plies,
                          "batch_id": next_id, "worker_pids": []}
    while pending or active:
        if STOP_REQUESTED:
            stopped_ids = list(active.keys()) + list(pending)
            for proc in active.values():
                if proc.is_alive():
                    proc.terminate()
            for proc in active.values():
                proc.join(timeout=2)
            active.clear()
            pending.clear()
            with LOCK:
                for gid in stopped_ids:
                    g = STATE["games"].setdefault(str(gid),
                        {"id": gid, "moves": [], "metrics": {}})
                    if g.get("status") not in ("finished", "failed"):
                        g.update(status="stopped", error="Stopped by user")
                STATE["batch"]["worker_pids"] = []
            break

        while pending and len(active) < concurrency:
            gid = pending.pop(0)
            proc = CTX.Process(target=_worker_entry,
                               args=(gid, 10000 + gid, time_ms, max_depth, max_plies, q, next_id))
            try:
                proc.start()
            except Exception as exc:
                with LOCK:
                    STATE["games"][str(gid)].update(status="failed", error=f"Could not start worker: {exc}")
                    STATE["batch"]["failed"] = STATE["batch"].get("failed", 0) + 1
                    STATE["last_error"] = f"Game {gid}: worker start failed: {exc}"
                continue
            active[gid] = proc
            with LOCK:
                STATE["games"].setdefault(str(gid),
                    {"id": gid, "status": "queued", "moves": [], "metrics": {}})
                STATE["games"][str(gid)].update(status="starting", worker_pid=proc.pid)
                STATE["batch"]["worker_pids"] = [p.pid for p in active.values()]

        for gid, proc in list(active.items()):
            if not proc.is_alive():
                proc.join()
                active.pop(gid)
                with LOCK:
                    STATE["batch"]["worker_pids"] = [p.pid for p in active.values()]
                # Queue delivery is asynchronous. Wait for the collector to apply
                # game_finished/worker_failed before deciding the process failed.
                terminal_deadline = time.monotonic() + 3.0
                while time.monotonic() < terminal_deadline:
                    with LOCK:
                        status = STATE["games"].get(str(gid), {}).get("status")
                    if status in ("finished", "failed", "stopped"):
                        break
                    time.sleep(0.03)
                with LOCK:
                    g = STATE["games"].get(str(gid), {})
                    if g.get("status") not in ("finished", "failed", "stopped"):
                        g.update(status="failed", error=f"worker exited with code {proc.exitcode} without a finish event")
                        STATE["batch"]["failed"] = STATE["batch"].get("failed", 0) + 1
                        STATE["last_error"] = f"Game {gid} worker exit code {proc.exitcode}; no terminal event"
        time.sleep(0.05)

    with LOCK:
        if STOP_REQUESTED:
            STATE["status"] = "stopped"
        elif STATE["batch"].get("failed", 0):
            STATE["status"] = "completed_with_errors"
        else:
            STATE["status"] = "completed"
        STATE["finished_at"] = time.time()
        STATE["batch"]["worker_pids"] = []
    ACTIVE.clear()



def _worker_entry(gid, seed, time_ms, max_depth, max_plies, q, batch_id):
    try:
        from .dashboard_worker import run_game_worker
        # 1.7 bugfix #6: book + batas knowledge dari config diteruskan ke
        # worker (sebelumnya worker selfplay dashboard memakai default semua).
        try:
            from .config import load_config
            _cfg = load_config()
            _book = (str(_cfg.knowledge_db_path)
                     if (_cfg.knowledge.enabled and _cfg.selfplay.use_book
                         and _cfg.knowledge_db_path.exists()) else None)
            _book_kw = {"book_max_plies": _cfg.knowledge.book_max_plies,
                        "book_min_games": _cfg.knowledge.book_min_games,
                        "book_max_moves": _cfg.knowledge.book_max_moves_per_position}
        except Exception:
            _book, _book_kw = None, {}
        run_game_worker(
            gid, seed, time_ms, max_depth, max_plies, q,
            batch_id=batch_id, book_path=_book,
            book_max_plies=_book_kw.get("book_max_plies"),
            book_min_games=_book_kw.get("book_min_games"),
            book_max_moves=_book_kw.get("book_max_moves"))
    except Exception as exc:  # Report the failure to the parent collector.
        import traceback
        q.put({"type": "worker_failed", "game_id": gid, "batch_id": batch_id,
               "timestamp": time.time(),
               "worker_pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(limit=8)})



class RunRequest(BaseModel):
    """Default mengikuti config.toml [selfplay] — bukan nilai hard-coded."""
    games: int = Field(default=int(_SELFPLAY_CONFIG.get("games_per_batch", 4)), ge=1, le=500)
    concurrency: int = Field(default=int(_SELFPLAY_CONFIG.get("concurrency", 2)), ge=1, le=16)
    time_ms: int = Field(default=int(_SELFPLAY_CONFIG.get("time_ms", 150)), ge=10, le=10000)
    max_depth: int = Field(default=int(_SELFPLAY_CONFIG.get("max_depth", 10)), ge=1, le=64)
    max_plies: int = Field(default=int(_SELFPLAY_CONFIG.get("max_plies", 160)), ge=20, le=500)



@app.on_event("startup")
def startup():
    global EVENT_QUEUE
    EVENT_QUEUE = CTX.Queue()
    threading.Thread(target=_collector, args=(EVENT_QUEUE,), daemon=True, name="kestrel-event-collector").start()



@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")



@app.get("/api/state")
def get_state():
    with LOCK:
        snap = _json_safe(STATE)
    snap["train"] = _current_train_snapshot()
    start = snap.get("started_at")
    end = snap.get("finished_at") or (time.time() if snap.get("status") == "running" else start)
    elapsed = max(0.0, (end-start)) if start and end else 0.0
    completed = snap["batch"].get("completed", 0)
    snap["runtime"] = {"logical_cpus": os.cpu_count() or 1,
                       "multiprocessing_start_method": CTX.get_start_method()}
    snap["analytics"] = {"elapsed_seconds": round(elapsed, 2),
        "games_per_minute": round(completed / max(elapsed/60.0, 1e-9), 2) if completed and elapsed else 0.0,
        "avg_depth": round(snap["totals"].get("depth_sum",0) / max(snap["totals"].get("moves",0),1), 2),
        "nodes_per_game": int(snap["totals"].get("nodes",0) / max(completed,1)),
        "failed_games": snap["batch"].get("failed", 0),
        "result_counts": {"wins":snap["totals"].get("wins",0), "draws":snap["totals"].get("draws",0), "losses":snap["totals"].get("losses",0)}}
    return snap



@app.post("/api/run")
def run(req: RunRequest):
    global COORDINATOR, STOP_REQUESTED
    with LOCK:
        if STATE["status"] in ("running", "starting"):
            raise HTTPException(409, "Batch sedang berjalan. Stop batch saat ini terlebih dahulu.")
        STATE["status"] = "starting"
        STOP_REQUESTED = False
    COORDINATOR = threading.Thread(target=_run_batch, args=(req.games, req.concurrency, req.time_ms, req.max_depth, req.max_plies), daemon=True)
    COORDINATOR.start()
    settings = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    return {"accepted": True, "message": "Batch dimulai", "settings": settings}



@app.post("/api/stop")
def stop():
    global STOP_REQUESTED
    with LOCK:
        if STATE["status"] not in ("running", "starting"):
            return {"accepted": False, "message": "Tidak ada batch yang sedang berjalan."}
        STOP_REQUESTED = True
    return {"accepted": True, "message": "Stop diminta; worker aktif akan dihentikan."}



# ------------------------- 1.6: trainer unified -------------------------

class TrainRequest(BaseModel):
    """Nilai default mengikuti config.toml [train]/[selfplay]/[eval]."""
    mode: str = "cross_engine"
    generations: int = Field(default=int(_CONFIG.get("train", {}).get("generations", 1)), ge=1, le=1000)
    games: int | None = Field(default=None, ge=1, le=500)
    skill: int = Field(default=5, ge=0, le=7)
    stockfish_time_ms: int = Field(default=200, ge=10, le=5000)
    quick: bool = False


def _launch_train_job(req: TrainRequest) -> bool:
    global TRAIN_THREAD
    from .config import load_config
    cfg = load_config()
    with _TRAIN_LOCK:
        if TRAIN_STATE.get("status") in ("starting", "running"):
            return False
        if STATE["status"] in ("running", "starting"):
            return False
        settings = {"mode": req.mode, "generations": req.generations,
                    "games": req.games, "skill": req.skill,
                    "stockfish_time_ms": req.stockfish_time_ms,
                    "quick": req.quick, "results_path": str(cfg.results_path)}
        run = train_start(RESULTS, entry_point="dashboard", settings=settings)
        rid = run["run_id"]
        TRAIN_STATE.update(status="starting", run_id=rid,
                           last_message="menunggu worker",
                           progress={}, result=None, error=None,
                           updated_at=run["updated_at"])
        TRAIN_STOP.clear()
    def task():
        with _TRAIN_LOCK:
            TRAIN_STATE["status"] = "running"
        try:
            if req.mode == "cross_engine":
                from .cross_engine import run_cross_engine_training  # pyright: ignore[reportMissingImports]
                result = run_cross_engine_training(
                    cfg, run_id=rid, skill=req.skill,
                    stockfish_time_ms=req.stockfish_time_ms,
                    games=req.games, generations=req.generations,
                    quick=req.quick, verify=True, stop_flag=TRAIN_STOP)
                status = str(result.get("overall", "FAIL")).lower()
                with _TRAIN_LOCK:
                    TRAIN_STATE.update(result=result, status=status,
                                       finished_at=time.time())
                return
            from .trainer import QuickOverrides, run_training
            run_cfg = QuickOverrides.apply(cfg) if req.quick else cfg
            if req.games is not None:
                run_cfg = run_cfg.with_overrides("selfplay", games_per_batch=req.games)
            if req.generations is not None:
                run_cfg = run_cfg.with_overrides("train", generations=req.generations)
            def progress(info):
                detail = dict(info)
                message = str(detail.pop("message", "training"))
                prog = {k: detail[k] for k in ("stage", "generation",
                                               "completed", "total", "result")
                        if k in detail}
                train_update(RESULTS, run_id=rid, message=message,
                             progress=prog, **detail)
                with _TRAIN_LOCK:
                    TRAIN_STATE.update(last_message=message, progress=prog)
            rep = run_training(run_cfg, quick=req.quick, verify=True,
                               stop_flag=TRAIN_STOP, progress=progress)
            result = {k: rep.get(k) for k in
                      ("overall", "promoted_count", "final_generation",
                       "report_path")}
            status = str(rep.get("overall", "FAIL")).lower()
            train_finish(RESULTS, run_id=rid, status=status, result=result)
            with _TRAIN_LOCK:
                TRAIN_STATE.update(result=result, status=status, finished_at=time.time())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            train_finish(RESULTS, run_id=rid, status="failed", error=error)
            with _TRAIN_LOCK:
                TRAIN_STATE.update(status="failed", error=error,
                                   finished_at=time.time())
    TRAIN_THREAD = threading.Thread(target=task, daemon=True, name="kestrel-train")
    TRAIN_THREAD.start()
    return True


@app.post("/api/train")
def train(req: TrainRequest):
    if not _launch_train_job(req):
        raise HTTPException(409, "Training/self-play sedang berjalan; tunggu hingga selesai.")
    label = ("Cross-engine Kestrel vs Stockfish" if req.mode == "cross_engine"
             else "Trainer unified selfplay")
    return {"accepted": True, "message": f"{label} dimulai",
            "settings": req.model_dump() if hasattr(req, "model_dump") else req.dict()}


@app.post("/api/train/stop")
def train_stop():
    with _TRAIN_LOCK:
        if TRAIN_STATE.get("status") not in ("running", "starting"):
            return {"accepted": False, "message": "Tidak ada training berjalan."}
        TRAIN_STOP.set()
    return {"accepted": True, "message": "Stop training diminta (selesai di akhir generasi)."}


@app.get("/api/train/state")
def train_state():
    snap = _current_train_snapshot()
    from .versioning import CheckpointStore
    try:
        snap["champion"] = _json_safe(CheckpointStore(RESULTS).summary())
    except Exception:
        snap["champion"] = None
    return snap



@app.get("/api/health")
def health():
    with LOCK:
        return {"ok": True, "status": STATE["status"],
                "logical_cpus": os.cpu_count() or 1,
                "active_workers": len(STATE.get("batch", {}).get("worker_pids", [])),
                "completed": STATE.get("batch", {}).get("completed", 0),
                "failed": STATE.get("batch", {}).get("failed", 0)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kestrel.dashboard:app", host="127.0.0.1", port=8765, reload=False)
