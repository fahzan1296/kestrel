"""Shared training status and structured logs for CLI and dashboard.

Both entry points write the same ``train_state.json`` and ``train.log`` files
inside the configured results directory. The dashboard polls those files, so a
training run started by ``python3 -m kestrel.cli train`` is visible while the
dashboard process keeps serving requests normally.

Every log line is one JSON object with a stable ``event`` name, a ``run_id``
correlating the whole run, and an ``entry_point`` identifying who started it.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_NAME = "train_state.json"
LOG_NAME = "train.log"
_REDACT = re.compile(r"(token|password|secret|api[_-]?key)", re.I)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ("<redacted>" if _REDACT.search(str(k)) else _safe(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _state_path(results_dir: Path) -> Path:
    return Path(results_dir) / STATE_NAME


def _log_path(results_dir: Path) -> Path:
    return Path(results_dir) / LOG_NAME


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_safe(payload), ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _emit(results_dir: Path, *, run_id: str, entry_point: str,
          event: str, message: str, echo: bool = True, **fields: Any) -> dict:
    record = {"ts": _now(), "run_id": run_id, "entry_point": entry_point,
              "event": event, "message": message}
    record.update(_safe(fields))
    path = _log_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    if echo:
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def start(results_dir: Path, *, entry_point: str,
          settings: dict | None = None, run_id: str | None = None,
          echo: bool = True) -> dict:
    """Mark a training run as started and write its first structured event."""
    rid = run_id or uuid.uuid4().hex[:12]
    state = {"run_id": rid, "entry_point": entry_point,
             "status": "running", "settings": _safe(settings or {}),
             "last_message": None, "progress": {}, "result": None,
             "error": None, "live_game": None,
             "started_at": _now(), "updated_at": _now(),
             "finished_at": None, "pid": os.getpid()}
    _atomic_json(_state_path(results_dir), state)
    _emit(results_dir, run_id=rid, entry_point=entry_point,
          event="train_started", message="training dimulai",
          settings=state["settings"], echo=echo)
    return state


def read_state(results_dir: Path) -> dict | None:
    path = _state_path(results_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_log(results_dir: Path, *, limit: int = 200) -> list[dict]:
    path = _log_path(results_dir)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-max(1, limit):]:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def update(results_dir: Path, *, run_id: str, message: str | None = None,
           progress: dict | None = None, live_game: dict | None = None,
           event: str = "train_update", echo: bool = True, **fields: Any) -> bool:
    """Append a progress event; ignore writes from a stale run id."""
    state = read_state(results_dir)
    if state is None or state.get("run_id") != run_id:
        return False
    if message is not None:
        state["last_message"] = str(message)
    if progress is not None:
        state["progress"] = _safe(progress)
    if live_game is not None:
        state["live_game"] = _safe(live_game)
    state["updated_at"] = _now()
    _atomic_json(_state_path(results_dir), state)
    _emit(results_dir, run_id=run_id, entry_point=state["entry_point"],
          event=event, message=message or event, echo=echo,
          progress=state["progress"], **fields)
    return True


def finish(results_dir: Path, *, run_id: str, status: str,
           result: dict | None = None, error: str | None = None,
           echo: bool = True) -> bool:
    """Record the terminal state and event for a run."""
    state = read_state(results_dir)
    if state is None or state.get("run_id") != run_id:
        return False
    state["status"] = status
    state["result"] = _safe(result or {})
    state["error"] = str(error) if error else None
    state["updated_at"] = _now()
    state["finished_at"] = _now()
    _atomic_json(_state_path(results_dir), state)
    _emit(results_dir, run_id=run_id, entry_point=state["entry_point"],
          event="train_finished", message=f"training selesai: {status}",
          status=status, result=state["result"], error=state["error"], echo=echo)
    return True
