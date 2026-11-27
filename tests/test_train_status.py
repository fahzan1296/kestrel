"""Contract tests for cross-process training status and structured logs."""
import json
from pathlib import Path

from kestrel.train_status import (  # pyright: ignore[reportMissingImports]
    finish,
    read_log,
    read_state,
    start,
    update,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_start_writes_dashboard_visible_state(tmp_path):
    state = start(tmp_path, entry_point="cli", settings={"generations": 2})

    assert state["status"] == "running"
    assert state["entry_point"] == "cli"
    assert state["settings"] == {"generations": 2}
    assert state["run_id"]
    disk = read_state(tmp_path)
    assert disk is not None
    assert disk["run_id"] == state["run_id"]
    assert disk["status"] == "running"
    assert disk["entry_point"] == "cli"


def test_start_appends_structured_start_event(tmp_path):
    state = start(tmp_path, entry_point="cli", settings={})

    lines = read_log(tmp_path, limit=10)
    assert lines
    assert lines[-1]["event"] == "train_started"
    assert lines[-1]["run_id"] == state["run_id"]
    assert lines[-1]["entry_point"] == "cli"


def test_update_keeps_state_and_appends_detail_event(tmp_path):
    state = start(tmp_path, entry_point="cli", settings={})

    update(tmp_path, run_id=state["run_id"], message="generasi 1 selfplay",
           progress={"generation": 1, "completed": 3, "total": 8},
           event="generation_progress")

    disk = read_state(tmp_path)
    assert disk is not None
    assert disk["last_message"] == "generasi 1 selfplay"
    assert disk["progress"] == {"generation": 1, "completed": 3, "total": 8}
    assert disk["status"] == "running"
    lines = read_log(tmp_path, limit=10)
    assert lines[-1]["event"] == "generation_progress"
    assert lines[-1]["progress"]["completed"] == 3


def test_ignores_update_from_stale_run(tmp_path):
    start(tmp_path, entry_point="cli", settings={})

    update(tmp_path, run_id="stale-run", message="should not land", event="stale")

    disk = read_state(tmp_path)
    assert disk is not None
    assert disk["last_message"] is None
    assert read_log(tmp_path, limit=10)[-1]["event"] == "train_started"


def test_finish_records_result_and_terminal_event(tmp_path):
    state = start(tmp_path, entry_point="cli", settings={})

    finish(tmp_path, run_id=state["run_id"], status="pass",
           result={"promoted_count": 1, "final_generation": 3})

    disk = read_state(tmp_path)
    assert disk is not None
    assert disk["status"] == "pass"
    assert disk["result"] == {"promoted_count": 1, "final_generation": 3}
    assert disk["finished_at"]
    lines = read_log(tmp_path, limit=10)
    assert lines[-1]["event"] == "train_finished"
    assert lines[-1]["status"] == "pass"
