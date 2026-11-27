"""Dashboard must surface CLI-started training and share its own run state."""
from pathlib import Path

from kestrel import dashboard  # pyright: ignore[reportMissingImports]
from kestrel.config import load_config  # pyright: ignore[reportMissingImports]
from kestrel.train_status import finish, read_state, start, update  # pyright: ignore[reportMissingImports]


def _cfg(tmp_path: Path):
    cfg = load_config()
    return cfg.with_overrides(
        "storage",
        results_dir=str(tmp_path / "results"),
        database=str(tmp_path / "results" / "kestrel.db"),
        pgn_file=str(tmp_path / "results" / "games.pgn"),
    )


def test_train_state_reports_cli_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RESULTS", tmp_path)
    run = start(tmp_path, entry_point="cli", settings={"generations": 2})
    update(tmp_path, run_id=run["run_id"], message="selfplay 1/4",
           progress={"generation": 1, "completed": 1, "total": 4},
           event="train_update", echo=False)

    snap = dashboard.train_state()

    assert snap["status"] == "running"
    assert snap["entry_point"] == "cli"
    assert snap["last_message"] == "selfplay 1/4"
    assert snap["progress"]["completed"] == 1
    assert snap["log"][-1]["message"] == "selfplay 1/4"


def test_dashboard_launched_cross_engine_streams_board(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RESULTS", tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("kestrel.config.load_config", lambda: cfg)
    seen = {}

    def fake_cross(cfg, *, run_id, progress=None, board_callback=None, **kwargs):
        state = read_state(tmp_path)
        assert state is not None
        seen["settings"] = state["settings"]
        update(tmp_path, run_id=run_id, event="game_board",
               message="live_game", live_game={
                   "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                   "moves": [], "white": "Kestrel",
                   "black": "Stockfish 19"})
        if progress is not None:
            progress({"stage": "cross_engine", "completed": 2, "total": 2})
        if board_callback is not None:
            board_callback({"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})
        update(tmp_path, run_id=run_id, event="cross_engine_done",
               message="cross-engine selesai")
        result = {"overall": "PASS", "corpus": {"games": 2},
                  "promoted_count": 0, "final_generation": 1,
                  "report_path": str(tmp_path / "report.json")}
        finish(tmp_path, run_id=run_id, status="pass", result=result)
        return result

    monkeypatch.setattr("kestrel.cross_engine.run_cross_engine_training", fake_cross)
    req = dashboard.TrainRequest(mode="cross_engine", generations=1,
                                 games=2, skill=1, stockfish_time_ms=50)
    assert dashboard._launch_train_job(req) is True
    thread = dashboard.TRAIN_THREAD
    assert thread is not None
    thread.join(timeout=5)

    settings = seen["settings"]
    assert isinstance(settings, dict)
    assert settings["mode"] == "cross_engine"
    state = read_state(tmp_path)
    assert state is not None
    assert state["live_game"]["black"] == "Stockfish 19"
    assert state["status"] == "pass"
    snap = dashboard.train_state()
    assert snap["live_game"]["white"] == "Kestrel"
    assert any(line["event"] == "game_board" for line in snap["log"])

def test_dashboard_launched_training_publishes_shared_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RESULTS", tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("kestrel.config.load_config", lambda: cfg)

    def fake_training(cfg, **kwargs):
        progress = kwargs.get("progress")
        assert progress is not None
        progress({"message": "verifikasi", "stage": "verification", "decision": "PROMOTE"})
        return {"overall": "PASS", "promoted_count": 1,
                "final_generation": 2, "report_path": str(tmp_path / "report.json")}

    monkeypatch.setattr("kestrel.trainer.run_training", fake_training)
    req = dashboard.TrainRequest(mode="selfplay", generations=1)
    assert dashboard._launch_train_job(req) is True
    thread = dashboard.TRAIN_THREAD
    assert thread is not None
    thread.join(timeout=5)

    state = read_state(tmp_path)
    assert state is not None
    assert state["entry_point"] == "dashboard"
    assert state["status"] == "pass"
    assert state["settings"]["generations"] == 1
    log_messages = [line["message"] for line in dashboard.train_state()["log"]]
    assert "verifikasi" in log_messages
