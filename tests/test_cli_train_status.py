"""CLI train must publish its progress to the shared dashboard state."""
from pathlib import Path

import chess

from kestrel import cli  # pyright: ignore[reportMissingImports]
from kestrel.config import load_config  # pyright: ignore[reportMissingImports]
from kestrel.train_status import finish, read_log, read_state, update  # pyright: ignore[reportMissingImports]


def _cfg(tmp_path: Path):
    cfg = load_config()
    return cfg.with_overrides(
        "storage",
        results_dir=str(tmp_path / "results"),
        database=str(tmp_path / "results" / "kestrel.db"),
        pgn_file=str(tmp_path / "results" / "games.pgn"),
    )


def _args(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_or_exit", lambda args: _cfg(tmp_path))
    monkeypatch.setattr(cli, "_apply_common", lambda cfg, args: cfg)
    return cli.build_parser().parse_args(["train", "--generations", "1"])


def test_selfplay_passes_champion_as_net_path(monkeypatch, tmp_path, capsys):
    from kestrel import selfplay

    captured = {}

    def fake_run_selfplay(**kwargs):
        captured.update(kwargs)
        return {"games_finished": 1}

    monkeypatch.setattr(selfplay, "run_selfplay", fake_run_selfplay)
    monkeypatch.setattr(cli, "_load_or_exit", lambda args: _cfg(tmp_path))
    monkeypatch.setattr(cli, "_apply_common", lambda cfg, args: cfg)
    args = cli.build_parser().parse_args(["selfplay"])

    assert cli.cmd_selfplay(args) == 0
    capsys.readouterr()
    assert "net_path" in captured
    assert "learned_path" not in captured


def test_cli_train_exposes_progress_to_dashboard(tmp_path, monkeypatch, capsys):
    args = _args(monkeypatch, tmp_path)
    seen_during_run = {}

    def fake_training(cfg, *, quick=False, verify=True, progress=None, **kwargs):
        state = read_state(cfg.results_path)
        assert state is not None
        seen_during_run["status"] = state["status"]
        seen_during_run["entry_point"] = state["entry_point"]
        assert progress is not None
        progress({"message": "selfplay 1/2", "stage": "selfplay",
                  "completed": 1, "total": 2})
        return {"overall": "PASS", "promoted_count": 0,
                "final_generation": 1, "report_path": str(tmp_path / "report.json")}

    monkeypatch.setattr("kestrel.trainer.run_training", fake_training)

    code = cli.cmd_train(args)

    assert code == 0
    assert seen_during_run["status"] == "running"
    assert seen_during_run["entry_point"] == "cli"
    state = read_state(_cfg(tmp_path).results_path)
    assert state is not None
    assert state["status"] == "pass"
    assert state["result"]["final_generation"] == 1
    assert state["last_message"] == "selfplay 1/2"
    assert state["settings"]["generations"] == 1
    events = [line["event"] for line in read_log(_cfg(tmp_path).results_path)]
    assert "train_started" in events
    assert "train_update" in events
    assert "train_finished" in events
    console = capsys.readouterr().out
    assert '"event": "train_update"' in console


def test_cli_cross_train_streams_live_board_to_shared_state(tmp_path, monkeypatch):
    args = cli.build_parser().parse_args([
        "cross-train", "--generations", "1", "--games", "2",
        "--skill", "1", "--stockfish-time-ms", "50", "--quick"])
    monkeypatch.setattr(cli, "_load_or_exit", lambda args: _cfg(tmp_path))
    seen = {}

    def fake_cross(cfg, *, run_id, progress=None, board_callback=None, **kwargs):
        state = read_state(cfg.results_path)
        assert state is not None
        seen["status"] = state["status"]
        seen["settings"] = state["settings"]
        assert isinstance(seen["settings"], dict)
        update(cfg.results_path, run_id=run_id, event="game_board",
               message="live_game", live_game={
                   "fen": chess.STARTING_FEN, "moves": [],
                   "white": "Kestrel", "black": "Stockfish 19"})
        if progress is not None:
            progress({"stage": "cross_engine", "game_id": 1,
                      "completed": 1, "total": 2, "result": "1-0"})
        if board_callback is not None:
            board_callback({"fen": chess.STARTING_FEN, "moves": []})
        update(cfg.results_path, run_id=run_id, event="cross_engine_done",
               message="cross-engine selesai",
               progress={"stage": "cross_engine", "completed": 2, "total": 2})
        result = {"overall": "PASS", "games": 2, "score_rate": 0.5,
                  "settings": {"mode": "cross_engine"}}
        finish(cfg.results_path, run_id=run_id, status="pass", result=result)
        return result

    monkeypatch.setattr("kestrel.cross_engine.run_cross_engine_training", fake_cross)

    code = cli.cmd_cross_train(args)
    assert code == 0
    assert seen["status"] == "running"
    settings = seen["settings"]
    assert isinstance(settings, dict)
    assert settings["mode"] == "cross_engine"
    assert settings["skill"] == 1
    state = read_state(_cfg(tmp_path).results_path)
    assert state is not None
    assert state["status"] == "pass"
    assert state["live_game"]["white"] == "Kestrel"
    events = [line["event"] for line in read_log(_cfg(tmp_path).results_path)]
    assert "game_board" in events
    assert "cross_engine_done" in events

def test_cli_train_records_failure_without_hiding_exception(tmp_path, monkeypatch):
    args = _args(monkeypatch, tmp_path)

    def broken(cfg, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("kestrel.trainer.run_training", broken)

    try:
        cli.cmd_train(args)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")

    state = read_state(_cfg(tmp_path).results_path)
    assert state is not None
    assert state["status"] == "failed"
    assert "boom" in state["error"]
