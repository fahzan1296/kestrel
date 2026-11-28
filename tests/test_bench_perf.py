import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def test_bench_perf_cli_emits_json_metrics():
    """Bench deterministik harus bisa dijalankan tanpa argument model."""
    proc = subprocess.run(
        [sys.executable, "scripts/bench_perf.py", "--nodes", "2000", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["nodes"] >= 2000
    assert payload["elapsed_ms"] > 0
    assert payload["nps"] > 0
    assert "depth" in payload

def test_bench_perf_suite_emits_five_results():
    """Benchmark suite harus menghasilkan lima FEN dengan metadata lengkap."""
    proc = subprocess.run(
        [sys.executable, "scripts/bench_perf.py", "--suite", "--nodes", "2000", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 5
    required = {"fen", "nodes", "elapsed_ms", "nps", "score_cp", "model", "blas_threads"}
    for entry in payload:
        assert required.issubset(entry)
        assert entry["nodes"] >= 2000
        assert entry["elapsed_ms"] > 0
        assert entry["nps"] > 0
    assert payload[0]["fen"] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
