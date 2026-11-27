"""KESTREL 1.6 — Pipeline end-to-end (kini berbasis trainer unified).

Tahap default: tests -> train (generasi unified: selfplay -> knowledge ->
learning -> REDA -> verification -> promote) -> benchmark -> stockfish.

1.5 lama punya tahap terpisah (selfplay, learning, match); di 1.6 ketiganya
digabung dalam satu tahap `train` supaya selalu satu config dan selalu ada
eval otomatis champion vs challenger. `--skip` tetap mendukung nama tahap.

Usage:
    python3 -m kestrel.pipeline [--quick] [--skip tests train ...]
    python3 -m kestrel.cli train --generations 10   # loop penuh via CLI
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def stage_tests(quick: bool) -> dict:
    """Jalankan test suite sebagai subprocess (isolasi state)."""
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                       capture_output=True, text=True, timeout=3600, cwd=ROOT)
    passed = r.returncode == 0
    output = (r.stdout.strip() or r.stderr.strip()).splitlines()
    return {"stage": "tests", "pass": passed,
            "summary": output[-1] if output else "no output",
            "tail": output[-8:], "seconds": round(time.time() - t0, 1)}


def stage_train(quick: bool) -> dict:
    """Satu generasi pipeline unified (selfplay+knowledge+learning+REDA+verify)."""
    from kestrel.config import load_config
    from kestrel.trainer import QuickOverrides, run_training
    cfg = load_config()
    if quick:
        cfg = QuickOverrides.apply(cfg)
    else:
        cfg = cfg.with_overrides("train", generations=1)
    rep = run_training(cfg, quick=quick)
    gens = rep.get("generations", [])
    last = gens[-1] if gens else {}
    return {"stage": "train", "pass": rep.get("overall") == "PASS",
            "generation": last.get("generation"),
            "status": last.get("status"),
            "selfplay": last.get("stages", {}).get("selfplay"),
            "verification": last.get("stages", {}).get("verification"),
            "report": rep.get("report_path")}


def stage_benchmark(quick: bool) -> dict:
    from kestrel.benchmark import (
        evaluation_consistency,
        performance_suite,
        special_move_suite,
        tactical_suite,
    )
    t = tactical_suite(time_ms=800 if quick else 3000,
                       node_limit=30000 if quick else 200000)
    perf = performance_suite(node_budget=10000 if quick else 50000)
    special = special_move_suite()
    cons = evaluation_consistency(samples=100 if quick else 200)
    return {"stage": "benchmark", "pass": t["accuracy"] >= 0.6 and special["pass"] and cons["within_epsilon"],
            "tactical": {k: t[k] for k in ("solved", "total", "accuracy")},
            "special_moves": special, "performance": perf, "consistency": cons}


def stage_stockfish(quick: bool) -> dict:
    """Opsional: hanya jalan bila binary tersedia."""
    from kestrel.arena import Player
    from kestrel.stockfish import find_stockfish, sparring_match
    if not find_stockfish():
        return {"stage": "stockfish", "pass": True, "skipped": True,
                "reason": "binary tidak tersedia di sistem ini"}
    p = Player("Kestrel-v1.0", time_ms=100 if quick else 300)
    r = sparring_match(p, games=2, skill=3 if quick else 5, time_limit=0.1)
    return {"stage": "stockfish", "pass": True, **r}


STAGES = {"tests": stage_tests, "train": stage_train,
          "benchmark": stage_benchmark, "stockfish": stage_stockfish}


def run_pipeline(quick: bool = False, skip: list[str] | None = None,
                 stages: list[str] | None = None) -> dict:
    skip = skip or []
    order = stages or ["tests", "train", "benchmark", "stockfish"]
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": quick,
              "stages": []}
    ok_all = True
    for name in order:
        if name in skip:
            report["stages"].append({"stage": name, "pass": None,
                                     "skipped": True})
            continue
        print(f"[pipeline] >>> {name} ...", flush=True)
        try:
            r = STAGES[name](quick)
        except Exception as e:  # noqa: BLE001
            r = {"stage": name, "pass": False, "error": str(e)}
        report["stages"].append(r)
        status = "PASS" if r.get("pass") else "FAIL" if r.get("pass") is False else "SKIP"
        print(f"[pipeline] <<< {name}: {status}", flush=True)
        if r.get("pass") is False:
            ok_all = False
    report["overall"] = "PASS" if ok_all else "FAIL"
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"pipeline_report_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    report["report_path"] = str(out)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    rep = run_pipeline(quick=args.quick, skip=args.skip)
    print(json.dumps({k: v for k, v in rep.items() if k != "stages"},
                     indent=2))
    for s in rep["stages"]:
        print(f"  {s['stage']:10s} -> {'SKIP' if s.get('skipped') else ('PASS' if s.get('pass') else 'FAIL')}")
    sys.exit(0 if rep["overall"] == "PASS" else 1)
