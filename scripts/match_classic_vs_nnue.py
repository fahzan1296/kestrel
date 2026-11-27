#!/usr/bin/env python3
"""Match jujur: evaluator klasik 1.8 vs evaluator AI TinyNNUE 2.0.

Keduanya memakai search stack yang sama (hanya evaluator berbeda), warna dan
opening seimbang, seed deterministik. TeacherEvaluator = evaluator klasik 1.8
identik bit-per-bit (dipindah ke kestrel/teacher.py), jadi perbandingan ini
adalah "1.8 vs 2.0" pada komponen yang benar-benar berubah.

Jalankan:  python3 scripts/match_classic_vs_nnue.py [--games 20] [--time-ms 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kestrel.arena import Player, match
from kestrel.nnue import load_default
from kestrel.teacher import TeacherEvaluator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--time-ms", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-depth", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.15)
    args = ap.parse_args()

    classic = Player("Klasik-1.8", net=TeacherEvaluator(),
                     time_ms=args.time_ms, max_depth=args.max_depth,
                     min_depth=args.min_depth, grace_ms=600)
    ai = Player("AI-TinyNNUE-2.0", net=load_default(),
                time_ms=args.time_ms, max_depth=args.max_depth,
                min_depth=args.min_depth, grace_ms=600)

    t0 = time.monotonic()
    rep = match(classic, ai, games=args.games, temperature=args.temperature,
                verbose=True)
    elapsed = time.monotonic() - t0
    out = {"games": rep["games"], "challenger_is_nnue": True,
           "nnue_W": rep["challenger"]["W"], "nnue_D": rep["challenger"]["D"],
           "nnue_L": rep["challenger"]["L"],
           "score_rate_nnue": rep["score_rate"],
           "elo_delta": rep["elo_delta"], "ci95": rep["ci95"],
           "sprt": rep["sprt"], "time_ms": args.time_ms,
           "max_depth": args.max_depth, "elapsed_seconds": round(elapsed, 1)}
    print(json.dumps(out, indent=2))
    target = ROOT / "results" / "match_classic_vs_nnue.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"laporan: {target}")


if __name__ == "__main__":
    main()
