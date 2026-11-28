#!/usr/bin/env python3
"""Benchmark deterministik untuk Kestrel.

Mengukur NPS dan waktu tempuh search pada FEN tetap tanpa bergantung
pada game Lichess. Model bisa dipilih eksplisit supaya champion tidak
berubah di tengah pengukuran.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chess  # noqa: E402

from kestrel.nnue import TinyNNUE, load_default  # noqa: E402
from kestrel.search import Searcher  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BENCH_FENS = (
    START_FEN,
    "r1bq1rk1/ppp2ppp/2n5/3np3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 9",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
)

def _blas_threads() -> str:
    """Best-effort label untuk thread BLAS dari environment."""
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value:
            return key + "=" + value
    return "default"

def run_bench(fen: str, nodes: int, model: str | None = None) -> dict:
    net = TinyNNUE.load(model) if model else load_default()
    searcher = Searcher(
        net=net,
        node_limit=nodes,
        time_ms=600000,
        max_depth=64,
        min_depth=0,
        grace_ms=0,
        hard_ext_ms=0,
        hard_ext_max=0,
        tt_size=1 << 16,
    )
    board = chess.Board(fen)
    started = time.perf_counter()
    _, info = searcher.best_move(board)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "fen": fen,
        "nodes": searcher.nodes,
        "elapsed_ms": round(elapsed_ms, 3),
        "nps": round(searcher.nodes / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0,
        "depth": info.get("depth"),
        "score_cp": info.get("score_cp"),
        "model": model or "bundled_baseline",
    }


def run_bench_suite(nodes: int, model: str | None = None) -> list[dict]:
    """Run fixed FEN suite and annotate each result with BLAS metadata."""
    out = []
    for fen in BENCH_FENS:
        row = run_bench(fen, nodes, model)
        row["blas_threads"] = _blas_threads()
        out.append(row)
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default=START_FEN)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--nodes", type=int, default=200000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.suite:
        rows = run_bench_suite(args.nodes, args.model)
        if args.json:
            print(json.dumps(rows))
        else:
            for i, row in enumerate(rows):
                print(f"--- fen {i + 1}/{len(rows)} ---")
                for key, value in row.items():
                    print(f"{key}: {value}")
        return

    result = run_bench(args.fen, args.nodes, args.model)
    if args.json:
        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
