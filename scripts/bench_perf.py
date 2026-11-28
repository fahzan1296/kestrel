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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default=START_FEN)
    parser.add_argument("--nodes", type=int, default=200000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_bench(args.fen, args.nodes, args.model)
    if args.json:
        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
