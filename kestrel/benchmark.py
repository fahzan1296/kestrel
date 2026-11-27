"""KESTREL — Benchmark: tactical suite, positional suite, performance (Sec. 29).

1.7: ``parallel_performance_suite`` menambahkan metrik NPS agregat lintas
proses (throughput nyata yang dipakai trainer/self-play pada mesin
multi-core). NPS single-search tetap dilaporkan terpisah lewat
``performance_suite`` — keduanya metrik berbeda dan jujur masing-masing.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing
import os
import time
import chess
from .search import Searcher

# FEN, best move (UCI), deskripsi
TACTICAL_SUITE = [
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "Mate in 1: Re8#"),
    ("rnb1kbnr/pppp1ppp/8/3q4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "e4d5",
     "Ambil ratu bebas: exd5"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8", "Back-rank mate: Ra8#"),
    ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6b7", "Mate in 1: Qb7#"),
]

SPECIAL_MOVE_SUITE = [
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "e1g1", "Kingside castling is legal"),
]

_KIWIPEDE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def tactical_suite(time_ms=3000, node_limit=200000, net=None) -> dict:
    """TacticalAccuracy = CorrectBestMove / TotalPositions.

    ``net=None`` memakai checkpoint baseline bawaan (evaluator TinyNNUE).
    """
    results = []
    for fen, expect, desc in TACTICAL_SUITE:
        s = Searcher(net=net, node_limit=node_limit, time_ms=time_ms)
        mv, info = s.best_move(chess.Board(fen))
        got = mv.uci() if mv else None
        ok = got == expect
        results.append({"fen": fen, "desc": desc, "expected": expect,
                        "got": got, "pass": ok,
                        "nodes": info["nodes"], "depth": info["depth"]})
    solved = sum(1 for r in results if r["pass"])
    return {"solved": solved, "total": len(results),
            "accuracy": solved / len(results), "results": results}


def special_move_suite() -> dict:
    """Verify special-move availability independently from best-move choice."""
    results = []
    for fen, expect, desc in SPECIAL_MOVE_SUITE:
        board = chess.Board(fen)
        move = chess.Move.from_uci(expect)
        ok = move in board.legal_moves
        results.append({"fen": fen, "desc": desc, "move": expect, "legal": ok})
    return {"legal": sum(1 for r in results if r["legal"]),
            "total": len(results), "pass": all(r["legal"] for r in results),
            "results": results}


def performance_suite(node_budget=50000, net=None) -> dict:
    """NPS single-search dan nodes/move pada posisi kiwipete (Sec. 28)."""
    b = chess.Board(_KIWIPEDE)
    s = Searcher(net=net, node_limit=node_budget, time_ms=600000)
    t0 = time.monotonic()
    s.best_move(b.copy())
    dt = time.monotonic() - t0
    return {"nodes": s.nodes, "seconds": round(dt, 2),
            "nps": int(s.nodes / max(dt, 1e-6)),
            "node_budget": node_budget}


def _aggregate_worker(node_budget: int) -> dict:
    """Worker proses paralel: satu pencarian kiwipete lengkap & independen."""
    b = chess.Board(_KIWIPEDE)
    s = Searcher(node_limit=node_budget, time_ms=600000)
    t0 = time.monotonic()
    s.best_move(b.copy())
    dt = time.monotonic() - t0
    return {"nodes": s.nodes, "seconds": dt,
            "nps": int(s.nodes / max(dt, 1e-6))}


def parallel_performance_suite(node_budget=50000, workers: int = 2) -> dict:
    """NPS agregat lintas proses — throughput nyata training multi-core (1.7).

    Setiap worker menjalankan pencarian kiwipete yang sama dan saling
    independen; NPS agregat = total node / waktu dinding. Laporan juga
    memuat NPS per worker dan efisiensi scaling agar pembacaan tetap jujur.
    """
    workers = max(1, int(workers))
    cpu = os.cpu_count() or 1
    if workers == 1:
        r = performance_suite(node_budget=node_budget)
        return {"workers": 1, "cpu_count": cpu, "single": r,
                "aggregate_nps": r["nps"], "scaling_efficiency": 1.0}
    methods = multiprocessing.get_all_start_methods()
    start_method = "fork" if "fork" in methods else "spawn"
    ctx = multiprocessing.get_context(start_method)
    t0 = time.monotonic()
    with ctx.Pool(processes=workers) as pool:
        results = pool.starmap(_aggregate_worker, [(node_budget,)] * workers)
    elapsed = max(time.monotonic() - t0, 1e-6)
    total_nodes = sum(r["nodes"] for r in results)
    agg_nps = int(total_nodes / elapsed)
    singles = [r["nps"] for r in results]
    mean_single = sum(singles) / len(singles)
    ideal = mean_single * workers
    eff = (agg_nps / ideal) if ideal else 0.0
    return {"workers": workers, "cpu_count": cpu, "start_method": start_method,
            "elapsed_seconds": round(elapsed, 2),
            "total_nodes": total_nodes,
            "aggregate_nps": agg_nps,
            "mean_single_nps": int(mean_single),
            "worker_nps": singles,
            "scaling_efficiency": round(eff, 3)}


def evaluation_consistency(samples=200, net=None) -> dict:
    """Antisimetri evaluator: posisi vs mirror-nya harus berlawanan tanda.

    2.0: TinyNNUE memakai perspektif STM sehingga antisimetri dijamin BY
    KONSTRUKSI — drift yang diukur harus 0 eksak untuk semua posisi.
    """
    import random
    from .eval import evaluate
    rng = random.Random(7)
    board = chess.Board()
    max_drift = 0
    checked = 0
    for _ in range(samples):
        if board.is_game_over():
            board = chess.Board()
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        if not board.is_valid():
            continue
        drift = abs(evaluate(board, net) + evaluate(board.mirror(), net))
        max_drift = max(max_drift, drift)
        checked += 1
    return {"samples": checked, "max_drift": max_drift,
            "within_epsilon": max_drift <= 1}


def main():
    parser = argparse.ArgumentParser(description="Run Kestrel tactical/performance sanity benchmarks")
    parser.add_argument("--quick", action="store_true", help="Use a smaller search budget")
    parser.add_argument("--workers", type=int, default=0,
                        help="jumlah proses paralel untuk NPS agregat (0 = lewati)")
    args = parser.parse_args()
    tactical = tactical_suite(time_ms=800 if args.quick else 3000,
                              node_limit=30000 if args.quick else 200000)
    special = special_move_suite()
    performance = performance_suite(node_budget=10000 if args.quick else 50000)
    consistency = evaluation_consistency(samples=40 if args.quick else 200)
    report = {"tactical": tactical, "special_moves": special,
              "performance": performance, "evaluation_consistency": consistency}
    if args.workers and args.workers > 1:
        report["parallel_performance"] = parallel_performance_suite(
            node_budget=10000 if args.quick else 50000, workers=args.workers)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if tactical["accuracy"] >= 0.6 and special["pass"] and consistency["within_epsilon"] else 1)


if __name__ == "__main__":
    main()
