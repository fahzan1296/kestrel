"""KESTREL — Test suite (Section 44: setiap formula wajib unit/boundary test)."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import chess
from kestrel.core import perft, is_terminal
from kestrel.eval import evaluate, stm_eval, TinyMLP, handcrafted
from kestrel.search import Searcher, _tt_key
from kestrel.arena import play_game, match, Player, wilson_ci, sprt
from kestrel.memory import AdaptiveMemory

PASS = 0; FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name} {extra}")
    else: FAIL += 1; print(f"  [FAIL] {name} {extra}")

def test_engine_acceptance():
    global PASS, FAIL
    PASS = 0
    FAIL = 0
    print("== 1. Chess Core / Perft ==")
    b = chess.Board()
    check("perft(1) == 20", perft(b, 1) == 20)
    check("perft(2) == 400", perft(b, 2) == 400)
    check("perft(3) == 8902", perft(b, 3) == 8902)
    b2 = chess.Board("8/8/8/8/8/5k2/6q1/6K1 w - - 0 1")
    check("checkmate detection U=-1", is_terminal(b2) == -1)
    b3 = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    check("stalemate detection U=0", is_terminal(b3) == 0)

    print("== 2. Evaluator ==")
    net = TinyMLP()
    check("param < 1M", net.params() < 1_000_000, f"({net.params()})")
    check("param di target 100k-300k", 100_000 <= net.params() <= 300_000)
    start_eval = evaluate(chess.Board(), net)
    check("eval finite", abs(start_eval) < 10**6, f"({start_eval})")
    check("output net finite", abs(net.forward(chess.Board())) < 10**6)
    # simetri: posisi mirrored harus ~sama
    pos = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mir = pos.mirror()
    check("mirror antisymmetry |H+H'| < 30cp", abs(handcrafted(pos) + handcrafted(mir)) < 30,
          f"({handcrafted(pos)} vs {handcrafted(mir)})")
    # Checkmate score must be positive for White when Black is mated, and vice versa.
    mate_black = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    check("mate score has correct White perspective", mate_black.is_checkmate() and handcrafted(mate_black) > 0)
    check("mirrored mate score is negative", mate_black.mirror().is_checkmate() and handcrafted(mate_black.mirror()) < 0)
    check("TinyMLP starts neutral", TinyMLP().forward(chess.Board()) == 0.0)
    check("TT key includes halfmove clock", _tt_key(chess.Board()) != _tt_key(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 20 1")))
    check("TinyMLP custom aux dimension is supported", TinyMLP(aux_dim=4).forward(chess.Board()) == 0.0)
    check("TinyMLP accepts integral float aux dimension", TinyMLP(aux_dim=4.0).forward(chess.Board()) == 0.0)  # pyright: ignore[reportArgumentType]

    print("== 3. Search ==")
    invalid_budget_rejected = False
    try:
        Searcher(node_limit=-1)
    except ValueError:
        invalid_budget_rejected = True
    check("invalid search budgets rejected", invalid_budget_rejected)
    s = Searcher(node_limit=60000, time_ms=5000)
    m1, info1 = s.best_move(chess.Board())
    assert m1 is not None
    check("legal best move dari startpos", m1 in chess.Board().legal_moves, f"({m1}, d{info1['depth']})")
    check("search returns telemetry metrics", all(k in info1 for k in ("pv", "tt_hit_rate", "qnode_ratio", "timed_out", "completed_depth")))
    check("PV starts with chosen move", bool(info1["pv"]) and info1["pv"][0] == m1.uci())
    # mate in 1
    mate1 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    s2 = Searcher(node_limit=200000, time_ms=8000)
    m2, info2 = s2.best_move(mate1)
    assert m2 is not None
    b4 = mate1.copy(); b4.push(m2)
    check("menemukan mate in 1 (Re8#)", b4.is_checkmate(), f"({m2})")
    # mengambil ratu gratis
    # ratu tak terlindungi di d5 -> pawn e4xd5
    free = chess.Board("rnb1kbnr/pppp1ppp/8/3q4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    s3 = Searcher(node_limit=150000, time_ms=60000)
    m3, info3 = s3.best_move(free)
    assert m3 is not None
    check("mengambil ratu gratis (exd5)", m3.uci() == "e4d5", f"({m3}, d{info3['depth']}, n{info3['nodes']})")
    # root tidak terkorupsi
    bref = chess.Board().fen()
    Searcher(node_limit=500).best_move(chess.Board())
    check("root state tidak berubah", chess.Board().fen() == bref)
    # Qsearch must search legal check evasions rather than stand pat/capture-only.
    check_pos = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    sq = Searcher(time_ms=1000, max_depth=2)
    sq.start = time.time()
    qv = sq.qsearch(check_pos, -100000, 100000)
    check("qsearch in-check evasion search", isinstance(qv, int) and sq.nodes > 1, f"(nodes={sq.nodes})")
    sq2 = Searcher(time_ms=1000, max_depth=2)
    sq2.start = time.time()
    fail_low = sq2.qsearch(check_pos, 100000, 100001)
    check("qsearch in-check fail-low returns searched score", fail_low < 100000,
          f"(score={fail_low})")
    # Threefold repetitions are treated as draw in search nodes.
    rep = chess.Board()
    for _ in range(2):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            rep.push_uci(uci)
    sr = Searcher(time_ms=1000, max_depth=1)
    sr.start = time.time()
    check("threefold repetition score=0", sr.search(rep, 1, -100000, 100000) == 0)
    # budget
    s4 = Searcher(node_limit=5000, time_ms=20000)
    s4.best_move(chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"))
    check("node budget dihormati (<= 1.3x)", s4.nodes <= 6500, f"(nodes={s4.nodes})")
    limited = Searcher(node_limit=1, time_ms=10000, max_depth=8)
    limited_move, limited_info = limited.best_move(chess.Board())
    assert limited_move is not None
    check("node-limit returns legal fallback move", limited_move in chess.Board().legal_moves)
    check("node-limit is reported distinctly", limited_info["node_limited"] and limited_info["stop_reason"] == "node_limit")
    check("incomplete depth is not published", limited_info["completed_depth"] == 0)

    print("== 4. Adaptive Memory ==")
    import os
    if os.path.exists("/tmp/kestrel_test.db"): os.remove("/tmp/kestrel_test.db")
    mem = AdaptiveMemory("/tmp/kestrel_test.db")
    bb = chess.Board(); mem.remember(bb, "e2e4", 0.5, 10)
    row = mem.db.execute("SELECT id FROM memory").fetchone()
    mem.activate(row[0])
    check("memory store+retrieve", len(mem.retrieve(bb)) > 0)
    mem.disable(row[0])
    check("disable memory", len(mem.retrieve(bb)) == 0)
    check("influence bounded A_max", mem.influence(bb) <= mem.a_max)

    print("== 5. Arena / Statistik ==")
    w_, d_, l_ = 5, 8, 2
    lo, hi = wilson_ci(w_, d_, l_)
    check("wilson CI masuk akal", 0 < lo < (w_+0.5*d_)/15 < hi < 1, f"({lo:.2f},{hi:.2f})")
    check("SPRT accepts decisive challenger wins", sprt([(200, 0, 0)]) == "H1")
    check("SPRT rejects decisive challenger losses", sprt([(0, 0, 200)]) == "H0")
    check("SPRT treats draws as no Elo evidence", sprt([(0, 200, 0)]) == "continue")

    p1 = Player("Champion", time_ms=120, max_depth=8)
    p2 = Player("Challenger", time_ms=120, max_depth=8)
    res, pgn, plies = play_game(p1, p2, temperature=0.0, seed=7)
    check("game selesai tanpa illegal move", res in ("1-0", "0-1", "1/2-1/2"), f"({res}, {plies} plies)")
    check("PGN valid", "1." in pgn)
    short_res, short_pgn, short_plies = play_game(p1, p2, seed=8, max_plies=1)
    check("ply-limit game is adjudicated as draw", short_res == "1/2-1/2" and short_plies == 1 and '[Termination "Ply limit adjudicated as draw"]' in short_pgn)
    class NoMovePlayer:
        name = "BrokenEngine"
        def play(self, board, *args, **kwargs):
            return None, {}
    forfeited, forfeit_pgn, _ = play_game(NoMovePlayer(), p2, seed=9)  # pyright: ignore[reportArgumentType]
    check("engine returning no move forfeits instead of drawing",
          forfeited == "0-1" and 'Engine forfeit: no move returned' in forfeit_pgn)
    mem.save_game("Champion", "Challenger", res, plies, pgn, 7)
    check("persistensi game", mem.db.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1)

    print("== 6. Match Champion vs Challenger (8 games) ==")
    r = match(p1, p2, games=8, temperature=0.2, verbose=True)
    check("match balanced & lengkap", r["games"] == 8, f"(score={r['score_rate']:.2f}, eloΔ={r['elo_delta']:+.1f}, sprt={r['sprt']})")

    assert FAIL == 0, f"{FAIL} acceptance check(s) failed out of {PASS + FAIL}"


if __name__ == "__main__":
    test_engine_acceptance()
    print(f"\nTOTAL: {PASS} pass, {FAIL} fail")
