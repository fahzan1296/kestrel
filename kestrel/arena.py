"""KESTREL — Arena, self-play, rating & statistical validation (Section 21-26)."""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

import chess
import chess.pgn

from .eval import stm_eval
from .memory import AdaptiveMemory
from .search import Searcher

OPENINGS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    # 1.7: pembukaan tambahan untuk keragaman korpus cross/self-play
    # (di-APPEND; urutan lama tetap stabil untuk determinisme seed).
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",  # Italian
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",   # Ruy Lopez
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",        # French
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 4",       # Caro-Kann
    "rnb1kbnr/ppp1pppp/8/3q4/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",         # Scandinavian
    "rnbqkbnr/pp3ppp/4p3/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",      # Sicilian e6
    "rnbqkbnr/pp1ppp1p/6p1/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",    # Sicilian g6
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",        # King's Gambit
    "rnbqkbnr/pp3ppp/4p3/2pp4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 5",     # QGD
    "rnbqkb1r/pp3ppp/2p5/3p4/2pPP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 4 4",     # Slav
    "rnbqkbnr/pp1ppppp/8/2p5/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2",       # King's Indian
    "rnbqkbnr/pp2pppp/3p4/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 4",       # Pirc
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Two Knights
]


def expected_score(delta_r: float) -> float:
    return 1.0 / (1.0 + 10 ** (-delta_r / 400))


def elo_delta(p: float) -> float:
    """Convert score rate to Elo with a conservative finite-sample extreme cap."""
    p = float(p)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("score rate must be finite and between 0 and 1")
    if p <= 0.0:
        return -800.0
    if p >= 1.0:
        return 800.0
    return 400 * math.log10(p / (1.0 - p))


def wilson_ci(wins: int, draws: int, losses: int, z: float = 1.96):
    """Two-sided Wilson interval for the score rate (a draw is half a point)."""
    if not math.isfinite(float(z)) or z <= 0:
        raise ValueError("z must be a finite positive value")
    values = (wins, draws, losses)
    if any(int(v) != v or v < 0 for v in values):
        raise ValueError("W/D/L counts must be non-negative integers")
    n = int(wins + draws + losses)
    if n == 0:
        return (0.0, 1.0)
    p = (wins + 0.5 * draws) / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = (z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))) / den
    return (max(0.0, center - half), min(1.0, center + half))


def sprt(wdl, elo0=0.0, elo1=35.0, alpha=0.05, beta=0.10,
         draw_rate: float | None = None):
    """Approximate trinomial SPRT from W/D/L counts.

    H0 and H1 define the challenger's expected score at ``elo0``/``elo1``.
    A shared draw probability is estimated from the observed sample (or supplied
    explicitly); wins/losses then carry the likelihood evidence. This is an
    approximation, but unlike the previous implementation it uses valid,
    normalized W/D/L probabilities and the correct Elo sign.
    """
    if not all(math.isfinite(float(v)) for v in (elo0, elo1, alpha, beta)):
        raise ValueError("elo0, elo1, alpha and beta must be finite")
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("alpha and beta must be between 0 and 1")
    counts = []
    for item in wdl:
        if len(item) != 3:
            raise ValueError("Each WDL item must be (wins, draws, losses)")
        w, d, losses = item
        if any(int(v) != v or v < 0 for v in (w, d, losses)):
            raise ValueError("W/D/L counts must be non-negative integers")
        counts.append((int(w), int(d), int(losses)))
    wins = sum(x[0] for x in counts)
    draws = sum(x[1] for x in counts)
    losses = sum(x[2] for x in counts)
    n = wins + draws + losses
    if n == 0:
        return "continue"

    e0, e1 = expected_score(elo0), expected_score(elo1)
    # Jeffreys-style smoothing avoids a zero draw-probability estimate on small
    # batches. Both hypotheses share this nuisance parameter.
    observed_draw_rate = ((draws + 0.5) / (n + 1.5)
                         if draw_rate is None else float(draw_rate))
    if not 0.0 <= observed_draw_rate < 1.0:
        raise ValueError("draw_rate must be in [0, 1)")
    max_draw = min(2.0 * min(e0, 1.0 - e0), 2.0 * min(e1, 1.0 - e1))
    dprob = min(observed_draw_rate, max_draw * (1.0 - 1e-9))
    eps = 1e-12
    p0w, p0l = max(e0 - dprob / 2.0, eps), max(1.0 - e0 - dprob / 2.0, eps)
    p1w, p1l = max(e1 - dprob / 2.0, eps), max(1.0 - e1 - dprob / 2.0, eps)

    llr = wins * math.log(p1w / p0w) + losses * math.log(p1l / p0l)
    log_a = math.log(beta / (1.0 - alpha))
    log_b = math.log((1.0 - beta) / alpha)
    if llr >= log_b:
        return "H1"
    if llr <= log_a:
        return "H0"
    return "continue"


class Player:
    """Agent Kestrel: config search + evaluator + memory."""

    def __init__(self, name, net=None, time_ms=300, node_limit=0, max_depth=64,
                 memory: AdaptiveMemory | None = None, use_null_move=True,
                 use_lmr=True, seed=42, learned=None, min_depth=0, grace_ms=1000,
                 tt_size=None, book=None):
        self.name = name
        self.net = net
        self.time_ms = time_ms
        self.node_limit = node_limit
        self.max_depth = max_depth
        self.memory = memory
        self.use_null_move = use_null_move
        self.use_lmr = use_lmr
        self.seed = seed
        self.learned = learned
        self.book = book  # KnowledgeBase opsional (langkah pembuka statistik)
        self.book_moves_used = 0
        # Persistent per-player search state: TT, killers and history survive across moves.
        searcher_kwargs: dict[str, Any] = {
            "node_limit": self.node_limit, "time_ms": self.time_ms,
            "max_depth": self.max_depth, "use_null_move": self.use_null_move,
            "use_lmr": self.use_lmr, "learned": self.learned,
            "min_depth": min_depth, "grace_ms": grace_ms,
        }
        if tt_size is not None:
            searcher_kwargs["tt_size"] = int(tt_size)
        self.searcher = Searcher(self.net, **searcher_kwargs)

    def play(self, board: chess.Board, temperature: float = 0.0, rng=None,
             search_callback: Callable[[dict[str, Any]], None] | None = None):
        """Greedy search, atau sampling softmax Q/tau (Section 21)."""
        # 1.6: langkah book statistik di ply awal — pengetahuan hasil self-play.
        if self.book is not None and rng is not None:
            pick = self.book.book_move(board, rng)
            if pick is not None:
                uci, san, games, score = pick
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    self.book_moves_used += 1
                    return mv, {"book": True, "book_uci": uci, "book_san": san,
                                "book_games": games, "book_score": round(score, 3),
                                "depth": 0, "score_cp": 0, "nodes": 0, "nps": 0,
                                "pv": [uci]}
        self.searcher.on_info = search_callback
        mv, info = self.searcher.best_move(board.copy())
        if temperature > 0 and rng is not None and mv is not None:
            legal = list(board.legal_moves)
            # Q(s,a) dari statis evaluasi posisi hasil, softmax numerik-stabil
            raw = []
            for m in legal:
                b2 = board.copy()
                b2.push(m)
                raw.append(-stm_eval(b2, self.net, self.learned) / max(temperature, 1e-3))
            mx = max(raw)
            weights = [math.exp(v - mx) for v in raw]
            mv = rng.choices(legal, weights=weights, k=1)[0]
        return mv, info


def play_game(white: Player, black: Player, fen: str | None = None,
              temperature: float = 0.0, seed: int = 42,
              memory: AdaptiveMemory | None = None, max_plies: int = 300):
    """Mainkan satu game lengkap; return (result, pgn, plies)."""
    if int(max_plies) != max_plies or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    board = chess.Board(fen) if fen else chess.Board()
    rng = random.Random(seed)
    game = chess.pgn.Game()
    if fen:
        game.setup(board)
    node = game
    plies = 0
    forfeit_result = None
    white_score = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        mover_color = board.turn
        player = white if mover_color == chess.WHITE else black
        mv, info = player.play(board, temperature if plies < 12 else 0.0, rng)
        if mv is None:
            # A live position with legal moves but no engine move is a forfeit,
            # not a draw. Counting it as a draw would hide engine failures.
            if next(iter(board.legal_moves), None) is not None:
                forfeit_result = "0-1" if mover_color == chess.WHITE else "1-0"
            break
        board.push(mv)
        node = node.add_variation(chess.Move.from_uci(mv.uci()))
        if info.get("book"):
            # Langkah book: catat sumbernya; tidak ada telemetry pencarian.
            node.comment = f"book {info['book_san']} ({info['book_games']}g,{info['book_score']:.2f})"
        else:
            white_score = info["score_cp"] if mover_color == chess.WHITE else -info["score_cp"]
            # PGN telemetry is always White-relative, matching the evaluator and
            # dashboard chart. Searcher.score_cp itself remains side-to-move.
            node.comment = f"{white_score}/d{info['depth']} n{info['nodes']}"
        plies += 1
        if memory and board.is_capture(mv):
            memory.remember(board, str(mv), 0, white_score)
    if forfeit_result is not None:
        result = forfeit_result
        game.headers["Termination"] = "Engine forfeit: no move returned"
    elif board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "1/2-1/2"
        game.headers["Termination"] = outcome.termination.name if outcome else "Draw"
    elif plies >= max_plies:
        # A ply cap is an artificial stop, not an unfinished PGN (*). Adjudicate
        # as a draw consistently with the live dashboard and record the reason.
        result = "1/2-1/2"
        game.headers["Termination"] = "Ply limit adjudicated as draw"
    else:
        result = "1/2-1/2"
        game.headers["Termination"] = "Agent returned no move; adjudicated as draw"
    game.headers["Result"] = result
    game.headers["White"], game.headers["Black"] = white.name, black.name
    return result, str(game), plies


def match(champ: Player, chall: Player, games: int = 20,
          openings=None, memory: AdaptiveMemory | None = None,
          temperature: float = 0.15, verbose=True):
    """Pairwise match balanced warna + opening (Section 24). Returns dict."""
    if int(games) != games or games < 1:
        raise ValueError("games must be a positive integer")
    openings = OPENINGS if openings is None else openings
    if not openings:
        raise ValueError("openings must contain at least one FEN")
    w = d = losses = 0
    results = []
    for i in range(games):
        fen = openings[i % len(openings)]
        seed = 1000 + i
        if i % 2 == 0:  # challenger putih
            res, _, _ = play_game(chall, champ, fen, temperature, seed, memory)
            c_is_white = True
        else:
            res, _, _ = play_game(champ, chall, fen, temperature, seed, memory)
            c_is_white = False
        if res == "1/2-1/2":
            d += 1
        elif (res == "1-0") == c_is_white:
            w += 1
        else:
            losses += 1
        results.append(res)
        if verbose:
            print(f"  game {i+1:3d}: {res}  (W:{w} D:{d} L:{losses})")
    n = max(w + d + losses, 1)
    p = (w + 0.5 * d) / n
    lo, hi = wilson_ci(w, d, losses)
    return {"games": n, "challenger": {"W": w, "D": d, "L": losses},
            "score_rate": p, "elo_delta": elo_delta(p),
            "ci95": (round(lo, 3), round(hi, 3)),
            "sprt": sprt([(w, d, losses)])}
