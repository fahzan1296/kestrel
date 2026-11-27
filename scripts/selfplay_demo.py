"""KESTREL — Self-play batch + Live Analytics (grafik matplotlib).

2.0: budget pencarian mengikuti config.toml [selfplay]; evaluator selalu
TinyNNUE (champion atau baseline bawaan). Untuk produksi gunakan
`python -m kestrel.cli selfplay` atau `train`.
"""
import contextlib
import os
import random
import re
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import chess
import chess.pgn
import matplotlib.pyplot as plt

from kestrel.arena import OPENINGS, Player, play_game
from kestrel.config import load_config
from kestrel.nnue import TinyNNUE
from kestrel.versioning import CheckpointStore

CFG = load_config()
SP = CFG.selfplay
# 2.0: evaluator AI selalu aktif — champion bila ada, else baseline bawaan.
net = CheckpointStore(CFG.results_path).load_champion()
with contextlib.suppress(ValueError, KeyError, TypeError):
    net = TinyNNUE.load(str(CFG.champion_model_path))
book = None
if SP.use_book and CFG.knowledge_db_path.exists():
    try:
        from kestrel.knowledge import KnowledgeBase

        book = KnowledgeBase(CFG.knowledge_db_path, CFG.knowledge.book_max_plies,
                             CFG.knowledge.book_min_games)
    except Exception:
        book = None

player_kwargs: dict[str, Any] = {
    "net": net, "time_ms": SP.time_ms, "max_depth": SP.max_depth,
    "book": book, "min_depth": SP.min_depth, "grace_ms": SP.grace_ms,
}
p_white = Player("Kestrel-v2.0", **player_kwargs)
p_black = Player("Kestrel-v2.0", **player_kwargs)

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rng = random.Random(2024)

t0 = time.time()
plies_hist, eval_hist, nodes_hist, depth_hist = [], [], [], []
w = d = losses = 0
for i in range(GAMES):
    fen = OPENINGS[rng.randrange(len(OPENINGS))]
    seed = 5000 + i
    res, pgn, plies = play_game(p_white, p_black, fen, temperature=0.25,
                                seed=seed)
    try:
        with open("results/games.pgn", "a") as f:
            f.write(pgn + "\n\n")
    except OSError as exc:
        print(f"gagal menulis PGN: {exc}")
    if res == "1-0":
        w += 1
    elif res == "0-1":
        losses += 1
    else:
        d += 1
    plies_hist.append(plies)
    print(f"game {i+1:3d}: {res} ({plies} plies)")

# Ekstrak telemetri dari komentar PGN (eval/depth/nodes per move)
def _walk(g):
    node = g
    while node.variations:
        node = node.variation(0)
        yield node


ev, dp, nd = [], [], []
n_moves = 0
try:
    with open("results/games.pgn") as stream:
        game = chess.pgn.read_game(stream)
        while game:
            for node in _walk(game):
                m = re.match(r"(-?\d+)/d(\d+) n(\d+)", node.comment)
                if m:
                    ev.append(int(m.group(1)))
                    dp.append(int(m.group(2)))
                    nd.append(int(m.group(3)))
                    n_moves += 1
            game = chess.pgn.read_game(stream)
except OSError as exc:
    print(f"gagal membaca PGN: {exc}")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0, 0].plot(ev, lw=0.7, color="#2a9d8f")
axes[0, 0].set_title("Evaluasi per langkah (cp)")
axes[0, 0].set_xlabel("Ply")
axes[0, 1].hist(dp, bins=range(1, 12), color="#e76f51", edgecolor="white")
axes[0, 1].set_title("Distribusi kedalaman pencarian")
axes[0, 1].set_xlabel("Depth")
axes[1, 0].plot(nd, lw=0.7, color="#264653")
axes[1, 0].set_title("Node per langkah")
axes[1, 0].set_xlabel("Ply")
axes[1, 1].hist(plies_hist, bins=max(6, GAMES // 2), color="#e9c46a", edgecolor="white")
axes[1, 1].set_title("Panjang game (plies)")
axes[1, 1].set_xlabel("Plies")
for a in axes.flat:
    a.grid(alpha=0.3)
fig.suptitle(f"KESTREL Live Analytics — {GAMES} self-play games "
             f"(W:{w} D:{d} L:{losses}, draw rate {d/GAMES:.0%})", fontsize=12)
fig.tight_layout()
fig.savefig("results/analytics.png", dpi=150)
print(f"\nDone in {time.time()-t0:.0f}s — W:{w} D:{d} L:{losses}")
print(f"Total moves analyzed: {n_moves}; avg depth {sum(dp)/max(len(dp),1):.1f}; "
      f"avg nodes {sum(nd)/max(len(nd),1):.0f}")
print("Grafik: results/analytics.png | DB: results/kestrel.db")
