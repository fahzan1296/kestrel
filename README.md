# KESTREL 1.6.0 Experimental

> **Baru di 1.6:** trainer unified (self-play + learning + REDA dalam SATU loop), eval otomatis & mendetail champion vs challenger (SPRT, Elo CI, per-opening, benchmark), knowledge base (opening book & pengalaman posisi hasil self-play), bot Lichess, CLI penuh pengganti konfigurasi UI, jaminan depth tinggi (`min_depth` + grace time), dan evaluator bitboard ~2x lebih cepat (NPS +29-36%). Lihat `QUICKSTART_1.6.md` dan `CHANGELOG_1.6.md`.

CPU-first chess engine and self-play lab written in Python. The project prioritizes a transparent search/evaluation stack, reproducible experiments, and an optional localhost dashboard. It is an experimental engine, not a Stockfish-strength engine and not an Elo claim.

## What is included

- **Chess state:** `python-chess` is the authority for legal moves, game outcomes, FEN and PGN.
- **Evaluation:** handcrafted material, tapered piece-square tables, mobility, pawn structure, bishop pair, rook placement and king safety. `OutcomeModel` can add a bounded residual learned from PGN outcomes.
- **Search:** iterative deepening, negamax alpha-beta, PVS, aspiration windows with progressive widening, transposition table, MVV-LVA move ordering, quiescence (capture-only generation, all check evasions), killer/history ordering, bounded late-move reduction and pruning, conservative late-move pruning, bounded check extensions, mate-distance pruning, adaptive null-move pruning, and a bounded per-search evaluation cache. Single move generation per node; only completed iterative-deepening depths are published. Results include PV, NPS, TT hit rate, qnode ratio, stop reason and completed depth.
- **Budgets and telemetry:** monotonic time and node limits; only completed iterative-deepening depths are published. Results include PV, NPS, TT hit rate, qnode ratio, stop reason and completed depth.
- **Arena:** paired-color matches, PGN output, score-rate confidence interval and an explicitly approximate trinomial SPRT.
- **Self-play monitor:** optional FastAPI dashboard with a live board per game, process-isolated workers, event logs, evaluation trend, batch progress and PGN/SQLite persistence.

The small NumPy MLP is retained as an experimental component. Its output head starts at zero; an untrained random network must not be treated as a trained evaluator. The dashboard and demo use the handcrafted baseline plus an optional validated `OutcomeModel` instead.

## Install

Python 3.10 or newer is required.

```bash
# Runtime dependencies and editable package
python3 -m pip install -e .

# Development and tests
python3 -m pip install -e ".[dev]"

# Optional web monitor
python3 -m pip install -e ".[monitor]"
```

For systems where editable installs are inconvenient, install runtime packages with `python3 -m pip install -r requirements.txt` and run commands from this directory.

## Unified training (the 1.6 way)

```bash
python3 -m kestrel.cli train --generations 10   # full loop from config.toml
python3 -m kestrel.cli train --quick            # smoke test (~2 min)
```

One command runs, per generation: self-play (parallel workers, champion
model + knowledge book) -> knowledge update -> outcome learning -> REDA
distillation (in-memory) -> automatic champion-vs-challenger verification
(balanced match, SPRT, Wilson/Elo CI, per-opening breakdown, tactical/NPS/
consistency benchmarks) -> atomic promote or keep-champion. All budgets come
from `config.toml`; any CLI flag overrides them per invocation — no more
1x-configuration-per-UI-batch.

Full CLI reference (config, selfplay, reda, eval, verify, bench, book, model,
lichess, dashboard): see `QUICKSTART_1.6.md`.

## Run tests and benchmarks

```bash
python3 -m pytest -q tests
python3 -m kestrel.pipeline --quick          # tests -> train(1 gen) -> benchmark -> stockfish
python3 -m kestrel.cli verify                # strength proof vs handcrafted baseline
python3 -m kestrel.cli bench
```

The acceptance test includes perft, terminal positions, evaluator symmetry, mate search, free-piece tactics, search budgets, repetition handling, arena statistics and a short match. The PGN learner has its own round-trip/training tests. A passing test suite checks correctness properties; it does **not** establish playing strength.

## Run self-play

```bash
python3 -m kestrel.cli selfplay --games 12 --learn   # recommended
python3 scripts/selfplay_demo.py 12                  # 1.5 demo wrapper (now config-driven)
```

`--learn` updates the knowledge base (opening book + position experience) from
the produced games; the book is then used automatically by later self-play,
training and the Lichess bot. Inspect it with `python3 -m kestrel.cli book show`.

Outputs are written under `results/`: `games.pgn`, `kestrel.db` and `analytics.png`. Game openings are selected deterministically from a seeded opening set, rather than repeating the same opening mechanically every five game IDs.

## Live dashboard

Install the optional dependency group, then start the monitor:

```bash
python3 -m pip install -e ".[monitor]"
python3 -m kestrel.dashboard
```

Open `http://127.0.0.1:8765` on the same machine. The service binds to loopback by default; it is not a public web service. To view a VPS dashboard from your local computer, use an SSH tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 USER@YOUR_SERVER
```

Then open `http://127.0.0.1:8765` locally. Do not expose the development dashboard directly to the public internet without adding authentication, TLS and deployment hardening.

The dashboard controls game count, parallel worker count, per-move time, depth and ply cap. Each game runs in its own child process. For CPU-bound search, start with a worker count at or below the VPS logical CPU count; increasing workers beyond available CPU capacity often reduces per-game search speed and may lower total throughput. Compare 1, 2, 4, ... workers with the same workload before choosing a production value.

API endpoints: `GET /api/state`, `GET /api/health`, `POST /api/run`, `POST /api/stop`, plus 1.6: `POST /api/train`, `POST /api/train/stop`, `GET /api/train/state` (the unified trainer behind the CLI). Dialog defaults follow `config.toml`.

## Train the outcome model

```bash
python3 -m kestrel.learning results/games.pgn \
  --epochs 10 --lr 0.015 --output results/learned_eval.json
```

The learner is a small logistic model trained on game outcomes. It is not reinforcement learning, does not provide move-level engine labels, and can overfit self-play data. The metadata records corpus size, the train/validation split method, metrics and epoch history. With fewer than five games, validation is position-level and is not independent of game identity. Do not promote a checkpoint based only on training metrics; compare it against the baseline in balanced games and retain a separate validation set.

## Important limitations

- Python and `python-chess` make this implementation suitable for experimentation, not competitive high-performance chess search.
- Self-play results are not independent samples and cannot, by themselves, establish Elo improvement.
- The SPRT helper uses a shared empirical draw-rate approximation. Treat it as a screening statistic, not a substitute for a carefully specified engine-testing protocol.
- The dashboard is an observability tool. Its `NPS` values and game throughput should be measured under controlled, comparable machine load.
- No NNUE or trained neural-network checkpoint is bundled as a validated strength model.

## REDA — Rare Experience Deep Adaptation (experimental)

REDA is a bounded sparse residual adapter that leaves the existing evaluator's
base `OutcomeModel.weights` and `bias` untouched. It can adjust only a small
`micro_weights` vector (default max 3 active features, each bounded to +/-0.02).
Updates require verified White-relative search-score targets and a held-out
position validation split; a candidate is committed only when validation MSE improves.

The residual is embedded in the same `learned_eval.json` model artifact, so no
separate experience database is required for the adapted weights. It is loaded by
`OutcomeModel` and automatically participates through the existing evaluator.

Current limitation: this repository does not contain a trained NNUE network.
`TinyMLP` is a neutral-output numpy MLP, and REDA currently adapts the compact
handcrafted-feature outcome model residual, not NNUE tensors. The dashboard now
has a candidate-mining/deep-verification job, manual trigger, and opt-in
"Run REDA after batch" checkbox. It scans PGNs in two passes with bounded fingerprint tracking (100k entries by default),
keeps only a small ranked heap of candidate board snapshots, then deep-searches
the selected candidates. It uses Kestrel's own deeper scores as targets (search-score
distillation, not independent truth), with a deterministic position-level holdout
that may still be correlated within games. MSE improvement is a gate, not proof of
strength gain. A balanced color-swapped mini-match (8 games by default) rejects
updates with score rate below 0.35 as a catastrophic-regression safety filter; this
is not an Elo estimate. Accepted checkpoints are atomically replaced and the
previous model is saved as `learned_eval.json.reda_backup` when one exists. The
evaluator residual is now applied even when the dashboard worker runs without a
TinyMLP; previously that code path silently skipped the learned model.


## Automatic model verification & versioning (1.6)

Every training generation produces a challenger checkpoint
(`results/checkpoints/gen_<lineage>_tryNN_learned_eval.json`) that is matched
against the reigning champion (color-balanced, fixed seeded opening set): W/D/L,
score rate with Wilson 95% CI, Elo delta with CI, trinomial SPRT screening
(H0 elo0 / H1 elo1 from `[eval]`), per-opening breakdown, and tactical/NPS/
consistency benchmarks for BOTH models. The decision — PROMOTE only when the
challenger clears the gates (SPRT H1 or score rate >= `eval.min_score_rate`,
no tactical regression, consistency intact, NPS not collapsed) — is recorded in
`results/verification_gen_<label>.json|.md` with its reasons. `results/champion.json`
tracks the champion lineage; `python3 -m kestrel.cli model rollback` restores
any previous champion. Reports state honestly that small-match score rates are
indicative, not final Elo estimates.

Manual head-to-head of any two model files: `python3 -m kestrel.cli eval`.

## Depth guarantee & raw performance (1.6)

`Searcher(min_depth=N, grace_ms=M)` keeps iterating (restarting time-cut
iterations, extending the deadline up to `time_ms + grace_ms`) until the
requested depth is COMPLETED; node limits are never relaxed and
`info["min_depth_met"]` is reported. The evaluator now reads bitboards directly
(piece masks + popcount) with identical outputs to 1.5 (verified: zero drift on
406 random positions, exact mirror antisymmetry): handcrafted eval ~1.9x faster,
feature extraction ~3.6x faster, search NPS +29-36%.

## Lichess bot (1.6)

```bash
python3 -m kestrel.cli lichess --token YOUR_TOKEN      # play on lichess.org
python3 -m kestrel.cli lichess --dry-run               # offline decision check
```

Accepts/declines challenges per `[lichess]` policy (standard chess only, rating
gap, bot/human), plays with the champion model + knowledge book + min-depth
guarantee, claims draws when drawable and not winning, resigns after repeated
losing moves, and saves each game's PGN under `results/lichess/`. The account
must be upgraded to a BOT account — the bot offers the upgrade automatically on
first run. Token scopes: `bot:play`.

## Result of the 1.6 acceptance run

A real end-to-end smoke run ships in `results/smoke/`: two verification reports
(one REJECT at score 0.25, one PROMOTE at 0.5 — both decision paths exercised),
`train_report_*.json`, `training_log.jsonl`, the learned knowledge database and
the console log. The full test suite (11 tests) passes.
