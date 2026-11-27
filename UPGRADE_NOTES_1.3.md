# Upgrade notes — Kestrel 1.3.0

1. Install/update package dependencies from the project root:
   `python3 -m pip install -e ".[dev,monitor]"`
2. Run tests before starting any self-play batch:
   `python3 -m pytest -q tests`
3. Start the optional monitor:
   `python3 -m kestrel.dashboard`
4. Begin worker scaling with a low concurrency (for example 1 or 2). Run the same game/time/depth configuration at several worker counts and compare total games/minute, aggregate nodes/second, per-move depth, and CPU usage. Do not assume more child processes means more throughput.
5. Keep `results/games.pgn`, `results/kestrel.db`, and any learned checkpoint backed up before replacing code or training a new model.

The 1.3.0 changes correct budget unwinding, node accounting, quiescence fail-low scoring, evaluation orientation, draw statistics, failed-engine adjudication, unfinished ply-cap outcomes, and worker event isolation. The approximate SPRT remains explicitly experimental. No strength claim is made by this update.
