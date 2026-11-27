# Kestrel Upgrade Notes (1.1 pass)

## Changes made
- Q-search no longer uses stand-pat or capture-only filtering while the side to move is in check; it considers all legal evasions and disables delta pruning during check.
- Search treats 3-fold repetition and claimable 50-move draws as draw scores.
- Mate-distance scores are normalized for transposition-table storage/retrieval.
- TT has a configurable fixed cap; it updates existing keys and stops inserting new unique keys after capacity instead of sorting/evicting the whole table repeatedly.
- Search metrics include TT hits/misses, beta cutoffs, qnodes, and elapsed time.
- `Player` keeps one `Searcher` per player instance so TT/killer/history data can be reused between moves in the same game.
- Added a reproducible PGN outcome learner (`kestrel.learning`): 11 hand-inspectable features, SGD logistic outcome training, JSON model persistence, capped evaluator residual.
- Pipeline learns from available PGNs, then compares baseline Champion against learned Challenger; the selfplay script loads a trained model when available.
- Live dashboard batch view/events are cleared at batch start while IDs remain sequential in-process. Added batch analytics (elapsed, throughput, average search depth, nodes/game, white/draw/black result counts) and worker PID labels.

## Limits and honest status
- This is an engineering upgrade, not evidence that Kestrel is now a strong chess engine.
- The outcome learner is deliberately modest and currently has no held-out validation, cross-validation, confidence calibration, or model rollback selection. Its residual is capped; it must earn its place through paired matches and benchmark results.
- The new tests were syntax-checked only in this environment. Full tests could not run because `python-chess` is not installed and package installation failed due to unavailable DNS/network. Run both test scripts in an environment with project dependencies before deploying.
- No SSH/server changes were performed; this ZIP is a local source package for you to deploy after testing.
