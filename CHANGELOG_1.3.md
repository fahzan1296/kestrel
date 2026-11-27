# Kestrel 1.3.0

## Search correctness and budget handling
- Stop propagation now unwinds active search branches immediately after time/node exhaustion instead of evaluating every remaining sibling branch.
- Root search retains a legal fallback move when the first iteration cannot complete, without reporting an incomplete depth as completed.
- Root move boundaries force deadline checks; internal nodes retain amortized monotonic-clock polling.
- Node accounting no longer counts depth-zero nodes twice when they enter quiescence; `qnode_ratio` and NPS are more meaningful.
- Transposition-table keys incorporate the half-move clock; repeated-path TT hits are used for move ordering but not score cutoffs.
- Full-table replacement samples a bounded number of keys without copying the entire table.
- LMR avoids reducing quiet checking moves. Delta pruning avoids discarding checking captures. Quiescence orders tactical moves and searches all legal check evasions.
- In-check quiescence fail-low now returns the best searched evasion score, not the caller's alpha bound; this fixes a subtle score-overstatement bug.
- History-heuristic counters are capped to prevent unbounded growth over very long self-play sessions, and search budget parameters are validated.

## Evaluation and learning
- Checkmate scores now use the correct White-relative sign.
- Untrained TinyMLP output is neutral instead of injecting random position bias. Dashboard/demo self-play defaults to the handcrafted evaluator.
- `TinyMLP` supports custom auxiliary feature dimensions safely.
- Outcome model load/save validates weight shape and finite values; trainer hyperparameters are checked before training.
- PGN sampling alternates side-to-move parity across games so the tempo feature is not constant in the training corpus.

## Arena and experiments
- Wilson intervals now use the Wilson score formula and return the full interval for an empty sample.
- SPRT uses normalized W/D/L outcome probabilities with correctly signed Elo hypotheses and a shared draw-rate approximation.
- Ply-limit games are explicitly adjudicated as draws and carry a PGN termination header instead of being saved as unfinished `*` results.
- A live game where an engine fails to return a move is recorded as a forfeit, not silently converted to a draw. PGN evaluation comments are consistently White-relative.
- Evaluation consistency benchmark now checks White-relative mirror antisymmetry; the former zero-drift placeholder was removed.

## Dashboard and test infrastructure
- Worker failures are reported to the collector and counted instead of silently disappearing.
- Batch IDs isolate late queue events from a subsequent run; completed PGNs are still persisted.
- Queued games are represented before worker start, stop marks queued/active games, and the coordinator waits for terminal collector events before marking a child failed.
- Worker startup errors are surfaced. Health/state endpoints expose CPU/runtime information and failure counts.
- Game openings are selected deterministically but pseudo-randomly per seed instead of repeating by fixed modulo game ID.
- Acceptance checks are pytest-discoverable; CI and the pipeline invoke the same test suite.

## Validation note

Source compilation and structural checks were run while preparing this archive. Runtime chess tests could not be executed in this environment because `python-chess` was unavailable and package-index networking was blocked. Install the dependencies and run `python3 -m pytest -q tests` before using this revision for experiments.
