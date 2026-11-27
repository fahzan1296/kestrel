# Kestrel 1.4 Experimental — REDA

- Added post-self-play rare-position candidate discovery from PGN telemetry.
- Deep-verifies selected candidates with bounded iterative search and White-relative score targets.
- Adds dashboard REDA job controls, progress status, validation MSE, accepted features, and optional auto-run after batch.
- Adds concurrency guards between REDA and self-play model loading/saving.
- Added a balanced color-swapped mini-match regression guard before commit (safety threshold; not an Elo estimate).
- PGN candidate mining is two-pass and bounded (fingerprint counter/processed set capped at 100k by default; only top candidate boards retained).
- Accepted updates persist as sparse residuals in `results/learned_eval.json`; base linear evaluator weights are immutable to REDA.
- REDA is explicitly experimental: score distillation, position-level validation, and not NNUE adaptation or proof of Elo gain.
