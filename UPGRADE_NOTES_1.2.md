# Kestrel Advanced 1.2 — Search/Training reliability update


## Search
- Monotonic timing instead of wall-clock timing, reducing sensitivity to system clock adjustments.
- Amortized deadline checks (clock polled every 64 nodes) to reduce hot-path timer overhead while retaining node-limit checks.
- Depth/age-aware transposition-table replacement with a deterministic sampled victim policy.
- Aspiration windows with full-window retry on fail-low/fail-high from depth 3 onward.
- Root PVS (full window for first move, null-window for subsequent moves with re-search).
- Principal variation extraction from legal TT moves.
- Search info now reports PV, TT hit rate, qnode ratio, timeout state, and completed depth.


## Learning
- Deterministic mini-batch SGD instead of per-sample updates.
- Validation split by whole games when at least five games exist; position split is used for smaller corpora and explicitly marked as non-independent.
- Early stopping and best-validation checkpoint selection.
- Log-loss, Brier score, coarse threshold accuracy, per-epoch train/validation history, corpus result counts, and split provenance are recorded.
- CLI accepts validation fraction, batch size, and patience.
- Learned evaluator bias is now included in the bounded residual evaluation.


## Limitations / validation
- These changes are source-level and syntax-validated in the development environment.
- Runtime chess tests require `python-chess`, which could not be installed because the environment had no network/DNS access.
- Do not treat the learning validation metrics as playing-strength measurements. For fewer than five games, the split is position-level and is not independent of game identity.
