# Kestrel 1.2.0


### Search quality/efficiency
- Root PVS pass introduced.
- Aspiration windows start at depth 3, with full-window re-search after fail-low/fail-high.
- TT replacement is depth/age aware instead of refusing inserts at full capacity.
- PV reconstructed from legal TT moves.
- Clock polling uses monotonic time and amortized polling to reduce timer cost.
- Added TT hit rate, qnode ratio, PV, completed-depth and timeout telemetry.

### Learning pipeline
- Vectorized mini-batch logistic training.
- Game-based validation split when the corpus is large enough; clearly labeled position-level split for tiny data.
- Early stopping, best checkpoint selection and validation metrics.
- Trainer reports sample/game counts, result distribution, metrics and per-epoch history.
- CLI exposes validation fraction, batch size and patience.
- Learned model bias now contributes to the bounded residual evaluator.

### Validation
- All Python sources and tests compile and parse.
- ZIP archive integrity verified.
- Runtime test suite not run: python-chess dependency unavailable in the isolated environment (network/DNS failure).
