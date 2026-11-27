"""Focused regression tests for the outcome learner (run: python tests/test_learning.py)."""
import os, tempfile, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
from kestrel.learning import features, OutcomeModel, train_pgn
from kestrel.reda import RareExperienceAdapter


def test_outcome_learning():
    b=chess.Board()
    assert features(b).shape == (11,)
    assert all(float(x)==float(x) for x in features(b))
    model=OutcomeModel()
    assert abs(model.score(b)) < 1e-9
    # The learned residual must still affect evaluation when the dashboard has no MLP.
    from kestrel.eval import evaluate
    model.micro_weights[10] = 0.1
    assert evaluate(b, net=None, learned=model) != evaluate(b, net=None)
    model.micro_weights[10] = 0.0
    # Feature score should stay finite and serialize round-trip without schema drift.
    model.weights[0]=0.1
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"model.json"); model.save(p,{"test":True})
        loaded=OutcomeModel().load(p)
        assert abs(loaded.score(b)-model.score(b)) < 1e-6
    # Minimal PGN outcome corpus tests trainer path end-to-end.
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"tiny.pgn")
        open(p,"w").write('[Event "Tiny"]\n[Result "1-0"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n')
        trained, meta=train_pgn([p],epochs=2,seed=7)
        assert meta["samples"] > 0 and trained.weights.shape == (11,)
    # REDA micro-adaptation: frozen base weights, sparse bounded residual, validation gate.
    base_model = OutcomeModel()
    base_weights = base_model.weights.copy()
    train_boards = [chess.Board("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1"),
                     chess.Board("4k3/8/8/8/3q4/8/8/4K3 w - - 0 1")]
    val_boards = [chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1"),
                  chess.Board("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1")]
    train_samples = [(b, 12.0 * float(features(b)[0])) for b in train_boards]
    validation_samples = [(b, 12.0 * float(features(b)[0])) for b in val_boards]
    result = RareExperienceAdapter(base_model, max_active=2, learning_rate=0.05,
                                 max_abs_weight=0.02, min_improvement=1e-5).adapt(
                                     train_samples, validation_samples, steps=100)
    assert result.accepted, result
    assert len(result.updated_features) <= 2
    assert (base_model.weights == base_weights).all(), "REDA must not mutate base model weights"
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"model_with_reda.json")
        base_model.save(p)
        restored=OutcomeModel().load(p)
        assert (restored.weights == base_weights).all()
        assert (restored.micro_weights == base_model.micro_weights).all()
        assert abs(restored.micro_bias - base_model.micro_bias) < 1e-12
    # Insufficient evidence should fail closed without touching the adapter.
    before_micro = base_model.micro_weights.copy()
    rejected = RareExperienceAdapter(base_model).adapt(train_samples[:1], validation_samples)
    assert not rejected.accepted and (base_model.micro_weights == before_micro).all()
    # REDA PGN mining deduplicates clocks and ranks unique positions (integration test).
    from kestrel.reda_pipeline import mine_candidates
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"mining.pgn")
        open(p,"w").write('[Event "Mine A"]\n[Result "1-0"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n\n[Event "Mine B"]\n[Result "0-1"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 0-1\n')
        mined, unique_count, frequency = mine_candidates(p, limit=12)
        assert mined and unique_count > 0
        assert len({row[2] for row in mined}) == len(mined)
        assert all(len(fp) == 24 for _, _, fp, _ in mined)
    # Training reports validation provenance and metrics, even with a tiny one-game corpus.
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"tiny.pgn")
        open(p,"w").write('[Event "Tiny"]\n[Result "1-0"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n')
        trained, meta=train_pgn([p],epochs=3,seed=9,batch_size=16)
        assert meta["samples"] > 0
        assert meta["epochs_completed"] >= 1
        assert meta["validation_mode"] == "position_holdout_small_corpus"
        assert meta["train_metrics"]["logloss"] is not None


if __name__ == "__main__":
    test_outcome_learning()
    print("learning tests: PASS")
