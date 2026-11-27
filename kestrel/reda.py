"""REDA: Rare Experience Deep Adaptation (experimental, bounded residual layer).

Base OutcomeModel weights are never modified. REDA updates only a small residual
vector using verified search-score targets and commits only if held-out MSE improves.
This is a feature residual for the current evaluator, NOT NNUE training.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .learning import FEATURES, features, OutcomeModel
from .eval import handcrafted



@dataclass
class REDAResult:
    accepted: bool
    baseline_mse: float
    candidate_mse: float
    updated_features: list[str]
    reason: str



class RareExperienceAdapter:
    """Bounded sparse residual updater.

    Samples are (board, target_cp), with target_cp a verified deep-search score
    in White-relative centipawns. Do not feed shallow/noisy scores or game outcomes.
    """
    def __init__(self, model: OutcomeModel, max_active: int = 3,
                 max_abs_weight: float = 0.02, learning_rate: float = 0.01,
                 l2: float = 0.002, min_improvement: float = 1e-5):
        self.model = model
        self.max_active = max(1, min(int(max_active), len(FEATURES)))
        self.max_abs_weight = float(max_abs_weight)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.min_improvement = float(min_improvement)
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.max_abs_weight, self.learning_rate, self.min_improvement)):
            raise ValueError("REDA limits/rates must be finite and positive")
        if not math.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")

    def _base_cp(self, board):
        return float(handcrafted(board)) + max(-220.0, min(220.0, (self.model.position_score(board) + self.model.bias) * 120.0))



    def _mse(self, samples, residual):
        if not samples:
            return float("inf")
        errs = []
        for board, target_cp in samples:
            target = float(target_cp)
            if not math.isfinite(target):
                raise ValueError("target_cp must be finite")
            learned_cp = ((self.model.position_score(board) + self.model.bias
                           + float(np.dot(residual, features(board))) + self.model.micro_bias) * 120.0)
            pred = float(handcrafted(board)) + max(-220.0, min(220.0, learned_cp))
            errs.append((pred - target) ** 2)
        return float(np.mean(errs))


    def adapt(self, train_samples, validation_samples, steps: int = 80) -> REDAResult:
        """Propose a sparse correction and validate it on held-out positions.

        1.5 upgrade: instead of one fixed SGD trajectory on a single feature
        set, the adapter solves closed-form ridge regression for every
        candidate active-set size (1..max_active), optionally polishes each
        with a few SGD steps, and commits the size with the best held-out MSE.
        The validation gate, sparsity cap and weight bounds are unchanged.
        """
        train_samples, validation_samples = list(train_samples), list(validation_samples)
        if len(train_samples) < 2 or len(validation_samples) < 2:
            return REDAResult(False, float("inf"), float("inf"), [], "insufficient_samples")
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        base_micro = self.model.micro_weights.copy()
        X = np.stack([features(b) for b, _ in train_samples]).astype(np.float32)
        y = np.asarray([float(target) - self._base_cp(b) - self.model.micro_bias * 120.0
                        for b, target in train_samples], dtype=np.float32) / 120.0
        if not np.all(np.isfinite(y)):
            raise ValueError("targets must be finite")
        centered = X - X.mean(axis=0, keepdims=True)
        corr = np.abs(centered.T @ (y - y.mean()))
        ranked = np.argsort(corr)[::-1]
        usable = [int(i) for i in ranked if corr[i] > 1e-9]
        if not usable:
            base_mse = self._mse(validation_samples, base_micro)
            return REDAResult(False, base_mse, base_mse, [], "no_signal")

        base_mse = self._mse(validation_samples, base_micro)
        best_mse, best_candidate, best_active = base_mse, base_micro.copy(), []
        # Try each active-set size; pick the one that validates best, not the biggest.
        for size in range(1, min(self.max_active, len(usable)) + 1):
            active = usable[:size]
            candidate = base_micro.copy()
            Xa = X[:, active]
            n = len(y)
            gram = Xa.T @ Xa + (self.l2 * n) * np.eye(len(active), dtype=np.float32) \
                + 1e-8 * np.eye(len(active), dtype=np.float32)
            rhs = Xa.T @ y
            try:
                w = np.linalg.solve(gram, rhs).astype(np.float32)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(Xa, y, rcond=None)[0].astype(np.float32)
            # Small SGD polish from the closed-form start (bounded, clipped).
            w = np.clip(w, -self.max_abs_weight, self.max_abs_weight)
            for _ in range(int(steps)):
                err = Xa @ w - y
                grad = (Xa.T @ err) / n + self.l2 * w
                w -= self.learning_rate * grad
                w = np.clip(w, -self.max_abs_weight, self.max_abs_weight)
            candidate[active] = w
            cand_mse = self._mse(validation_samples, candidate)
            if math.isfinite(cand_mse) and cand_mse < best_mse:
                best_mse, best_candidate, best_active = cand_mse, candidate, list(active)

        improvement = base_mse - best_mse
        candidate_mse = best_mse
        candidate = best_candidate
        active = best_active
        accepted = (improvement >= self.min_improvement
                    and np.count_nonzero(np.abs(candidate) > 1e-12) <= self.max_active
                    and np.max(np.abs(candidate)) <= self.max_abs_weight)
        if accepted:
            self.model.micro_weights = candidate.astype(np.float32)
        return REDAResult(accepted, base_mse, candidate_mse,
                         [FEATURES[i] for i in active] if accepted else [],
                         "validation_improved" if accepted else "validation_gate_rejected")
