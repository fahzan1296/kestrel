"""REDA discovery and deep-verification pipeline for self-play PGNs.


Targets are deeper search scores (White-relative centipawns), not game outcomes.
This is search-score distillation into the sparse residual adapter, not NNUE training.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

from .learning import OutcomeModel
from .reda import RareExperienceAdapter
from .search import Searcher


def _fingerprint(board: chess.Board) -> str:
    # Include halfmove clock because the 50-move rule changes position value.
    payload = f"{board.board_fen()}|{int(board.turn)}|{board.castling_rights}|{board.ep_square}|{board.halfmove_clock}"
    return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()


def mine_candidates(pgn_path: str | Path, limit: int = 24, memory_cap: int = 100_000):
    """Two-pass bounded-memory miner; retains only a small ranked candidate heap."""
    import heapq
    pgn_path = Path(pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN corpus not found: {pgn_path}")
    memory_cap = max(1000, min(500_000, int(memory_cap)))
    frequency: Counter[str] = Counter()
    # Pass 1: count only tracked fingerprints, with a hard cap on RAM usage.
    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        while game := chess.pgn.read_game(handle):
            board = game.board()
            for node in game.mainline():
                board.push(node.move)
                fp = _fingerprint(board)
                if fp in frequency:
                    frequency[fp] += 1
                elif len(frequency) < memory_cap:
                    frequency[fp] = 1
    tracked_unique = len(frequency)

    # Pass 2: score positions and retain only top-K board snapshots.
    heap = []
    processed: set[str] = set()
    serial = 0
    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        while game := chess.pgn.read_game(handle):
            board = game.board()
            prev_cp = None
            for node in game.mainline():
                move = node.move
                is_capture = board.is_capture(move)
                board.push(move)
                fp = _fingerprint(board)
                cp = None
                comment = (node.comment or "").split()
                if comment:
                    with contextlib.suppress(ValueError, IndexError):
                        cp = int(comment[0].split("/")[0])
                swing = abs(cp - prev_cp) if cp is not None and prev_cp is not None else 0
                prev_cp = cp
                # 1.7: komentar PGN dari semua penulis (arena, selfplay worker,
                # cross-engine) kini dinormalisasi White-relative, jadi swing
                # membandingkan dua nilai dalam perspektif yang sama. Korpus
                # lama mover-relative hanya menurunkan kualitas ranking
                # kandidat, bukan target distilasi (target selalu dihitung
                # ulang lewat deep search di posisi kandidat).
                if fp in processed or len(processed) >= memory_cap or not board.is_valid():
                    continue
                processed.add(fp)
                tactical = int(board.is_check()) + int(move.promotion is not None) + int(is_capture)
                rarity = 1.0 / (1.0 + frequency.get(fp, 1))
                surprise = min(swing, 1200) / 1200.0
                score = rarity + surprise + tactical * 0.35
                serial += 1
                meta = {"eval_swing_cp": swing, "tactical_flags": tactical}
                if len(heap) < limit:
                    heapq.heappush(heap, (score, serial, board.copy(), fp, meta))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, serial, board.copy(), fp, meta))
    ranked = sorted(heap, key=lambda x: x[0], reverse=True)
    return [(score, board, fp, meta) for score, _, board, fp, meta in ranked], tracked_unique, frequency



def _match_safety_gate(candidate_model, baseline_model, *, pairs=4, time_ms=150,
                        depth=8, min_score_rate=0.35, progress=None):
    """Balanced color-swapped mini-match; regression guard, not an Elo estimate."""
    from .arena import OPENINGS, Player, play_game
    wins = draws = losses = 0
    total = max(1, int(pairs))
    for pair_index in range(total):
        opening = OPENINGS[pair_index % len(OPENINGS)]
        seed = 81000 + pair_index
        for candidate_white in (True, False):
            cand = Player("REDA candidate", time_ms=time_ms, max_depth=depth, learned=candidate_model)
            base = Player("Frozen baseline", time_ms=time_ms, max_depth=depth, learned=baseline_model)
            white, black = (cand, base) if candidate_white else (base, cand)
            result, _, plies = play_game(white, black, fen=opening, temperature=0.0,
                                        seed=seed, max_plies=100)
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == candidate_white:
                wins += 1
            else:
                losses += 1
            if progress:
                progress({"status": "match_gate", "match_games": wins + draws + losses,
                         "match_total": total * 2, "match_wdl": {"wins": wins, "draws": draws,
                         "losses": losses}, "latest_plies": plies})
    score_rate = (wins + 0.5 * draws) / max(1, wins + draws + losses)
    return {"games": wins + draws + losses, "wins": wins, "draws": draws,
            "losses": losses, "score_rate": round(score_rate, 4),
            "minimum_score_rate": float(min_score_rate),
            "passed": score_rate >= float(min_score_rate),
            "note": "small paired safety match; not a strength/Elo proof"}



def reda_adapt_model(pgn_path, base_model: OutcomeModel, *, limit=24, depth=10,
                     time_ms=1500, max_active=3, max_abs_weight=0.02,
                     learning_rate=0.01, l2=0.002, min_improvement=1e-5,
                     steps=160, memory_cap=100_000, min_verified=6,
                     progress=None):
    """Distilasi skor pencarian mendalam ke dalam residual REDA — in-memory.

    Mengembalikan ``(candidate_model, report)``. Model dasar TIDAK diubah dan
    TIDAK ada file yang ditulis; keputusan commit ada di pemanggil (trainer
    menjalankan verification match dulu, baru promote). Ini pemisahan yang
    membuat alur unified aman: adaptasi gagal/lemah tidak pernah menyentuh
    champion.
    """
    started = time.monotonic()

    def emit(status, **info):
        if progress:
            progress({"status": status, **info})

    model = OutcomeModel(base_model.weights, base_model.bias,
                         base_model.micro_weights, base_model.micro_bias)
    net = None  # Self-play saat ini memakai handcrafted + learned evaluator.
    candidates, unique_count, freq = mine_candidates(pgn_path, limit, memory_cap=memory_cap)
    emit("discovering", candidate_count=len(candidates), unique_positions=unique_count)
    verified = []
    for i, (_, board, fp, discovery) in enumerate(candidates, 1):
        searcher = Searcher(net, time_ms=time_ms, max_depth=depth, learned=model)
        _, info = searcher.best_move(board)
        # Skor searcher relatif-terhadap-pemain; konversi ke White-relative cp.
        white_cp = int(info["score_cp"] if board.turn == chess.WHITE else -info["score_cp"])
        verified.append((board, white_cp, fp, {**discovery, "depth": info["depth"],
                         "nodes": info["nodes"], "white_score_cp": white_cp,
                         "timed_out": info.get("timed_out", False)}))
        emit("verifying", verified=i, total=len(candidates), latest_depth=info.get("depth", 0),
             latest_nodes=info.get("nodes", 0))
    if len(verified) < min_verified:
        report = {"accepted": False, "reason": f"need_at_least_{min_verified}_deep_verified_positions",
                  "candidate_count": len(candidates), "verified_count": len(verified),
                  "unique_positions": unique_count,
                  "elapsed_seconds": round(time.monotonic() - started, 2)}
        emit("rejected", **report)
        return model, report
    ranked = sorted(verified, key=lambda x: x[2])
    validation = [x for i, x in enumerate(ranked) if i % 4 == 0]
    train = [x for i, x in enumerate(ranked) if i % 4 != 0]
    train_samples = [(b, cp) for b, cp, _, _ in train]
    validation_samples = [(b, cp) for b, cp, _, _ in validation]
    adapter = RareExperienceAdapter(model, max_active=max_active,
        max_abs_weight=max_abs_weight, learning_rate=learning_rate, l2=l2,
        min_improvement=min_improvement)
    result = adapter.adapt(train_samples, validation_samples, steps=steps)
    report = {"accepted": result.accepted, "reason": result.reason,
        "baseline_validation_mse_cp2": result.baseline_mse,
        "candidate_validation_mse_cp2": result.candidate_mse,
        "updated_features": result.updated_features,
        "candidate_count": len(candidates), "verified_count": len(verified),
        "training_count": len(train_samples), "validation_count": len(validation_samples),
        "unique_positions": unique_count,
        "experience_samples": [x[3] for x in verified],
        "elapsed_seconds": round(time.monotonic() - started, 2)}
    emit("accepted" if result.accepted else "rejected", **report)
    return model, report


def run_reda(pgn_path, model_path, *, limit=24, depth=10, time_ms=1500,
            max_active=3, max_abs_weight=0.02, learning_rate=0.01,
            l2=0.002, min_improvement=1e-5, steps=160, match_pairs=4,
            match_time_ms=150, match_depth=8, min_match_score_rate=0.35,
            memory_cap=100_000, progress=None):
    """Alur REDA satu-tembak (kompatibilitas dashboard 1.5): adaptasi langsung
    commit bila lolos gate validasi + match keamanan kecil."""
    started = time.monotonic()
    def emit(status, **info):
        if progress:
            progress({"status": status, **info})
    model = OutcomeModel()
    model_path = Path(model_path)
    if model_path.exists():
        model.load(model_path)
    baseline_model = OutcomeModel(model.weights, model.bias, model.micro_weights, model.micro_bias)
    net = None  # Dashboard self-play currently uses handcrafted + learned evaluator.
    candidates, unique_count, freq = mine_candidates(pgn_path, limit, memory_cap=memory_cap)
    emit("discovering", candidate_count=len(candidates), unique_positions=unique_count)
    verified = []
    for i, (_, board, fp, discovery) in enumerate(candidates, 1):
        searcher = Searcher(net, time_ms=time_ms, max_depth=depth, learned=model)
        _, info = searcher.best_move(board)
        # Searcher's score is side-to-move-relative. Convert to White-relative cp.
        white_cp = int(info["score_cp"] if board.turn == chess.WHITE else -info["score_cp"])
        verified.append((board, white_cp, fp, {**discovery, "depth": info["depth"],
                         "nodes": info["nodes"], "white_score_cp": white_cp,
                         "timed_out": info.get("timed_out", False)}))
        emit("verifying", verified=i, total=len(candidates), latest_depth=info.get("depth", 0),
             latest_nodes=info.get("nodes", 0))
    # Deterministic fingerprint split, minimum 2 held-out samples. Do not update
    # with unverified/no-sample data; fail closed on small corpora.
    if len(verified) < 6:
        result = {"accepted": False, "reason": "need_at_least_6_deep_verified_positions",
                  "candidate_count": len(candidates), "verified_count": len(verified),
                  "unique_positions": unique_count, "elapsed_seconds": round(time.monotonic()-started, 2)}
        emit("rejected", **result)
        return result
    ranked = sorted(verified, key=lambda x: x[2])
    validation = [x for i, x in enumerate(ranked) if i % 4 == 0]
    train = [x for i, x in enumerate(ranked) if i % 4 != 0]
    train_samples = [(b, cp) for b, cp, _, _ in train]
    validation_samples = [(b, cp) for b, cp, _, _ in validation]
    adapter = RareExperienceAdapter(model, max_active=max_active,
        max_abs_weight=max_abs_weight, learning_rate=learning_rate, l2=l2,
        min_improvement=min_improvement)
    result = adapter.adapt(train_samples, validation_samples, steps=steps)
    report = {"accepted": result.accepted, "reason": result.reason,
        "baseline_validation_mse_cp2": result.baseline_mse,
        "candidate_validation_mse_cp2": result.candidate_mse,
        "updated_features": result.updated_features,
        "candidate_count": len(candidates), "verified_count": len(verified),
        "training_count": len(train_samples), "validation_count": len(validation_samples),
        "unique_positions": unique_count,
        "experience_samples": [x[3] for x in verified],
        "elapsed_seconds": round(time.monotonic()-started, 2)}
    if result.accepted:
        match_gate = _match_safety_gate(model, baseline_model, pairs=match_pairs,
            time_ms=match_time_ms, depth=match_depth,
            min_score_rate=min_match_score_rate, progress=progress)
        report["strength_safety_match"] = match_gate
        if not match_gate["passed"]:
            model.micro_weights = baseline_model.micro_weights.copy()
            report.update(accepted=False, reason="match_safety_gate_rejected",
                         updated_features=[], model_saved=None)
            emit("rejected", **report)
            return report
        # Preserve the previous checkpoint and replace the active file atomically.
        backup_path = model_path.with_name(model_path.name + ".reda_backup")
        if model_path.exists():
            shutil.copy2(model_path, backup_path)
        temp_path = model_path.with_name(model_path.name + ".reda_tmp")
        model.save(temp_path, metadata={"reda": {"updated_at": time.time(),
            "verified_positions": len(verified), "updated_features": result.updated_features,
            "validation_mse_before": result.baseline_mse,
            "validation_mse_after": result.candidate_mse,
            "validation_mode": "position_holdout_same_corpus"}})
        os.replace(temp_path, model_path)
        report["model_saved"] = str(model_path)
        report["rollback_checkpoint"] = str(backup_path) if backup_path.exists() else None
        report["validation_mode"] = "position_holdout_same_corpus"
    emit("accepted" if result.accepted else "rejected", **report)
    return report
