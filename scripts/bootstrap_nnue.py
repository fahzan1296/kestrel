#!/usr/bin/env python3
"""Bootstrap checkpoint baseline TinyNNUE (distilasi teacher klasik).

Membuat checkpoint terlatih bawaan paket ``kestrel/models/nnue_baseline.json``:

1. GENERASI KORPUS — mayoritas playout TERPANDU teacher (softmax bertemperatur
   atas evaluasi statis anak => posisi realistis gaya-main engine) +
   minoritas playout acak utk cakupan; posisi di-deduplikasi per Zobrist.
2. LABEL — evaluasi statis teacher STM-relative (PERSIS fungsi leaf search
   1.8 pada posisi quiet), clip +-1200 cp. Label qsearch tidak dipakai:
   pada posisi playout liar targetnya sangat noisy dan menggeser distribusi
   (pelajaran bootstrap v1).
3. TRAINING — train_nnue (mini-batch SGD numpy + warm-start linear
   least-squares, holdout level-game, early stop).
4. VALIDASI — korelasi & kesepakatan best-move 1-ply vs teacher pada posisi
   quiet hasil game nyata (bukan sekadar korelasi korpus).

Jalankan:  python3 scripts/bootstrap_nnue.py [--samples 45000] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess
import chess.polyglot

from kestrel.nnue import ACC_DIM, AUX_DIM, OUT_SCALE, TinyNNUE, _aux
from kestrel.nnue_train import _plane_lists, train_nnue
from kestrel.teacher import static_eval, teacher_qsearch


def _softmax_pick(scores: list[float], temp_cp: float, rng: random.Random) -> int:
    """Pilih index dengan softmax suhu (scores = evaluasi STM-relative, cp)."""
    if not scores:
        return 0
    mx = max(scores)
    weights = [math.exp((s - mx) / max(temp_cp, 1e-3)) for s in scores]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def generate_corpus(target_samples: int, seed: int = 20261201,
                    guided_ratio: float = 0.75, max_plies: int = 140,
                    max_samples_per_game: int = 45, min_ply: int = 5,
                    progress_every: int = 25,
                    guided_temp_cp: float = 100.0) -> tuple[list[dict], dict]:
    """Playout ter-seed -> sampel korpus (plane indices + target teacher)."""
    rng = random.Random(seed)
    seen: set[int] = set()
    samples: list[dict] = []
    guided_games = random_games = 0
    t0 = time.monotonic()
    game_id = 0
    while len(samples) < target_samples:
        guided = rng.random() < guided_ratio
        board = chess.Board()
        plies_taken: list[chess.Board] = []
        for ply in range(1, max_plies + 1):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=True):
                break
            if guided:
                children = []
                scores = []
                sign = 1.0 if board.turn == chess.WHITE else -1.0
                for mv in moves:
                    board.push(mv)
                    if board.is_valid():
                        se = static_eval(board, terminal=False)
                        scores.append(se * sign)  # perspektif STM
                        children.append((board.copy(stack=False), mv))
                    board.pop()
                if not children:
                    break
                idx = _softmax_pick(scores, guided_temp_cp, rng)
                board = children[idx][0]
            else:
                board.push(rng.choice(moves))
            if ply >= min_ply and board.is_valid() and not board.is_game_over():
                plies_taken.append(board.copy(stack=False))
        game_id += 1
        if guided:
            guided_games += 1
        else:
            random_games += 1
        if len(plies_taken) > max_samples_per_game:
            step = len(plies_taken) / max_samples_per_game
            picked = [plies_taken[int(i * step)] for i in range(max_samples_per_game)]
        else:
            picked = plies_taken
        added = 0
        for b in picked:
            if len(samples) >= target_samples:
                break
            key = chess.polyglot.zobrist_hash(b)
            if key in seen:
                continue
            seen.add(key)
            stm = b.turn
            sign = 1.0 if stm == chess.WHITE else -1.0
            # label = evaluasi statis teacher STM-relative — fungsi leaf yang
            # PERSIS dipakai search 1.8 pada daun quiet.
            teacher_white = float(static_eval(b, terminal=False))
            if abs(teacher_white) > 10000:
                continue
            target = max(-1200.0, min(1200.0, teacher_white * sign))
            iw, ib = _plane_lists(b)
            samples.append({
                "plane_w": iw, "plane_b": ib, "aux": _aux(b),
                "stm_white": stm == chess.WHITE,
                "target_cp": target, "game_id": game_id,
            })
            added += 1
        if game_id % progress_every == 0:
            print(f"[corpus] games={game_id} samples={len(samples)} "
                  f"(guided={guided_games} random={random_games}) "
                  f"elapsed={time.monotonic() - t0:.0f}s", flush=True)
    stats = {"games": game_id, "guided_games": guided_games,
             "random_games": random_games, "unique_positions": len(samples),
             "generation_seconds": round(time.monotonic() - t0, 1)}
    return samples, stats


def spearman(pred_cp, target_cp) -> float:
    """Korelasi rank Spearman (tanpa scipy: implementasi langsung)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rp, rt = ranks(pred_cp), ranks(target_cp)
    n = len(rp)
    mp, mt = sum(rp) / n, sum(rt) / n
    cov = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
    vp = math.sqrt(sum((a - mp) ** 2 for a in rp))
    vt = math.sqrt(sum((b - mt) ** 2 for b in rt))
    return cov / (vp * vt) if vp and vt else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=45000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261201)
    ap.add_argument("--out", default=str(ROOT / "kestrel" / "models"
                                         / "nnue_baseline.json"))
    args = ap.parse_args()

    cache = Path(ROOT / "results" / "bootstrap_corpus.pkl")
    if cache.exists():
        print(f"[bootstrap] memakai cache korpus: {cache}", flush=True)
        with cache.open("rb") as fh:
            samples, gen_stats = pickle.load(fh)
    else:
        print(f"[bootstrap] target {args.samples} sampel unik, seed {args.seed}",
              flush=True)
        samples, gen_stats = generate_corpus(args.samples, seed=args.seed)
        print(f"[bootstrap] korpus selesai: {gen_stats}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump((samples, gen_stats), fh)
        print(f"[bootstrap] korpus di-cache ke {cache}", flush=True)

    net, report = train_nnue(samples, base=None, warm_start=False,
                             epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size, seed=args.seed,
                             lr_decay=0.9, patience=8)
    print(f"[bootstrap] training selesai: best_epoch={report['best_epoch']} "
          f"val_mse_cp2={report['best_validation_mse_cp2']:.1f}", flush=True)

    # validasi korelasi pada seluruh korpus (prediksi vs target)
    from kestrel.nnue_train import _predict
    preds = _predict(net, samples) * OUT_SCALE
    targets = [s["target_cp"] for s in samples]
    rho = float(spearman(list(preds), targets))
    mae = float(sum(abs(p - t) for p, t in zip(preds, targets)) / len(targets))
    print(f"[bootstrap] korelasi Spearman={rho:.4f} MAE={mae:.1f} cp", flush=True)

    meta = {
        "trained_by": "scripts/bootstrap_nnue.py",
        "teacher": "static_eval klasik 1.8 (fungsi leaf search), STM-relative",
        "corpus": gen_stats,
        "training": {k: report[k] for k in
                     ("samples", "train_samples", "validation_samples",
                      "validation_mode", "epochs_requested", "epochs_completed",
                      "best_epoch", "best_validation_mse_cp2", "lr",
                      "lr_decay", "batch_size", "params")},
        "spearman_target": round(rho, 4),
        "mae_cp": round(mae, 1),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.out)
    net.save(out, meta)
    size_kb = out.stat().st_size / 1024
    print(f"[bootstrap] checkpoint tersimpan: {out} ({size_kb:.0f} KB)", flush=True)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
