"""KESTREL 2.0 — Trainer TinyNNUE: distilasi + hasil game (ganti OutcomeModel/REDA).

Dua sumber supervisi per posisi, digabung jadi satu target STM:

    y_stm = alpha * clip(search_stm, +-distill_clip)      # distilasi skor search
          + (1-alpha) * outcome_scale * outcome_stm        # sinyal hasil game

- ``search_stm``: komentar skor PGN self-play (``{score/dN nM}``, White-relative)
  yang ditulis worker saat bermain — label "gratis" tanpa pencarian ulang.
  Posisi tanpa komentar memakai teacher klasik (``kestrel.teacher``) sebagai
  fallback label.
- ``outcome_stm``: hasil game (1-0 / seri / 0-1) diproyeksikan ke perspektif
  STM dan diskalakan outcome_scale centipawn — kalibrasi jangka panjang.

Training: minibatch SGD numpy murni, forward/backward vektorisasi penuh
(matriks one-hot S @ W1 untuk akumulator), holdout level-game, early stop,
lr decay, gradient clipping, weight decay, warm start dari champion.

Jumlah parameter net: 106.689 — tetap di bawah batas 1.000.000 spesifikasi.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .nnue import (ACC_DIM, AUX_DIM, HIDDEN, OUT_SCALE, PLANES, TinyNNUE,
                   plane_b, plane_w)
from .teacher import static_eval

# W1 training memakai 769 baris: baris ke-768 adalah baris-NOL untuk padding
# slot bidak kosong (forward tetap eksak, grad ke baris padding dibuang).
_PAD_ROW = PLANES
_W1_TRAIN_ROWS = PLANES + 1
_SCORE_RE = re.compile(r"(-?\d+)/d\d+")
_CLAIM_RE = re.compile(r"\{([^}]*)\}")
TARGET_CLIP = 1500.0


# ---------------- korpus ----------------
def _plane_lists(board: chess.Board) -> tuple[list[int], list[int]]:
    """(plane indices perspektif putih, perspektif hitam) untuk posisi."""
    iw: list[int] = []
    ib: list[int] = []
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
               chess.QUEEN, chess.KING):
        base = (pt - 1) * 128
        m = board.pieces_mask(pt, chess.WHITE)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            iw.append(base + sq)
            ib.append(base + 64 + (sq ^ 56))
        m = board.pieces_mask(pt, chess.BLACK)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            iw.append(base + 64 + sq)
            ib.append(base + (sq ^ 56))
    return iw, ib


def _aux_vector(board: chess.Board) -> np.ndarray:
    from .nnue import _aux as _aux_fn
    return _aux_fn(board)


def corpus_from_pgn(paths, *, max_samples_per_game: int = 120,
                    min_ply: int = 6, distill_weight: float = 0.75,
                    outcome_scale: float = 350.0, distill_clip: float = 1200.0,
                    teacher_fallback: bool = True,
                    seed: int = 42) -> tuple[list[dict], dict]:
    """Kumpulkan sampel training dari PGN. Return (samples, stats).

    Satu sampel = dict(plane_w, plane_b, aux, target_stm_cp, game_id).
    """
    rng = random.Random(seed)
    samples: list[dict] = []
    games_read = skipped = 0
    labels_comment = labels_teacher = 0
    result_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            while game := chess.pgn.read_game(f):
                res = game.headers.get("Result")
                if res not in result_counts:
                    skipped += 1
                    continue
                games_read += 1
                result_counts[res] += 1
                white_outcome = {"1-0": 1.0, "1/2-1/2": 0.0, "0-1": -1.0}[res]
                board = game.board()
                ply = 0
                pos: list[tuple[chess.Board, float, float | None]] = []
                node = game
                while node.variations:
                    node = node.variation(0)
                    mv = node.move
                    if mv is None:
                        break
                    board.push(mv)
                    ply += 1
                    if ply < min_ply or not board.is_valid():
                        continue
                    comment = node.comment or ""
                    m = _SCORE_RE.search(comment)
                    search_white = float(m.group(1)) if m else None
                    pos.append((board.copy(stack=False), white_outcome,
                                search_white))
                if not pos:
                    continue
                if len(pos) > max_samples_per_game:
                    pos = rng.sample(pos, max_samples_per_game)
                for b, wout, swhite in pos:
                    stm = b.turn
                    sign = 1.0 if stm == chess.WHITE else -1.0
                    outcome_stm = wout * sign
                    if swhite is not None:
                        search_stm = swhite * sign
                        target = (distill_weight * max(-distill_clip,
                                  min(distill_clip, search_stm))
                                  + (1.0 - distill_weight) * outcome_scale
                                  * outcome_stm)
                        labels_comment += 1
                    elif teacher_fallback:
                        teacher_white = float(static_eval(b))
                        if abs(teacher_white) > 10000:  # posisi terminal
                            continue
                        target = (distill_weight * max(-distill_clip,
                                  min(distill_clip, teacher_white * sign))
                                  + (1.0 - distill_weight) * outcome_scale
                                  * outcome_stm)
                        labels_teacher += 1
                    else:
                        target = outcome_scale * outcome_stm
                        labels_teacher += 1
                    iw, ib = _plane_lists(b)
                    samples.append({
                        "plane_w": iw, "plane_b": ib,
                        "aux": _aux_vector(b),
                        "stm_white": stm == chess.WHITE,
                        "target_cp": float(max(-TARGET_CLIP,
                                               min(TARGET_CLIP, target))),
                        "game_id": games_read,
                    })
    stats = {"games_read": games_read, "games_skipped": skipped,
             "result_counts": result_counts,
             "samples": len(samples),
             "labels_from_comments": labels_comment,
             "labels_from_teacher": labels_teacher}
    return samples, stats


# ---------------- batch encode ----------------
def _encode_batch(samples: list[dict]):
    """Bangun (S_w, S_b, aux, y) untuk satu minibatch.

    S_w/S_b: (B, 769) one-hot float32; baris padding memakai kolom 768 pada
    W1-train (baris nol) sehingga forward eksak dan grad bersih.
    """
    b = len(samples)
    sw = np.zeros((b, _W1_TRAIN_ROWS), np.float32)
    sb = np.zeros((b, _W1_TRAIN_ROWS), np.float32)
    aux = np.empty((b, AUX_DIM), np.float32)
    y = np.empty((b, 1), np.float32)
    for i, s in enumerate(samples):
        sw[i, s["plane_w"]] = 1.0
        sb[i, s["plane_b"]] = 1.0
        aux[i] = s["aux"]
        y[i, 0] = s["target_cp"]
    return sw, sb, aux, y


def _predict(net: TinyNNUE, samples: list[dict], batch_size: int = 512) -> np.ndarray:
    """Prediksi centipawn STM untuk list sampel (tanpa grad)."""
    outs = []
    for begin in range(0, len(samples), batch_size):
        chunk = samples[begin:begin + batch_size]
        sw, sb, aux, _y = _encode_batch(chunk)
        # kolom padding (index 769) tidak pernah berisi plane nyata — aman dislice
        acc_w = sw[:, :PLANES] @ net.w1
        acc_b = sb[:, :PLANES] @ net.w1
        n = len(chunk)
        x = np.empty((n, ACC_DIM * 2 + AUX_DIM), np.float32)
        white_mask = np.fromiter((s["stm_white"] for s in chunk), bool, n)
        x[white_mask, :ACC_DIM] = acc_w[white_mask]
        x[white_mask, ACC_DIM:ACC_DIM * 2] = acc_b[white_mask]
        bm = ~white_mask
        x[bm, :ACC_DIM] = acc_b[bm]
        x[bm, ACC_DIM:ACC_DIM * 2] = acc_w[bm]
        x[:, ACC_DIM * 2:] = aux
        h = np.maximum(x @ net.w2 + net.b2, 0.0)
        outs.append((h @ net.w3 + net.b3)[:, 0])
    return np.concatenate(outs) if outs else np.empty(0, np.float32)


# ---------------- linear warm start (bootstrap dari nol) ----------------
def _linear_warmstart(samples: list[dict], ridge: float = 1e-2,
                      chunk: int = 2048) -> np.ndarray:
    """Solusi least-squares tertutup utk evaluator linear atas plane STM/NTM.

    Melatih W1 (98rb parameter) murni lewat SGD dari noise acak itu terlalu
    lambat (gradien ter-atteniasi lewat bottleneck L2). Kita selesaikan
    dulu model linear ``y_norm ~ [S_stm | S_ntm] @ theta`` secara closed-form
    (normal equation + ridge), lalu embed theta ke W1 sebagai jalur linear
    EKSAK: kolom akumulator 0 = proyeksi theta_stm, kolom 1 = theta_ntm —
    sehingga L2/L3 hanya perlu mempelajari readout (mudah), dan SGD
    melanjutkan penyempurnaan non-linear. Dimensi akumulator lain tetap noise
    kecil untuk eksplorasi.
    """
    d = 2 * PLANES
    n_total = len(samples)
    # Optimisasi 2.0.4: normal equation primal berdimensi 1536x1536 dipecah
    # lewat identitas Woodbury. Untuk n sampel kecil (kasus umum smoke test &
    # generasi awal) menyelesaikan sistem n x n jauh lebih murah: O(n^3)
    # versus O(d^3), tanpa mengubah solusi (selisih numerik ~1e-13).
    # Untuk korpus besar pakai jalur primal chunked yang lama agar memori
    # tidak meletup (n x n float64 = 8*n^2 byte).
    if n_total <= 1024:
        a = np.zeros((n_total, d), np.float64)
        y = np.empty(n_total, np.float64)
        for i, s in enumerate(samples):
            pw = np.asarray(s["plane_w"], dtype=np.intp)
            pb = np.asarray(s["plane_b"], dtype=np.intp)
            if s["stm_white"]:
                a[i, pw] = 1.0
                a[i, PLANES + pb] = 1.0
            else:
                a[i, pb] = 1.0
                a[i, PLANES + pw] = 1.0
            y[i] = s["target_cp"] / OUT_SCALE
        # Skala regularisasi mengikuti trace(A^T A) = jumlah fitur aktif,
        # identik dengan jalur primal di bawah (~2 * n_total).
        scale = max(1.0, float(np.count_nonzero(a)) / d)
        reg = ridge * scale
        alpha = np.linalg.solve(a @ a.T + reg * np.eye(n_total), y)
        theta = a.T @ alpha
    else:
        ata = np.zeros((d, d), np.float64)
        aty = np.zeros(d, np.float64)
        for begin in range(0, n_total, chunk):
            part = samples[begin:begin + chunk]
            n = len(part)
            s_stm = np.zeros((n, PLANES), np.float32)
            s_ntm = np.zeros((n, PLANES), np.float32)
            yy = np.empty(n, np.float64)
            for i, s in enumerate(part):
                pw = s["plane_w"]; pb = s["plane_b"]
                if s["stm_white"]:
                    s_stm[i, pw] = 1.0
                    s_ntm[i, pb] = 1.0
                else:
                    s_stm[i, pb] = 1.0
                    s_ntm[i, pw] = 1.0
                yy[i] = s["target_cp"] / OUT_SCALE
            aa = np.concatenate((s_stm, s_ntm), axis=1).astype(np.float64)
            ata += aa.T @ aa
            aty += aa.T @ yy
        reg = ridge * max(1.0, float(np.trace(ata)) / d)
        theta = np.linalg.solve(ata + reg * np.eye(d), aty)
    w1 = (np.random.default_rng(11).standard_normal((PLANES, ACC_DIM))
          * 0.01).astype(np.float32)
    w1[:, 0] = theta[:PLANES].astype(np.float32)
    w1[:, 1] = theta[PLANES:].astype(np.float32)
    return w1


# ---------------- training ----------------
def train_nnue(samples: list[dict], *, base: TinyNNUE | None = None,
               epochs: int = 12, lr: float = 0.05, batch_size: int = 256,
               patience: int = 3, validation_fraction: float = 0.20,
               seed: int = 42, lr_decay: float = 0.7,
               weight_decay: float = 1e-5, grad_clip: float = 5.0,
               warm_start: bool = True) -> tuple[TinyNNUE, dict]:
    """Latih TinyNNUE dari sampel korpus. Return (net, report).

    Holdout level-game; early stop pada MSE validasi (satuan cp²).
    """
    if not samples:
        raise ValueError("Korpus training kosong")
    for s in samples:
        s.setdefault("stm_white", True)
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    # holdout level-game
    game_ids = sorted({s["game_id"] for s in samples})
    rng.shuffle(game_ids)
    val_count = max(1, int(round(len(game_ids) * float(np.clip(
        validation_fraction, 0.05, 0.45))))) if len(game_ids) >= 5 else 0
    val_ids = set(game_ids[:val_count]) if val_count else set()
    train = [s for s in samples if s["game_id"] not in val_ids]
    val = [s for s in samples if s["game_id"] in val_ids]
    if not train:
        raise ValueError("Split training kosong; korpus terlalu kecil")

    # init: warm start dari champion/baseline; tanpa base, kepala output
    # di-init acak KECIL — zero-head mengunci gradien upstream (dh = d_out @
    # w3.T == 0) sehingga layer 1/2 tidak pernah belajar. Kepala acak hanya
    # untuk fase training; runtime tetap hanya memuat checkpoint terlatih.
    if base is not None and warm_start:
        net = base.copy()
    else:
        # Bootstrap dari nol: W1 warm-start linear (least-squares) + kepala
        # acak kecil. Zero-head mengunci gradien upstream (dh = d_out @ w3.T
        # == 0) sehingga layer 1/2 tidak pernah belajar.
        head_rng = np.random.default_rng(seed + 1)
        net = TinyNNUE()
        net.w3 = (head_rng.standard_normal((HIDDEN, 1)) * 0.5).astype(np.float32)
        net.b3 = np.zeros(1, np.float32)
        net.w1 = _linear_warmstart(train)
    # W1 training buffer: 768 baris nyata + 1 baris nol utk padding
    w1t = np.zeros((_W1_TRAIN_ROWS, ACC_DIM), np.float32)
    w1t[:PLANES] = net.w1
    w2 = net.w2.copy()
    b2 = net.b2.copy()
    w3 = net.w3.copy()
    b3 = net.b3.copy()

    y_train = np.asarray([s["target_cp"] for s in train], np.float32)
    y_val = np.asarray([s["target_cp"] for s in val], np.float32) if val else None

    def mse_cp2(pred: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0:
            return float("nan")
        return float(np.mean((pred * OUT_SCALE - y) ** 2))

    best = {"loss": float("inf"), "params": None, "epoch": 0}
    history = []
    bs = max(8, int(batch_size))
    lr0 = float(lr)

    def snapshot():
        w1 = w1t[:PLANES].copy()
        return (w1, w2.copy(), b2.copy(), w3.copy(), b3.copy())

    for ep in range(max(1, int(epochs))):
        lr_ep = lr0 * (float(lr_decay) ** ep) if float(lr_decay) < 1.0 else lr0
        order = nprng.permutation(len(train))
        for begin in range(0, len(order), bs):
            ix = order[begin:begin + bs]
            chunk = [train[i] for i in ix]
            sw, sb, aux, yb = _encode_batch(chunk)
            n = len(chunk)
            # ---- forward ----
            acc_w = sw @ w1t
            acc_b = sb @ w1t
            x = np.empty((n, ACC_DIM * 2 + AUX_DIM), np.float32)
            white_mask = np.fromiter((s["stm_white"] for s in chunk), bool, n)
            x[white_mask, :ACC_DIM] = acc_w[white_mask]
            x[white_mask, ACC_DIM:ACC_DIM * 2] = acc_b[white_mask]
            bm = ~white_mask
            x[bm, :ACC_DIM] = acc_b[bm]
            x[bm, ACC_DIM:ACC_DIM * 2] = acc_w[bm]
            x[:, ACC_DIM * 2:] = aux
            h_pre = x @ w2 + b2
            h = np.maximum(h_pre, 0.0)
            out = h @ w3 + b3
            # ---- loss & backward (target NORMALISASI raw: y / OUT_SCALE —
            # gradien O(1), tidak meledak oleh faktor OUT_SCALE^2) ----
            y_norm = yb / OUT_SCALE
            d_out = 2.0 / n * (out - y_norm)
            dw3 = h.T @ d_out
            db3 = d_out.sum(axis=0)
            dh = d_out @ w3.T
            dh *= (h_pre > 0)
            dw2 = x.T @ dh
            db2 = dh.sum(axis=0)
            dx = dh @ w2.T
            dacc_w = dx[:, :ACC_DIM].copy()
            dacc_b = dx[:, ACC_DIM:ACC_DIM * 2].copy()
            # undo swap STM
            dacc_w[bm], dacc_b[bm] = dacc_b[bm], dacc_w[bm]
            dw1 = sb.T @ dacc_b
            dw1[:PLANES] += (sw.T @ dacc_w)[:PLANES]
            # ---- clip (SEMUA tensor termasuk bias) + weight decay + update ----
            for arr, g in ((w1t, dw1), (w2, dw2), (b2, db2), (w3, dw3), (b3, db3)):
                norm = float(np.linalg.norm(g))
                if norm > grad_clip:
                    g *= grad_clip / norm
                arr -= lr_ep * (g + weight_decay * arr)
        # ---- evaluasi epoch ----
        net_ep = TinyNNUE(w1t[:PLANES].copy(), w2.copy(), b2.copy(),
                          w3.copy(), b3.copy())
        tr_pred = _predict(net_ep, train)
        val_pred = _predict(net_ep, val) if val else tr_pred
        train_mse = mse_cp2(tr_pred, y_train)
        val_mse = mse_cp2(val_pred, y_val) if y_val is not None else train_mse
        monitored = val_mse
        history.append({"epoch": ep + 1, "lr": round(lr_ep, 6),
                        "train_mse_cp2": round(train_mse, 2),
                        "validation_mse_cp2": round(val_mse, 2)})
        if monitored < best["loss"] - 0.5:
            best = {"loss": monitored, "params": snapshot(), "epoch": ep + 1}
        # early stop: berhenti bila tidak ada epoch terbaik baru dalam
        # `patience` epoch terakhir (dihitung dari history).
        best_idx = min(range(len(history)),
                       key=lambda i: history[i]["validation_mse_cp2"])
        if ep - best_idx >= max(1, int(patience)):
            break

    if best["params"] is not None:
        w1b, w2b, b2b, w3b, b3b = best["params"]
        net = TinyNNUE(w1b, w2b, b2b, w3b, b3b)
    best_idx = min(range(len(history)),
                   key=lambda i: history[i]["validation_mse_cp2"])
    report = {
        "samples": len(samples),
        "train_samples": len(train),
        "validation_samples": len(val),
        "validation_mode": "game_holdout" if val_count else "single_split_small_corpus",
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "best_epoch": history[best_idx]["epoch"],
        "best_validation_mse_cp2": history[best_idx]["validation_mse_cp2"],
        "train_mse_cp2_final": history[-1]["train_mse_cp2"],
        "lr": lr0, "lr_decay": float(lr_decay), "batch_size": bs,
        "warm_start": bool(base is not None and warm_start),
        "params": net.params(),
        "history": history,
    }
    return net, report


def train_from_pgn(paths, *, base: TinyNNUE | None = None, warm_start: bool = True,
                   epochs: int = 12, lr: float = 0.05, batch_size: int = 256,
                   patience: int = 3, validation_fraction: float = 0.20,
                   seed: int = 42, lr_decay: float = 0.7,
                   distill_weight: float = 0.75, outcome_scale: float = 350.0,
                   distill_clip: float = 1200.0,
                   max_samples_per_game: int = 120,
                   teacher_fallback: bool = True) -> tuple[TinyNNUE, dict]:
    """Pipeline penuh PGN -> korpus -> TinyNNUE terlatih. Return (net, report)."""
    samples, stats = corpus_from_pgn(
        paths, max_samples_per_game=max_samples_per_game,
        distill_weight=distill_weight, outcome_scale=outcome_scale,
        distill_clip=distill_clip, teacher_fallback=teacher_fallback,
        seed=seed)
    if not samples:
        raise ValueError("Tidak ada posisi valid di PGN korpus")
    net, report = train_nnue(samples, base=base, warm_start=warm_start,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             patience=patience,
                             validation_fraction=validation_fraction,
                             seed=seed, lr_decay=lr_decay)
    report["corpus"] = stats
    report["distill_weight"] = distill_weight
    report["outcome_scale_cp"] = outcome_scale
    report["distill_clip_cp"] = distill_clip
    return net, report


def main():
    ap = argparse.ArgumentParser(
        description="Distilasi TinyNNUE dari korpus PGN (ganti learning/REDA lama)")
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output", default="results/nnue_eval.json")
    ap.add_argument("--base-model", default=None,
                    help="checkpoint TinyNNUE utk warm start")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base = TinyNNUE.load(args.base_model) if args.base_model else None
    net, report = train_from_pgn(args.pgn, base=base, epochs=args.epochs,
                                 lr=args.lr, batch_size=args.batch_size,
                                 seed=args.seed)
    out = net.save(args.output, {"trained_by": "nnue_train.main",
                                 "report_summary": {
                                     k: report[k] for k in
                                     ("samples", "best_epoch",
                                      "best_validation_mse_cp2")}})
    print(json.dumps({"output": str(out), **{k: report[k] for k in
                                             ("samples", "best_epoch",
                                              "best_validation_mse_cp2",
                                              "corpus")}}, indent=2))


if __name__ == "__main__":
    main()
