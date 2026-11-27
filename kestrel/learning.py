"""Outcome-based learning from PGN, using bounded linear evaluation features.

The model learns from completed games; no claims of deep RL are made. It is an
inspectable, CPU-cheap baseline that can be trained/retrained reproducibly.

1.8 upgrade (pembelajaran lebih efektif & efisien):
- Skema fitur v2 (11 -> 14): ``doubled_pawns``, ``isolated_pawns``,
  ``king_shield`` — dihitung dari agregat file-pion & mask pion yang sudah
  ada, biaya marginal per sampel hampir nol.
- MIGRASI model lama: bobot 1.6/1.7 (11 fitur) dimuat sebagai prefix;
  fitur baru mulai dari bobot 0 sehingga model lama berperilaku identik
  sampai training berikutnya mengisi bobotnya.
- WARM START: training generasi berikutnya dimulai dari bobot champion
  (bukan dari nol) — konvergensi lebih cepat, sampel lebih sedikit cukup.
- LR DECAY per epoch (default 0.85^epoch): langkah besar di awal, halus
  di akhir — validasi logloss turun lebih stabil.
- CLASS BALANCE: bobot sampel sebanding frekuensi terbalik kelas hasil
  (1-0 / seri / 0-1) agar korpus timpang tidak membiaskan model.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import chess, chess.pgn
import numpy as np
from .core import PIECE_VALUES, phase


FEATURES = ["material_pawns", "pawn_advantage", "passed_pawns", "center_control",
             "mobility", "king_exposure", "bishop_pair", "rook_open_files",
             "knight_outposts", "phase", "tempo",
             # 1.8 (skema v2): tiga fitur struktur pion / keamanan raja.
             "doubled_pawns", "isolated_pawns", "king_shield"]

# Zona pion-lolos dipakai bersama dengan evaluator (satu definisi, dua pemakai).
from .eval import _PASS_ZONE_W as _PASS_ZONE_W, _PASS_ZONE_B as _PASS_ZONE_B
from .eval import _FILES as _FILES

# Mask file d+e (pion tengah): kolom vertikal, satu bit per rank.
_FILE_D_E = (0x0101010101010101 << 3) | (0x0101010101010101 << 4)
_RANK_MASKS = [0xFF << (8 * r) for r in range(8)]


def features(board: chess.Board) -> np.ndarray:
    """Fitur linier 14 dimensi (bitboard-fast, fitur 0-10 identik dengan 1.5/1.7)."""
    black_occ = board.occupied_co[chess.BLACK]
    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & black_occ
    # [0] material (int, sama dengan core.material): putih - hitam
    #     = (mask total - 2 * mask hitam) per jenis bidak.
    v0 = (100 * (board.pawns.bit_count() - 2 * black_pawns.bit_count())
          + 320 * (board.knights.bit_count() - 2 * (board.knights & black_occ).bit_count())
          + 335 * (board.bishops.bit_count() - 2 * (board.bishops & black_occ).bit_count())
          + 500 * (board.rooks.bit_count() - 2 * (board.rooks & black_occ).bit_count())
          + 900 * (board.queens.bit_count() - 2 * (board.queens & black_occ).bit_count())) / 1000.0
    # Agregat file-pion (16 popcount mask): dipakai fitur 12/13 (1.8).
    w_pf = [(white_pawns & _FILES[f]).bit_count() for f in range(8)]
    b_pf = [(black_pawns & _FILES[f]).bit_count() for f in range(8)]
    v1 = 0.0
    v2 = 0.0
    v3 = 0
    v5 = 0
    v6 = 0
    v7 = 0
    v8 = 0
    v4 = 0.0
    attacks_mask = board.attacks_mask
    # Iterasi pion per warna sekali: rank-sum, passer, center, tercakup di sini.
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        own = white_pawns if color == chess.WHITE else black_pawns
        enemy = black_pawns if color == chess.WHITE else white_pawns
        # [1] pawn_advantage: jumlah rank pion (relatif) / 16
        rank_sum = 0
        m = own
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            rank_sum += (sq >> 3) if color else 7 - (sq >> 3)
        v1 += sign * (rank_sum / 16.0)
        # [2] passed_pawns dengan zona pra-hitung
        zone = _PASS_ZONE_W if color == chess.WHITE else _PASS_ZONE_B
        m = own
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            if not (enemy & zone[sq]):
                prog = (sq >> 3) if color else 7 - (sq >> 3)
                v2 += sign * (1 + prog) / 8.0
        # [3] pion di file tengah d/e
        v3 += sign * (own & _FILE_D_E).bit_count()
        # [4] mobility tanpa bobot untuk N/B/R/Q
        mob = 0
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            m = board.pieces_mask(pt, color)
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                mob += attacks_mask(sq).bit_count()
        v4 += sign * (mob / 40.0)
        # [5] king_exposure
        king = board.king(color)
        if king is not None:
            v5 += sign * board.attackers_mask(not color, king).bit_count()
        # [6] bishop pair
        if board.pieces_mask(chess.BISHOP, color).bit_count() >= 2:
            v6 += sign
        # [7] benteng di file tanpa pion sendiri
        own_pawn_files = 0
        m = own
        while m:
            own_pawn_files |= 1 << ((m & -m).bit_length() - 1 & 7)
            m &= m - 1
        m = board.pieces_mask(chess.ROOK, color)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            if not (own_pawn_files >> (sq & 7)) & 1:
                v7 += sign
        # [8] knight outposts (rank relatif >= 4, tidak diserang lawan)
        m = board.pieces_mask(chess.KNIGHT, color)
        while m:
            sq = (m & -m).bit_length() - 1
            m &= m - 1
            rank = (sq >> 3) if color else 7 - (sq >> 3)
            if rank >= 4 and not board.is_attacked_by(not color, sq):
                v8 += sign
    v9 = phase(board) * float(np.float32(v0))
    v10 = 1.0 if board.turn == chess.WHITE else -1.0
    # [11] doubled_pawns: kelebihan pion di atas satu per file (putih - hitam).
    w_doubled = sum(c - 1 for c in w_pf if c > 1)
    b_doubled = sum(c - 1 for c in b_pf if c > 1)
    v11 = (w_doubled - b_doubled) / 4.0
    # [12] isolated_pawns: pion di file tanpa pion kawan bertetangga.
    w_iso = sum(c for f, c in enumerate(w_pf)
                if c and (f == 0 or not w_pf[f - 1]) and (f == 7 or not w_pf[f + 1]))
    b_iso = sum(c for f, c in enumerate(b_pf)
                if c and (f == 0 or not b_pf[f - 1]) and (f == 7 or not b_pf[f + 1]))
    v12 = (w_iso - b_iso) / 4.0
    # [13] king_shield: pion sendiri pada tiga file depan raja (putih - hitam)/2.
    w_shield = b_shield = 0
    kw = board.king(chess.WHITE)
    if kw is not None:
        kf = kw & 7
        w_shield = sum(1 for f in (kf - 1, kf, kf + 1)
                       if 0 <= f <= 7 and (white_pawns >> chess.square(f, 1)) & 1)
    kb = board.king(chess.BLACK)
    if kb is not None:
        kf = kb & 7
        b_shield = sum(1 for f in (kf - 1, kf, kf + 1)
                       if 0 <= f <= 7 and (black_pawns >> chess.square(f, 6)) & 1)
    v13 = (w_shield - b_shield) / 2.0
    return np.asarray([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                       v11, v12, v13], dtype=np.float32)


class OutcomeModel:
    def __init__(self, weights=None, bias=0.0, micro_weights=None, micro_bias=0.0):
        values = np.asarray(weights if weights is not None else np.zeros(len(FEATURES)), dtype=np.float32)
        if values.shape != (len(FEATURES),) or not np.all(np.isfinite(values)):
            raise ValueError(f"weights must be {len(FEATURES)} finite values")
        bias = float(bias)
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        self.weights = values.copy()
        self.bias = bias
        micro = np.asarray(micro_weights if micro_weights is not None else np.zeros(len(FEATURES)), dtype=np.float32)
        if micro.shape != (len(FEATURES),) or not np.all(np.isfinite(micro)):
            raise ValueError(f"micro_weights must be {len(FEATURES)} finite values")
        micro_bias = float(micro_bias)
        if not math.isfinite(micro_bias):
            raise ValueError("micro_bias must be finite")
        self.micro_weights = micro.copy()
        self.micro_bias = micro_bias
    def position_score(self, board: chess.Board) -> float:
        return float(np.dot(self.weights, features(board)))
    def micro_score(self, board: chess.Board) -> float:
        return float(np.dot(self.micro_weights, features(board)) + self.micro_bias)
    def score(self, board: chess.Board) -> float:
        # Fitur dihitung sekali untuk kedua kepala (1.6: hilangkan komputasi ganda).
        f = features(board)
        return float(np.dot(self.weights, f) + self.bias
                     + np.dot(self.micro_weights, f) + self.micro_bias)
    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_features = data.get("features")
        if saved_features == FEATURES:
            # Reuse constructor validation to reject truncated, malformed or NaN models.
            validated = OutcomeModel(data["weights"], data.get("bias", 0.0),
                                     data.get("micro_weights"), data.get("micro_bias", 0.0))
        elif (isinstance(saved_features, list) and len(saved_features) < len(FEATURES)
              and FEATURES[:len(saved_features)] == saved_features):
            # Migrasi skema 1.6/1.7 -> 1.8: fitur lama adalah prefix skema baru;
            # fitur tambahan mulai dengan bobot 0 (perilaku model lama identik).
            pad = len(FEATURES) - len(saved_features)
            weights = list(data["weights"]) + [0.0] * pad
            micro_saved = data.get("micro_weights")
            micro = (list(micro_saved) if isinstance(micro_saved, list)
                     and len(micro_saved) == len(saved_features)
                     else [0.0] * len(saved_features))
            micro = micro + [0.0] * pad
            validated = OutcomeModel(weights, data.get("bias", 0.0), micro,
                                     data.get("micro_bias", 0.0))
        else:
            raise ValueError("Feature schema mismatch")
        self.weights, self.bias = validated.weights, validated.bias
        self.micro_weights, self.micro_bias = validated.micro_weights, validated.micro_bias
        return self

    def save(self, path, metadata=None):
        # Revalidate mutable public arrays before persisting a checkpoint.
        validated = OutcomeModel(self.weights, self.bias, self.micro_weights, self.micro_bias)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"version": 2, "schema_version": 2,
                                      "features": FEATURES,
                                      "weights": validated.weights.tolist(),
                                      "bias": validated.bias,
                                      "micro_weights": validated.micro_weights.tolist(),
                                      "micro_bias": validated.micro_bias,
                                      "metadata": metadata or {}}, indent=2), encoding="utf-8")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _metrics(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> dict:
    if len(y) == 0:
        return {"samples": 0, "logloss": None, "brier": None, "accuracy": None}
    p = _sigmoid(X @ w + b)
    eps = 1e-7
    logloss = float(-np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
    brier = float(np.mean((p - y) ** 2))
    # Outcomes include draws (0.5); accuracy uses the decisive threshold only as a
    # coarse diagnostic and deliberately is not treated as a strength metric.
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    return {"samples": int(len(y)), "logloss": logloss, "brier": brier, "accuracy": acc}


def train_pgn(paths, epochs=8, lr=0.015, seed=42, max_samples_per_game=120,
              validation_fraction=0.20, batch_size=64, patience=4,
              init_weights=None, init_bias=0.0, lr_decay=1.0, class_balance=True):
    """Train a white-outcome logistic evaluator with deterministic game-level holdout.


    For tiny corpora (fewer than five games), fall back to a position-level split
    and explicitly report that validation is not game-independent. Mini-batch SGD,
    clipping, regularization, early stopping, and best-validation checkpointing make
    training faster and less prone to overfit than per-position unbounded SGD.

    1.8: ``init_weights``/``init_bias`` melanjutkan dari bobot champion
    (warm start — konvergensi lebih cepat antar generasi); ``lr_decay`` < 1
    menurunkan learning rate per epoch (lr * decay^epoch); ``class_balance``
    memberi bobot sampel frekuensi-terbalik per kelas hasil (1-0/seri/0-1)
    agar korpus timpang tidak membiaskan model.
    """
    if int(epochs) < 1 or int(batch_size) < 1 or int(max_samples_per_game) < 1:
        raise ValueError("epochs, batch_size and max_samples_per_game must be positive")
    if not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError("lr must be a finite positive value")
    if not math.isfinite(float(lr_decay)) or float(lr_decay) <= 0 or float(lr_decay) > 1.0:
        raise ValueError("lr_decay must be in (0, 1]")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    game_samples = []
    total_games = 0
    skipped_games = 0
    result_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            while game := chess.pgn.read_game(f):
                res = game.headers.get("Result")
                if res not in result_counts:
                    skipped_games += 1
                    continue
                total_games += 1
                result_counts[res] += 1
                target = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
                board = game.board()
                positions = []
                ply = 0
                for mv in game.mainline_moves():
                    if board.is_game_over():
                        break
                    board.push(mv)
                    ply += 1
                    # Alternate sampled ply parity across games. Sampling only even
                    # plies made every training position White-to-move, turning the
                    # tempo feature into a constant and weakening its usefulness.
                    sample_parity = total_games % 2
                    if ply >= 6 and ply % 2 == sample_parity and board.is_valid():
                        positions.append((features(board), target))
                if not positions:
                    continue
                if len(positions) > max_samples_per_game:
                    positions = rng.sample(positions, max_samples_per_game)
                game_samples.append(positions)

    if not game_samples:
        raise ValueError("No valid labeled positions found in PGN files")

    val_fraction = float(np.clip(validation_fraction, 0.05, 0.45))
    independent_validation = len(game_samples) >= 5
    if independent_validation:
        order = list(range(len(game_samples)))
        rng.shuffle(order)
        val_count = max(1, int(round(len(order) * val_fraction)))
        val_ids = set(order[:val_count])
        train_groups = [g for i, g in enumerate(game_samples) if i not in val_ids]
        val_groups = [g for i, g in enumerate(game_samples) if i in val_ids]
    else:
        # For small corpora, split positions within games. This is useful for
        # debugging training but can leak game-level patterns; flag it in metadata.
        flat = [sample for group in game_samples for sample in group]
        rng.shuffle(flat)
        val_count = max(1, int(round(len(flat) * val_fraction))) if len(flat) > 3 else 0
        val_groups = [flat[:val_count]] if val_count else []
        train_groups = [flat[val_count:] if val_count else flat]

    train_flat = [sample for group in train_groups for sample in group]
    val_flat = [sample for group in val_groups for sample in group]
    if not train_flat:
        raise ValueError("Training split is empty; add more PGN positions")
    X_train = np.stack([x for x, _ in train_flat]).astype(np.float32)
    y_train = np.asarray([y for _, y in train_flat], dtype=np.float32)
    X_val = np.stack([x for x, _ in val_flat]).astype(np.float32) if val_flat else np.empty((0, len(FEATURES)), dtype=np.float32)
    y_val = np.asarray([y for _, y in val_flat], dtype=np.float32) if val_flat else np.empty(0, dtype=np.float32)

    # 1.8: warm start dari champion bila dimensi cocok (migrasi skema sudah
    # menjamin champion selalu berdimensi penuh); best checkpoint epoch 0
    # = champion itu sendiri, jadi early-stop tidak pernah lebih buruk.
    if init_weights is not None:
        w0 = np.asarray(init_weights, dtype=np.float32).ravel()
        if w0.shape != (X_train.shape[1],) or not np.all(np.isfinite(w0)):
            raise ValueError("init_weights must match the feature schema")
        b0 = float(init_bias)
        if not math.isfinite(b0):
            raise ValueError("init_bias must be finite")
    else:
        w0 = np.zeros(X_train.shape[1], dtype=np.float32)
        b0 = 0.0
    w = w0.copy()
    b = b0
    best_w, best_b = w.copy(), b
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    bs = max(8, int(batch_size))
    # 1.8: bobot kelas frekuensi-terbalik (hasil: 1.0 putih menang, 0.5 seri,
    # 0.0 hitam menang) — korpus timpang tidak lagi mendominasi gradient.
    # Normalisasi hanya atas kelas yang HADIR (rata-rata bobot = 1): korpus
    # satu-kelas (korpus awal kecil) tidak menyusutkan gradien, dan model
    # berperilaku sama dengan 1.7 pada kasus degenerate itu.
    class_weights = None
    sw_train = None
    if class_balance and len(y_train):
        counts = {cls: int(np.sum(y_train == cls)) for cls in (0.0, 0.5, 1.0)}
        present = {c: n for c, n in counts.items() if n > 0}
        k = len(present)
        class_weights = {c: (len(y_train) / (k * n)) for c, n in present.items()}
        for cls in (0.0, 0.5, 1.0):
            class_weights.setdefault(cls, 1.0)
        sw_train = np.asarray([class_weights[y] for y in y_train],
                              dtype=np.float32)
    for ep in range(max(1, int(epochs))):
        lr_ep = lr * (float(lr_decay) ** ep) if float(lr_decay) < 1.0 else lr
        indices = np_rng.permutation(len(y_train))
        for begin in range(0, len(indices), bs):
            ix = indices[begin:begin + bs]
            xb, yb = X_train[ix], y_train[ix]
            pred = _sigmoid(xb @ w + b)
            grad = (pred - yb)
            if sw_train is not None:
                swb = sw_train[ix]
                grad = grad * swb
                denom = max(1e-6, float(np.sum(swb)))
                gw = (xb.T @ grad) / denom + 0.0005 * w
                gb = float(np.sum(grad)) / denom
            else:
                gw = (xb.T @ grad) / max(1, len(ix)) + 0.0005 * w
                gb = float(np.mean(grad))
            # Prevent extreme updates from corrupting the model on noisy labels.
            norm = float(np.linalg.norm(gw))
            if norm > 5.0:
                gw *= 5.0 / norm
            w -= lr_ep * gw
            b -= lr_ep * gb
        train_m = _metrics(X_train, y_train, w, b)
        val_m = _metrics(X_val, y_val, w, b) if len(y_val) else train_m
        monitored = val_m["logloss"]
        history.append({"epoch": ep + 1, "train": train_m, "validation": val_m})
        if monitored < best_loss - 1e-5:
            best_loss, best_w, best_b = monitored, w.copy(), b
            best_epoch = ep + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max(1, patience):
                break
    model = OutcomeModel(best_w, best_b)
    metadata = {
        "samples": int(len(y_train) + len(y_val)),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "games_sources": len(paths),
        "games_read": total_games,
        "games_skipped": skipped_games,
        "result_counts": result_counts,
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "lr": float(lr),
        "seed": int(seed),
        # 1.8: pencatatan knob pembelajaran baru (audit + reproducibility).
        "warm_start": bool(init_weights is not None),
        "lr_decay": float(lr_decay),
        "class_balance": bool(class_balance),
        "class_weights": ({str(k): round(v, 6) for k, v in class_weights.items()}
                          if class_weights else None),
        "validation_mode": "game_holdout" if independent_validation else "position_holdout_small_corpus",
        "best_validation_logloss": best_loss,
        "train_metrics": _metrics(X_train, y_train, best_w, best_b),
        "validation_metrics": _metrics(X_val, y_val, best_w, best_b) if len(y_val) else None,
        "history": history,
    }
    return model, metadata


def main():
    ap=argparse.ArgumentParser(description="Train Kestrel's outcome-based evaluator from PGN games")
    ap.add_argument("pgn", nargs="+", help="PGN source files")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr",type=float,default=0.015)
    ap.add_argument("--output",default="results/learned_eval.json"); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--validation-fraction",type=float,default=0.20)
    ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--patience",type=int,default=4)
    ap.add_argument("--init-model",default=None, help="model untuk warm start (skema kompatibel)")
    ap.add_argument("--lr-decay",type=float,default=1.0); ap.add_argument("--no-class-balance",action="store_true")
    args=ap.parse_args()
    init_w = init_b = None
    if args.init_model:
        m = OutcomeModel().load(args.init_model)
        init_w, init_b = m.weights, m.bias
    model,meta=train_pgn(args.pgn,args.epochs,args.lr,args.seed,
        validation_fraction=args.validation_fraction,batch_size=args.batch_size,patience=args.patience,
        init_weights=init_w, init_bias=init_b if init_b is not None else 0.0,
        lr_decay=args.lr_decay, class_balance=not args.no_class_balance)
    model.save(args.output,meta); print(json.dumps({"output":args.output,**meta},indent=2))
if __name__ == "__main__": main()
