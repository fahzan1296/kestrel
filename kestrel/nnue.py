"""KESTREL 2.0 — TinyNNUE: evaluator AI tinyML (menggantikan evaluator klasik).

Arsitektur (gaya NNUE versi mini, numpy FP32, CPU-only):

    fitur : 12 bidak-plane x 64 kotak = 768 plane per perspektif
    L1    : 768 -> 128 (BOBOT DIBAGI antara dua perspektif, NNUE-style)
    acc   : dua akumulator 128-dim (perspektif putih & hitam), inkremental
    L2    : concat(acc_stm, acc_ntm, aux[4]) = 260 -> 32, ReLU
    L3    : 32 -> 1, linear, diskalakan OUT_SCALE=400 -> centipawn STM

Perbedaan kunci dari TinyMLP eksperimental 1.x:

1. AKUMULATOR INKREMENTAL. Layer pertama tidak pernah dihitung ulang penuh:
   setiap langkah hanya menambah/mengurangi baris W1 bidak yang berpindah /
   tertangkap (<= 4 baris per langkah). Searcher memelihara stack akumulator
   bersama stack Zobrist-nya.
2. PERSPEKTIF STM. Jaringan menerima (acc_stm, acc_ntm), bukan (acc_putih,
   acc_hitam) + fitur giliran. Konsekuensinya antisimetri mirror sempurna
   BY KONSTRUKSI: raw(mirror(s)) == raw(s), sehingga
   eval_white(mirror(s)) == -eval_white(s) tanpa drift (diverifikasi test).
3. SELALU TERLATIH. Output tidak pernah di-inisialisasi acak; model
   checkpoint hasil training (distilasi skor search + hasil game) yang
   di-load. Tanpa checkpoint valid, `load_default()` gagal keras — tidak
   ada jalur "net acak" yang diam-diam dipakai engine.
4. TERKUANTISASI int8 untuk penyimpanan (~110 KB vs ~430 KB FP32) dengan
   skala per-tensor; runtime memuat kembali ke FP32.

Jumlah parameter: 768*128 + (256+4)*32 + 32 + 32 + 1 = 106.689 (< 1M).
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

import chess
import numpy as np

# ---------------- dimensi arsitektur ----------------
PLANES = 768          # 6 jenis bidak x 2 warna x 64 kotak
ACC_DIM = 128         # dimensi akumulator (layer 1)
AUX_DIM = 4           # fitur aux: fase, halfmove, castle stm, castle ntm
HIDDEN = 32           # layer 2
OUT_SCALE = 400.0     # raw -> centipawn
EVAL_CLIP = 1800      # batas centipawn output (score mate bukan domain net)

PARAM_COUNT = (PLANES * ACC_DIM) + (ACC_DIM * 2 + AUX_DIM) * HIDDEN + HIDDEN \
    + HIDDEN * 1 + 1  # 106.689

_B64 = base64.b64encode, base64.b64decode


def plane_w(pt: int, color: bool, sq: int) -> int:
    """Index plane perspektif PUTIH: warna sama dgn view -> offset 0, lawan -> 64."""
    return (pt - 1) * 128 + (0 if color else 64) + sq


def plane_b(pt: int, color: bool, sq: int) -> int:
    """Index plane perspektif HITAM (kotak di-mirror vertikal)."""
    return (pt - 1) * 128 + (64 if color else 0) + (sq ^ 56)


def _aux(board: chess.Board) -> np.ndarray:
    """Fitur aux 4-dim, simetris terhadap mirror (perspektif STM)."""
    stm = board.turn
    n_kn = board.knights.bit_count()
    n_bi = board.bishops.bit_count()
    n_ro = board.rooks.bit_count()
    n_qu = board.queens.bit_count()
    phase = min(24, n_kn + n_bi + 2 * n_ro + 4 * n_qu) / 24.0
    rights = board.clean_castling_rights()
    # Hak castling dari perspektif STM; mirror menukar putih<->hitam, jadi
    # fitur ini tetap simetris perspektif-STM.
    own = (rights & chess.BB_RANK_1 if stm == chess.WHITE
           else rights & chess.BB_RANK_8)
    opp = (rights & chess.BB_RANK_8 if stm == chess.WHITE
           else rights & chess.BB_RANK_1)
    return np.asarray([
        phase,
        min(board.halfmove_clock, 100) / 100.0,
        own.bit_count() / 2.0,
        opp.bit_count() / 2.0,
    ], dtype=np.float32)


class TinyNNUE:
    """Evaluator tinyML NNUE-lite. FP32 runtime, int8 storage."""

    kind = "tinynnue"

    def __init__(self, w1: np.ndarray | None = None, w2: np.ndarray | None = None,
                 b2: np.ndarray | None = None, w3: np.ndarray | None = None,
                 b3: np.ndarray | None = None, metadata: dict | None = None):
        if w1 is None:
            # Skala init ditata agar aktivasi hidup (lihat scripts/grad_check):
            # 32 bidak x std 0.05 -> acc std ~0.28; He-init utk L2; kepala
            # output tetap NOL (net acak tak boleh jadi evaluator runtime —
            # training menggantinya dengan kepala acak-kecil, lihat
            # nnue_train.train_nnue).
            rng = np.random.default_rng(42)
            w1 = rng.standard_normal((PLANES, ACC_DIM)) * 0.05
            w2 = rng.standard_normal((ACC_DIM * 2 + AUX_DIM, HIDDEN)) * \
                math.sqrt(2.0 / (ACC_DIM * 2 + AUX_DIM))
            b2 = np.zeros(HIDDEN, np.float32)
            w3 = np.zeros((HIDDEN, 1), np.float32)
            b3 = np.zeros(1, np.float32)
        self.w1 = np.ascontiguousarray(w1, dtype=np.float32)
        self.w2 = np.ascontiguousarray(w2, dtype=np.float32)
        self.b2 = np.ascontiguousarray(b2, dtype=np.float32)
        self.w3 = np.ascontiguousarray(w3, dtype=np.float32)
        self.b3 = np.ascontiguousarray(b3, dtype=np.float32)
        self.metadata = dict(metadata or {})
        self._validate()

    def _validate(self) -> None:
        shapes = {
            "w1": (self.w1.shape, (PLANES, ACC_DIM)),
            "w2": (self.w2.shape, (ACC_DIM * 2 + AUX_DIM, HIDDEN)),
            "b2": (self.b2.shape, (HIDDEN,)),
            "w3": (self.w3.shape, (HIDDEN, 1)),
            "b3": (self.b3.shape, (1,)),
        }
        for name, (got, want) in shapes.items():
            if got != want:
                raise ValueError(f"{name} shape {got} != {want}")
        for name, arr in (("w1", self.w1), ("w2", self.w2), ("b2", self.b2),
                          ("w3", self.w3), ("b3", self.b3)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} mengandung nilai non-finite")

    # ---------------- param ----------------
    def params(self) -> int:
        return PARAM_COUNT

    def copy(self) -> "TinyNNUE":
        return TinyNNUE(self.w1.copy(), self.w2.copy(), self.b2.copy(),
                        self.w3.copy(), self.b3.copy(), self.metadata)

    # ---------------- akumulator ----------------
    def refresh(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
        """Akumulator (acc_putih, acc_hitam) dari nol — bitboard loop.

        Dipakai saat sinkronisasi root / setelah push eksternal. Jalur panas
        search memakai update inkremental lewat ``delta()``.
        """
        w1 = self.w1
        aw = np.zeros(ACC_DIM, np.float32)
        ab = np.zeros(ACC_DIM, np.float32)
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
                   chess.QUEEN, chess.KING):
            base = (pt - 1) * 128
            m = board.pieces_mask(pt, chess.WHITE)
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                aw += w1[base + sq]
                ab += w1[base + 64 + (sq ^ 56)]
            m = board.pieces_mask(pt, chess.BLACK)
            while m:
                sq = (m & -m).bit_length() - 1
                m &= m - 1
                aw += w1[base + 64 + sq]
                ab += w1[base + (sq ^ 56)]
        return aw, ab

    def delta(self, removed: list[tuple[int, int, int]],
              added: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """Delta akumulator (d_accw, d_accb) untuk bidak hilang/muncul.

        Setiap item = (piece_type, color, square). Perspektif putih memakai
        kotak apa adanya, perspektif hitam memakai kotak mirrored.
        """
        dw = np.zeros(ACC_DIM, np.float32)
        db = np.zeros(ACC_DIM, np.float32)
        w1 = self.w1
        for pt, color, sq in removed:
            base = (pt - 1) * 128
            dw -= w1[base + (0 if color else 64) + sq]
            db -= w1[base + (64 if color else 0) + (sq ^ 56)]
        for pt, color, sq in added:
            base = (pt - 1) * 128
            dw += w1[base + (0 if color else 64) + sq]
            db += w1[base + (64 if color else 0) + (sq ^ 56)]
        return dw, db

    # ---------------- forward ----------------
    def raw(self, board: chess.Board,
            accs: tuple[np.ndarray, np.ndarray] | None = None) -> float:
        """Skor mentah (pra-clip) RELATIF STM. ``accs`` = (acc_putih, acc_hitam)."""
        aw, ab = accs if accs is not None else self.refresh(board)
        if board.turn == chess.WHITE:
            x = np.concatenate((aw, ab, _aux(board)))
        else:
            x = np.concatenate((ab, aw, _aux(board)))
        h = np.maximum(x @ self.w2 + self.b2, 0.0)
        return float((h @ self.w3 + self.b3)[0])

    def stm_score(self, board: chess.Board,
                  accs: tuple[np.ndarray, np.ndarray] | None = None) -> int:
        """Centipawn relatif side-to-move (ter-clip)."""
        return int(max(-EVAL_CLIP, min(EVAL_CLIP, OUT_SCALE * self.raw(board, accs))))

    def white_score(self, board: chess.Board,
                    accs: tuple[np.ndarray, np.ndarray] | None = None) -> int:
        """Centipawn relatif PUTIH: posisi & mirror-nya anti-simetrik eksak."""
        return (self.stm_score(board, accs) if board.turn == chess.WHITE
                else -self.stm_score(board, accs))

    # ---------------- persistensi ----------------
    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "version": 3,
            "arch": {"planes": PLANES, "acc": ACC_DIM, "hidden": HIDDEN,
                     "aux": AUX_DIM, "out_scale": OUT_SCALE},
            "params": int(PARAM_COUNT),
            "w1_int8": {
                "scale": float(np.max(np.abs(self.w1)) / 127.0),
                "data": base64.b64encode(
                    np.clip(np.round(self.w1 / max(np.max(np.abs(self.w1)) / 127.0, 1e-12)),
                            -127, 127).astype(np.int8).tobytes()).decode("ascii"),
            },
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "metadata": {**self.metadata, **(metadata or {})},
        }
        import json
        target.write_text(json.dumps(payload, default=float), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TinyNNUE":
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("kind") != cls.kind or data.get("version") != 3:
            raise ValueError(f"Bukan checkpoint TinyNNUE v3: {path}")
        arch = data["arch"]
        if (arch["planes"], arch["acc"], arch["hidden"], arch["aux"]) != \
                (PLANES, ACC_DIM, HIDDEN, AUX_DIM):
            raise ValueError("Arsitektur checkpoint tidak cocok")
        q = data["w1_int8"]
        w1 = np.frombuffer(base64.b64decode(q["data"]), dtype=np.int8) \
            .astype(np.float32).reshape(PLANES, ACC_DIM) * q["scale"]
        net = cls(w1, np.asarray(data["w2"], np.float32),
                  np.asarray(data["b2"], np.float32),
                  np.asarray(data["w3"], np.float32),
                  np.asarray(data["b3"], np.float32),
                  data.get("metadata"))
        return net


# ---------------- model bawaan (baseline terlatih) ----------------
_DEFAULT_NET: TinyNNUE | None = None
BASELINE_PATH = Path(__file__).resolve().parent / "models" / "nnue_baseline.json"


def load_default() -> TinyNNUE:
    """Muat checkpoint baseline terlatih yang dibundel dengan paket.

    Tidak ada fallback acak: bila file baseline hilang/korup, error keras —
    engine 2.0 tidak pernah berjalan dengan evaluator tidak terlatih.
    """
    global _DEFAULT_NET
    if _DEFAULT_NET is None:
        if not BASELINE_PATH.exists():
            raise RuntimeError(
                "Checkpoint baseline TinyNNUE tidak ditemukan di "
                f"{BASELINE_PATH}. Jalankan scripts/bootstrap_nnue.py atau "
                "`kestrel nnue distill` untuk membuatnya.")
        _DEFAULT_NET = TinyNNUE.load(BASELINE_PATH)
    return _DEFAULT_NET
