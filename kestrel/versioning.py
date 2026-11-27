"""KESTREL 2.0 — Versioning checkpoint TinyNNUE (champion/challenger per generasi).

Kontrak file:
- ``results/nnue_eval.json``        : net champion aktif (dipakai worker
  dashboard/selfplay/bot Lichess/cross-engine).
- ``results/champion.json``         : metadata champion (generasi, asal,
  laporan verifikasi, riwayat promote).
- ``results/checkpoints/gen_<label>_nnue.json`` : snapshot kandidat.

Semua penggantian file atomik (write temp + os.replace) dan checkpoint lama
dipertahankan sesuai ``keep_last_checkpoints``. Rollback mengembalikan
snapshot sebelumnya tanpa menghapusnya.

``load_champion()`` TIDAK pernah mengembalikan net acak: tanpa champion,
yang dipakai adalah checkpoint baseline terlatih bawaan paket.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

from .nnue import TinyNNUE, load_default


class CheckpointStore:
    def __init__(self, results_dir, keep_last: int = 5):
        self.results = Path(results_dir)
        self.results.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.results / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)
        self.keep_last = max(1, int(keep_last))
        self.champion_model_path = self.results / "nnue_eval.json"
        self.champion_meta_path = self.results / "champion.json"

    # ---------------- champion metadata ----------------
    def champion_meta(self) -> dict:
        """Metadata champion; kalau belum ada, inisialisasi generasi 0."""
        if self.champion_meta_path.exists():
            try:
                return json.loads(self.champion_meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        meta = {"generation": 0, "source": "baseline",
                "created_at": time.time(), "history": []}
        if self.champion_model_path.exists():
            meta["model_file"] = str(self.champion_model_path)
        self._write_json(self.champion_meta_path, meta)
        return meta

    def _write_json(self, path: Path, data):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    # ---------------- candidates ----------------
    def candidate_path(self, generation: int | str) -> Path:
        # `generation` bisa int (champion lineage) atau str label unik
        # (mis. "0000_try02") supaya kandidat yang DITOLAK tidak saling
        # menimpa checkpoint/laporan antar iterasi trainer.
        return self.checkpoints / f"gen_{str(generation)}_nnue.json"

    def save_candidate(self, net: TinyNNUE, generation: int | str,
                       metadata: dict | None = None) -> Path:
        path = self.candidate_path(generation)
        meta = dict(metadata or {})
        meta.setdefault("generation", generation)
        meta.setdefault("created_at", time.time())
        net.save(path, metadata=meta)
        self._prune_checkpoints()
        return path

    def load_model(self, path) -> TinyNNUE:
        return TinyNNUE.load(Path(path))

    def load_champion(self) -> TinyNNUE:
        """Champion aktif; tanpa champion file → baseline bawaan (terlatih)."""
        if self.champion_model_path.exists():
            return TinyNNUE.load(self.champion_model_path)
        return load_default()

    def has_trained_champion(self) -> bool:
        return self.champion_model_path.exists()

    def candidates(self) -> list[Path]:
        return sorted(self.checkpoints.glob("gen_*_nnue.json"))

    # ---------------- promote / rollback ----------------
    def promote(self, candidate_path, reason: str, report: dict | None = None) -> dict:
        """Jadikan kandidat champion baru (atomik + catat riwayat)."""
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        meta = self.champion_meta()
        previous = {"generation": meta.get("generation", 0),
                    "promoted_at": time.time(),
                    "model_file": None}
        # Snapshot champion lama ke file stabil: history tidak boleh menunjuk
        # path live champion (file itu tertimpa kandidat baru). Promosi PERTAMA
        # (belum ada champion) menyimpan None — rollback ke entri ini dilewati
        # otomatis.
        if self.champion_model_path.exists():
            snap = self.checkpoints / f"champion_gen_{meta.get('generation', 0):04d}.json"
            shutil.copy2(self.champion_model_path, snap)
            previous["model_file"] = str(snap)
        tmp = self.champion_model_path.with_name(self.champion_model_path.name + ".promote_tmp")
        shutil.copy2(candidate_path, tmp)
        os.replace(tmp, self.champion_model_path)
        meta["generation"] = meta.get("generation", 0) + 1
        meta["source"] = str(candidate_path)
        meta["promoted_at"] = time.time()
        meta["promote_reason"] = reason
        meta["last_verification"] = report or {}
        meta["history"] = (meta.get("history", []) + [previous])[-50:]
        self._write_json(self.champion_meta_path, meta)
        self._prune_checkpoints()
        return meta

    def rollback(self, steps: int = 1) -> dict:
        """Kembalikan champion ke snapshot kandidat `steps` langkah sebelumnya."""
        meta = self.champion_meta()
        history = meta.get("history", [])
        if not history:
            raise FileNotFoundError("Tidak ada riwayat untuk rollback")
        steps = max(1, min(int(steps), len(history)))
        # entri pertama riwayat bisa tanpa snapshot (promosi pertama). Cari
        # snapshot terdekat yang benar-benar ada.
        entry = history[-steps]
        idx = -steps
        while entry.get("model_file") is None or not Path(entry["model_file"]).exists():
            idx -= 1
            if abs(idx) > len(history):
                raise FileNotFoundError(
                    "Tidak ada snapshot champion yang bisa di-rollback")
            entry = history[idx]
        src = Path(entry["model_file"])
        tmp = self.champion_model_path.with_name(self.champion_model_path.name + ".rollback_tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, self.champion_model_path)
        meta["generation"] = entry.get("generation", meta.get("generation", 1) - steps)
        meta["source"] = f"rollback:{src}"
        meta["rolled_back_at"] = time.time()
        meta["history"] = history[:-steps]
        self._write_json(self.champion_meta_path, meta)
        return meta

    def _prune_checkpoints(self):
        cands = self.candidates()
        excess = len(cands) - self.keep_last
        for path in cands[:max(0, excess)]:
            with contextlib.suppress(OSError):
                path.unlink()

    def summary(self) -> dict:
        meta = self.champion_meta()
        return {"generation": meta.get("generation", 0),
                "source": meta.get("source"),
                "champion_exists": self.champion_model_path.exists(),
                "candidates": [p.name for p in self.candidates()],
                "keep_last": self.keep_last,
                "promoted_at": meta.get("promoted_at"),
                "promote_reason": meta.get("promote_reason")}
