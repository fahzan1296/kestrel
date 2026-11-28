# Kestrel performance phase 2 — design

## Intent

Lanjutkan optimasi Kestrel 2.0.2 selama ~24 jam dengan fokus pada kecepatan
pencarian realtime saat bot tetap bermain. Tujuannya bukan sekadar angka NPS,
tetapi langkah yang tetap kuat dan bisa dibuktikan tidak merusak evaluasi.

## Goals

- Turunkan waktu per langkah pada posisi start, middlegame, taktik, endgame,
  dan posisi check.
- Pertahankan perilaku catur: legalitas, mate/stalemate, TT, killer, history,
  dan skor NNUE tetap sama.
- Setiap perubahan memiliki pengukuran sebelum/sesudah dengan 20 run
  interleaved A/B/A/B setelah harness dikunci.
- Perubahan yang netral atau lebih lambat di-revert.

## Non-goals

- Tidak menurunkan `move_time_ms`, `min_depth`, `grace_ms`, `hard_ext_ms`.
- Tidak mengganti TinyNNUE dengan evaluator klasik.
- Tidak menyentuh token, `.env`, atau konfigurasi Lichess.
- Tidak menggabungkan beberapa optimasi dalam satu pengukuran.

## Constraints

- Mesin produksi: InstaBox `kestrel-2`, 4 vCPU; bot dan training berjalan
  bersamaan.
- Token hanya ada di `.env` mode 600 dan tidak boleh dicetak.
- Full test suite harus hijau sebelum deploy.
- Benchmark memakai model baseline yang sama dan FEN tetap.
- Champion promotion tidak boleh terjadi di tengah benchmark.
- Baseline 5-run awal `[5897, 5852, 5743, 5485, 5804]` punya spread ~7.5%,
  terlalu bising untuk klaim gain kecil.

## Approaches

1. **Harness dulu (dipilih)** — kunci `scripts/bench_perf.py --suite` dengan 5
   FEN tetap, model baseline eksplisit, catat environment dan BLAS thread.
   Ukur 20 run interleaved A/B/A/B. Tanpa ini, semua angka berikutnya tidak
   bisa dipercaya.
2. **Runtime BLAS single-thread** — uji `OPENBLAS_NUM_THREADS=1`,
   `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Ini operasional, murah, dan bisa
   lebih berdampak daripada micro-optimasi evaluator.
3. **qsearch legal-move cache** — `main search` sudah memakai `_moves_cache`,
   tetapi qsearch masih memanggil `list(board.legal_moves)` dan
   `next(iter(board.legal_moves))`. Legal moves adalah fungsi murni posisi,
   jadi cache aman secara teori dan menyentuh hotspot lebih besar daripada
   NNUE.
4. **NNUE scratch buffer** — hapus `np.concatenate` dan array `_aux` baru per
   leaf dengan scratch buffer thread-local. Ditunda sampai harness dan
   qsearch cache selesai.

Risiko dan kontrol:

- qsearch cache: hasil harus identik pada normal/check/mate/stalemate/
  repetition; list hasil tidak boleh dimutasi caller.
- NNUE buffer: parity exakt untuk `raw`, `stm_score`, `white_score`; semua slice
  ditimpa tiap call; aman per thread.
- BLAS: jangan gabungkan dengan perubahan kode lain dalam satu pengukuran.

## Builder decisions

- Python 3.13, python-chess, numpy, pytest — mengikuti proyek yang ada.
- Benchmark tetap di `scripts/bench_perf.py`; tidak membuat framework baru.
- Setiap eksperimen dicatat di `docs/PERF.md`, termasuk yang di-revert.
- Eksekusi inline di sesi ini; tidak memakai workflow multi-agent.
- Karena `/root/kestrel` bukan git repo, commit dilakukan di staging repo
  `/root/kestrel-release-2.0` setelah perubahan terverifikasi.

## Remaining to builder

- Tulis test RED untuk tiap perubahan.
- Ukur 20 run interleaved sebelum/sesudah dengan perintah identik.
- Revert bila tidak melewati threshold.
- Deploy ke mesin hanya setelah suite hijau.
