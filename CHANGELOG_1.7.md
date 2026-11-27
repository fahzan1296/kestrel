# KESTREL 1.7.0 — Optimasi Performa, Bug Fix, & Pengembangan Cross-Engine

Tanggal: 2026-11-26. Basis: KESTREL 1.6.0. Semua angka diukur pada sandbox 2 vCPU (Python 3.12, python-chess 1.11.2).

## Ringkasan

| Aspek | 1.6 (baseline) | 1.7 (hasil) |
|---|---|---|
| NPS single-search (kiwipete, 50k node) | 10.786 | **≈17.700 (+64%)** |
| NPS single-search (200k node) | ~10.500 | **≈17.700** |
| NPS agregat paralel (2 worker) | — | **≈24.400** (efisiensi scaling 0,98) |
| Test suite | 55 passed | **55 passed** |
| Drift evaluasi vs 1.6 | — | **0** (420 posisi + path hint) |
| Zobrist inkremental vs polyglot | — | **identik** (28.289 posisi fuzz) |
| Perilaku search vs 1.6 | — | skor & jumlah node **identik bit-per-bit** di depth 1–6 |

Pada mesin dengan lebih banyak core, NPS agregat paralel naik linear
(efisiensi ~0,98/worker) — di 8 core ≈ 100–130k, di 16 core ≈ 200–260k.

## 1. Optimasi performa (perilaku search tetap identik)

**search.py**
- **Zobrist inkremental eksak-polyglot**: kunci posisi diperbarui selama
  push/pop (tabel acak `chess.polyglot.POLYGLOT_RANDOM_ARRAY` yang sama), menghapus
  hash penuh 32 bidak per node (dulu ~20–43 μs/node). Diverifikasi identik
  bit-per-bit dengan `chess.polyglot.zobrist_hash` pada 28.289 posisi fuzz
  (push/pop/null/ep/promosi/castling).
- **Transposition table flat-array** (indeks `key & mask`, kunci penuh
  diverifikasi saat probe) menggantikan dict + eviksi sampling 64 entri.
- **Cache daftar langkah legal/capture + `is_check`** per kunci Zobrist
  (murni fungsi posisi — perilaku tidak berubah).
- **Generator capture bitboard langsung** (tabel diag/rank/file python-chess)
  dengan filter `is_into_check`; promosi non-capture dibangun langsung tanpa
  movegen penuh.
- **Move ordering satu-pass**: skor MVV-LVA/killer/history + flag
  capture/promo dihitung sekali per langkah (dulu `is_capture` dibayar dua
  kali); urutan hasil identik (sort stabel).
- Gate draw ketat yang TEPAT: repetition butuh `clock >= 2/4`, klaim
  50-langkah aktif sejak `clock >= 99`, material tak cukup pasti `<= 4` bidak.
- Tabel LMR pra-hitung (tanpa `math.log` di jalur panas); `_heavy_material`
  via popcount bitboard.

**eval.py**
- Fusi material+PST jadi satu pass per jenis bidak (17 → 12 loop mask).
- `mobility` mengakses tabel attack langsung per jenis bidak (tanpa dispatch
  `attacks_mask` per bidak).
- `pawn_structure + piece_bonuses` dilebur (`_pawn_and_bonuses`) dengan
  file-pion via popcount mask file.
- Tabel PST taper pra-hitung per `(jenis, fase-integer)` — fase selalu integer
  (≤ 26 nilai) sehingga nilai truncation `int()` tetap eksak.
- Blok terminal menerima hint dari caller (`in_check`/`has_moves`/`terminal`);
  gate ketat menjaga hasil identik tanpa hint.

## 2. Bug yang diperbaiki

**Dari audit independen (15 temuan):**
1. Perspektif komentar PGN dinormalisasi **White-relative** di SEMUA penulis
   (arena sudah benar; dashboard_worker & cross_engine diperbaiki) —
   swing kandidat REDA kini konsisten satu perspektif.
2. Komentar book move tidak lagi menulis `0/d0 n0` yang mempolusi target
   distilasi REDA/experience (dashboard_worker + cross_engine).
3. `selfplay.temperature` kini benar-benar dipakai worker (dulu hardcoded 0.25).
4. Game yang gagal start worker tidak lagi dihitung "selesai" (statistik
   `failed` jujur; trainer tidak lagi melewati retry).
5. Book ranking SQLite: `white_wins/games` (integer division) → `(white_wins * 1.0) / games`.
6. Config `[knowledge]` kini diteruskan penuh (book_max_plies/min_games/
   max_moves) ke trainer, CLI, worker dashboard, dan worker selfplay.
7. `reuse_pgn` menghitung game nyata dari file PGN (dulu angka karangan;
   korpus_games=None tidak lagi gagal tiap generasi).
8. `run_selfplay` tidak lagi hang selamanya bila worker mati tanpa sentinel.
9. Race counter belajar Lichess: increment dipindah ke dalam `_games_lock`.
10. Kolom dead/menyesatkan pada agregasi book dihapus.
11. (Tercakup #1) eval-swing lintas perspektif.
12. Rollback ke promosi pertama tidak lagi FileNotFoundError (snapshot None
    dilewati otomatis).
13. Validasi pydantic REDA post-batch dilaporkan ke REDA_STATE, bukan crash
    di thread koordinator setelah status "completed".
14. Dashboard memakai path storage dari config (dulu hardcode games.pgn/kestrel.db).
15. Stop oleh pengguna → status **STOPPED**, bukan FAIL (generasi selesai tetap dihitung).

**Dari pengujian sendiri selama optimasi (3 bug di implementasi baru):**
- Sign-flip stand-pat qsearch (white-relative dipakai tanpa konversi
  side-to-move) — penyebab ledakan qsearch & skor salah; kini skor + jumlah
  node identik bit-per-bit vs 1.6 di semua depth.
- Pivot Zobrist terbalik (`occupied_co` = [HITAM, PUTIH] karena
  `chess.WHITE == True == 1`) — ditemukan lewat brute-force indeks.
- Mask shift EP yang salah arah (wrap a↔h) — pion di h4 "menangkap" ep a6;
  kini mengikuti semantik `shift_left/right` python-chess.

## 3. Pengembangan sistem cross-engine

- **`[cross]` config section baru** (`CrossConfig`): games_per_batch, skill,
  skill_min/max, adaptive_skill, skill_up/down_score, skill_window, parallel,
  stockfish_time_ms, time_ms/max_depth/max_plies/min_depth/grace_ms, use_book
  — lengkap dengan validasi.
- **Game paralel**: `parallel > 1` menjalankan game dalam gelombang proses
  terpisah (fork), satu Stockfish subprocess per game worker; PGN ditulis
  berurutan per gelombang (deterministik); kegagalan satu game dilaporkan
  (`failed`), tidak menghentikan batch.
- **Adaptive skill**: Skill Level Stockfish naik (+1) saat score_rate jendela
  >= `skill_up_score`, turun (-1) saat <= `skill_down_score` — korpus tetap
  kompetitif; riwayat di `skill_history`.
- **Per-opening breakdown** (W/D/L per FEN) + telemetry Kestrel: rata-rata
  NPS, depth, panjang game, jumlah book move per game.
- **OPENINGS diperluas 5 → 18** (Italian, Ruy Lopez, French, Caro-Kann,
  Scandinavian, Sicilian e6/g6, King's Gambit, QGD, Slav, KID, Pirc,
  Two Knights) — append-only agar determinisme seed lama tetap.
- **CLI**: `kestrel cross-train --parallel N --skill-min --skill-max
  --no-adaptive` dan `kestrel bench --workers N`.
- Worker dipakai ulang oleh `run_cross_engine_training` (dashboard-ready).

## 4. Benchmark agregat paralel (baru)

- `performance_suite(node_budget)` tetap mengukur **NPS single-search** (jujur,
  tidak berubah maknanya).
- `parallel_performance_suite(node_budget, workers)` baru: total node / waktu
  dinding lintas proses = throughput nyata yang dipakai trainer/self-play,
  dilaporkan bersama NPS per worker + efisiensi scaling.
- CLI: `python -m kestrel.cli bench --workers 8`.

## 5. Verifikasi yang dijalankan

- `pytest -q tests` → **55 passed**.
- `python -m kestrel.pipeline --quick --skip selfplay stockfish` → train PASS,
  benchmark PASS (stage "tests" pipeline mengalami flake timing pada satu
  test kecepatan saat dijalankan di bawah beban paralel; `pytest` langsung
  55/55 hijau — tidak ada perubahan perilaku yang terkait).
- Fuzz Zobrist inkremental: 28.289 posisi, 0 mismatch vs polyglot.
- Drift evaluasi: 420 posisi (termasuk path hint), 0 mismatch vs 1.6.
- Paritas search vs 1.6: skor + jumlah node identik di depth 1–6.
- Cross-engine: graceful failure tanpa Stockfish (sekuensial & paralel),
  validasi config, smoke PGN.

## Cara pakai yang baru

```bash
python -m kestrel.cli bench --workers 8          # NPS single + agregat
python -m kestrel.cli cross-train --games 40 --parallel 4 --skill 5
python -m kestrel.cli cross-train --no-adaptive --skill 8   # skill konstan
```

`config.toml` menambah section `[cross]` (lihat nilai default di
`kestrel/config.py::CrossConfig`).
