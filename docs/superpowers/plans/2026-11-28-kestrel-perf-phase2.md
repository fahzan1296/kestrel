# Kestrel performance phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kunci harness benchmark multi-FEN, lalu ukur dan terapkan optimasi yang terbukti mempercepat search tanpa mengubah perilaku catur atau skor NNUE.

**Architecture:** Fase ini memisahkan pengukuran dari optimasi. Task 1 mengunci `scripts/bench_perf.py --suite` dengan 5 FEN tetap dan metadata environment. Task 2 menguji BLAS single-thread sebagai perubahan operasional. Task 3 menambah cache legal moves di qsearch. Task 4 baru menyentuh NNUE scratch buffer. Setiap task memakai 20 run interleaved A/B/A/B dan full test suite.

**Tech Stack:** Python 3.13, numpy, python-chess, pytest.

**Spec:** `docs/superpowers/specs/2026-11-28-kestrel-perf-phase2-design.md`

## Global Constraints

- Jangan turunkan `move_time_ms`, `min_depth`, `grace_ms`, `hard_ext_ms`.
- Jangan ubah bobot, arsitektur, atau aturan search kecuali task yang menyatakan demikian.
- Token tidak boleh dicetak atau masuk repo.
- Full suite `python3 -m pytest -q` harus hijau.
- Benchmark 20 run interleaved A/B/A/B setelah Task 1.
- Perubahan netral/lebih lambat harus di-revert.
- Jangan gabungkan beberapa optimasi dalam satu pengukuran.

## Review Focus

- FEN suite mencakup startpos, middlegame, taktik, endgame, dan in-check.
- Model baseline tetap; champion promotion tidak aktif saat benchmark.
- BLAS thread count tercatat dan identik saat A/B.
- qsearch cache tidak mengembalikan list yang bisa dimutasi caller.
- qsearch cache tidak mengubah mate, stalemate, repetition, atau score.
- NNUE buffer parity exakt untuk `raw`, `stm_score`, `white_score`.

---

### Task 1: Kunci harness benchmark multi-FEN

**Files:**
- Modify: `scripts/bench_perf.py`
- Test: `tests/test_bench_perf.py`

**Interfaces:**
- Consumes: `run_bench(fen, nodes, model)` dari benchmark lama.
- Produces: `BENCH_FENS` (5 FEN), `--suite`, output JSON list dengan field lama + `blas_threads`.

- [ ] **Step 1: Write failing test**

Tambah `test_bench_perf_suite_emits_five_results` di `tests/test_bench_perf.py`:
- panggil `run_bench_suite(nodes=2000)`;
- assert `len(result) == 5`;
- assert setiap entry punya `fen`, `nodes`, `elapsed_ms`, `nps`, `score_cp`, `model`, `blas_threads`;
- assert FEN pertama sama dengan `START_FEN`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_bench_perf.py -k suite`
Expected: FAIL because `run_bench_suite` belum ada.

- [ ] **Step 3: Implement harness**

Tambah `BENCH_FENS` dengan 5 FEN tetap. Tambah `run_bench_suite(nodes, model=None) -> list[dict]` yang memanggil `run_bench` untuk tiap FEN dan menambah `blas_threads` dari `os.environ` atau `threadpoolctl` bila tersedia. Tambah flag `--suite`; tanpa flag, perilaku lama tetap.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest -q tests/test_bench_perf.py -k suite`
Expected: PASS.

- [ ] **Step 5: Run harness sanity**

Run: `python3 scripts/bench_perf.py --suite --nodes 2000 --json`
Expected: JSON list panjang 5, tidak ada error.

- [ ] **Step 6: Full suite and commit**

Run `python3 -m pytest -q`. Expected PASS.
Commit `test: kunci benchmark suite lima FEN`.

---

### Task 2: Uji BLAS single-thread

**Files:**
- Modify: `docs/PERF.md`

**Interfaces:**
- Consumes: harness dari Task 1.
- Produces: keputusan apakah BLAS single-thread dipakai di benchmark/runtime.

- [ ] **Step 1: Capture baseline 20 run interleaved**

Jalankan 20 run `--suite --nodes 10000 --json` dengan environment default, dan 20 run dengan `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, bergantian A/B/A/B. Simpan ke workspace.

- [ ] **Step 2: Analyze**

Hitung median dan p95 per FEN dan agregat. Jika single-thread tidak menurunkan p95 atau median dengan jelas di luar spread, jangan pakai.

- [ ] **Step 3: Document**

Tulis tabel ke `docs/PERF.md`. Jika dipakai, nyatakan perubahan runtime yang diperlukan. Jika tidak, nyatakan eksperimen ditutup.

- [ ] **Step 4: Full suite and commit**

Run `python3 -m pytest -q`. Expected PASS.
Commit `docs: uji pengaturan BLAS single-thread`.

---

### Task 3: Cache legal moves di qsearch

**Files:**
- Modify: `kestrel/search.py`
- Test: `tests/test_search_stand_pat.py` atau file baru `tests/test_search_qsearch_cache.py`

**Interfaces:**
- Consumes: `Searcher._moves_cache`, `_zkey`, `qsearch`.
- Produces: qsearch hasil identik dengan legal move list yang di-cache.

- [ ] **Step 1: Write failing test**

Tambah test `test_qsearch_cache_preserves_score_and_returns_immutable_moves`:
- buat Searcher dengan net default;
- jalankan qsearch pada 5 FEN dari `BENCH_FENS`;
- bandingkan skor dengan baseline yang dihitung sebelum perubahan;
- pastikan `_moves_cache` terisi untuk posisi qsearch in-check;
- pastikan caller tidak bisa mengubah list cache lewat hasil helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_search_qsearch_cache.py -k qsearch_cache`
Expected: FAIL because helper cache belum ada.

- [ ] **Step 3: Implement minimal cache**

Tambah helper `_legal_moves_cached(board)` yang memakai `self._moves_cache` seperti main search, lalu pakai di qsearch untuk `list(board.legal_moves)` dan deteksi stalemate. Jangan menyalin list ke caller.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest -q tests/test_search_qsearch_cache.py`
Expected: PASS.

- [ ] **Step 5: Benchmark 20 run interleaved**

BandingkanTask 3 vs Task 2 baseline dengan harness yang sama. Expected: median NPS naik ≥5%, p95 tidak naik, skor benchmark identik. Jika tidak, revert commit Task 3.

- [ ] **Step 6: Full suite and commit**

Run `python3 -m pytest -q`. Expected PASS.
Commit `perf: cache legal moves di qsearch`.

---

### Task 4: NNUE forward scratch buffer (conditional)

**Files:**
- Modify: `kestrel/nnue.py`
- Test: `tests/test_nnue.py`

**Interfaces:**
- Consumes: `TinyNNUE.raw(board, accs=None)`.
- Produces: internal per-thread buffer, API publik tidak berubah.

- [ ] **Step 1: Write failing parity test**

Tambah `test_raw_scratch_buffer_reused_without_score_change`:
- ambil `buf1 = net._forward_buffer()`, `buf2 = net._forward_buffer()` di thread sama, assert identik;
- bandingkan `raw`, `stm_score`, `white_score` dengan referensi `np.concatenate` lama pada 5 FEN;
- assert equality ketat.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_nnue.py -k scratch_buffer`
Expected: FAIL.

- [ ] **Step 3: Implement scratch buffer**

Gunakan `threading.local`, buffer `float32` ukuran `256 + AUX_DIM`, timpa semua slice tiap call.

- [ ] **Step 4: Run test and full suite**

Run targeted then `python3 -m pytest -q`. Expected PASS.

- [ ] **Step 5: Benchmark 20 run interleaved**

Expected: median NPS naik ≥5% dan skor identik. Jika tidak, revert.

- [ ] **Step 6: Commit**

`perf: reuse NNUE forward buffer per thread`
