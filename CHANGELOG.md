# CHANGELOG

Riwayat versi KESTREL. Entri pra-2.0 diringkas dari changelog per-versi lama
(1.2–1.8) yang kini dihapus dari repo; detail lengkapnya tersimpan di riwayat
git proyek asal.

## 2.0.4 — Optimasi training & penuntasan gate baseline

### Fixed
- Artifact `results/train_state.json` dan `results/train.log` tidak lagi
  menunjuk path mesin lama (`/home/z/.../kestrel-1.8/results`).
- `results/HASIL_BOOTSTRAP.md`: klaim "verifikasi baseline belum dijalankan"
  diganti hasil aktual (8W-2D-10L, score 0,45, Elo −35, SPRT continue).

### Changed
- `nnue_train._linear_warmstart`: normal equation primal 1536×1536
  diganti identitas Woodbury (sistem n×n) untuk korpus ≤ 1024 sampel.
  Ekuivalen numerik (Δ ~1e-13) dan **~68x lebih cepat** pada korpus kecil;
  jalur primal chunked tetap dipakai untuk korpus besar.
- Versi disamakan ke 2.0.4 di `pyproject.toml`, `kestrel/__init__.py`,
  `docs/PERF.md`, dan `docs/deployment.md`.

### Added
- `tests/test_regressions.py`: gate 75-langkah, evaluator produksi
  (antisimetri + nonzero), dan validasi warmstart Woodbury.

## 2.0.3 — Perbaikan bot Lichess

### Fixed
- Auto-seek tidak lagi menolak challenge kiriman sendiri ketika event Lichess
  tidak menyertakan field `direction`. Fallback mencocokkan nama challenger
  dengan username bot, sehingga challenge sendiri tidak di-decline dan
  gameStart berikutnya tetap memakai reservasi slot.

## 2.0.0 — Graduasi: evaluator AI TinyNNUE sepenuhnya

### Added
- **TinyNNUE** (`kestrel/nnue.py`): jaringan NNUE-lite 768-plane → 128
  akumulator ×2 perspektif (bobot L1 dibagi) → 32 (ReLU) → 1; **106.689
  parameter** (< batas 1.000.000 spesifikasi), FP32 runtime, checkpoint int8
  terkuantisasi (~110 KB).
- **Akumulator inkremental** terintegrasi `Searcher`: update ≤ 4 baris bobot
  per langkah, tersinkron dengan stack Zobrist inkremental 1.7 (castling,
  en passant, promosi, null move ditangani benar; diverifikasi identik dengan
  hitung-dari-nol pada ratusan posisi acak).
- **Perspektif STM** pada jaringan: antisimetri mirror **eksak by
  construction** (`eval(mirror(s)) == -eval(s)`, drift 0 — bukan ±epsilon).
- **Checkpoint baseline terlatih dibundel** (`kestrel/models/nnue_baseline.json`):
  distilasi evaluasi statis klasik (fungsi leaf search 1.8, STM-relative)
  pada ~45rb posisi playout realistis (mayoritas terpandu teacher). Runtime
  tanpa champion memakai baseline ini; tidak ada jalur "net acak" lagi.
- **Trainer TinyNNUE** (`kestrel/nnue_train.py`): target gabungan
  `alpha*skor search + (1-alpha)*hasil game` (komentar PGN self-play sebagai
  label "gratis"; teacher klasik sebagai fallback), mini-batch SGD numpy
  tervektorisasi penuh (forward/backward via matriks one-hot), holdout
  level-game, early stop, lr decay, gradient clip, weight decay, warm start
  champion.
- **Bootstrap script** (`scripts/bootstrap_nnue.py`): korpus playout realistis
  → label evaluasi statis teacher → training (warm-start linear) → checkpoint
  baseline + laporan korelasi & kesepakatan best-move.
- CLI `kestrel nnue distill|info`; benchmark evaluator kini menerima parameter
  net sehingga verifikasi benar-benar menguji net yang dinilai (1.x menjalankan
  benchmark pada evaluator default — bug kejujuran yang diperbaiki).
- Token Lichess kini dibaca dari environment `KESTREL_LICHESS_TOKEN` / `--token`.

### Changed
- **Evaluator runtime = TinyNNUE sepenuhnya.** Blend handcrafted (ALPHA=0.85),
  residual linear OutcomeModel (14 fitur), dan REDA dihapus dari jalur
  evaluasi. Gate terminal (skakmat/pat/material/75-langkah/5x repetisi)
  dipertahankan sebagai aturan permainan.
- Evaluator klasik 1.8 (identik bit-per-bit) dipindah ke `kestrel/teacher.py`
  sebagai teacher distilasi + pembanding benchmark, bukan jalur runtime.
- `Searcher(net=...)` tanpa argumen `learned`; seluruh pemanggil
  (arena, selfplay worker, dashboard, cross-engine, bot Lichess, verifikasi)
  memuat net champion atau baseline bawaan.
- Trainer unified: tahap learning+REDA diganti satu tahap NNUE; checkpoint
  kini `results/nnue_eval.json` + `results/checkpoints/gen_*_nnue.json`.
- Config: section `[learning]`/`[reda]` diganti `[nnue]` (kompat: `[learning]`
  lama dipetakan sebagian ke `[nnue]`).
- Verifikasi champion-vs-challenger memakai net eksplisit untuk kedua sisi.

### Removed
- `kestrel/reda.py`, `kestrel/reda_pipeline.py` (adapter residual eksperimental).
- `kestrel/learning.py` (OutcomeModel + skema fitur linear) — digantikan
  sepenuhnya oleh TinyNNUE + trainer baru.
- `kestrel/memory.py` (AdaptiveMemory) — tidak pernah dipakai jalur runtime.
- TinyMLP eksperimental (768→128→64→1, kepala nol) dari `kestrel/eval.py`.
- CHANGELOG/UPGRADE_NOTES per-versi lama (diringkas ke file ini),
  QUICKSTART_1.6.md, script doc-generator one-off
  (`scripts/make_infra_doc.py`, `make_report.py`, `make_tidy_memo.py`).
- Endpoint dashboard `/api/reda/run`, `/api/reda/rollback`, state REDA, dan
  UI REDA; checkbox "Run REDA after batch".
- Token Lichess dari `config.toml` (tereksposi di versi lama — revoke token tsb).

### Security
- Token Lichess yang tertanam di config lama dianggap terekspos publik;
  sejak 2.0 token tidak pernah disimpan ke file konfigurasi.

## 1.8.0 — ringkasan
- Search: continuation/capture history + penalti gravity, IIR, LMR adaptif
  sejarah, futility, null-move R adaptif 2/3/4, delta-SEE; ekstensi waktu
  posisi sulit (+5 detik, audited); evaluator fusi satu-pass 1.37x (nilai
  identik bit-per-bit); pembelajaran 14 fitur + migrasi model lama, warm
  start champion, class balance, lr decay.

## 1.7.0 — ringkasan
- TT flat-array + Zobrist inkremental eksak-polyglot; move ordering satu-pass;
  cache langkah/check; LMR tabel pra-hitung; bitboard eval ~1.9x lebih cepat,
  NPS +29-36%; lichess auto-seek + pembelajaran korpus lichess.

## 1.6.0 — ringkasan
- Trainer unified (selfplay→knowledge→learning→REDA→verify→promote) satu
  perintah; verifikasi otomatis + versioning champion/rollback; jaminan
  min_depth + grace time; bot Lichess; dashboard train endpoint.

## 1.2–1.5 — ringkasan
- Arena/SPRT/Wilson CI, dashboard live self-play, cross-engine vs Stockfish
  adaptive skill, bitboard evaluator, knowledge base (book + pengalaman posisi),
  perft/acceptance suite.
