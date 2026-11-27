# Kestrel 1.8 Experimental — "Kekuatan & Akurasi, Tradeoff Kecil yang Disengaja"

Fokus empat permintaan besar: (1) cara bermain dan akurasi lebih tinggi dengan
tradeoff yang sedikit, (2) optimasi agresif lanjutan, (3) pembelajaran yang
lebih efektif dan efisien, (4) posisi sangat sulit mendapat tambahan waktu
(standar 5 detik) bila budget terdahulu tidak cukup.

Kontrak publik tidak berubah: `Searcher(...)`, `best_move()`, `stm_eval()`,
`Player`, skema `config.toml` 1.6/1.7 tetap diterima, format
`learned_eval.json` dinaikkan ke **schema_version 2** dengan migrasi otomatis
(model lama 11 fitur di-load sebagai 14 fitur dengan zero-pad).

## 1. Search — kekuatan & akurasi, tradeoff kecil yang disengaja

Semua fitur baru punya saklar dan **default ON**: `use_iir`,
`use_futility`, `use_cont_history` (kompatibel mundur: set `False` untuk
perilaku 1.7). Perubahan:

- **Continuation history (1-ply):** skor quiet move menggabungkan butterfly
  history dengan riwayat pasangan (langkah sebelumnya → langkah ini) —
  ordering mengikuti pola respons yang benar-benar meng-cutoff, bukan hanya
  statistik langkah tunggal.
- **Capture history:** reorder antar capture berdasar riwayat cutoff per
  (victim, from, to), bukan hanya MVV-LVA statis.
- **Penalti gravity dua arah:** langkah quiet/capture yang dicoba di node
  yang sama tetap gagal meng-cutoff menerima bonus NEGATIF, bukan hanya bonus
  positif untuk yang berhasil — ordering cepat "belajar" dari kegagalan.
- **IIR (internal iterative reduction):** node dalam tanpa hash move disearch
  satu ply lebih dangkal; memangkas pencarian buta di subtree tanpa info TT.
- **LMR adaptif sejarah:** langkah quiet ber-sejarah buruk direduksi +1 ply,
  sejarah sangat baik -1 ply (di atas tabel LMR dasar).
- **Futility pruning parent (depth ≤ 2):** quiet move dilewati bila evaluasi
  statis + margin jelas tidak mencapai alpha.
- **Null-move R adaptif:** R=2 (depth <6), R=3 (6–9), R=4 (≥10) — sebelumnya
  hanya R=2/3.
- **Delta pruning qsearch dengan aproksimasi SEE** (victim − attacker):
  capture yang jelas rugi material setelah prun delta tak disearch.

**Tradeoff terukur (dua vCPU, skrip benchmark sama, 40k node):**

| Posisi    | NPS 1.7 → 1.8   | depth 40k 1.7 → 1.8 |
| --------- | --------------- | ------------------- |
| startpos  | 16.856 → 16.532 (−2%) | 8 → 9         |
| kiwipete  | 17.322 → 18.696 (+8%) | 5 → 5         |
| endgame   | 22.823 → 21.886 (−4%) | 7 → 7         |
| complex   | 23.941 → 19.227 (−20%) | 5 → 5        |

NPS turun di posisi kompleks karena per-node kini membayar pemeriksaan
pruning/sejarah ekstra; kompensasinya ordering lebih akurat (history gravity,
continuation) dan subtree lebih dangkal bila tak ada info. Posisi startpos
malah mencapai depth 9 (sebelumnya 8) pada budget node sama. Suite taktik 6/6
pada kedua versi — tidak ada regresi taktik.

## 2. Ekstensi waktu posisi sulit (+5 detik, default)

- **Trigger (tiga alasan, semua tercatat di `info`):**
  (a) `hard_position_timeout` — budget waktu habis, `min_depth` belum
  tercapai, dan grace time sudah habis;
  (b) `root_instability` — best move root berubah antar iterasi pada
  depth ≥ 4 dan sisa budget tinggal < 40%;
  (c) `score_swing` — skor root berayun ≥ 150 cp antar iterasi pada
  depth ≥ 4 dengan sisa budget < 40%. Ekstensi diberikan SEBELUM iterasi
  berikutnya, agar keputusan tidak diambil dari pencarian dangkal.
- **Parameter** (`[search]` di `config.toml` + kwargs `Searcher`):
  `hard_ext_ms = 5000` (tambahan waktu, default 5 detik sesuai permintaan),
  `hard_ext_max = 1` per langkah, `hard_ext_min_budget_ms = 500`
  (budget terlalu kecil — mis. 100–400 ms — sengaja TIDAK diekstensi; ekstensi
  adalah untuk langkah serius dengan budget ≥ 500 ms).
- **Jaminan:** hanya budget WAKTU yang diperpanjang — limit node tidak pernah
  dilonggaran; `hard_ext_max` membatasi jumlah ekstensi; semua ekstensi
  tercatat di `info["extensions"]` dan `info["extension_reasons"]` sehingga
  hasilnya selalu bisa diaudit.
- **Passthrough lengkap:** `dashboard_worker.run_game_worker`,
  config `[search]`, dan demo benchmark. Demo terukur (kiwipete, budget 40 ms
  + min_depth 5): tanpa ekstensi berhenti 105 ms tanpa mencapai min_depth;
  dengan ekstensi, budget diperpanjang dan depth 5 tuntas
  (`min_depth_met: true`, 1 ekstensi, alasan `hard_position_timeout`).

## 3. Optimasi agresif: evaluator fusi satu-pass penuh

- `handcrafted` memindai board **hanya sekali per (jenis bidak, warna)**:
  material, PST taper, mobility berbobot, istilah pion
  (doubled/isolated/passed), bishop pair, rook file terbuka/semi-terbuka,
  dan king safety diakumulasi dalam satu loop — sebelumnya board dipindai
  ±4 kali. Semua term tetap integer penjumlahan komutatif.
- **Identitas terverifikasi:** 599 posisi acak 1.8 vs 1.7 → **0 mismatch**
  (bit-per-bit); antisimetri mirror 420 posisi → **0 pelanggaran, eksak**.
- Kecepatan: 43,8 → 32,1 µs per evaluasi (**1,37×**) pada run benchmark
  resmi (run beban lain 28,5–30 µs).
- Fungsi publik per-term (`pst`, `mobility`, `pawn_structure`, …) tetap ada
  untuk pengujian, namun search memakai jalur fusi.

## 4. Pembelajaran lebih efektif & efisien

- **Skema fitur v2 — 14 fitur:** tambahan `doubled_pawns`, `isolated_pawns`,
  `king_shield` di atas 11 fitur 1.7 (v2 tetap 100% fitur papan nyata, tanpa
  telemetry hasil).
- **Migrasi otomatis:** `OutcomeModel.load` menerima model lama (11 fitur)
  dan menaikkan skema ke 14 via zero-pad; champion lama tetap valid.
- **Warm start:** trainer 1.8 bisa memulai dari bobot champion saat ini
  (`[learning] warm_start = true`); checkpoint epoch 0 = champion, jadi
  early-stop tidak pernah lebih buruk dari champion.
- **Class balance:** bobot kelas 1/(frekuensi) untuk hasil W/D/L yang timpang
  (korpus self-play dominan menang putih/seri).
- **lr decay:** `lr_epoch = lr * lr_decay^epoch` (default 0,85) — konvergensi
  lebih halus di akhir pelatihan.
- **Hasil terukur (korpus sama — games.pgn 1.7, 5.110 sampel, validasi
  game-holdout 948 sampel, 12 epoch, seed sama):**

| Run                        | LogLoss val | Brier | Akurasi val |
| -------------------------- | ----------- | ----- | ----------- |
| 1.7 cold (11 fitur)        | 0,5836      | 0,1062 | 67,4%      |
| 1.8 cold (14 fitur)        | 0,5814      | 0,1052 | 67,4%      |
| **1.8 warm (migrasi champion 1.7)** | **0,5743** | **0,1019** | **69,0%** |

Warm start + class balance + lr decay memperbaiki LogLoss validasi ~1,6% dan
akurasi +1,6 pp dibanding trainer 1.7 pada korpus yang persis sama.

## 5. Validasi

- **Test suite: 67/67 hijau** — 55 test 1.7 + 12 test baru
  (`tests/test_kestrel18.py`): trigger ekstensi (timeout & node-budget
  guard, max/min-budget, validasi parameter), saklar fitur ON/OFF tetap
  menemukan mate, batas history int-key, continuation mengubah ordering
  setelah cutoff, skema v2 + migrasi, warm start/class balance/lr decay,
  identitas fusi eval vs komposisi term, field config baru.
- **Benchmark final vs baseline 1.7** dengan skrip yang sama persis
  (`results/bench_report_baseline_1_7.json` vs
  `results/bench_report_final_1_8.json`): taktik 6/6 keduanya, antisimetri
  eksak 0, tabel tradeoff NPS di atas, eval 1,37× lebih cepat.
- **Match head-to-head 1.7 vs 1.8** (`results/match_1_7_vs_1_8.json`):
  24 game — 6 pembukaan × 2 warna × 2 repetisi, 400 ms/langkah, budget sama
  persis, tanpa model, adjudikasi evaluasi statis ≥ 800 cp dua langkah
  berturut (penggaris sama untuk kedua sisi). Hasil: **1.8 menang 9, remis
  10, kalah 5 → score rate 0,583 (nominal ≈ +58 Elo; Wilson 95% CI
  [0,388, 0,755] masih mencakup 0,5)** — keunggulan adalah indikasi arah,
  belum bukti statistik; rincian per game tersimpan penuh di JSON.
- **Pembandingan pembelajaran** pada korpus identik:
  `results/learning_1_7_vs_1_8.json`.

## 6. Status jujur

Tradeoff NPS adalah pilihan desain yang disengaja (pemeriksaan per-node
lebih mahal untuk ordering/akurasi lebih baik), bukan regresi — dan bisa
dimatikan lewat saklar fitur. Score rate dari match pendek adalah indikasi,
bukan estimasi Elo final; SPRT tetap skrining trinomial eksperimental. Tidak
ada klaim kekuatan absolut — semua angka di changelog ini bisa direproduksi
dari artefak di `results/`.
