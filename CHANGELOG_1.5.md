# Kestrel 1.5 Experimental — "Ganas & Cerdas" upgrade

Fokus: kekuatan permainan (search + evaluasi) dan kecerdasan adaptasi (REDA),
tanpa mengubah kontrak API publik (`Searcher(...)`, `best_move()`, `stm_eval()`,
`Player`, skema fitur `learning.FEATURES`, format `learned_eval.json`).

## Search — lebih ganas

- **Satu move generation per node.** Deteksi terminal (skakmat/patu) dan urutan
  langkah kini memakai daftar legal moves yang sama; sebelumnya setiap node
  meng-generate legal moves dua kali.
- **`gives_check` dihapus dari move ordering.** Predikat termahal python-chess
  dibayar untuk setiap langkah quiet di setiap node; kini hanya dihitung malas
  (lazy) ketika keputusan LMR sudah memenuhi semua syarat lain.
- **Qsearch hanya generate capture/promotion** (`generate_legal_captures` +
  pre-check pion di rank terakhir) alih-alih mem-sort semua langkah lalu
  membuang langkah quiet. Stand-pat dan delta pruning tetap.
- **Check extension terbatas** (maks. 24 per jalur, budget `ext_left`): posisi
  skak disearch satu ply lebih dalam, sehingga mate/taktik ditemukan jauh lebih
  cepat — mate-in-1 kini selesai sampai depth 64 dalam waktu nyaris nol.
- **Mate-distance pruning:** `alpha/beta` dipangkas jarak mate, memangkas cabang
  yang tidak mungkin mengalahkan mate yang sudah ditemukan.
- **Null-move adaptif:** R=2 untuk depth < 6, R=3 untuk depth >= 6.
- **Late-move pruning konservatif** (depth <= 3, ambang `4 + 3*depth`, tidak
  pernah memangkas capture/promotion/cek, tidak jalan sebelum ada best move).
- **Urutan root dari iterasi sebelumnya:** skor per langkah root disimpan dan
  dipakai mengurutkan iterasi berikutnya (bukan hanya hash move).
- **Aspiration window melebar bertahap** (48 → 192 → penuh) sebelum fail yang
  memicu re-search penuh.
- **Cache evaluasi white-relative per search** (kunci Zobrist, maks 262k entri,
  dilewati saat halfmove_clock >= 90): transposisi dan daun qsearch yang berulang
  tidak menghitung evaluasi ulang. Deteksi 75-langkah/fivefold tetap berjalan
  sebelum cache, sehingga nilai draw path-dependent tidak pernah ter-cache.
- **TT replacement lebih baik:** entri EXACT hanya menurunkan entri EXACT yang
  lebih dangkal; entri bound lebih dalam tidak diturunkan.

## Evaluasi — lebih cerdas

- **Bonus bishop pair** (+30 cp) dan **rook di file terbuka/semi-terbuka**
  (+16/+8 cp), dihitung simetris per warna — antisimetri mirror evaluator tetap
  eksak (drift 0).
- **`pawn_structure` ditulis ulang:** rank pion lawan di-bucket per file satu
  kali, tes passed-pawn menjadi lookup kamus alih-alih scan pion-x-pion bersarang.
  Semantik identik (diverifikasi 300 posisi acak, 0 mismatch).
- **Popcount native** (`int.bit_count()`) menggantikan `bin().count("1")` di mobility.

## REDA — adaptasi lebih cerdas

- **Closed-form ridge regression** per ukuran active-set kandidat (1..max_active)
  sebagai titik awal, lalu polish SGD terbatas; dipilih ukuran set dengan MSE
  validasi terbaik — bukan lagi satu lintasan SGD tetap pada satu set fitur.
  Gerbang validasi MSE, batas sparsity, dan bound bobot tidak berubah.

## Hasil pengukuran (2 vCPU sandbox, budget identik)

- NPS kiwipete (40k node): **2.990 → 8.584 (2,87x)**
- Kedalaman selesai rata-rata dalam 1.000 ms pada 6 posisi: **3,33 → 4,83**
  (kiwipete: depth 0 → 3)
- Tactical suite: **4/4** tetap lulus; seluruh acceptance test tetap hijau.

Lihat `results/bench_1_5.json` untuk laporan penuh bila tersedia.
