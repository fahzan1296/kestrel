# Kestrel performance notes

Catatan ini mendokumentasikan optimasi performa 2026-11-28 untuk Kestrel 2.0.1.
Prinsipnya: ukur dulu, ubah satu hal, ukur ulang, lalu keep/revert.

## Lingkungan

- Mesin: InstaBox `kestrel-2`, 4 vCPU.
- Bot: `Azchess-1`, `concurrency=2`, `move_time_ms=1000`, `max_depth=18`,
  `min_depth=10`, `grace_ms=1500`, `hard_ext_ms=5000`.
- Training: self-play 100 game/batch, 2 worker, berjalan terus.
- Token Lichess hanya di `.env` mode `600`; tidak pernah dicetak atau commit.

## Baseline live

Observasi awal saat bot dan training aktif:

- 2 worker self-play masing-masing ~100% CPU.
- Bot ~100% CPU saat sedang berpikir.
- Semua proses `nice=0`.
- `loadavg` ~2.44–2.9 pada 4 vCPU.

Dengan perf log aktif (`KESTREL_PERF_LOG=1`), 10 sampel move saat hanya
satu bot aktif:

- `elapsed_ms` rata-rata: 7132.8 ms
- p95 `elapsed_ms`: 7502.7 ms
- NPS rata-rata: 20501
- NPS median: 20300

## Masalah operasional yang ditemukan

Ada **dua proses bot** yang hidup bersamaan setelah restart supervisor:
satu proses orphan dari supervisor lama dan satu proses baru. Gejalanya:

- `game ... tidak punya slot`
- `rate-limited; tidur 60s`
- CPU bot terbagi dua

Orphan di-hentikan eksplisit dan supervisor direstart. Sekarang hanya satu
proses `kestrel.cli lichess` yang berjalan, dan status Lichess `online=true`.

## P0 — Resource isolation

Training supervisor dan child process sekarang dijalankan dengan `nice -n 10`;
bot tetap `nice=0`. Perubahan ini diterapkan di `train-supervisor.sh` dan pada
proses yang sedang berjalan.

Pengukuran pembanding tidak menunjukkan gain yang bisa dipastikan:

- benchmark dengan training `nice=10`: NPS median 17439
- benchmark proses benchmark sendiri `nice=10`: NPS median 17864

Jadi `nice` **tidak diklaim sebagai percepatan**. Ia dipertahankan sebagai
guard murah: jika bot, training, dan benchmark bersamaan, bot tetap lebih
prioritas. Risiko kekuatan model: nol.

## Benchmark deterministik

Ditambahkan `scripts/bench_perf.py`:

- FEN tetap, default startpos.
- `--nodes` dan `--model` eksplisit.
- Output JSON: `nodes`, `elapsed_ms`, `nps`, `depth`, `score_cp`, `model`.
- Memakai `load_default()` sehingga jalur eval sama dengan bot (akumulator
  inkremental), bukan `net=None` yang memaksa refresh 768-plane per node.

Contoh:

```bash
python3 scripts/bench_perf.py --nodes 30000 --json
```

## Profiling

`cProfile` pada 30k node menunjukkan hotspot:

- `qsearch`: ~6.2 s kumulatif
- legal move generation: ~4.4 s
- evaluasi: ~4.1 s
- `_push`: ~3.4 s
- move ordering: ~2.5 s
- `nnue.raw`: ~2.0 s

Artinya bottleneck bukan I/O PGN dan bukan rebuild akumulator NNUE.

## P1 — Stand-pat qsearch tanpa terminal check duplikat

`qsearch` sudah menangani insufficient material, 75-langkah, repetition, dan
mate/stalemate di pintu masuk. Namun cabang stand-pat masih memanggil
`evaluate(..., terminal=True)`, yang mengulang predikat terminal mahal di leaf.

Ditambahkan `Searcher._stand_pat()`:

```python
return self._stm(board, in_check=False, has_moves=True, terminal=False)
```

Hasil benchmark 5 run, 30k node, FEN startpos:

| Versi | NPS 5 run | Median |
|---|---|---|
| lama | 5367, 5376, 5271, 5345, 5127 | 5345 |
| baru | 5714, 5678, 5668, 5687, 5422 | 5678 |

Gain median ~6.2%, dengan rentang yang tidak overlap. Gate kebenaran:

- test parity skor stand-pat vs evaluasi biasa pada 3 posisi non-terminal;
- full suite: **82 passed, 2 warnings**;
- tidak ada perubahan arsitektur, bobot, atau aturan search.

Keputusan: **keep**.

## Perf logging

Bot mencatat baris berikut hanya bila `KESTREL_PERF_LOG=1`:

```
[perf] game=<id> depth=<d> nodes=<n> elapsed_ms=<ms> nps=<nps> score_cp=<cp>
```

Flag ini berada di `.env` mesin, bukan di repo. Tidak ada token yang dicetak.

## Keep / revert rule

Optimasi hanya dipertahankan bila:

1. benchmark identik diulang minimal 5 kali;
2. median NPS naik atau p95 latency turun di luar noise;
3. full test suite hijau;
4. parity evaluasi terjaga bila menyentuh jalur eval;
5. tidak ada perubahan kekuatan/model tanpa head-to-head.

Semua eksperimen, termasuk yang tidak terbukti, dicatat di sini.

## Status

- Optimasi diterapkan di mesin `kestrel-2`.
- Bot direstart sekali dengan supervisor bersih; hanya satu instance aktif.
- Training tetap running.
- Tidak ada perubahan token atau config Lichess.
