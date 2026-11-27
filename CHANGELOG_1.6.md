# Kestrel 1.6 Experimental — "Unified Trainer, Auto-Verification, Lichess"

Upgrade ini menjawab enam permintaan besar: (1) REDA & self-play dijadikan SATU
pipeline training, (2) eval otomatis & mendetail antar versi model, (3) fitur
jadi AI Lichess, (4) pipeline pembuktian konsistensi & kekuatan, (5) model yang
benar-benar belajar & menghasilkan pengetahuan dari self-play, (6) CLI
pengganti konfigurasi UI + optimasi depth tinggi & performa raw.

## 1. Trainer unified (REDA + self-play jadi satu)

**Modul baru: `kestrel/trainer.py`.** Satu perintah menjalankan loop per
generasi dengan SEMUA budget mengikuti satu config (`config.toml`, section
`[selfplay]`/`[learning]`/`[reda]`/`[eval]`/`[train]`):

```
selfplay → knowledge → learning → REDA → kandidat → verify → promote/reject
```

- Self-play proses-paralel memakai model champion saat itu (+ langkah book),
  korpus PGN tumbuh inkremental (`results/games.pgn`).
- REDA tidak lagi menyentuh file model secara langsung: distilasi skor
  pencarian mendalam kini in-memory (`reda_adapt_model` di `reda_pipeline.py`);
  commit hanya terjadi setelah match verifikasi lolos. Alur lama
  `run_reda` tetap ada untuk kompatibilitas dashboard.
- Kandidat = bobot outcome hasil learning + residual REDA champion
  (`_champion_candidate_merge`) — pengetahuan dua jalur tidak pernah hilang.
- Checkpoint per generasi di `results/checkpoints/` dengan label unik
  `gen_<lineage>_tryNN` — kandidat yang DITOLAK tidak saling menimpa.
- Laporan per generasi: `results/verification_gen_<label>.json|.md`,
  `results/training_log.jsonl` (append), dan `results/train_report_<ts>.json`.
- Deadlock `multiprocessing.Queue` yang lazim diilustrasikan oleh worker yang
  macet di finalizer kini dicegah dengan protokol sentinel (`worker_done`
  selalu event terakhir; parent berhenti membaca hanya setelah semua sentinel).

## 2. Eval otomatis & mendetail lama vs baru

**Modul baru: `kestrel/verification.py`.** Setiap kandidat diadu vs champion:

- Match warna-seimbang pada set opening tetap, seed deterministik.
- W/D/L, score rate, **Wilson 95% CI**, **Elo delta + CI**, **SPRT trinomial**
  (elo0/elo1/alpha/beta dari config), rincian W-D-L **per opening**.
- Benchmark untuk KEDUA model: akurasi taktik, NPS kiwipete, drift konsistensi
  (antisimetri mirror) → regresi terdeteksi otomatis.
- Keputusan `PROMOTE`/`REJECT` berbasis gate config: SPRT H1 atau score rate ≥
  `min_score_rate`, tanpa regresi taktik, konsistensi harus lolos, NPS challenger
  ≥ `min_nps_ratio` champion. Alasan keputusan selalu tercantum.
- Catatan kejujuran statistik tertanam di laporan: score rate match kecil
  adalah indikasi, bukan estimasi Elo final.

Contoh nyata (run smoke 1.6): percobaan 1 REJECT `score_rate 0.25 < 0.45`,
percobaan 2 PROMOTE `score_rate 0.5` — dua jalur keputusan terbukti berjalan.

## 3. Bot Lichess

**Modul baru: `kestrel/lichess_bot.py`** (REST + SSE via `urllib`, tanpa
dependensi tambahan; `requests` tetap opsional sebagai extra `bot`).

- Upgrade akun ke tipe BOT otomatis pada run pertama (`/api/bot/account/upgrade`).
- Stream event + stream game, accept/decline challenge sesuai policy config
  (variant standard saja, bot/human, rentang rating, bukan self-challenge).
- Main memakai engine Kestrel: champion model + knowledge book + jaminan
  min-depth, budget waktu dari config `[lichess]`.
- Claim draw (repetisi/50 langkah) hanya bila tidak unggul; resign setelah
  `resign_moves` langkah berturut ≤ `resign_cp`.
- PGN tiap game disimpan ke `results/lichess/<gameid>.pgn`.
- Token: `--token`, `config.toml [lichess] token`, atau env
  `KESTREL_LICHESS_TOKEN`.
- `--dry-run`: simulasi lokal tanpa jaringan (teruji: menang vs lawan acak,
  mengambil ratu bebas, depth hingga 18 dengan min_depth 10).

## 4. Pipeline pembuktian konsistensi & kekuatan

- `kestrel verify` (CLI): adu champion vs **baseline handcrafted** (engine
  tanpa model) + benchmark taktik/NPS/konsistensi penuh → laporan
  `results/verify_<ts>.json|.md`. Ini bukti "model belajar > tanpa model".
- `kestrel eval`: head-to-head manual dua file model apa pun dengan laporan
  lengkap yang sama.
- `kestrel bench`: akurasi taktik, special moves, NPS, konsistensi — satu angka
  `pass` untuk CI.
- `pipeline.py` kini menjalankan `tests → train(1 generasi unified) →
  benchmark → stockfish` sehingga CI selalu membuktikan kekuatan, bukan hanya
  kebenaran.

## 5. Pengetahuan nyata dari self-play

**Modul baru: `kestrel/knowledge.py`** (SQLite `results/knowledge.db`).

- **Opening book statistik**: frekuensi + skor (White-relative) tiap langkah ≤
  `book_max_plies` dari seluruh korpus; hanya langkah dengan dukungan ≥
  `book_min_games` yang dipakai; dipilih deterministik (terbaik) atau
  proporsional skor² (dengan rng) — keragaman opening tanpa langkah lemah.
- **Experience positions**: agregat evaluasi per posisi (kunci Zobrist) dari
  telemetry PGN — korpus target distilasi REDA generasi berikutnya.
- **Lessons**: log peristiwa (korpus dipelajari, dsb.) untuk audit.
- Book dipakai di self-play (`use_book`), CLI `selfplay`, `selfplay_demo.py`,
  dan bot Lichess; langkah book ditandai di komentar PGN sehingga tidak
  tercampur telemetry pencarian.
- `kestrel book build|show|clear` untuk mengelolanya. Run smoke menghasilkan
  book 50 langkah & 333 posisi experience dari 8 game — bukti pengetahuan
  terakumulasi dan benar-benar dipakai bermain.

## 6. CLI pengganti konfigurasi UI

**Modul baru: `kestrel/cli.py`** + entry point `kestrel` (console script).

```
kestrel config show|validate|path|init
kestrel train  [--generations N] [--games N] [--time-ms N] [--depth N]
               [--min-depth N] [--grace-ms N] [--eval-games N] [--quick]
               [--no-verify] [--config PATH]
kestrel selfplay [--games N] [--learn] [+override umum]
kestrel reda                       # adaptasi satu-tembak pada korpus ada
kestrel eval   [--champion F --challenger F --eval-games N]
kestrel verify [--challenger F] [--quick]
kestrel bench  [--quick]
kestrel book   build|show|clear [--pgn ...]
kestrel model  show|history|rollback [--steps N]|promote --path F
kestrel lichess [--token T] [--max-games N] [--dry-run]
kestrel dashboard [--host H --port P]
```

Prioritas nilai: default kode < `config.toml` < flag CLI — eksperimen tidak
lagi 1× per 1 konfigurasi UI. Semua perintah mengeluarkan JSON dan exit code
yang jelas (0 sukses).

## 7. Jaminan depth tinggi + performa raw

**Search (`search.py`)**

- `min_depth` + `grace_ms`: selama completed depth < `min_depth`, iterasi yang
  terpotong waktu diulang dan deadline diperpanjang hingga `time_ms + grace_ms`.
  Batas node TIDAK dilonggar. `info["min_depth_met"]` dilaporkan per langkah.
- Teruji: startpos 50 ms biasanya selesai depth ~3; dengan `min_depth=7,
  grace_ms=3000` depth 7 selalu tuntas (1.08 s). Endgame ringan mencapai depth 12.

**Evaluator (`eval.py`, `core.py`, `learning.py`)** — nilai keluaran identik
dengan 1.5 (drift 0 pada 406 posisi uji; antisimetri mirror eksak 0):

- Bitboard langsung (`pieces_mask`, popcount) menggantikan konstruksi
  `SquareSet` per bidak di `pst/mobility/pawn_structure/king_safety/
  piece_bonuses/material/phase/features`.
- `handcrafted`: predikat mahal hanya dibayar saat perlu (cek skak dulu, baru
  movegen) + gate `halfmove_clock >= 8` sebelum travers fivefold.
- `OutcomeModel.score`: fitur dihitung SEKALI untuk dua kepala (sebelumnya dua
  kali), `features()` 3.6× lebih cepat.
- Hasil ukur (2 CPU): handcrafted 94.8 → 50.4 µs (1.9×), features 61.3 →
  16.9 µs (3.6×), NPS pencarian +29–36% (startpos 7.9k → 10.5k, kiwipete
  8.3k → 11.3k, endgame 11.9k → 16.2k).

## Lain-lain

- `config.toml` ditulis ulang untuk 1.6 (section `[search] [selfplay]
  [learning] [reda] [eval] [train] [knowledge] [arena] [lichess] [storage]`);
  section lama 1.4/1.5 masih diterima (mapping `[engine] → [search]`).
- Dashboard: default dialog self-play kini mengikuti `config.toml [selfplay]`;
  endpoint baru `/api/train`, `/api/train/stop`, `/api/train/state` menjalankan
  trainer unified dari UI; state champion ikut dilaporkan.
- `dashboard_worker.run_game_worker` menerima `learned_path`, `book_path`,
  `min_depth`, `grace_ms`, `temperature` (kompatibel ke belakang).
- `pyproject.toml`: versi 1.6.0, console script `kestrel`, extra `bot`.
- Makefile baru dengan target `train train-smoke selfplay eval verify bench
  book lichess`.
- Test: `tests/test_unified.py` (9 test — config, knowledge, versioning,
  verification, trainer end-to-end dua generasi, jaminan min-depth, CLI,
  bot Lichess tanpa jaringan). Suite penuh 11 test hijau.
- Artefak bukti run nyata tersimpan di `results/smoke/`: dua laporan
  verifikasi (REJECT + PROMOTE), `train_report_*.json`, `training_log.jsonl`,
  book/experience DB, dan log konsol.

## Hotfix lapangan (laporan Termux: "bot tidak menerima tantangan")

Dua bug pada rilis pertama 1.6 membuat bot terlihat "diam" saat ditantang:

1. **`accept_human` default `false`** — bot otomatis MENGANJUKAN tantangan
   dari akun manusia tanpa log apa pun. Kini default `true` (di
   `config.py` dan `config.toml`), dan setiap challenge masuk selalu
   di-print: diterima / ditolak + alasannya (tidak ada lagi diam).
2. **Parser SSE hanya mengenali `data: ` ber-spasi.** Lichess dapat
   mengirim `data:` tanpa spasi sehingga challenge tidak pernah terlihat.
   Parser baru (`parse_sse`, ada unit test-nya) mengenali kedua format,
   me-yield per baris data, dan mengabaikan ping/komentar.
3. **Langkah pertama saat `gameFull`** — jika giliran kita sudah tiba saat
   snapshot game pertama tiba (kita putih / challenge berlanjut ke game),
   bot kini langsung bergerak (`_take_turn`), tidak menunggu `gameState`.
4. **Retry selfplay saat spawn flaky** — spawn worker multiprocessing bisa
   gagal sesaat di perangkat kecil (Termux/VPS 2 core); tahap selfplay
   trainer kini mencoba ulang 1× sebelum menyatakan generasi gagal.
5. Tambahan info policy saat start: bot mencetak `accept_bot/accept_human/
   max_rating_gap` dan pengingat bahwa lichess tidak mengizinkan game
   RATED vs BOT (gunakan mode Santai/casual saat menantang bot).

Suite penuh: 12 test hijau (tambah 1 test parser SSE dua format).
