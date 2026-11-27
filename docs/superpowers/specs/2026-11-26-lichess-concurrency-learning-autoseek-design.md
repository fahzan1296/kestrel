# Design: Lichess multi-game, learning dari PGN Lichess, dan auto-seek lawan

Tanggal: 2026-11-26
Status: disetujui — dieksekusi 2026-11-26

## Intent & why

Bot Lichess saat ini hanya bisa menerima tantangan dan memainkan satu game per
waktu. Pemilik meminta tiga hal:

1. Beberapa lawan bisa dimainkan bersamaan, tidak antre.
2. Model Kestrel tidak hanya belajar dari self-play lokal, tetapi juga dari
   game yang benar-benar dimainkan di Lichess.
3. Bot mencari lawan secara otomatis.

Ini memperluas peran bot dari “klien API yang patuh” menjadi loop latih-main:
game Lichess menghasilkan data, data memperbarui kandidat model, kandidat
diverifikasi sebelum mengganti champion, lalu champion dipakai game berikutnya.

## Goals

- `lichess.concurrency` benar-benar dipakai; `gameStart` tidak memblokir event
  stream.
- Tiap game punya `Player`/`Searcher` sendiri; tidak ada TT/killers/history
  yang dipakai bersama antar game.
- PGN Lichess yang selesai bisa masuk ke jalur `train_pgn`, knowledge book, dan
  REDA dengan label yang benar.
- Champion tetap dilindungi: update model dari Lichess tidak langsung menimpa
  champion tanpa verifikasi.
- Auto-seek mengambil kandidat dari `GET /api/bot/online`, membatasi laju
  challenge, dan tidak mengirim challenge ke diri sendiri.
- Perilaku lama (dry-run, `max_games`, PGN output, token tidak diubah) tetap
  berfungsi.

## Non-goals

- Tidak mengubah token di `config.toml` atau menampilkan token.
- Tidak mengikuti pool/tournament Lichess; Bot API melarangnya.
- Tidak menjamin lawan menerima challenge; auto-seek hanya mengirim undangan.
- Tidak mengganti arsitektur `run()` ke async/await.
- Tidak menjalankan training penuh di dalam thread game.

## Anti-goals

- Jangan mempromosikan model hanya karena jumlah game Lichess bertambah.
- Jangan mengabaikan `max_games` sampai bot menyentuh batas tanpa sengaja
  membuka game baru.
- Jangan mengirim challenge bertubi-tubi ke bot yang sama.

## Constraints

- Host: aarch64 Linux, 8 vCPU, ~5.7 GiB RAM, sekitar 2 GiB tersedia. Build
  berat mudah OOM; verifikasi tetap ringan (Ruff, py_compile, targeted test).
- Python 3.11+, python-chess, NumPy. Tidak ada dependensi async baru.
- Lichess API meminta **satu request pada satu waktu** untuk endpoint non-stream
  dan memberi 429 bila dilanggar. [Sumber](https://lichess.org/api)
- Bot API hanya untuk akun BOT; bot hanya bisa main challenge game, bukan pool
  atau tournament. [Sumber](https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml)
- `POST /api/challenge/{username}` memakai `clock.limit`, `clock.increment`,
  `rated`, `variant`, `color`, `days`, `keepAliveStream`, dan `rules`.
  [Sumber](https://github.com/lichess-org/api/blob/master/doc/specs/tags/challenges/api-challenge-username.yaml)
- `GET /api/bot/online?nb=N` mengembalikan NDJSON bot online, `nb` 1..512,
  default 100, throttle 50 bot/detik.
  [Sumber](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-online.yaml)
- Scope token saat ini belum diverifikasi. CREATE challenge mungkin perlu
  `challenge:write` atau `bot:play`; bila 401/403, auto-seek harus mati dengan
  pesan jelas, bukan crash. [Sumber](https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml)
- `train_pgn` melabeli dari header `Result` PGN (1-0 / 0-1 / 1/2-1/2) dan
  mengabaikan game tanpa label valid. [Sumber: `kestrel/learning.py`]
- `LichessAPI.request` memakai `urllib`, satu `Request` per panggilan; aman
  dipanggil dari thread berbeda asal endpoint non-stream diserialisasi.

## Approaches considered

### 1. Concurrency: ThreadPoolExecutor vs async rewrite

- **ThreadPoolExecutor (dipilih)**: `gameStart` dijadikan future; event stream
  tetap membaca. Perubahan lokal di `lichess_bot.py`, kompatibel dengan
  `urllib`. Perlu lock untuk API write dan counter.
- Async rewrite: teori lebih rapi, tapi harus mengganti stream `urllib`,
  menambah dependency, dan menyentuh banyak jalur. Tidak sepadan.
- Multi-process: isolasi penuh, tapi koordinasi `max_games` dan PGN rumit.

### 2. Learning: inline adapter vs batch candidate

- **Batch candidate dari kumpulan PGN Lichess (dipilih)**: kumpulkan PGN yang
  sudah ditulis `_save_pgn`, latih kandidat, REDA adapt, verifikasi, promote.
  Champion tidak berubah sampai verifikasi lolos.
- Inline adapter tiap game: lebih cepat bereaksi, tapi training bisa bertabrakan
  dengan game yang masih jalan dan sulit dibatalkan bila kandidat buruk.
- Manual CLI saja: aman, tapi tidak memenuhi permintaan “mekanisme otomatis”.

### 3. Auto-seek: config whitelist vs online bot discovery

- **Online bot discovery (dipilih sesuai pilihan user)**:
  `GET /api/bot/online?nb=N`, filter diri sendiri + cooldown, lalu create
  challenge dengan rate limit.
- Whitelist username: lebih prediktabel, tapi butuh daftar manual dan tidak
  “mencari lawan”.
- Open-ended challenge: pihak lain yang datang, bukan bot yang mencari.

## Decisions

### D-1: concurrency default 2, diambil dari config

- `LichessConfig.concurrency` dipakai sebagai `max_workers` ThreadPoolExecutor.
- `config.toml` diubah ke `concurrency = 2`.
- CLI `lichess` mendapat `--concurrency` yang menimpa `lichess.concurrency`.
- Future game disimpan di `self._games` (dict game_id -> Future) dengan lock.
- `games_played` dihitung saat `gameStart`, bukan saat game selesai.
- Reservation counter dipakai supaya `max_games` tidak terlampaui: sebelum
  menerima challenge atau mengirim seek, cek
  `games_played + active_games + pending_challenges < max_games`.
- Challenge yang sudah dikirim dan masih pending dibatalkan lewat
  `POST /api/challenge/{challengeId}/cancel` bila budget penuh.
  [Sumber](https://github.com/lichess-org/api/blob/master/doc/specs/tags/challenges/api-challenge-challengeId-cancel.yaml)
- Saat `max_games` tercapai, bot berhenti menerima/mencari lawan baru, lalu
  menunggu game aktif selesai sebelum return.

### D-2: isolasi Player per game

- `play_game(game_id, player=None)` menerima instance `Player` yang dibuat oleh
  pemanggil. Bila None (dry-run/test), tetap memanggil `_make_player()`.
- `self._player` tidak lagi dipakai jalur live; `choose_move(board, rng, player)`
  menerima player eksplisit.
- `rng` dibuat per game, bukan `random.Random(int(time.time()))` yang bisa
  sama untuk dua game yang mulai bersamaan.
- `self._games_lock`, `self._api_lock`, dan `self._learn_lock` dibuat di
  `__init__`.

### D-3: serialisasi API non-stream

- Semua `api.get`/`api.post` dari thread mana pun lewat helper `_api_call`
  yang memakai `self._api_lock`.
- `api.stream` tidak memakai lock karena streaming panjang; satu event stream
  di main thread dan N game stream di worker thread diizinkan.
- HTTP 429 dari challenge/create: exponential backoff ringan (2s, 4s, 8s,
  maks 60s) atau skip sesuai `seek_interval`.

### D-4: PGN Lichess sebagai korpus training

- `_save_pgn` tetap menulis `results/lichess/{game_id}.pgn`.
- Tambah config `[lichess]`:
  - `learn_from_games = true`
  - `learn_every_games = 5` (0 = nonaktif)
  - `learn_min_games = 8` (minimal PGN sebelum training pertama)
  - `learn_max_games = 200` (jendela PGN terakhir)
- Setelah sebuah game selesai, worker thread memeriksa jumlah PGN. Bila
  ambang tercapai dan tidak ada game aktif, jalankan satu siklus:
  1. kumpulkan PGN terakhir dari `pgn_dir`, dedup by game id
  2. `kb.learn_from_pgn(paths)` bila `cfg.knowledge.enabled`; tulis DB hanya
     dari worker ini, dan latch daftar game id yang sudah dipelajari supaya
     siklus berikutnya tidak mengulang
  3. tulis korpus gabungan ke `pgn_dir/.learn_corpus.pgn` (atomic replace)
  4. `train_pgn(paths, epochs/lr dari cfg.learning)` → `trained`
  5. `_champion_candidate_merge(champion, trained)` → `candidate`
  6. `reda_adapt_model(corpus_path, candidate, ...)` bila `cfg.reda.enabled`
  7. `verify_challenger(champion, candidate, cfg, quick=True)` dengan budget
     `cfg.eval.games` penuh; tidak menurunkan jumlah game verifikasi
  8. `store.save_candidate(...)`, lalu `store.promote(...)` hanya bila
     `decision == "PROMOTE"` dan `cfg.train.auto_promote`
- Verifikasi sengaja menunggu sampai tidak ada game aktif. Ini menjaga champion
  dan menghindari OOM; Trade-off: siklus bisa tertunda saat bot terus ramai.
- Siklus training berjalan di thread terpisah, maksimal satu siklus aktif
  (`_learn_lock`). Tidak memblokir game stream.
- Model baru otomatis dipakai game berikutnya karena `play_game()` memuat
  champion dari checkpoint setiap kali game dimulai.

### D-5: kualitas label PGN

- `_save_pgn` harus menulis header `Result` yang benar untuk semua status
  akhir (mate, resign, outoftime, draw, stalemate), bukan `*` bila pemenang
  diketahui.
- `state.get("winner")` dipakai bila tersedia; fallback ke status + `my_color`.
- Hanya PGN dengan `Result` valid yang dipakai `train_pgn`.
- Game tanpa `gameFull`, variant non-standard, atau < 6 ply dilewati.
- `_save_pgn` dipanggil di semua jalur akhir `play_game`, termasuk
  `resign_sent` dan `stream_ended`, dengan `gameFull`/state terakhir yang
  tersedia. Ini mencegah game resign hilang dari korpus.

### D-6: auto-seek dari bot online

Tambah config `[lichess]`:

- `auto_seek = true`
- `seek_interval_seconds = 60.0`
- `seek_pool_size = 100` (maks 512 sesuai docs)
- `seek_cooldown_games = 10` (jangan challenge bot yang sama lagi sebelum N
  game lewat)
- `seek_rated = false` (bishop testing aman; docs menyarankan casual saat
  menguji bot)
- `seek_clock_seconds = 300`
- `seek_increment_seconds = 3`
- `seek_color = "random"`

Algoritma:

1. Worker thread terpisah menunggu `seek_interval_seconds`, bukan bergantung
   pada event stream yang bisa idle.
2. Ambil `/api/bot/online?nb={seek_pool_size}` lewat `_api_call`.
3. Parse NDJSON dengan `parse_sse` yang sudah ada.
4. Filter: bukan `self.username`, belum dalam cooldown, punya `username`.
5. Pilih satu secara acak; POST
   `/api/challenge/{username}` dengan form:
   `clock.limit`, `clock.increment`, `rated=false`, `variant=standard`,
   `color=random`.
6. Catat challenge id; anggap pending sampai `gameStart` dengan id sama atau
   `seek_interval` berikutnya lewat. Docs resmi menyatakan game ID sama dengan
   challenge ID, jadi korelasi ini reliable.
   [Sumber](https://github.com/lichess-org/api/blob/master/doc/specs/tags/challenges/api-challenge-username.yaml)
7. Jika `active_games >= concurrency` atau `max_games` tercapai, skip.
8. HTTP 401/403 → matikan auto-seek untuk sesi ini dengan pesan scope token.
9. HTTP 429 → tunggu `min(60, 2 ** retry)` detik.

Challenge ke bot lain adalah undangan; bot lawan boleh menolak. Itu bukan bug.

### D-7: keamanan dan observability

- Semua username dari API dipakai sebagai path segment; validasi regex
  `^[A-Za-z0-9_-]{1,50}$` sebelum membangun URL.
- Tidak ada token yang dicetak. Pesan error hanya kode status.
- Log ringkas: `[lichess] seek -> user`, `[lichess] learn cycle #n`,
  `[lichess] promoted`, `[lichess] auto-seek disabled (scope)`.
- Tambah ringkasan di return `run()`: games_started, games_finished,
  learn_cycles, promoted, seek_sent, seek_failed.

## Files yang akan disentuh

- `kestrel/lichess_bot.py`: pool, lock, API helper, auto-seek, learning hook,
  PGN result fix.
- `kestrel/config.py`: field baru `LichessConfig`.
- `config.toml`: nilai default konkret.
- `kestrel/cli.py`: `--concurrency` untuk subcommand `lichess`.
- `tests/test_unified.py`: test offline untuk parser bot online, validasi
  username, filtering cooldown, result PGN, dan scheduling concurrency.
- `docs/superpowers/specs/...design.md`: dokumen ini.

## Testing & verification

- Targeted pytest: `tests/test_unified.py -q -k lichess`.
- `python3 -m py_compile` untuk file Python yang berubah.
- Ruff core config pada file yang berubah.
- LSP diagnostics error-only pada file yang berubah.
- Tidak menjalankan `make train`, `make test`, atau live Lichess challenge
  tanpa izin terpisah.

## What is left to the builder

- Pilih struktur helper internal (mis. `_LearningWorker` vs fungsi) selama
  kontrak di atas dipenuhi.
- Tentukan detail backoff dan format log selama tidak membocorkan token.
- Pastikan perubahan tidak memakai `asyncio` atau dependency baru.
