# QUICKSTART KESTREL 1.6 — Panduan CLI (Bahasa Indonesia)

Dokumen ini menggantikan alur "isi konfigurasi di UI lalu klik batch" yang
dulu 1× per 1 konfigurasi. Semua kini lewat CLI; nilai default diambil dari
`config.toml`, dan setiap flag menimpanya per pemanggilan.

Instalasi:

```bash
pip install -e ".[dev]"          # + ".[monitor]" untuk dashboard
```

## 1. Training unified (self-play + REDA jadi satu)

```bash
# Loop penuh, 10 generasi (default config.toml):
python -m kestrel.cli train

# Eksperimen cepat tanpa menyentuh config file:
python -m kestrel.cli train --generations 5 --games 40 --time-ms 200 \
    --depth 10 --min-depth 6 --eval-games 30

# Smoke test (~2 menit) untuk memastikan semuanya jalan:
python -m kestrel.cli train --quick
```

Per generasi yang terjadi otomatis:

1. self-play paralel (budget dari `[selfplay]`) dengan champion + book,
2. update knowledge base (opening book + pengalaman posisi),
3. learning bobot outcome dari korpus,
4. distilasi REDA (skor pencarian mendalam → residual mikro, in-memory),
5. match verifikasi champion vs kandidat (SPRT, Elo CI, per-opening, benchmark),
6. PROMOTE hanya bila lolos gate; kalau tidak, champion tetap.

Hasil per generasi: `results/verification_gen_*.md|.json`,
`results/training_log.jsonl`, checkpoint `results/checkpoints/`,
ringkasan `results/train_report_<ts>.json`.

## 2. Eval antar versi model (manual)

```bash
# Champion vs kandidat terakhir:
python -m kestrel.cli eval --eval-games 40

# Dua file model apa pun:
python -m kestrel.cli eval --champion results/learned_eval.json \
    --challenger results/checkpoints/gen_0001_learned_eval.json
```

Output: score rate + CI95, Elo delta + CI, SPRT, W-D-L per opening,
benchmark taktik/NPS/konsistensi kedua model, keputusan + alasan.
Laporan tersimpan di `results/eval_manual_<ts>.json|.md`.

## 3. Bukti kekuatan & konsistensi

```bash
python -m kestrel.cli verify          # adu champion vs baseline + benchmark penuh
python -m kestrel.cli bench --quick   # sanity taktik/NPS/konsistensi
```

`verify` menjawab "apakah model benar-benar lebih kuat dari engine tanpa
model?" dengan match + SPRT + benchmark — laporan Markdown siap dibaca.

## 4. Knowledge base (pengetahuan hasil self-play)

```bash
python -m kestrel.cli selfplay --games 20 --learn   # main + langsung belajar
python -m kestrel.cli book build                    # bangun book dari korpus
python -m kestrel.cli book show                     # statistik + garis terbaik
```

Book hanya memakai langkah dengan dukungan ≥ `knowledge.book_min_games`, jadi
bukan memory book buta. Book otomatis dipakai self-play dan bot Lichess
(`use_book` / `book_moves`).

## 5. Model & checkpoint

```bash
python -m kestrel.cli model show        # bobot champion aktif
python -m kestrel.cli model history     # riwayat generasi + alasan promote
python -m kestrel.cli model rollback --steps 1
python -m kestrel.cli model promote --path results/checkpoints/gen_0002_learned_eval.json
```

## 6. Jadi AI Lichess

```bash
# 1) Buat token di https://lichess.org/account/oauth/token (scope bot:play)
# 2) Jalankan (pilih satu cara token):
python -m kestrel.cli lichess --token LICH_TOKEN
KESTREL_LICHESS_TOKEN=LICH_TOKEN python -m kestrel.cli lichess
# atau isi [lichess] token di config.toml

# Tanpa jaringan (uji keputusan engine):
python -m kestrel.cli lichess --dry-run
```

Catatan: akun harus (di)upgrade ke tipe BOT — bot menawarkan upgrade otomatis
pada run pertama. Policy challenge, budget waktu, resign/claim draw diatur di
`[lichess]`. PGN per game tersimpan di `results/lichess/`.

## 7. Dashboard (opsional)

```bash
python -m kestrel.cli dashboard            # atau: make monitor
```

UI tetap tersedia untuk memantau; tombol/dilog self-play kini mengikuti
`config.toml`, dan ada endpoint baru `POST /api/train` yang menjalankan
trainer unified yang sama dengan CLI.

## 8. Tuning depth tinggi & performa

- `[search] min_depth` + `grace_ms` (atau flag `--min-depth/--grace-ms`):
  langkah tidak berhenti sebelum depth itu selesai, dengan anggaran ekstra
  `time_ms + grace_ms`. Batas node tidak dilonggar.
- `[selfplay] min_depth/grace_ms` menjaga korpus self-play selalu "deep".
- `[lichess] min_depth` (default 10) menjaga kualitas langkah online.
- Optimasi evaluator 1.6 (bitboard, single-pass features) menaikkan NPS ~30%+
  pada budget sama — depth lebih tinggi dalam waktu sama.

## 9. Resep harian yang disarankan

```bash
# Malam hari: latih 6 generasi dengan game banyak, eval 40 game
python -m kestrel.cli train --generations 6 --games 120 --time-ms 250 \
    --depth 12 --min-depth 8 --eval-games 40

# Pagi hari: buktikan versi baru lebih kuat
python -m kestrel.cli verify

# Sebelum online: cek bot
python -m kestrel.cli lichess --dry-run
```
