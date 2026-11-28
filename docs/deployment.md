# Deployment: training + bot Lichess di InstaBox

Kestrel 2.0.4 dijalankan di mesin InstaBox `kestrel-2`
(`m_agrb3oehifkozpqnikonsnaysu`) sebagai proses background. Render dan Hugging Face
Space tidak dipakai lagi.

## Lokasi di mesin

| Path | Isi |
|---|---|
| `/data/home/kestrel` | source Kestrel 2.0.4 |
| `/data/home/kestrel/.venv` | virtualenv Python 3.13 |
| `/data/home/kestrel/.env` | `KESTREL_LICHESS_TOKEN`, mode `0600` |
| `/data/home/kestrel/train-supervisor.sh` | loop restart training |
| `/data/home/kestrel/bot-supervisor.sh` | loop restart bot Lichess |
| `/data/home/kestrel/train-run.log` | stdout training |
| `/data/home/kestrel/kestrel-bot.log` | stdout bot |
| `/data/home/kestrel/train-supervisor.log` | start/exit training |
| `/data/home/kestrel/bot-supervisor.log` | start/exit bot |
| `/data/home/kestrel/results/train_state.json` | progres training terbaru |

`config.toml` memakai `token = ""`; CLI membaca `KESTREL_LICHESS_TOKEN` dari `.env`.
File `.env` tidak masuk git dan tidak pernah dicetak. Evaluator runtime adalah
TinyNNUE dengan baseline `kestrel/models/nnue_baseline.json`; bila training
mempromosikan champion, file `results/nnue_eval.json` dipakai otomatis oleh game
baru. Self-play trainer berjalan terlepas dari apakah bot sedang dipakai di
Lichess atau tidak.

## Menjalankan ulang

Dari lokal dengan `INSTABOX_API_KEY` sudah diset:

```bash
instabox get kestrel-2
instabox jobs ls kestrel-2
```

Jika mesin suspend, bangunkan dulu:

```bash
instabox start kestrel-2
```

Jika supervisor berhenti:

```bash
instabox exec --background kestrel-2 -- bash -lc 'exec /data/home/kestrel/train-supervisor.sh'
instabox exec --background kestrel-2 -- bash -lc 'exec /data/home/kestrel/bot-supervisor.sh'
```

## Verifikasi

```bash
instabox exec kestrel-2 -- bash -lc 'tail -n 30 /data/home/kestrel/kestrel-bot.log'
instabox exec kestrel-2 -- bash -lc 'head -c 600 /data/home/kestrel/results/train_state.json'
curl -s 'https://lichess.org/api/users/status?ids=azchess-1'
```

`Azchess-1` harus `online: true`, log bot harus mengandung `memainkan ...`,
dan `train_state.json` harus `status: running`. Log tidak boleh berisi
worker error, `401`, `403`, atau token `lip_...`.

## Batasan free plan

Mesin free InstaBox:

- idle pause setelah 30 menit tanpa aktivitas,
- max run mulai 120 menit dan bisa tumbuh sampai 480 menit,
- setelah max run ada cooldown 5–20 menit atau lebih.

Proses training dan bot mencegah idle pause, tetapi tidak menghilangkan
max-run cooldown. Jadi mesin bisa sesekali suspend sampai dibangunkan manual
dengan `instabox start kestrel-2`. Ini sudah disetujui sebagai konsekuensi
memakai paket free.

## Rollback

Hentikan kedua supervisor tanpa menghapus mesin:

```bash
instabox exec kestrel-2 -- bash -lc 'pkill -f /data/home/kestrel/train-supervisor.sh || true'
instabox exec kestrel-2 -- bash -lc 'pkill -f /data/home/kestrel/bot-supervisor.sh || true'
instabox exec kestrel-2 -- bash -lc 'pkill -f "kestrel.cli lichess" || true'
```

Lalu verifikasi Lichess status tidak lagi `online`. Jangan pernah menyalin
token ke `config.toml`. Untuk menghapus semua data, langkah terakhir adalah
`instabox rm kestrel-2` (destruktif).
