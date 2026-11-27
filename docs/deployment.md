# Deployment: bot Lichess di InstaBox

Bot Lichess Kestrel 1.8.0 dijalankan di mesin InstaBox `box-001d46`
(`m_agrbxnvui5c57gc5fz655cqniu`) sebagai proses background. Render dan Hugging Face
Space tidak dipakai lagi.

## Lokasi di mesin

| Path | Isi |
|---|---|
| `/data/home/kestrel` | source Kestrel 1.8.0 |
| `/data/home/kestrel/.venv` | virtualenv Python 3.13 |
| `/data/home/kestrel/.env` | `KESTREL_LICHESS_TOKEN` dan `IB_KEY`, mode `0600` |
| `/data/home/kestrel/supervisor.sh` | loop restart bot |
| `/data/home/kestrel/lease.sh` | loop renew activity lease |
| `/data/home/kestrel/kestrel-bot.log` | stdout bot |
| `/data/home/kestrel/bot-supervisor.log` | start/exit bot |
| `/data/home/kestrel/lease.log` | hasil renew lease |

`config.toml` memakai `token = ""`; CLI membaca `KESTREL_LICHESS_TOKEN` dari `.env`.
File `.env` tidak masuk git dan tidak pernah dicetak.

## Menjalankan ulang

Dari lokal dengan `INSTABOX_API_KEY` sudah diset:

```bash
instabox get box-001d46
instabox ps box-001d46
instabox jobs ls box-001d46
```

Jika supervisor berhenti:

```bash
instabox exec --background box-001d46 -- bash -lc '/data/home/kestrel/supervisor.sh'
```

Jika lease loop berhenti:

```bash
instabox exec --background box-001d46 -- bash -lc '/data/home/kestrel/lease.sh'
```

## Verifikasi

```bash
instabox exec box-001d46 -- bash -lc 'tail -n 20 /data/home/kestrel/kestrel-bot.log'
curl -s 'https://lichess.org/api/users/status?ids=Azchess-1'
```

`Azchess-1` harus `online: true`. Log tidak boleh berisi `401` atau token `lip_...`.

## Batasan free plan

Mesin free InstaBox:

- idle pause setelah 30 menit tanpa aktivitas,
- max run mulai 120 menit dan bisa tumbuh sampai 480 menit,
- setelah max run ada cooldown 5–20 menit atau lebih.

`lease.sh` mencegah idle pause, tetapi tidak menghilangkan max-run cooldown.
Jadi bot bisa sesekali offline sampai mesin bisa dibangunkan lagi. Ini sudah
disetujui sebagai konsekuensi memakai paket free.

## Rollback

Matikan keduanya tanpa menghapus mesin:

```bash
instabox exec box-001d46 -- bash -lc 'pkill -f /data/home/kestrel/supervisor.sh || true'
instabox exec box-001d46 -- bash -lc 'pkill -f /data/home/kestrel/lease.sh || true'
instabox exec box-001d46 -- bash -lc 'pkill -f "kestrel.cli lichess" || true'
```

Lalu verifikasi Lichess status tidak lagi `online`. Untuk menghapus semua data,
langkah terakhir adalah `instabox rm box-001d46` (destruktif).
