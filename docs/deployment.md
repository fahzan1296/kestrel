# Deployment: bot Lichess di Render

Bot Lichess dijalankan sebagai **background worker** Render dengan runtime Docker.
Layanan ini sengaja tidak mengekspos port HTTP; doesn't ada dashboard di dalamnya.

## Environment

| Nama | Wajib | Catatan |
|---|---|---|
| `KESTREL_LICHESS_TOKEN` | ya | Token Bot API Lichess (scope `bot:play`). Disimpan sebagai env var Render, **bukan** di `config.toml` repo. |

`config.toml` di repo memakai `token = ""` supaya tidak ada kredensial di VCS.
Bila menjalankan lokal, export env var di atas sebelum perintah CLI.

## Menjalankan lokal

```bash
python3 -m pip install -e .
export KESTREL_LICHESS_TOKEN=<token>
python3 -m kestrel.cli lichess
```

Sanity check tanpa koneksi nyata:

```bash
python3 -m kestrel.cli lichess --dry-run
```

## Deploy / redeploy

```bash
render services list
render deploys create <service-id> --wait --confirm
render logs -r <service-id> --limit 80
```

Auto-deploy aktif untuk branch `main`; commit baru memicu deploy otomatis.

## Rollback

Sebelum rilis, catat commit yang masih dikenal baik:

```bash
git rev-parse HEAD
```

Jika deploy baru bermasalah, deploy ulang commit lama:

```bash
render deploys create <service-id> --commit <sha> --wait --confirm
```

Bila worker gagal total dan perlu dihentikan sementara:

```bash
render services delete <service-id>
```

## Verifikasi setelah deploy

- Log worker tidak berisi `Token belum ada`, `401`, atau traceback Python.
- `https://lichess.org/api/users/status?ids=Azchess-1` mengembalikan entri akun.
- Akun `Azchess-1` menerima/melanjutkan challenge sesuai `config.toml`.
