# Design: Rilis KESTREL 1.7 ke GitHub dan bot Lichess always-on di Render

Tanggal: 2026-11-27
Status: disetujui — owner memberi akses penuh dan memilih "gas" (brainstorming dilewati)

## Intent & why

Owner ingin kode KESTREL 1.7 yang ada di `/root/kestrel/kestrel-1.7` dipublikasikan
sebagai rilis GitHub, lalu bot Lichess-nya dijalankan terus-menerus lewat Render
sebagai proses CLI (bukan dashboard/web). Bot saat ini tidak aktif: tidak ada
proses kestrel yang berjalan, dan `results/train_state.json` di root 1.6 adalah
state stale dari sesi training lama.

## Goals

- Repo GitHub baru berisi sumber 1.7 yang bersih, tanpa token atau artefak lokal.
- Tag rilis `v1.7.0` dan GitHub Release dengan catatan perubahan yang dibaca pengguna.
- Render background worker menjalankan bot Lichess 24/7 dengan token dari environment.
- Ada verifikasi nyata (test, deploy status, log, status akun Lichess) dan rollback yang jelas.

## Non-goals

- Tidak mengubah arsitektur search/eval, model, atau sistem training.
- Tidak menyalakan ulang training CLI lama; state stale sengaja dibiarkan.
- Tidak membuat layanan web/dashboard di Render.
- Tidak mengganti atau merotasi token Lichess yang masih valid.

## Anti-goals

- Jangan pernah memasukkan token Lichess ke repo, commit, log, atau keluaran perintah.
- Jangan mengganggu jalur model/champion yang sedang dipakai.
- Jangan mengirim kode yang belum lolos test ke rilis publik.

## Constraints

- Token Lichess asli ada di `config.toml` (awalan `lip_`, 24 karakter) di dua direktori;
  keduanya tidak boleh masuk ke repo baru.
- `kestrel.cli lichess` saat ini hanya membaca token dari config, bukan env;
  `kestrel.lichess_bot` (entrypoint langsung) sudah membaca env. Perbaikan kecil
  pada CLI diperlukan agar start command Render memakai env dengan aman.
- Render CLI v2.39.0 sudah login, workspace "My Workspace", belum ada service/project.
- GitHub CLI sudah login sebagai `fahzan1296`, belum ada repo `kestrel`.
- Host lokal aarch64, ~2 GiB RAM tersedia; hindari build/test berat yang OOM.

## Approaches

1. **Repo baru + Render background worker (dipilih).** Sumber 1.7 disalin ke
   direktori bersih, token disanitasi, di-push ke repo baru, lalu Render membuat
   worker Docker dari repo itu. Reproducible, audit trail jelas, rollback = deploy
   tag/commit sebelumnya. Trade-off: biaya Render plan berbayar untuk always-on.
2. **Deploy image Docker lokal ke Render.** Lebih cepat tetapi tidak ada sumber
   publik yang bisa diaudit dan sulit diulang bila image hilang. Ditolak.
3. **Biarkan bot jalan di mesin lokal (systemd/cron).** Tidak memenuhi permintaan
   Render dan mesin tidak guaranteed online. Ditolak.

## Decisions

- Versi: **1.7.0** dari `/root/kestrel/kestrel-1.7` (bukan root 1.6).
- Repo: `fahzan1296/kestrel`, publik, branch default `main`.
- Versi rilis: tag `v1.7.0`, angka diambil dari `CHANGELOG_1.7.md`.
- Render: **background worker** dengan plan `starter` (berbayar) agar tidak sleep;
  `runtime: docker`, repo GitHub yang sama, auto-deploy dari `main`.
- Token: `KESTREL_LICHESS_TOKEN` sebagai env var Render (di-set dari nilai lokal
  tanpa pernah dicetak); `config.toml` di repo berisi `token = ""`.
- Start command: `python3 -m kestrel.cli lichess` (clarity dengan dokumentasi CLI),
  dengan fallback env ditambahkan lewat test-first.
- Rollback: `render deploys create <service> --commit <commit lama> --wait`, atau
  `render services delete` bila deploy baru gagal total.

## What's left to the builder

- Jalankan plan di `docs/superpowers/plans/2026-11-27-kestrel-gh-render-release.md`.
- Jika token Lichess ternyata invalid (401 pada `/api/account`), hentikan deploy dan
  minta token baru dari owner — jangan menebak atau menulis token ke file.
