# KESTREL 1.7 GitHub Release + Render Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean KESTREL 1.7.0 release on GitHub and keep the Lichess bot online as a Render background worker, using an environment variable for its token.

**Architecture:** Copy the 1.7 tree into a fresh release directory, sanitize secrets, commit atomically, push to a new public repo, tag `v1.7.0`, then create a Render Docker background worker pointed at `main` with `KESTREL_LICHESS_TOKEN` injected as a secret env var. A small CLI fix makes `python3 -m kestrel.cli lichess` honor that env var.

**Tech Stack:** Python 3.12, pytest, git/gh CLI, Render CLI v2.39.0, Docker, Lichess Bot API.

**Spec:** `docs/superpowers/specs/2026-11-27-kestrel-gh-render-release-design.md`

## Global Constraints

- Sumber rilis adalah `/root/kestrel/kestrel-1.7` versi 1.7.0, bukan root 1.6.
- Token Lichess tidak boleh muncul di file, commit, log, atau output perintah. Nilai lokal dibaca ke shell var tanpa dicetak.
- Repo: `fahzan1296/kestrel`, publik, default branch `main`; rilis: tag `v1.7.0`.
- Render: background worker, plan `starter` (always-on), auto-deploy dari `main`.
- Test ringan saja; host punya ~2 GiB RAM bebas, hindari training penuh.
- Jangan menyentuh model/champion/results training lama.

## Review Focus

- Token masih bisa masuk repo lewat `config.toml`, `results/`, `.env`, atau history → secret scan sebelum setiap commit.
- `python3 -m kestrel.cli lichess` gagal walau env token benar → test env fallback di Task 2.
- Worker Render sleep saat idle → pakai plan `starter`, verifikasi lewat Lichess status endpoint.
- Deploy sukses tapi bot crash karena token invalid → cek `/api/account` sebelum deploy dan log sesudahnya.
- Rollback tidak jelas → simpan commit SHA sebelum rilis dan catat perintah redeploy di `docs/deployment.md`.

---

### Task 1: Siapkan pohon rilis bersih dan commit dasar

**Files:**
- Create: `/root/kestrel-release/` (dari salinan 1.7)
- Modify: `/root/kestrel-release/.gitignore`
- Modify: `/root/kestrel-release/config.toml`
- Create: `/root/kestrel-release/.env.example`
- Create: `/root/kestrel-release/docs/deployment.md`

- [ ] **Step 1: Salin sumber 1.7 ke direktori rilis baru**

```bash
mkdir -p /root/kestrel-release
cp -a /root/kestrel/kestrel-1.7/. /root/kestrel-release/
cd /root/kestrel-release
rm -rf results .ruff_cache .pytest_cache build dist
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
```

Expected: `ls` hanya berisi kode, docs, tests, scripts, config, dan file proyek; tidak ada `results/`.

- [ ] **Step 2: Sanitasi token di config**

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('config.toml')
s = p.read_text()
import re
s2 = re.sub(r'(?m)^(\s*token\s*=\s*)"[^"]*"', r'\1""', s)
assert s2 != s and 'lip_' not in s2
p.write_text(s2)
PY
```

Expected: `grep -n 'token' config.toml` menampilkan `token = ""`, tanpa nilai `lip_`.

- [ ] **Step 3: Perketat `.gitignore` dan tambah `.env.example`**

Tambahkan ke `.gitignore`: `.env`, `.env.*`, `!.env.example`, `results/`, `*.db`, `*.db-wal`, `*.db-shm`, `.ruff_cache/`.
Buat `.env.example` berisi hanya `KESTREL_LICHESS_TOKEN=` sebagai placeholder.

- [ ] **Step 4: Tulis `docs/deployment.md`**

Isi singkat: env var yang dibutuhkan, perintah start lokal (`python3 -m kestrel.cli lichess`),
cara deploy (`render deploys create <service> --wait`), dan rollback
(`git rev-parse HEAD` sebelum rilis; `render deploys create <service> --commit <sha> --wait`).

- [ ] **Step 5: Init git, scan secret, commit dasar**

```bash
cd /root/kestrel-release
git init -b main
git add -A
git diff --cached -i -G'lip_[A-Za-z0-9]{10,}|token[[:space:]]*=[[:space:]]*"[^"]{8,}' --name-only
git commit -m "chore: import KESTREL 1.7.0 source for public release"
```

Expected: perintah `git diff ... --name-only` tidak menampilkan file apa pun; commit berhasil.

---

### Task 2: Token Lichess dari env var di CLI (TDD)

**Files:**
- Modify: `/root/kestrel-release/tests/test_lichess_extensions.py`
- Modify: `/root/kestrel-release/kestrel/cli.py` (`cmd_lichess`)

**Interfaces:**
- Consumes: parser `build_parser()` dan `cmd_lichess(args)` yang sudah ada.
- Produces: `cmd_lichess` memakai `os.environ["KESTREL_LICHESS_TOKEN"]` bila `[lichess].token` kosong; `--token` masih menang.

- [ ] **Step 1: Tulis test yang gagal**

Tambahkan `test_cli_lichess_uses_env_token`: monkeypatch `kestrel.lichess_bot.LichessBot`
dengan kelas palsu yang merekam `token`, set env `KESTREL_LICHESS_TOKEN` ke nilai
dummy `from-env`, parse `["lichess"]`, panggil `cmd_lichess`, lalu
assert token yang tercatat sama dengan dummy.

- [ ] **Step 2: Verifikasi RED**

Run: `python3 -m pytest -q tests/test_lichess_extensions.py::test_cli_lichess_uses_env_token`
Expected: FAIL karena `cmd_lichess` mengembalikan 2 (token belum ada) dan token tidak tercatat.

- [ ] **Step 3: Implementasi minimal**

Di `cmd_lichess`, ganti pembacaan token menjadi:
`token = cfg.lichess.token or os.environ.get("KESTREL_LICHESS_TOKEN", "")`.
Tambahkan `import os` di bagian import bila belum ada. Jangan mengubah perilaku `--token`.

- [ ] **Step 4: Verifikasi GREEN dan suite lengkap**

```bash
python3 -m pytest -q tests/test_lichess_extensions.py::test_cli_lichess_uses_env_token
python3 -m pytest -q tests
```

Expected: keduanya PASS, tanpa error/warning baru. Laporkan test gagal apa pun yang muncul.

- [ ] **Step 5: Scan secret dan commit**

```bash
git add tests/test_lichess_extensions.py kestrel/cli.py
git diff --cached -i -G'lip_[A-Za-z0-9]{10,}|token[[:space:]]*=[[:space:]]*"[^"]{8,}' --name-only
git commit -m "fix: baca token Lichess dari env saat menjalankan CLI"
```

Expected: scan tidak menampilkan file. Pakai nilai dummy `from-env` di test supaya
regex scan tidak mengira ada token asli.

---

### Task 3: Push repo dan terbitkan rilis v1.7.0

**Files:**
- Remote: `https://github.com/fahzan1296/kestrel.git`
- Create: `/tmp/kestrel-v1.7.0-notes.md` (catatan rilis, tidak di-commit)

- [ ] **Step 1: Cek status bersih dan catat SHA**

```bash
cd /root/kestrel-release
git status --short
git rev-parse HEAD | tee /tmp/kestrel_pre_release_sha
```

Expected: `git status --short` kosong; SHA tersimpan untuk rollback.

- [ ] **Step 2: Buat repo GitHub dan push**

```bash
gh repo create fahzan1296/kestrel --public \
  --description "CPU-first chess engine, self-play lab, and Lichess bot (KESTREL 1.7)" \
  --disable-wiki
git remote add origin https://github.com/fahzan1296/kestrel.git
git push -u origin main
```

Expected: `gh repo view fahzan1296/kestrel` menampilkan repo; `main` terpush.

- [ ] **Step 3: Tag dan rilis**

Tulis `/tmp/kestrel-v1.7.0-notes.md` dari bagian ringkasan `CHANGELOG_1.7.md`
(diformat untuk pembaca: performa +64% NPS, test 55 passed, drift evaluasi 0,
cross-engine, auto-seek Lichess, dan instruksi env `KESTREL_LICHESS_TOKEN`).
Lalu:

```bash
git tag -a v1.7.0 -m "KESTREL 1.7.0"
git push origin v1.7.0
gh release create v1.7.0 --title "KESTREL 1.7.0" --notes-file /tmp/kestrel-v1.7.0-notes.md
```

Expected: `gh release view v1.7.0` menampilkan tag dan catatan.

- [ ] **Step 4: Verifikasi repo publik bersih**

```bash
git ls-files | xargs grep -n 'lip_[A-Za-z0-9]\{10,\}' || true
```

Expected: tidak ada keluaran.

---

### Task 4: Deploy bot Lichess ke Render background worker

**Files:**
- Remote: Render service `kestrel-lichess-bot` di workspace "My Workspace"

- [ ] **Step 1: Cek token Lichess tanpa menampilkannya**

```bash
TOKEN=$(python3 - <<'PY'
import re
print(re.search(r'(?m)^\s*token\s*=\s*"([^"]+)"', open('/root/kestrel/kestrel-1.7/config.toml').read()).group(1))
PY
)
curl -s -o /tmp/kestrel_lichess_account.json -w '%{http_code}\n' \
  -H "Authorization: Bearer $TOKEN" https://lichess.org/api/account
python3 - <<'PY'
import json
d = json.load(open('/tmp/kestrel_lichess_account.json'))
print('username:', d.get('username'))
PY
```

Expected: HTTP 200 dan username `Azchess-1`. HTTP 401/403 → berhenti, minta token baru dari owner.

- [ ] **Step 2: Buat service background worker**

```bash
render services create \
  --name kestrel-lichess-bot \
  --type background_worker \
  --runtime docker \
  --plan starter \
  --repo https://github.com/fahzan1296/kestrel \
  --branch main \
  --auto-deploy \
  --env-var "KESTREL_LICHESS_TOKEN=$TOKEN" \
  --output json | tee /tmp/kestrel_render_service.json
```

Expected: JSON berisi `id` service; simpan ke `/tmp/kestrel_render_service_id`.
Bila `--type background_worker` ditolak, jalankan `render services create --help`,
pilih nama tipe worker yang tercantum, ulangi perintah dengan tipe itu.

- [ ] **Step 3: Tunggu deploy dan baca log**

```bash
SERVICE_ID=$(python3 -c 'import json;print(json.load(open("/tmp/kestrel_render_service.json"))["id"])')
echo "$SERVICE_ID" > /tmp/kestrel_render_service_id
render deploys create "$SERVICE_ID" --wait --confirm
render logs -r "$SERVICE_ID" --limit 80 2>&1 | tail -40
```

Expected: deploy sukses; log tidak berisi "Token belum ada", exception Python, atau 401 Lichess.

---

### Task 5: Verifikasi bot aktif dan serahkan status

**Files:** tidak ada perubahan kode

- [ ] **Step 1: Cek keberadaan service dan status akun Lichess**

```bash
render services --output json
curl -s 'https://lichess.org/api/users/status?ids=Azchess-1'
```

Expected: service `kestrel-lichess-bot` ada dengan plan `starter`; endpoint status
mengembalikan entri untuk `Azchess-1`. Catat apakah field `online` bernilai benar;
bila tidak, bot masih bisa offline karena tidak ada lawan — sertakan bukti log koneksi.

- [ ] **Step 2: Uji dry-run lokal sebagai sanity check**

```bash
cd /root/kestrel-release
python3 -m kestrel.cli lichess --dry-run 2>&1 | head -20
```

Expected: tidak ada error import/token; bot mulai tanpa mengirim challenge nyata.

- [ ] **Step 3: Laporkan hasil**

Sertakan: URL repo, URL release, service ID Render, bukti deploy sukses,
bukti log/status Lichess, biaya plan Render, dan perintah rollback dari Task 1.

---

## Self-review

- Spec coverage: goals (repo bersih, tag, worker env token, verifikasi, rollback) terbaca di Task 1–5.
- Step scan: setiap langkah punya satu hasil yang bisa dicek; fallback worker type hanya untuk ketidakpastian flag CLI.
- Type consistency: `KESTREL_LICHESS_TOKEN`, `cmd_lichess`, `v1.7.0`, dan `kestrel-lichess-bot` konsisten di semua task.
- Review Focus: token, env fallback, sleep, invalid token, rollback semuanya punya langkah pengecekan.
- Proportion: plan lebih panjang dari spec karena memuat perintah persis untuk infra eksternal; bukan transkrip kode.
