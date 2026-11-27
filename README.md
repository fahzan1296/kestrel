# KESTREL 2.0 — CPU-native chess engine dengan evaluator AI TinyNNUE

> **Baru di 2.0 (graduasi dari experimental):** evaluator **AI TinyNNUE
> sepenuhnya** — jaringan NNUE-lite (768-plane → 128 akumulator ×2 perspektif
> → 32 → 1, **106.689 parameter**, int8 terkuantisasi) menggantikan seluruh
> blend evaluator klasik di jalur runtime. Akumulator **inkremental** (update
> ≤ 4 baris bobot per langkah, tersinkron dengan stack Zobrist search),
> antisimetri mirror **eksak by construction**, dan checkpoint baseline
> **terlatih dibundel** dengan paket. Skema eksperimental (TinyMLP lama,
> OutcomeModel linear, REDA, AdaptiveMemory) **dihapus**; trainer unified
> kini melatih satu net utuh per generasi lewat distilasi skor search +
> sinyal hasil game.

CPU-first chess engine and self-play lab written in Python. The project
prioritizes a transparent search/evaluation stack, reproducible experiments,
and an optional localhost dashboard. It is not a Stockfish-strength engine and
there is no Elo claim — but every evaluator change is now gated by automatic
verification (balanced match + SPRT + tactical/NPS/consistency benchmarks).

## Arsitektur evaluator (2.0)

```
fitur   : 12 bidak-plane × 64 kotak = 768 plane per perspektif (NNUE-style)
L1      : 768 → 128, BOBOT DIBAGI antara dua perspektif
acc     : dua akumulator 128-dim (perspektif putih & hitam) — inkremental
L2      : concat(acc_stm, acc_ntm, aux[4]) → 32, ReLU
L3      : 32 → 1, linear × 400 → centipawn STM-relative
```

- **Perspektif STM**: jaringan menerima akumulator berurutan sisi yang
  bergerak, sehingga `eval(mirror(s)) == -eval(s)` **tanpa drift** (bukan
  "±epsilon") — diuji eksak di test suite.
- **Inkremental**: `Searcher` memelihara akumulator bersama stack Zobrist-nya;
  setiap langkah hanya menambah/mengurangi baris bobot bidak yang berpindah
  atau tertangkap (castling, en passant, promosi, null move semuanya benar).
- **Selalu terlatih**: runtime tidak punya jalur "net acak". Tanpa champion
  di `results/`, checkpoint baseline terlatih bawaan
  (`kestrel/models/nnue_baseline.json`) yang dipakai; bila file itu hilang,
  program berhenti dengan pesan jelas (jalankan `scripts/bootstrap_nnue.py`).
- **Kuantisasi int8**: checkpoint tersimpan ~110 KB (per-tensor scale), runtime
  memuat kembali ke FP32; drift evaluasi ≤ 12 cp (diuji).

## What is included

- **Chess state:** `python-chess` is the authority for legal moves, game outcomes, FEN and PGN.
- **Evaluation:** `TinyNNUE` (di atas) — satu-satunya jalur evaluasi posisional. Gate terminal (skakmat, pat, material tak cukup, 75-langkah, lima-kali repetisi) tetap dikenakan sebagai ATURAN permainan, bukan evaluasi.
- **Teacher (distilasi saja):** evaluator klasik 1.8 dipindah ke `kestrel/teacher.py` — dipakai untuk bootstrap checkpoint (`scripts/bootstrap_nnue.py`), fallback label korpus, dan match pembanding "klasik vs AI". Tidak pernah menyentuh jalur runtime.
- **Search:** iterative deepening, negamax alpha-beta, PVS, aspiration windows with progressive widening, transposition table (incremental exact-polyglot Zobrist), MVV-LVA + killer/butterfly/continuation/capture-history ordering with two-way gravity penalties, quiescence (capture-only generation, delta/SEE pruning, all check evasions), adaptive history-aware late-move reduction and pruning, futility pruning, internal iterative reduction, bounded check extensions, mate-distance pruning, adaptive null-move (R=2/3/4), bounded per-search evaluation cache, hard-position time extension (+5 s by default, audited), single move generation per node, only completed iterative-deepening depths published.
- **Budgets and telemetry:** monotonic time and node limits (node limits are never relaxed); results include PV, NPS, TT hit rate, qnode ratio, stop reason, completed depth, and extension count/reasons.
- **Arena:** paired-color matches, PGN output, score-rate confidence interval and an explicitly approximate trinomial SPRT.
- **Trainer unified:** self-play → knowledge book → NNUE distillation (search-score comments + game outcomes, warm start champion) → automatic champion-vs-challenger verification (balanced match, SPRT, Wilson/Elo CI, per-opening breakdown, tactical/NPS/consistency benchmarks — kini benar-benar dijalankan pada net yang diverifikasi) → atomic promote or keep-champion.
- **Self-play monitor:** optional FastAPI dashboard with a live board per game, process-isolated workers, event logs, evaluation trend, batch progress and PGN/SQLite persistence. Endpoint REDA dihapus (fitur dihapus).

## Install

Python 3.10 or newer is required.

```bash
# Runtime dependencies and editable package
python3 -m pip install -e .

# Development and tests
python3 -m pip install -e ".[dev]"

# Optional web monitor
python3 -m pip install -e ".[monitor]"
```

For systems where editable installs are inconvenient, install runtime packages with `python3 -m pip install -r requirements.txt` and run commands from this directory.

## Unified training

```bash
python3 -m kestrel.cli train --generations 10   # full loop from config.toml
python3 -m kestrel.cli train --quick            # smoke test (~2 min)
```

One command runs, per generation: self-play (parallel workers, champion net +
knowledge book) → knowledge update → **TinyNNUE training** (distilasi skor
search dari komentar PGN + hasil game, warm start champion) → automatic
champion-vs-challenger verification (balanced match, SPRT, Wilson/Elo CI,
per-opening breakdown, tactical/NPS/consistency benchmarks) → atomic promote
or keep-champion. All budgets come from `config.toml`; any CLI flag overrides
them per invocation.

## Bootstrap / distilasi manual evaluator

```bash
python3 scripts/bootstrap_nnue.py --samples 30000   # buat baseline dari teacher
python3 -m kestrel.cli nnue distill results/games.pgn --epochs 12
python3 -m kestrel.cli nnue info                    # net aktif + metadata
```

Bootstrap menghasilkan korpus playout realistis (mayoritas terpandu teacher),
memberi label evaluasi statis klasik STM-relative — persis fungsi leaf search
1.8 — lalu melatih net dengan warm-start linear least-squares. Checkpoint
hasil disimpan terkuantisasi int8.

## Run tests and benchmarks

```bash
python3 -m pytest -q tests
python3 -m kestrel.pipeline --quick          # tests -> train(1 gen) -> benchmark -> stockfish
python3 -m kestrel.cli verify                # bukti kekuatan vs baseline bawaan
python3 -m kestrel.cli bench
```

The acceptance test includes perft, terminal positions, evaluator mirror
antisymmetry (eksak), mate search, free-piece tactics, search budgets,
repetition handling, arena statistics and a short match. A passing test suite
checks correctness properties; it does **not** establish playing strength.

## Run self-play

```bash
python3 -m kestrel.cli selfplay --games 12 --learn   # recommended
```

`--learn` updates the knowledge base (opening book + position experience) from
the produced games; the book is then used automatically by later self-play,
training and the Lichess bot. Inspect it with `python3 -m kestrel.cli book show`.

Outputs are written under `results/`: `games.pgn`, `kestrel.db`. Setiap game
mencantumkan komentar `score/depth` per langkah — ini label distilasi untuk
generasi trainer berikutnya (tanpa pencarian ulang).

## Live dashboard

Install the optional dependency group, then start the monitor:

```bash
python3 -m pip install -e ".[monitor]"
python3 -m kestrel.dashboard
```

Open `http://127.0.0.1:8765` on the same machine. The service binds to loopback
by default; it is not a public web service. Untuk akses dari komputer lokal ke
VPS, gunakan SSH tunnel (`ssh -L 8765:127.0.0.1:8765 USER@SERVER`). Do not
expose the development dashboard directly to the public internet without adding
authentication, TLS and deployment hardening.

The dashboard controls game count, parallel worker count, per-move time, depth
and ply cap. Each game runs in its own child process. API endpoints:
`GET /api/state`, `GET /api/health`, `POST /api/run`, `POST /api/stop`,
`POST /api/train`, `POST /api/train/stop`, `GET /api/train/state`.

## Train / verify evaluator

```bash
python3 -m kestrel.cli nnue distill results/games.pgn   # satu-tembak dari korpus
python3 -m kestrel.cli verify                            # match vs baseline bawaan
python3 -m kestrel.cli eval --champion A --challenger B  # head-to-head manual
python3 -m kestrel.cli model history                     # riwayat champion
python3 -m kestrel.cli model rollback --steps 1          # rollback champion
```

Setiap kandidat diverifikasi sebelum dipromosikan: W/D/L warna-seimbang,
score rate dengan Wilson 95% CI, Elo delta dengan CI, SPRT trinomial
screening, per-opening breakdown, dan benchmark taktik/NPS/konsistensi untuk
KEDUA net. Laporan tersimpan di `results/verification_gen_*.json|.md`.
`results/champion.json` melacak lineage champion.

## Lichess bot

```bash
export KESTREL_LICHESS_TOKEN=lip_xxx     # JANGAN ditulis ke config.toml
python3 -m kestrel.cli lichess           # atau --token YOUR_TOKEN
python3 -m kestrel.cli lichess --dry-run # offline decision check
```

Accepts/declines challenges per `[lichess]` policy (standard chess only, rating
gap, bot/human), plays with the champion net + knowledge book + min-depth
guarantee, claims draws when drawable and not winning, resigns after repeated
losing moves, and saves each game's PGN under `results/lichess/`. The account
must be upgraded to a BOT account — the bot offers the upgrade automatically on
first run. Token scopes: `bot:play`.

> **Catatan keamanan:** versi ≤ 1.8 pernah mengirim contoh config dengan token
> nyata di dalamnya. Anggap token tersebut terekspos: revoke di
> lichess.org/account/oauth/token. Sejak 2.0, token hanya dibaca dari
> environment variable atau flag CLI.

Deploy produksi di InstaBox: lihat [docs/deployment.md](docs/deployment.md).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Fahzan Dewantara.

## Important limitations

- Python and `python-chess` make this implementation suitable for
  experimentation, not competitive high-performance chess search.
- Self-play results are not independent samples and cannot, by themselves,
  establish Elo improvement.
- The SPRT helper uses a shared empirical draw-rate approximation. Treat it as
  a screening statistic, not a substitute for a carefully specified
  engine-testing protocol.
- The dashboard is an observability tool. Its `NPS` values should be measured
  under controlled, comparable machine load.
- Checkpoint baseline adalah distilasi evaluator klasik (bukan hasil
  supertraining): kekuatannya setara teacher-nya, dan generasi trainer
  berikutnya yang menaikkan lewat verifikasi otomatis.
