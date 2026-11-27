# KESTREL 2.0 — Quickstart

Referensi cepat CLI. Semua perintah membaca `config.toml` sebagai default;
flag baris-perintah menimpanya per pemanggilan.

## Instalasi

```bash
python3 -m pip install -e .            # runtime
python3 -m pip install -e ".[dev]"     # + pytest
python3 -m pip install -e ".[monitor]" # + dashboard (fastapi/uvicorn)
python3 -m pip install -e ".[bot]"     # + bot Lichess (requests)
```

## Evaluator AI (TinyNNUE)

| Perintah | Fungsi |
| --- | --- |
| `python3 -m kestrel.cli nnue info` | net aktif + metadata checkpoint |
| `python3 -m kestrel.cli nnue distill results/games.pgn --epochs 12` | distilasi satu-tembak dari korpus PGN |
| `python3 scripts/bootstrap_nnue.py --samples 30000` | buat checkpoint baseline bawaan (teacher klasik) |

Checkpoint champion: `results/nnue_eval.json`; snapshot kandidat:
`results/checkpoints/gen_<label>_nnue.json`; metadata lineage:
`results/champion.json`.

## Training unified

```bash
python3 -m kestrel.cli train                    # 10 generasi dari config
python3 -m kestrel.cli train --generations 3    # override
python3 -m kestrel.cli train --quick            # smoke test ~2 menit
python3 -m kestrel.cli train --no-verify        # tanpa match verifikasi
```

Per generasi: selfplay → knowledge book → **training TinyNNUE**
(distilasi skor search + hasil game, warm start champion) → verifikasi
(warna seimbang + SPRT + benchmark taktik/NPS/konsistensi kedua net) →
promote atomik atau keep-champion.

## Cross-engine (vs Stockfish)

```bash
python3 -m kestrel.cli cross-train --games 20 --skill 5
```

Korpus game vs Stockfish (adaptive skill) langsung dipakai untuk training
TinyNNUE generasi berikutnya.

## Self-play & knowledge

```bash
python3 -m kestrel.cli selfplay --games 12 --learn
python3 -m kestrel.cli book show
python3 scripts/selfplay_demo.py 12        # wrapper visual matplotlib
```

## Verifikasi & benchmark

```bash
python3 -m kestrel.cli verify              # match vs baseline bawaan + benchmark
python3 -m kestrel.cli bench               # taktik/NPS/antisimetri
python3 -m kestrel.cli eval --champion A.json --challenger B.json
python3 -m kestrel.cli model history       # lineage champion
python3 -m kestrel.cli model rollback --steps 1
python3 -m kestrel.cli model promote --path gen_0001_try01_nnue.json
```

## Bot Lichess

```bash
export KESTREL_LICHESS_TOKEN=lip_xxx   # JANGAN menulis token ke config.toml
python3 -m kestrel.cli lichess
python3 -m kestrel.cli lichess --dry-run
```

## Dashboard

```bash
python3 -m kestrel.dashboard             # http://127.0.0.1:8765 (loopback saja)
```

Endpoint: `GET /api/state`, `GET /api/health`, `POST /api/run`,
`POST /api/stop`, `POST /api/train`, `POST /api/train/stop`,
`GET /api/train/state`.
