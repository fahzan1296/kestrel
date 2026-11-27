# KESTREL 1.6 — satu perintah untuk semua (CLI pengganti konfigurasi UI)
.PHONY: install test train quick train-smoke selfplay benchmark verify bench book eval match pipeline monitor lichess clean zip

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest -q tests

## --- CLI baru (config dari config.toml; flag menimpa per pemanggilan) ---
train:              ## loop unified penuh (default 10 generasi dari config)
	python3 -m kestrel.cli train

train-smoke:        ## 1 generasi smoke-test ~2 menit
	python3 -m kestrel.cli train --quick --generations 1

selfplay:           ## batch self-play saja
	python3 -m kestrel.cli selfplay --games 12

eval:               ## eval detail champion vs kandidat terakhir
	python3 -m kestrel.cli eval --eval-games 20

verify:             ## bukti kekuatan: match vs baseline + benchmark penuh
	python3 -m kestrel.cli verify

bench:
	python3 -m kestrel.cli bench

book:               ## bangun/tampilkan knowledge base dari korpus
	python3 -m kestrel.cli book build
	python3 -m kestrel.cli book show

benchmark:
	python3 -m kestrel.pipeline --quick --skip tests train

match:
	python3 -m kestrel.pipeline --quick --skip tests train benchmark

pipeline:           ## full: tests -> train(1 gen unified) -> benchmark -> stockfish
	python3 -m kestrel.pipeline

quick:              ## smoke test cepat seluruh pipeline
	python3 -m kestrel.pipeline --quick

nnue-distill:       ## distilasi TinyNNUE dari korpus self-play
	python3 -m kestrel.cli nnue distill results/games.pgn

bootstrap:          ## buat checkpoint baseline bawaan
	python3 scripts/bootstrap_nnue.py

monitor:            ## dashboard monitoring localhost (optional dependency)
	python3 -m kestrel.dashboard

lichess:            ## bot Lichess (butuh token di config.toml / env)
	python3 -m kestrel.cli lichess

clean:
	rm -rf __pycache__ kestrel/__pycache__ results/pipeline_report_*.json

zip:
	cd .. && rm -f kestrel_python.zip && zip -r kestrel_python.zip kestrel_python -x "*__pycache__*"
