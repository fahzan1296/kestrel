FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python3", "-m", "kestrel.pipeline", "--quick", "--skip", "selfplay"]
