FROM python:3.11-slim

WORKDIR /workspace/trading_guy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN pip install uv \
  && uv sync --frozen --no-dev || uv sync --no-dev

COPY . .

ENV PYTHONPATH=/workspace/trading_guy
CMD ["uv", "run", "python", "-m", "trading.platform.runner"]