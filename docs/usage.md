# Usage

## Requirements

- Python 3.11 or newer.
- `uv` or an equivalent Python environment manager.
- Docker, MongoDB, MLflow, Ray, and Alpaca credentials only for workflows that need them.

Install dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

## Common Commands

Backtest:

```bash
uv run python run.py backtest --config configs/example_backtest.yaml
```

Backtest with aggregation:

```bash
uv run python run.py backtest --config configs/example_backtest_agg.yaml
```

Live trading:

```bash
uv run python run.py live --config configs/example_live_spy_trend_macd.yaml --account paper
```

Session replay:

```bash
uv run python run.py session-replay --config configs/example_session_replay.yaml --session-id <session_id>
```

Walk-forward:

```bash
uv run python run.py walk-forward --config configs/example_walk_forward.yaml
```

HPO:

```bash
uv run python run.py hpo --config configs/example_hpo.yaml
uv run python run.py hpo-split --config configs/example_hpo_split.yaml --validation-period-days 30
```

Release pipeline:

```bash
uv run python run.py pipeline research --config configs/example_hpo_split.yaml --account paper
uv run python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<candidate_run_id> --account paper
uv run python run.py pipeline review --config trading/promoted/<paper_bundle>/<paper_bundle>.yaml --account paper --session-id <paper_session_id>
uv run python run.py pipeline live --config trading/promoted/<approved_bundle>/<approved_bundle>.yaml --account live --session-id <live_session_id>
```

## Config Pattern

Most commands are driven by YAML profiles in `configs/`. A typical config chooses:

- Data provider.
- Algorithm class and parameters.
- Portfolio class and risk settings.
- Order manager.
- Analysis and MLflow settings.
- Optional aggregation, optimization, walk-forward, and pipeline gates.

Common flags include `--config`, `--account`, `--cash`, `--symbol`, `--algorithm`, `--portfolio`, `--no-mlflow`, `--run-name`, `--agg-period`, `--data`, and `--session-id`.

## Writing Components

Algorithms usually subclass `trading.core.algorithm.Algorithm` and implement `on_data_logic`. Multi-timeframe algorithms subclass `MultiTimeframeAlgorithm` and implement `on_mtf_data`.

Portfolios subclass `trading.core.portfolio.Portfolio` and implement `process_tick_market_signals_logic`.

See the root `README.md` and `docs/RUN_PY_GUIDE.md` for examples.

