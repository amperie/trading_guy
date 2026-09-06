# Trading Guy Current State

`trading_guy` is a modular, event-driven trading framework for backtesting, live trading, optimization, promotion, replay, and analysis. It is also the execution engine used by the Quant Crucible platform API through `trading.platform.runner`.

## Core Flow

```text
DataProvider -> optional aggregation -> Algorithm -> Portfolio -> OrderManager -> Analysis
```

Each major component is selected by dotted class path in YAML config, which keeps strategies swappable without changing engine code.

## Major Capabilities

- CSV, Alpaca, MongoDB, and session replay data providers.
- Single-timeframe and multi-timeframe algorithms.
- Built-in portfolios, including single-symbol, risk-managed, day-boundary, dual-symbol switch, oscillator, and risk-target variants.
- Backtesting and Alpaca live order managers.
- Backtest, live, session replay, Mongo backtest, walk-forward, HPO, split HPO, and walk-forward-window HPO commands.
- Release pipeline from research to paper to review to live.
- MLflow tracking, promoted bundle storage, and run reconstruction from MLflow URLs.
- MongoDB-backed live session state and replay.
- Algo Crucible validation milestones and platform runner integration.
- Session analyzer web UI under `web/session_analyzer`.

## Important Entry Points

- `run.py`: main CLI entry point.
- `trading/commands/*`: command implementations.
- `trading/engines/*`: backtest, live, walk-forward, and optimization engines.
- `trading/core/*`: algorithm, portfolio, order, and shared model abstractions.
- `trading/platform/runner.py`: structured runner used by `qc-platform-api`.
- `trading/pipeline.py` and `trading/commands/pipeline.py`: promotion pipeline.
- `configs/*`: runnable example profiles.
- `docs/*`: existing detailed guides.

## Existing Detailed Docs

- `docs/RUN_PY_GUIDE.md`: deep CLI guide.
- `docs/ALGO_CRUCIBLE.md`: Crucible validation workflow.
- `docs/README_MLFLOW.md`: MLflow setup and usage.
- `docs/REMOTE_RAY_SETUP.md`: remote Ray setup.
- `docs/ALPACA_SETUP.md`: Alpaca account and live trading setup.
- `docs/TEST_STRUCTURE.md`: test organization.
- `docs/DEBUG_INSTRUCTIONS.md`: debug workflow.

