# Architecture

`trading_guy` is built around a small set of runtime contracts and engines. The architecture is intentionally config-driven: YAML selects concrete classes, while the engines call stable base interfaces.

## Component Pipeline

```text
DataProvider -> TickAggregation -> Algorithm -> Portfolio -> OrderManager -> AnalysisEngine
```

- `DataProvider` yields grouped `PriceData` ticks by timestamp.
- `TickAggregationPassthroughEngine` can fold raw minute bars into larger bars.
- `Algorithm` receives data/history and emits `MarketSignal` objects.
- `Portfolio` converts signals into `Order` or `BracketOrder` instructions.
- `OrderManager` executes orders in backtest or live contexts.
- `AnalysisEngine` computes metrics, charts, trade logs, and MLflow artifacts.

## Key Packages

- `trading/core`: base classes, shared data models, portfolios, and order managers.
- `trading/algorithms`: built-in strategy algorithms.
- `trading/data_providers`: CSV, Alpaca, MongoDB, and replay providers.
- `trading/engines`: backtest, Alpaca live, self-optimizing live, walk-forward, tick aggregation, and dataset builders.
- `trading/commands`: CLI command implementations behind `run.py`.
- `trading/launchers`: Ray, MLflow, promotion, and backtest launcher utilities.
- `trading/analysis`: metrics, charts, regime analysis, and portfolio analysis.
- `trading/config`: config loading, validation, component registry, and component loader.
- `trading/reporting`: reporting models and sinks.
- `trading/platform`: platform runner bridge for Quant Crucible.
- `algo_crucible`: adversarial validation workflow and related jobs.

## Runtime Modes

The same component model supports:

- Historical backtesting from CSV or Alpaca bars.
- Live Alpaca trading.
- Live self-optimization.
- Historical walk-forward validation.
- Hyperparameter optimization with Ray Tune and Optuna.
- Session replay from MongoDB or MLflow-backed session metadata.
- Promotion pipeline runs that materialize and approve bundles.

## Platform Integration

`qc-platform-api` invokes `python -m trading.platform.runner` in Docker. The runner accepts stage, run id, strategy id, tenant id, config, output directory, account, run name, and experiment name. It emits structured progress JSON and writes artifacts that the API worker uploads.

