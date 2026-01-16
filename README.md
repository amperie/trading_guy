# Trading Guy

Modular trading backtesting and algorithm framework with an event-driven pipeline and MLflow-backed analysis.

## Architecture

Data flows through a configurable pipeline. Each component is swappable via `config.yaml`.

```
DataProvider -> Algorithm -> Portfolio -> OrderManager -> AnalysisEngine
```

- **DataProvider**: loads data and yields `PriceData` ticks.
- **Algorithm**: inspects ticks + history and emits `MarketSignal`s.
- **Portfolio**: converts signals into `Order`s based on cash/positions.
- **OrderManager**: executes orders (backtest or live).
- **AnalysisEngine**: extracts trades, computes metrics, and generates charts/reports.

Key modules:

- `core/algorithm.py` base class for strategies.
- `core/portfolio.py` base class for execution logic.
- `core/order_manager.py` base class for order execution.
- `data_providers/` sources (CSV/test providers).
- `engines/analysis_engine.py` performance analysis + MLflow logging.
- `utils/` shared config/logging/helpers.

## Backtesting flow (quickstart)

### Option 1: Use `config.yaml`

```python
from engines.backtest_engine import BacktestEngine

engine = BacktestEngine(cfg_section_to_use="simulator")
engine.run()
```

### Option 2: Wire components manually

```python
from engines.backtest_engine import BacktestEngine
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from algorithms.test_algorithm import TestAlgorithm

om = BacktestingOM()
pf = SingleSymbolPortfolio({
    "symbol": "SPXU",
    "cash": 100000,
    "keep_history": True,
    "stop_pct": 3.0,
    "profit_pct": 6.0
})
pf.set_order_manager(om)

algo = TestAlgorithm({"history_length": 10})

engine = BacktestEngine(al=algo, om=om, pf=pf)
engine.run()
```

## Core data objects

- `PriceData`: OHLCV tick input.
- `MarketSignal`: algorithm output.
- `Order`: portfolio output.
- `Position`: portfolio holdings.

These live in `core/classes.py`.

## Configuration

`config.yaml` controls the pipeline. Components are created dynamically using dotted paths.

Example:

```yaml
simulator:
  data_provider:
    provider: "data_providers.test_data_provider.TestDataProvider"
    path: "data/SPXU.csv"
  algorithm:
    algorithm: "algorithms.test_algorithm.TestAlgorithm"
    history_length: 10
  order_manager:
    order_manager: "core.om.backtesting_om.BacktestingOM"
  portfolio:
    portfolio: "core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    symbol: "SPXU"
    cash: 100000
    keep_history: true
```

## Analysis and reports

The analysis engine computes metrics (Sharpe, drawdown, win rate, etc.), extracts trades,
and can generate plots and MLflow artifacts.

```python
from engines.analysis_engine import AnalysisEngine

engine = AnalysisEngine(portfolio, order_manager)
metrics = engine.calculate_metrics()
engine.plot_equity_curve(save_path="equity_curve.png")
report = engine.generate_report()
```

Examples of generated artifacts:

![Equity curve](examples/equity_curve.png)
![Drawdown](examples/drawdown.png)
![Dashboard](examples/dashboard.png)

## Where to look next

- `algorithms/` for strategy examples.
- `engines/analysis_engine.py` for metrics/visualizations.
- `tests/README.md` for test coverage and running tests.
