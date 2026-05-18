# Trading Guy

A modular, event-driven trading framework. Swap any component via config — write an **Algorithm**, plug it in, and test it against historical data or live markets in minutes.

---

## Architecture

```
DataProvider → [TickAggregation] → Algorithm → Portfolio → OrderManager
```

| Component | Responsibility |
|---|---|
| **DataProvider** | Loads market data, yields `PriceData` ticks grouped by timestamp |
| **TickAggregationPassthroughEngine** | Optional — folds raw 1-min bars into N-min bars before the algorithm sees them |
| **Algorithm** | Receives ticks + rolling price history, emits `MarketSignal` objects |
| **Portfolio** | Converts signals into `Order` objects based on cash, positions, and risk rules |
| **OrderManager** | Executes orders — `BacktestingOM` fills instantly; `AlpacaOM` routes to a live broker |
| **AnalysisEngine** | Extracts trades, computes 30+ metrics, generates charts, logs to MLflow |

Every component is swappable via its dotted class path in the config file.

---

## Testing Any Algorithm

**1. Subclass Algorithm — implement one method:**

```python
from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType

class SmaCrossover(Algorithm):
    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        prices = list(self.price_history["SPY"])
        if len(prices) < 20:
            return []
        if sum(prices[-5:]) / 5 > sum(prices[-20:]) / 20:
            return [MarketSignal(type=SignalType.BUY, symbol="SPY", strength=75)]
        return [MarketSignal(type=SignalType.SELL, symbol="SPY", strength=75)]
```

**2. Point a config at it:**

```yaml
simulator:
  data_provider:
    provider: "trading.data_providers.test_data_provider.TestDataProvider"
    path: "data/SPY_1min.csv"

  algorithm:
    algorithm: "my_strategies.SmaCrossover"
    history_length: 20

  portfolio:
    portfolio: "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    symbol: "SPY"
    cash: 100000
    keep_history: true

  order_manager:
    order_manager: "trading.core.om.backtesting_om.BacktestingOrderManager"

analysis:
  enabled: true
  log_to_mlflow: true
  experiment_name: "SMA Tests"
```

**3. Run:**

```bash
python run.py backtest --config my_config.yaml
```

Results (metrics, equity curve, trade log) are printed and logged to MLflow automatically.

---

## All Run Modes

```bash
# Backtest from CSV
python run.py backtest --config configs/example_backtest.yaml

# Backtest with tick aggregation (1-min data → 5-min bars)
python run.py backtest --config configs/example_backtest_agg.yaml

# Live trading (Alpaca)
python run.py live --config configs/example_live_spy_trend_macd.yaml

# Live with background self-optimization
python run.py live --config configs/example_live_self_optimizing.yaml

# Live with walk-forward optimization
python run.py live --config configs/example_live_walk_forward.yaml --session-id <id>

# Replay a stored live session against Alpaca historical bars
python run.py session-replay --config configs/example_session_replay.yaml --session-id <id>

# Run a promoted or live-style config as a normal backtest using bars stored in MongoDB
python run.py mongo-backtest --config trading/promoted/<bundle>/<bundle>.yaml --session-id <id>

# Walk-forward backtest (rolling optimize + validate + continuous trade simulation)
python run.py walk-forward --config configs/example_walk_forward.yaml

# Standalone hyperparameter optimization
python run.py hpo --config configs/example_hpo.yaml
```

Common flags (all modes):

| Flag | Description |
|---|---|
| `--config` | YAML config profile (required) |
| `--cash` | Override starting cash |
| `--symbol` | Override trading symbol |
| `--algorithm` | Override algorithm class (dotted path) |
| `--no-mlflow` | Disable MLflow logging |
| `--run-name` | Override MLflow run name |
| `--agg-period N` | Set aggregation bar size in minutes (also enables aggregation) |
| `--data` | Override data file path (backtest / walk-forward / hpo) |
| `--session-id` | MongoDB session ID |

Live optimization modes:

| Mode | How to activate |
|---|---|
| Plain live | No `optimization.enabled` block |
| Self-optimizing live | `optimization.enabled: true` without `mode: walk_forward_live` |
| Live walk-forward | `optimization.enabled: true` and `optimization.mode: walk_forward_live` |

---

## Tick Aggregation

Trade on N-min bars without preprocessing data or changing strategy code:

```yaml
aggregation:
  enabled: true
  aggregation_period_minutes: 5   # 1, 3, 5, 10, 15, …
  use_market_open: true
```

Or via CLI: `--agg-period 10`

---

## Portfolio and Orders

Use a built-in portfolio or subclass `Portfolio` and implement `process_tick_market_signals_logic()`:

```python
class MyPortfolio(Portfolio):
    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal], tick: list[PriceData]) -> TickResults:
        signal = next((s for s in signals if s.symbol == "SPY"), None)
        price  = self.get_price("SPY", tick)
        if signal and signal.type == SignalType.BUY and self.cash >= price:
            qty = int(self.cash / price)
            return TickResults(orders=[BracketOrder.create_bracket_order(
                "SPY", price, price * 1.06, price * 0.97, qty, 0.0, tick
            )])
        return TickResults(orders=[])
```

Order types: **Market** (instant fill) and **Bracket** (entry + stop-loss + profit-taker; when one child triggers the other is canceled).

---

## Analysis

After a backtest, metrics are computed and logged automatically. For a stored live session:

```python
from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

analyzer = PortfolioAnalyzer.from_mongodb("your-session-uuid")
analyzer.run_full_analysis(output_dir="output/", log_to_mlflow=True)

# Merge multiple sessions (e.g. bot restarted)
analyzer = PortfolioAnalyzer.from_mongodb_multi(["sid1", "sid2"])
```

**Metrics:** total return, Sharpe, Sortino, max drawdown, win rate, profit factor, Calmar, Ulcer index, bracket effectiveness, and 20+ more.

## Walk-Forward Notes

The historical walk-forward mode now works in two phases:

- it computes rolling optimization and validation decisions period by period
- it then runs one continuous backtest over the full span, applying approved config changes at each trading-window boundary

That means the final MLflow run and final metrics come from the real end-to-end portfolio object, not from stitched or averaged per-period summaries.

Artifacts for walk-forward runs now include:

- one full-run MLflow entry with end-to-end metrics
- an equity curve chart annotated with optimization/adoption markers
- optimization event tables and JSON exports
- MongoDB `optimization_events` records keyed by event id so chart markers can be traced back to the stored decision

---

## Directory Structure

```
trading/
  core/
    algorithm.py              # Algorithm base (subclass this)
    portfolio.py              # Portfolio base (subclass this)
    classes.py                # PriceData, MarketSignal, Order, BracketOrder
    pf/                       # Portfolio implementations
    om/                       # OrderManager implementations
  algorithms/                 # Built-in strategies
  analysis/
    analysis_engine.py        # 30+ metrics, charts, MLflow
    portfolio_analyzer.py     # Drop-in alternative; adds from_mongodb()
  data_providers/             # CSV, Alpaca, SessionReplay
  engines/
    backtest_engine.py
    alpaca_engine.py
    tick_aggregation_passthrough_engine.py
    walk_forward_engine.py
    self_optimizing_live_engine.py
configs/                      # Example YAML profiles
utils/                        # Config, logging, MLflow, MongoDB helpers
run.py                        # Main entry point
data/                         # Market data CSV files
```

---

## Further Reading

- [Detailed `run.py` Guide](docs/RUN_PY_GUIDE.md)
- [Alpaca Live Trading Setup](docs/ALPACA_SETUP.md)
- [Remote Ray Cluster Setup](docs/REMOTE_RAY_SETUP.md)
- [MLflow Experiment Tracking](docs/README_MLFLOW.md)
- [Technical Indicators](docs/TECHNICAL_INDICATORS.md)
