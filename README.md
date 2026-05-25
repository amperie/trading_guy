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

### Multi-Timeframe Algorithms

`MultiTimeframeAlgorithm` extends `Algorithm` to maintain separate bar histories for N timeframes simultaneously. Raw ticks are aggregated in-process — no external aggregation engine is needed.

```python
from trading.core.multi_timeframe_algorithm import MultiTimeframeAlgorithm
from trading.core.classes import PriceData, MarketSignal, SignalType

class RegimeEntryAlgo(MultiTimeframeAlgorithm):
    def on_mtf_data(
        self,
        tick: list[PriceData],
        new_bars: dict[int, list[PriceData]],
    ) -> list[MarketSignal]:
        # new_bars only contains keys for timeframes that completed this tick
        if 60 not in new_bars:
            return []   # wait for hourly regime bar

        hourly = list(self.bar_history[60]["SPY"])
        intraday = list(self.bar_history[5]["SPY"])
        if not hourly or not intraday:
            return []

        if intraday[-1].close > hourly[-1].close:
            return [MarketSignal(SignalType.BUY, "SPY", 75)]
        return []
```

Config:
```yaml
algorithm:
  algorithm: "my_strategies.RegimeEntryAlgo"
  timeframes: [5, 60]         # periods in minutes; can be any number of timeframes
  history_length: 200         # bars to keep per timeframe
  market_open_hour: 9
  market_open_minute: 30
```

Key properties:
- `self.bar_history[period_minutes][symbol]` — deque of completed `PriceData` bars per timeframe
- `self.price_history[symbol]` / `self.price_data_history[symbol]` — raw tick history (inherited)
- `required_warmup_bars` defaults to `history_length × max(timeframes)` so the slowest TF is fully populated before signals fire
- Implement `on_mtf_data()` — do **not** override `on_data_logic()` (it is `@final` in this class)

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

## Recommended Workflow

The intended release path is now:

```text
research -> paper -> review -> live
```

Use the high-level pipeline commands when you want the lowest-friction path:

```bash
# 1. Research: backtest -> split HPO -> walk-forward -> gate checks
python run.py pipeline research --config configs/example_hpo_split.yaml --account paper

# 2. Paper: build a paper bundle from the winning MLflow run and launch it
python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<candidate_run_id> --account paper

# 3. Review: replay the paper session and approve a live bundle if drift gates pass
python run.py pipeline review --config trading/promoted/<paper_bundle>/<paper_bundle>.yaml --account paper --session-id <paper_session_id>

# 4. Live: launch from local bundle YAML or directly from the approved MLflow bundle URL
python run.py pipeline live --config trading/promoted/<approved_bundle>/<approved_bundle>.yaml --account live --session-id <live_session_id>
python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account live --session-id <live_session_id>
```

What the pipeline does:

- `pipeline research` prints the backtest, split-HPO, and walk-forward MLflow URLs, evaluates `pipeline.gates.research`, and can auto-register a candidate bundle in the dedicated pipeline MLflow experiment.
- `pipeline paper` materializes a paper bundle from a source MLflow run, stores that bundle locally under `trading/promoted/`, logs the same bundle into the pipeline MLflow experiment, and prints both launch paths.
- `pipeline review` runs `session-replay`, prints replay MLflow links and drift numbers, evaluates `pipeline.gates.review`, and when the session passes, registers an approved live bundle.
- `pipeline live` launches from either the local filesystem after a git pull or directly from an MLflow run URL containing a reconstructable promoted bundle config.

The root [config.yaml](/E:/Programming/trading_guy/config.yaml) now includes a dedicated `pipeline` section:

```yaml
pipeline:
  experiment_name: "Trading Pipeline Bundle Registry"
  auto_promote_research: true
  gates:
    research:
      min_val_annualized_return: 0
      max_val_max_drawdown_pct: 25
      min_val_total_trades: 5
      min_wf_annualized_return: 0
      max_wf_max_drawdown_pct: 30
    review:
      max_alpaca_live_equity_drift_pct: 5
      max_mongo_live_equity_drift_pct: 5
```

That experiment is where bundle-registration runs live. Those runs store promoted/approved bundle artifacts, manifests, and launch metadata. They are not intended to hold strategy performance metrics.

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

# Walk-forward window HPO (search optimization/validation/trading window sizes)
python run.py walk-forward-hpo --config configs/example_walk_forward_hpo.yaml

# Standalone hyperparameter optimization
python run.py hpo --config configs/example_hpo.yaml

# Split HPO with an out-of-sample validation holdout
python run.py hpo-split --config configs/example_hpo_split.yaml --validation-period-days 30

# Recreate HPO settings from a prior MLflow run
python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id>

# Full release pipeline
python run.py pipeline research --config configs/example_hpo_split.yaml --account paper
python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<candidate_run_id> --account paper
python run.py pipeline review --config trading/promoted/<paper_bundle>/<paper_bundle>.yaml --account paper --session-id <paper_session_id>
python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account live --session-id <live_session_id>
```

Common flags (all modes):

| Flag | Description |
|---|---|
| `--config` | Local YAML config profile, or an MLflow run URL with a reconstructable config |
| `--account` | Account name from `accounts.yaml` |
| `--cash` | Override starting cash |
| `--symbol` | Override trading symbol |
| `--algorithm` | Override algorithm class (dotted path) |
| `--algorithm-url` | Load algorithm code from an HTTP(S) URL |
| `--portfolio` | Override portfolio class (dotted path) |
| `--portfolio-url` | Load portfolio code from an HTTP(S) URL |
| `--no-mlflow` | Disable MLflow logging |
| `--run-name` | Override MLflow run name |
| `--agg-period N` | Set aggregation bar size in minutes (also enables aggregation) |
| `--data` | Override data file path for commands that expose it |
| `--session-id` | MongoDB session ID |

## Debug REPL And Runtime Logging

Set the global root config flag in [config.yaml](/E:/Programming/trading_guy/config.yaml) to enable the Ctrl-C debug REPL:

```yaml
debug_on_sigint: true
```

When enabled, pressing `Ctrl-C` during an engine-driven run opens the REPL instead of terminating immediately. Useful commands:

- `progress`: print the engine progress snapshot with elapsed time and estimated remaining work
- `loglevel`: show current root, console, and file logger levels
- `loglevel DEBUG`: raise both console and file handlers to `DEBUG`
- `loglevel console INFO`: change only console verbosity
- `loglevel file DEBUG`: change only file logging verbosity
- `c`: resume execution
- `q`: stop the engine

Interactive terminals also show a single-line live status bar for walk-forward modes. It updates progress in place and automatically falls back to normal logging when stdout is not a TTY.

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

### RiskTargetPortfolio

`RiskTargetPortfolio` is a built-in long-only, single-symbol portfolio for
volatility-based position sizing. It treats the algorithm as the directional
decision maker and handles the sizing decision itself:

- `BUY` signal: rebalance toward a risk-sized long position
- `SELL` signal: liquidate the current position
- no signal: hold unless the drawdown guard is triggered

Target exposure is:

```text
min(max_exposure, target_volatility / realized_volatility) * (signal.strength / 100)
```

Before enough price history exists to estimate volatility, it uses
`default_exposure`.

Example:

```yaml
portfolio:
  portfolio: "trading.core.pf.risk_target_portfolio.RiskTargetPortfolio"
  symbol: "SPY"
  cash: 100000
  keep_history: true

  # Sizing
  target_volatility: 0.15     # 15% annualized target volatility
  volatility_lookback: 20     # close-to-close returns used for realized vol
  annualization_factor: 252   # daily bars; use ~19656 for 5-min regular-session bars
  max_exposure: 1.0           # cap at 100% long exposure
  default_exposure: 0.5       # startup exposure before vol history is ready

  # Trading filters
  min_trade_value: 100        # skip tiny rebalance orders
  min_signal_strength: 50     # ignore weak signals; strength also scales size
  tx_cost: 0.0

  # Risk guard
  drawdown_limit_pct: 0.10    # liquidate after 10% drawdown from peak equity
  halt_on_drawdown: true      # ignore later BUY signals after that breach
```

Parameter notes:

- `target_volatility` and `drawdown_limit_pct` use decimal form, so `0.15`
  means 15%.
- `annualization_factor` must match the bar size. Common values are `252` for
  daily bars, about `1638` for hourly regular-session bars, about `19656` for
  5-minute regular-session bars, and about `98280` for 1-minute regular-session
  bars.
- `max_exposure` is an equity multiple. `1.0` means fully invested at most;
  values above `1.0` request leverage, subject to available cash or buying power.
- `min_signal_strength` filters signals and signal strength also scales the
  final target exposure.
- The portfolio emits market orders and integer share quantities only.

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
- optimization event artifacts (`optimization_events.json`, `.csv`, `.md`)

### Walk-Forward Window HPO

Use `walk-forward-hpo` when you want to tune the walk-forward schedule itself,
not just the algorithm and portfolio parameters inside each optimization
window.

```bash
python run.py walk-forward-hpo --config configs/example_walk_forward_hpo.yaml --account paper
```

This runs an outer Optuna search over:

- `optimization_window_days`
- `validation_window_days`
- `trading_window_days`

Each sampled schedule runs the normal historical walk-forward flow. That means
each candidate still performs the configured inner Ray Tune HPO over
`walk_forward.search_space`, validates incumbent vs challenger, and scores the
final continuous walk-forward result.

The candidate runs are logged to a temporary MLflow experiment. After the best
window schedule is selected, the winner is rerun into the permanent experiment
from `walk_forward_window_hpo.final_experiment_name`.

Typical config block:

```yaml
walk_forward_window_hpo:
  num_samples: 8
  objective_metric: wf_annualized_return
  min_periods: 3
  final_experiment_name: "walk_forward_runs"

  cleanup_staging_experiment: true
  run_mlflow_gc: true
  cleanup_s3_prefix: false

  search_space:
    optimization_window_days: { type: choice, values: [90, 120, 180, 252] }
    validation_window_days:   { type: choice, values: [20, 30, 45, 60] }
    trading_window_days:      { type: choice, values: [10, 20, 30] }
```

Cleanup notes:

- `cleanup_staging_experiment` marks the temporary MLflow experiment deleted.
- `run_mlflow_gc` attempts permanent MLflow cleanup after that deletion.
- `cleanup_s3_prefix` directly removes `staging_artifact_location` with the AWS
  CLI, so only use it with a dedicated staging prefix.
- optimization event tables and JSON exports
- MongoDB `optimization_events` records keyed by event id so chart markers can be traced back to the stored decision

---

## MLflow-Backed Bundles

Promoted and approved bundles now have two valid launch paths:

- local filesystem: `trading/promoted/<bundle>/<bundle>.yaml`
- MLflow run URL: `http://.../#/experiments/<id>/runs/<bundle_run_id>`

That means a second node can either:

1. `git pull` and run the local promoted YAML
2. skip the local promoted files and run directly from the MLflow bundle URL

Example:

```bash
python run.py live --config trading/promoted/spy_macd_live_v1/spy_macd_live_v1.yaml --account paper --session-id live-20260513-b
python run.py live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account paper --session-id live-20260513-c
```

The pipeline commands print both the local paths they created and the MLflow links they logged so there is no ambiguity about what artifact was produced.

---

## Directory Structure

```
trading/
  core/
    algorithm.py                    # Algorithm base (subclass this)
    multi_timeframe_algorithm.py    # MultiTimeframeAlgorithm base (multi-TF strategies)
    portfolio.py                    # Portfolio base (subclass this)
    classes.py                      # PriceData, MarketSignal, Order, BracketOrder
    pf/                             # Portfolio implementations
    om/                             # OrderManager implementations
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
