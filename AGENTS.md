# AGENTS.md

## Architecture Overview

Modular trading backtesting and algorithm framework with event-driven architecture supporting simulation and real-time trading.

**Data Flow Pipeline:** DataProvider → Algorithm → Portfolio → OrderManager
- **DataProvider** streams `PriceData` from sources (CSV, APIs, live feeds)
- **Algorithm** generates `MarketSignal` objects from market data
- **Portfolio** converts signals into `Order` objects based on cash/positions/strategy
- **OrderManager** executes orders (simulated or live via broker APIs)

An optional **TickAggregationPassthroughEngine** can sit between the DataProvider and the downstream engine to fold raw 1-min ticks into N-min bars before the algorithm sees them.

Components are swappable via config using `utils.utils.instantiate_from_string()` with dotted paths.

### Configuration System
- Singleton `ConfigManager` in `utils/config_manager.py` reads `config.yaml`
- `ConfigManager().get_as_object()` for dot-notation access, `.get(key)` for dict-style nested access
- All components receive their config dict on initialization

### Base Class Contracts

**Algorithm** (`core.algorithm.Algorithm`):
- Override `on_data_logic(data: list[PriceData]) -> list[MarketSignal]` (do NOT override `on_data()` — it's `@final`)
- Config: `history_length` (deque maxlen), `full_history` (bool)
- `self.price_history[symbol]` (deque of closes), `self.price_data_history[symbol]` (deque of PriceData)
- **Warmup gate**: `on_data()` always calls `on_data_logic()` (builds internal state) but returns `[]` until `is_warmed_up`.
  - `_ticks_seen` — incremented on every `on_data()` call (including warmup ticks)
  - `required_warmup_bars` — property, defaults to `history_length`; override to set a larger threshold
  - `is_warmed_up` — `True` once `_ticks_seen >= required_warmup_bars`
  - This gate enables session replay: warmup bars and live bars flow through `BacktestingEngine` as one stream; signals are suppressed automatically during the warmup window
- `reconfigure(new_params)` updates params without losing history. Override if `__init__` caches config into instance vars:
  ```python
  def reconfigure(self, new_params):
      super().reconfigure(new_params)
      for attr in ("my_param",):
          if attr in new_params: setattr(self, attr, new_params[attr])
  ```

**MultiTimeframeAlgorithm** (`core.multi_timeframe_algorithm.MultiTimeframeAlgorithm`):
- Extends `Algorithm` — all warmup gate, price history, and reconfigure machinery is inherited
- Override `on_mtf_data(tick, new_bars) -> list[MarketSignal]` (do NOT override `on_data_logic()` — it's `@final` here)
- Config: `timeframes` (list[int] of minute periods, required), plus all `Algorithm` config keys
- `self.bar_history[period_minutes][symbol]` — deque of completed `PriceData` bars per timeframe
- `new_bars: dict[int, list[PriceData]]` — only contains keys for timeframes that completed a bar this tick
- `required_warmup_bars` defaults to `history_length × max(timeframes)` so the slowest TF is fully populated before signals fire; override to adjust

**DataProvider** (`data_providers.data_provider.DataProvider`):
- Override `load_data()` to populate `self.data` (DataFrame: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`; optional: `trade_count`, `vwap`, `exchange`)
- `iterate()` yields `list[PriceData]` per timestamp, auto-detects chronological order, groups by timestamp
- `TestDataProvider`: `path` required, `truncate` optional (default 0 = load all rows), `start_date`/`end_date` optional

**Portfolio** (`core.portfolio.Portfolio`):
- Override `process_tick_market_signals_logic(signals, tick) -> TickResults`
- `get_price(symbol, tick) -> float | None` — `@final` price lookup: checks tick first, falls back to `self.previous_price` cache. Essential for live trading where ticks contain single symbols.
- Tracks: `cash`, `positions`, `total_value`, `previous_price`
- History (when `keep_history: true`): `tick_history`, `cash_history`, `value_history`, `signals_history`
- `process_market_signals_for_tick()` is `@final` — auto-calls OM pending order updates, broker sync, portfolio value update
- `reconfigure(new_params)` preserves cash/positions/history
- Impl: `SingleSymbolPortfolio` (core/pf/) — buys with all cash, sells all positions
- Impl: `DualSymbolSwitchPortfolio` (core/pf/) — switches between two symbols (e.g. UPRO/SPXU) with bracket orders, holding periods, and manual sale switching

**OrderManager** (`core.order_manager.OrderManager`):
- Override three backend methods:
  - `_submit_order_to_backend(order, tick, positions, pf_cash)`
  - `_update_order_status_from_backend(order, tick, positions, pf_cash)`
  - `_update_orders_statuses_from_backend(orders, tick, positions, pf_cash)`
- Public API: `submit_order()`, `update_order_status()`, `update_pending_orders()`
- Tracks: `_all_orders`, `_pending_orders_by_id`, `_filled_orders_by_id`
- Impl: `BacktestingOM` (core/om/) — instant market fills, bracket order stop-loss/profit-taker logic

### Order Types and Lifecycle

**OrderType:** `MARKET`, `BRACKET`, `STOP_LOSS`, `PROFIT_TAKER`
**OrderStatus:** `PENDING` → `PENDING_SALE` (bracket filled, awaiting child trigger) → `FILLED` / `CANCELED`
**OrderAction:** `BUY`, `SELL`

**Bracket Orders:**
- `BracketOrder.create_bracket_order(symbol, price, high_sell_price, low_sell_price, quantity, tx_cost, current_tick=None)` → single `BracketOrder`
- Child orders via `get_child_order("STOP")`, `get_child_order("PROFIT")`, `get_child_order("MANUAL_ORDER")`
- Props: `MANUAL_SALE` (bool), `SOLD_ORDER` (Order ref). Children have `parent_id` → main order
- When one child triggers, the other is auto-canceled

### Engines

**BacktestingEngine** (`engines/backtest_engine.py`): Synchronous backtesting — iterates DataProvider → Algorithm → Portfolio → OM.
- `BacktestingEngine(cfg, dp, al, om, pf).run()` — main entry point
- `on_tick(tick)` — processes one tick through the pipeline; also callable directly as a downstream target

**AlpacaRealTimeEngine** (`engines/alpaca_engine.py`): Live streaming via Alpaca WebSocket.
- `on_tick(tick)` — if `_agg_engine` is set, delegates to it; otherwise calls `_run_pipeline()`
- `_run_pipeline(tick)` — the core algo → portfolio → persist logic; also used as the shim target when aggregation is enabled in live mode
- `_agg_engine` — set by `run.py` when `aggregation.enabled: true`; routes raw ticks through `TickAggregationPassthroughEngine` before the pipeline
- Config keys: `api_key`, `secret_key`, `symbols_to_subscribe`, `subscribe_to_bars/quotes/trades`, `warmup`, `override_url`

**TickAggregationPassthroughEngine** (`engines/tick_aggregation_passthrough_engine.py`): Folds raw ticks into N-min bars before forwarding to a downstream engine.
- Sits between the DataProvider and the real downstream engine (BacktestingEngine or AlpacaRealTimeEngine pipeline)
- **Backtest usage** (`run.py cmd_backtest`): `agg_engine.dp = dp; agg_engine.run()` drives the loop; `BacktestingEngine` is the downstream (never calls `engine.run()`)
- **Live usage** (`run.py cmd_live`): `alpaca_engine._agg_engine = agg_engine` where the downstream is a `SimpleNamespace(on_tick=alpaca_engine._run_pipeline)` shim
- Config keys: `downstream_engine` (required), `aggregation_period_minutes` (default 5), `use_market_open` (default true), `market_open_hour/minute` (default 9:30), `batch_symbols` (default false), `expected_symbols` (list), `batch_timeout_seconds` (default 2.0)
- Window alignment: anchors to market open (9:30); window end = first multiple of period after the anchor that is ≥ tick timestamp. A tick that lands exactly on a boundary is its own window and flushes immediately.
- OHLCV accumulation: open=first tick, high=max, low=min, close=last, volume=sum; aggregated bar timestamp = window end

**AnalysisEngine** (`trading/analysis/analysis_engine.py`): Requires Portfolio with `keep_history=True`.
- `extract_trades()` → `Trade` objects (FIFO buy/sell pairing), `calculate_metrics()` → `PerformanceMetrics` (30+ metrics)
- Returns: `get_tick_returns()`, `get_daily_returns()`, `get_monthly_returns()`
- Plots: `plot_equity_curve()`, `plot_portfolio_with_trades()`, `plot_drawdown()`, `plot_trade_pnl()`, `plot_returns_distribution()`, `plot_stock_performance()`, `plot_interactive_portfolio()` (Plotly), `plot_comprehensive_dashboard()`
- Reports: `generate_report()`, `generate_signals_orders_report()`, `generate_signals_orders_dataframe()` (metadata objects auto-exploded to dot-notation columns)
- `run_full_analysis(log_to_mlflow=True, experiment_name, run_name, parameters, ...)` — complete analysis + MLflow logging (7 PNGs + interactive HTML + signals + DataFrame CSV/Parquet)
- `log_to_mlflow(...)` — manual MLflow logging with fine-grained control

**PortfolioAnalyzer** (`trading/analysis/portfolio_analyzer.py`): Drop-in replacement for AnalysisEngine — same interface, cleaner internals. Preferred for post-mortem analysis of stored sessions.
- `PortfolioAnalyzer(portfolio)` — from in-memory portfolio (backtest/live)
- `PortfolioAnalyzer.from_mongodb(session_id, connection_uri=None, database=None, start=None, end=None)` — reconstruct from a single stored MongoDB session; falls back to `state_store` config for connection details
- `PortfolioAnalyzer.from_mongodb_multi(session_ids, ...)` — merge multiple sessions (e.g. live bot restarted)
- Session metadata (algo class, config params) stored by the engine is automatically included as MLflow parameters — no extra caller work required
- `run_analysis(output_dir, log_to_mlflow=True, ...)` → `{"metrics", "trades", "files"}`
- `run_full_analysis(log_to_mlflow=True, ...)` → same return dict as AnalysisEngine

**WalkForwardEngine** (`engines/walk_forward_engine.py`): Rolling optimization + out-of-sample trading.
- Splits data into overlapping `optimization_window_days` + `trading_window_days` periods
- Each period: HPO via Ray Tune → compare vs current → adopt if improvement > threshold → backtest trading window
- MLflow: parent aggregate run + nested per-period runs
- Config keys: `walk_forward.{optimization_window_days, trading_window_days, improvement_threshold_pct, num_trials, max_concurrent_trials, search_space, algorithm_param_keys, portfolio_param_keys}`

**SelfOptimizingLiveEngine** (`engines/self_optimizing_live_engine.py`): Wraps any `AsyncEngine`, adds periodic background HPO.
- Schedule: `daily` | `weekly` | `monthly`. HPO runs in daemon thread, does not block ticks
- `_hpo_lock` guards reconfigure (~1ms). Only one optimization at a time (`_is_optimizing` flag)
- Config keys: `optimization.{enabled, schedule, rolling_window_days, improvement_threshold_pct, num_trials, search_space, algorithm_param_keys, portfolio_param_keys, historical_data_provider}`

### Utilities

- `utils.utils`: `instantiate_from_string()`, `find_pricedata_in_list()`, `find_marketsignal_in_list()`, `aggregate_stock_data(df, interval)`, `parse_search_space()`, `compute_warmup_start_date(warmup_bars, timeframe, reference_dt)` → datetime lookback start
- `utils/logger.py`: `Logger().get_logger(__name__)` — configured via `config.yaml` logging section. Supports per-call color override: `logger.info("msg", color="magenta")`. Colors: red, bold_red, green, yellow, blue, magenta, cyan, white
- `utils/mlflow_client.py`: `MLflowClient.from_config()` — tracking at `http://z440.lan:5000`. Context manager `start_run()`, log params/metrics/artifacts/charts/text/json/html
- `utils/performance_tracker.py`: Rolling portfolio value tracker used by SelfOptimizingLiveEngine

**HPO search space types** (`parse_search_space()`): `randint`, `uniform`, `choice`, `loguniform`

**Data aggregation** (`aggregate_stock_data`): Multi-symbol OHLCV to any pandas resample frequency. Timestamps = end of interval. Symbols kept separate. OHLCV: first/max/min/last/sum.

### run.py CLI Reference

All subcommands (`backtest`, `live`, `walk-forward`, `hpo`, `session-replay`) share these flags:

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config profile (required) |
| `--symbol` | Override portfolio symbol |
| `--cash` | Override starting cash |
| `--algorithm` | Override algorithm class (dotted path) |
| `--no-mlflow` | Disable MLflow logging |
| `--run-name` | Override analysis run name |
| `--session-id` | MongoDB state_store session ID |
| `--agg-period N` | Override `aggregation.aggregation_period_minutes`; also sets `aggregation.enabled=true` |

`backtest` and `walk-forward` additionally accept `--data` to override the data provider path.
`live` additionally accepts `--alpaca-override-url`.
`hpo` additionally accepts `--num-samples` and `--max-concurrent-trials`.
`session-replay` additionally accepts `--timeframe` (bar size override for sessions missing metadata, e.g. `Minute`, `Hour`, `Day`).

### Directory Structure

```
trading/
  core/
    classes.py                    # Enums/dataclasses: PriceData, MarketSignal, Order, Position, BracketOrder
    algorithm.py                  # Algorithm base class
    multi_timeframe_algorithm.py  # MultiTimeframeAlgorithm base class (aggregates N timeframes in-process)
    portfolio.py                  # Portfolio base class
    pf/                      # Portfolio implementations
      single_symbol_portfolio.py
      dual_symbol_switch_portfolio.py
      long_short_oscillator_portfolio.py
    om/                      # OrderManager implementations
      order_manager.py       # Base class
      backtesting_om.py      # Instant fills, bracket stop/profit logic
      alpaca_om.py           # Live Alpaca order routing
  algorithms/                # Algorithm implementations
    spy_trend_macd_algorithm.py
    macd_rsi_algorithm.py
    spy_trend_switch_algorithm.py
    multi_algorithm.py
    test_algorithm.py
  data_providers/
    data_provider.py         # DataProvider base class
    test_data_provider.py    # CSV reader (truncate optional, defaults to 0)
    alpaca_data_provider.py  # Alpaca historical bars
  engines/
    base_engine.py           # BaseEngine + AsyncEngine base classes
    backtest_engine.py       # Synchronous backtesting engine
    alpaca_engine.py         # Alpaca live engine (_agg_engine / _run_pipeline split)
    tick_aggregation_passthrough_engine.py  # Folds N raw ticks into 1 aggregated bar
    walk_forward_engine.py   # Rolling HPO + out-of-sample backtest
    self_optimizing_live_engine.py  # Live engine + background HPO
    split_period_backtest_engine.py
  analysis/
    analysis_engine.py       # 30+ metrics, charts, MLflow (primary)
    portfolio_analyzer.py    # Drop-in alternative; from_mongodb() support
configs/
  example_backtest.yaml
  example_backtest_agg.yaml            # Backtest with 1-min data + aggregation
  example_live.yaml
  example_live_spy_trend_macd.yaml
  example_live_spy_trend_macd_agg.yaml # Live MACD + aggregation
  example_live_self_optimizing.yaml
  example_walk_forward.yaml
  example_hpo.yaml
utils/
  config_manager.py          # Singleton config loader
  logger.py                  # Singleton logger
  mlflow_client.py           # MLflow tracking client
  performance_tracker.py     # Rolling performance tracker
  trading_state_store.py     # MongoDB session persistence
  utils.py                   # Helpers (instantiate_from_string, find_pricedata_in_list, aggregate_stock_data, parse_search_space, compute_warmup_start_date)
tests/
  unit/
    test_aggregate_stock_data.py      # 18/18
    test_indicators.py                # 66/66
    test_technical_analyzer.py        # 20/20
    test_bracket_order_progression.py # 6/6
    test_portfolio.py                 # 26/26
    test_get_price.py                 # 11/11 — Portfolio.get_price() tick-or-cache lookup
    test_dual_symbol_switch_portfolio.py  # 21/21 — DualSymbolSwitchPortfolio + live trading fallback
    test_macd_calculation.py          # 23/23 — one-shot MACD calculation, EMA, signal generation, reconfigure
    test_macd_algorithm.py            # 6/6
    test_tick_aggregation.py          # 40/40 — TickAggregationPassthroughEngine (window alignment, OHLCV, timing, multi-symbol, BacktestingEngine integration, AlpacaRealTimeEngine routing)
    test_warmup.py                    # 21/21 — Algorithm warmup gate (required_warmup_bars, is_warmed_up, _ticks_seen, signal suppression)
    test_session_replay.py            # 24/24 — compute_warmup_start_date, SessionReplayDataProvider, PortfolioAnalyzer._contribute_to_run
    test_analysis_engine.py           # needs fixtures
scratch/
  run_agg_sweep.py           # Sweep [1,3,5,10,15]-min aggregation periods; logs to MLflow
data/                        # Market data CSV files
run.py                       # Main entry point (backtest / live / hpo / walk-forward / session-replay)
```

### Running Tests

```bash
.venv/Scripts/pytest tests/ -v                    # All tests
.venv/Scripts/pytest tests/unit/ -v               # Unit tests only
# Only passing suites:
.venv/Scripts/pytest tests/unit/test_aggregate_stock_data.py tests/unit/test_indicators.py tests/unit/test_technical_analyzer.py tests/unit/test_bracket_order_progression.py tests/unit/test_portfolio.py tests/unit/test_get_price.py tests/unit/test_dual_symbol_switch_portfolio.py tests/unit/test_macd_calculation.py tests/unit/test_macd_algorithm.py tests/unit/test_tick_aggregation.py tests/unit/test_warmup.py tests/unit/test_session_replay.py -v
# Coverage:
.venv/Scripts/pytest tests/ --cov=core --cov=utils --cov-report=html
```

### Important Implementation Details

**Circular imports:** `core/classes.py` has NO imports from other core modules. Put utility functions referencing core classes in `utils/utils.py`. Inline lookups in classes.py: `next((x for x in tick if x.symbol == symbol), None)`.

**Timestamps:** Use `pd.to_datetime()`, `.dt.tz_localize()` / `.dt.tz_convert()`. DataProvider auto-detects order via `df['timestamp'].iat[0] > df['timestamp'].iat[1]`.

**DataFrame iteration:** `df.itertuples()` (fast), not `df.iterrows()`. Scalar access: `df['col'].iat[index]`.

**Order IDs:** Auto-generated `f"local-{uuid.uuid4()}"`.

**Config format (config.yaml):**
```yaml
logging: { level: INFO, console: true, folder: "logs", filename: "trading.log" }
mlflow: { enabled: true, tracking_uri: "http://z440.lan:5000", experiment_name: "Trading Backtest" }
simulator:
  data_provider: { provider: "data_providers.test_data_provider.TestDataProvider", path: "data/test_data.csv" }
  algorithm: { algorithm: "algorithms.test_algorithm.TestAlgorithm", history_length: 10 }
  order_manager: { order_manager: "core.om.backtesting_om.BacktestingOM" }
  portfolio: { portfolio: "core.pf.single_symbol_portfolio.SingleSymbolPortfolio", symbol: "SPXU", cash: 100000, keep_history: true }
```

**Aggregation config section** (optional, any profile):
```yaml
aggregation:
  enabled: true
  aggregation_period_minutes: 5   # target bar size (1, 3, 5, 10, 15, ...)
  use_market_open: true           # align windows to 9:30 market open
  market_open_hour: 9
  market_open_minute: 30
  # live-only options:
  batch_symbols: true             # wait for all expected symbols before forwarding
  expected_symbols: ["SPY", "UPRO", "SPXU"]
  batch_timeout_seconds: 2.0      # max wait before flushing incomplete batch
```
Can also be activated without editing the config via `--agg-period N` on the CLI.
