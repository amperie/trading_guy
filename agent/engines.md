# Engines (backtest, live, aggregation)

**BacktestingEngine** (`engines/backtest_engine.py`): DataProvider → Algorithm → Portfolio → OM.
- `BacktestingEngine(cfg, dp, al, om, pf).run()`
- `on_tick(tick)` — one tick; also a downstream target

**AlpacaRealTimeEngine** (`engines/alpaca_engine.py`): Alpaca WebSocket.
- `on_tick(tick)` — delegates to `_agg_engine` if set; else `_run_pipeline()`
- `_run_pipeline(tick)` — algo → portfolio → persist; aggregation shim target
- `_agg_engine` — set by `run.py` when `aggregation.enabled: true`
- Config: `api_key`, `secret_key`, `symbols_to_subscribe`, `subscribe_to_bars/quotes/trades`, `warmup`, `override_url`

**TickAggregationPassthroughEngine** (`engines/tick_aggregation_passthrough_engine.py`): ticks → N-min bars.
- Backtest: `agg_engine.dp = dp; agg_engine.run()`; BacktestingEngine is downstream (do not call `engine.run()`)
- Live: `alpaca_engine._agg_engine = agg_engine`; downstream `SimpleNamespace(on_tick=alpaca_engine._run_pipeline)`
- Config: `downstream_engine` (required), `aggregation_period_minutes` (5), `use_market_open` (true), `market_open_hour/minute` (9:30), `batch_symbols` (false), `expected_symbols`, `batch_timeout_seconds` (2.0)
- Anchor windows at market open; boundary ticks flush immediately. OHLCV: first/max/min/last/sum; bar timestamp = window end
