# Walk-forward and live HPO

**WalkForwardEngine** (`engines/walk_forward_engine.py`): rolling optimize + out-of-sample trade.
- Windows: overlapping `optimization_window_days` + `trading_window_days`
- Per period: Ray Tune HPO → adopt if improvement > threshold → backtest trading window
- MLflow: parent aggregate run + nested per-period runs
- Config under `walk_forward.`: `optimization_window_days`, `trading_window_days`, `improvement_threshold_pct`, `num_trials`, `max_concurrent_trials`, `search_space`, `algorithm_param_keys`, `portfolio_param_keys`

**SelfOptimizingLiveEngine** (`engines/self_optimizing_live_engine.py`): wraps `AsyncEngine`; background HPO.
- Schedule: `daily` | `weekly` | `monthly`. Daemon thread; ticks are not blocked
- `_hpo_lock` around reconfigure (~1ms). One run at a time (`_is_optimizing`)
- Config under `optimization.`: `enabled`, `schedule`, `rolling_window_days`, `improvement_threshold_pct`, `num_trials`, `search_space`, `algorithm_param_keys`, `portfolio_param_keys`, `historical_data_provider`

HPO types via `parse_search_space()`: `randint`, `uniform`, `choice`, `loguniform`.
