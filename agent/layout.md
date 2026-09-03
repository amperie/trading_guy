# Repository layout

Catalogue of source locations. Not an instruction file (no 30-line cap).

```
trading/
  core/
    classes.py                    # PriceData, MarketSignal, Order, Position, BracketOrder
    algorithm.py                  # Algorithm base
    multi_timeframe_algorithm.py  # MultiTimeframeAlgorithm
    portfolio.py                  # Portfolio base
    pf/                           # Portfolio implementations
      single_symbol_portfolio.py
      dual_symbol_switch_portfolio.py
      long_short_oscillator_portfolio.py
    om/                           # OrderManager implementations
      order_manager.py            # Base
      backtesting_om.py           # Instant fills, bracket stop/profit
      alpaca_om.py                # Live Alpaca routing
  algorithms/
    spy_trend_macd_algorithm.py
    macd_rsi_algorithm.py
    spy_trend_switch_algorithm.py
    multi_algorithm.py
    test_algorithm.py
  data_providers/
    data_provider.py
    test_data_provider.py         # CSV; truncate default 0
    alpaca_data_provider.py
  engines/
    base_engine.py                # BaseEngine, AsyncEngine
    backtest_engine.py
    alpaca_engine.py
    tick_aggregation_passthrough_engine.py
    walk_forward_engine.py
    self_optimizing_live_engine.py
    split_period_backtest_engine.py
  analysis/
    analysis_engine.py
    portfolio_analyzer.py
configs/
  example_backtest.yaml
  example_backtest_agg.yaml
  example_live.yaml
  example_live_spy_trend_macd.yaml
  example_live_spy_trend_macd_agg.yaml
  example_live_self_optimizing.yaml
  example_walk_forward.yaml
  example_hpo.yaml
utils/
  config_manager.py
  logger.py
  mlflow_client.py
  performance_tracker.py
  trading_state_store.py
  utils.py
tests/
  unit/                           # see agent/tests.md and tests/README.md
scratch/
  run_agg_sweep.py
data/                             # Market data CSVs
run.py                            # backtest / live / hpo / walk-forward / session-replay
```
