# Tests

```bash
.venv/bin/pytest tests/ -v
.venv/bin/pytest tests/unit/ -v
.venv/bin/pytest tests/ --cov=core --cov=utils --cov-report=html
```

On Windows, use `.venv/Scripts/pytest` instead of `.venv/bin/pytest`.

Passing unit suites (run these when changing the matching code):
`test_aggregate_stock_data`, `test_indicators`, `test_technical_analyzer`, `test_bracket_order_progression`, `test_portfolio`, `test_get_price`, `test_dual_symbol_switch_portfolio`, `test_macd_calculation`, `test_macd_algorithm`, `test_tick_aggregation`, `test_warmup`, `test_session_replay`, `test_multi_timeframe_algorithm`.

`tests/unit/test_analysis_engine.py` still needs fixtures. See `agent/TODO.md`.
