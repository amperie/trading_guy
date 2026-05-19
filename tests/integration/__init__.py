"""
Integration tests for trading framework.

Integration tests verify components working together with real external services
and data files. These tests may require:
- API credentials (stored in .env file)
- Network connectivity
- External services to be available (Alpaca, MLflow)
- Data files (data/SPY_UPRO_SPXU_5min_sorted.csv)

Test Files:
- test_alpaca_om_integration.py: Alpaca API connection and order execution (10 tests)
- test_bracket_holding_period.py: Full backtest with bracket orders (3 tests)
- test_ray_imports.py: Ray library import smoke tests (3 tests)
- test_signals_debug.py: Signal generation end-to-end (1 test)
Total: 17 integration tests

Run with: pytest tests/integration/ -v
"""
