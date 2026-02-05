"""
Trading Backtest Framework - Test Suite

This package contains comprehensive tests for all framework components.

Test Organization:
- unit/: Unit tests for individual components (158 tests)
  - test_aggregate_stock_data.py: Data aggregation utilities (18 tests)
  - test_alpaca_om.py: Alpaca order manager (18 tests)
  - test_analysis_engine.py: Backtesting analysis engine (10 tests)
  - test_bracket_order_progression.py: Bracket order lifecycle (6 tests)
  - test_ema_function.py: EMA/SMA calculations (7 tests)
  - test_indicators.py: Technical indicators - EMA, MACD, RSI (50 tests)
  - test_macd_algorithm.py: MACD algorithm signal generation (6 tests)
  - test_portfolio.py: Portfolio management (23 tests)
  - test_technical_analyzer.py: Technical analyzer API (16 tests)
  - test_tick_aggregation_passthrough_engine.py: Tick aggregation engine (4 tests)

- integration/: Integration tests with external services (29 tests)
  - test_alpaca_om_integration.py: Alpaca API integration (10 tests)
  - test_bracket_holding_period.py: Bracket orders with holding period (3 tests)
  - test_ray_imports.py: Ray library import smoke tests (3 tests)
  - test_signals_debug.py: Signal generation end-to-end (1 test)
  - test_split_period_engine.py: Split-period backtest engine (9 tests)
  - test_split_period_simple.py: Simple split-period backtest (3 tests)

Total: 187 tests (158 unit + 29 integration)

Running Tests:
    pytest tests/                    # Run all tests
    pytest tests/unit/               # Run unit tests only
    pytest tests/integration/        # Run integration tests only
    pytest tests/ -v                 # Verbose output
    pytest tests/ --cov=trading      # With coverage
"""
