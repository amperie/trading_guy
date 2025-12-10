"""
Trading Backtest Framework - Test Suite

This package contains comprehensive tests for all framework components.

Test Organization:
- unit/: Unit tests for individual components (123 tests)
  - test_aggregate_stock_data.py: Data aggregation utilities (18/18 passing ✅)
  - test_analysis_engine.py: Backtesting analysis engine (6/10 passing - fixture needs completed trades)
  - test_bracket_order_progression.py: Bracket order lifecycle (6/6 passing ✅)
  - test_indicators.py: Technical indicators (EMA, MACD, RSI) (66/66 passing ✅)
  - test_portfolio.py: Portfolio management (22/23 passing - 1 edge case)
  - test_technical_analyzer.py: Technical analyzer wrapper (20/20 passing ✅)

Test Status: 118/123 passing (96%)

Running Tests:
    pytest tests/                    # Run all tests
    pytest tests/unit/              # Run unit tests only
    pytest tests/ -v                # Verbose output
    pytest tests/ --cov=core        # With coverage

Test Data:
    data/SPXU_GDXU_UPRO_1min.csv   # Real market data (1.5M rows, 3 symbols)

Known Issues (5 tests):
    - 1 portfolio test: Bracket order filling edge case
    - 4 analysis engine tests: Fixture requires completed trade lifecycle
"""
