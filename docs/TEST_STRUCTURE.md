# Test Directory Structure

## Overview

The test suite is organized into **unit tests** (fast, isolated) and **integration tests** (slower, with external services).

## Directory Tree

```
tests/
├── __init__.py                          # Package initialization
├── README.md                            # Comprehensive test documentation
├── TEST_STRUCTURE.md                    # This file - directory organization
├── download_test_data.py                # Utility to download test data
│
├── unit/                                # Unit tests (fast, no external dependencies)
│   ├── __init__.py
│   ├── test_aggregate_stock_data.py     # ✅ 18/18 - Data aggregation utilities
│   ├── test_alpaca_om.py                # ✅ 18/18 - AlpacaOrderManager (mocked API)
│   ├── test_analysis_engine.py          # ⏸️  6/10 - Performance analysis
│   ├── test_bracket_order_progression.py # ✅ 6/6  - Bracket order lifecycle
│   ├── test_indicators.py               # ✅ 66/66 - Technical indicators (EMA, MACD, RSI)
│   ├── test_portfolio.py                # ⏸️ 22/23 - Portfolio management
│   └── test_technical_analyzer.py       # ✅ 20/20 - Technical analyzer API wrapper
│
└── integration/                         # Integration tests (requires external services)
    ├── __init__.py
    └── test_alpaca_om_integration.py    # ✅ 9/10 - AlpacaOrderManager with real Alpaca API
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test components in isolation with mocked dependencies
**Speed**: Fast (< 5 seconds total)
**Requirements**: None - all dependencies mocked
**Run**: `pytest tests/unit/ -v`

| File | Tests | Status | Description |
|------|-------|--------|-------------|
| `test_aggregate_stock_data.py` | 18/18 | ✅ 100% | OHLCV data aggregation |
| `test_alpaca_om.py` | 18/18 | ✅ 100% | AlpacaOrderManager (mocked) |
| `test_bracket_order_progression.py` | 6/6 | ✅ 100% | Bracket order state machine |
| `test_indicators.py` | 66/66 | ✅ 100% | EMA, MACD, RSI indicators |
| `test_technical_analyzer.py` | 20/20 | ✅ 100% | Technical analyzer wrapper |
| `test_analysis_engine.py` | 6/10 | ⏸️ 60% | Performance analysis (needs fixtures) |
| `test_portfolio.py` | 22/23 | ⏸️ 96% | Portfolio management (1 edge case) |

**Total Unit Tests**: 156/161 passing (97%)

### Integration Tests (`tests/integration/`)

**Purpose**: Test components with real external services
**Speed**: Slower (10-30 seconds)
**Requirements**:
- Alpaca API credentials in `.env` file
- Network connectivity
- External services available

**Run**: `pytest tests/integration/ -v -m integration`

| File | Tests | Status | Description |
|------|-------|--------|-------------|
| `test_alpaca_om_integration.py` | 9/10 | ✅ 90% | AlpacaOrderManager with real Alpaca API |

**Total Integration Tests**: 9/10 passing (90%) + 1 skipped

## Key Features

### AlpacaOrderManager Tests ⭐

**Unit Tests (Mocked API)**:
- Initialization & configuration
- Market order submission (BUY/SELL)
- Bracket order submission
- Order status updates
- Order cancellation
- Edge cases & error handling

**Integration Tests (Real API)**:
- ✅ **Independent verification**: Every test queries Alpaca API to prove orders exist
- ✅ **Bracket order validation**: Verifies STOP and PROFIT child orders captured locally
- ✅ **Platform ID mapping**: Verifies local order IDs mapped to Alpaca IDs
- ✅ **Child order verification**: Independently queries each bracket leg in Alpaca

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Unit Tests Only
```bash
pytest tests/unit/ -v
```

### Integration Tests Only
```bash
pytest tests/integration/ -v -m integration
```

### Specific Component
```bash
# AlpacaOrderManager only
pytest tests/ -k alpaca -v

# Indicators only
pytest tests/unit/test_indicators.py -v

# Bracket orders only
pytest tests/unit/test_bracket_order_progression.py -v
```

### With Coverage
```bash
pytest tests/ --cov=core --cov=utils --cov-report=html
open htmlcov/index.html
```

## Test Markers

Tests are marked for selective execution:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (requires external services)
- `@pytest.mark.slow` - Slow tests (> 5 seconds)

**Run by marker**:
```bash
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
```

## Setup for Integration Tests

### Alpaca API Setup

1. **Get Credentials**:
   - Sign up at [alpaca.markets](https://alpaca.markets) (free)
   - Navigate to Paper Trading section
   - Generate API keys (start with `PK` and `PS`)

2. **Configure Locally**:
   ```bash
   # Copy template
   cp .env.example .env

   # Edit .env
   ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
   ALPACA_SECRET_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

3. **Run Tests**:
   ```bash
   pytest tests/integration/ -v -m integration
   ```

**Files**:
- `.env.example` - Template (committed to git)
- `.env` - Your credentials (gitignored, safe)
- `ALPACA_SETUP.md` - Detailed setup guide

## Test Statistics

- **Total Tests**: 151
- **Passing**: 146 (97%)
- **Failing**: 4 (3%)
- **Skipped**: 1 (integration - market hours)

**By Category**:
- Unit Tests: 156/161 (97%)
- Integration Tests: 9/10 (90%) + 1 skipped

**By Component**:
- Data Aggregation: 18/18 ✅
- Technical Indicators: 66/66 ✅
- Technical Analyzer: 20/20 ✅
- Bracket Orders: 6/6 ✅
- AlpacaOrderManager: 27/28 ✅ (1 skipped)
- Portfolio: 22/23 ⏸️
- Analysis Engine: 6/10 ⏸️

## Contributing Tests

### Adding Unit Tests

1. Create file in `tests/unit/`
2. Name file `test_[component].py`
3. Use fixtures for test data
4. Mock external dependencies
5. Follow AAA pattern (Arrange, Act, Assert)

### Adding Integration Tests

1. Create file in `tests/integration/`
2. Name file `test_[component]_integration.py`
3. Mark with `@pytest.mark.integration`
4. Document external requirements
5. Clean up resources in teardown

### Test Naming Convention

```python
def test_[action]_[expected_result]():
    """
    [One-line description]

    [Optional detailed explanation]
    """
```

Examples:
- `test_submit_market_buy_order()`
- `test_bracket_stop_loss_trigger()`
- `test_invalid_credentials_raises_error()`

## Documentation

- **README.md** - Comprehensive test documentation with examples
- **TEST_STRUCTURE.md** - This file (directory organization)
- **ALPACA_SETUP.md** - Integration test setup guide
- **pytest.ini** - Pytest configuration and markers

## Support

For questions about tests:
- See `tests/README.md` for detailed test documentation
- See `ALPACA_SETUP.md` for integration test setup
- See `agent/AGENTS.md` for framework architecture
