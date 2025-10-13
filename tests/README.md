# Trading Framework Test Suite

Comprehensive test suite for the trading backtesting and algorithm framework.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Unit tests for individual components
│   ├── test_core_classes.py       # PriceData, Order, Position, MarketSignal
│   ├── test_algorithm.py          # Algorithm base class
│   ├── test_portfolio.py          # Portfolio and SingleSymbolPortfolio
│   ├── test_order_manager.py      # OrderManager and PerfectOm
│   ├── test_data_provider.py      # DataProvider and TestDataProvider
│   └── test_utils.py              # Utility functions
├── integration/             # Integration tests
│   └── test_simulator.py          # End-to-end simulation tests
├── fixtures/                # Test fixtures and helpers
│   └── mock_broker.py             # Mock broker for testing
└── README.md               # This file
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/unit/test_algorithm.py
```

### Run specific test class
```bash
pytest tests/unit/test_algorithm.py::TestAlgorithm
```

### Run specific test method
```bash
pytest tests/unit/test_algorithm.py::TestAlgorithm::test_algorithm_initialization_default
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=core --cov=data_providers --cov=engines --cov=algorithms
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

## Test Coverage

### Unit Tests

#### Core Classes (`test_core_classes.py`)
- ✅ PriceData creation and from_dict method
- ✅ Position creation and modification
- ✅ MarketSignal types and validation
- ✅ Order creation, ID generation, and parent-child relationships
- ✅ All enum types (SignalType, OrderType, OrderAction, OrderStatus)

#### Algorithm (`test_algorithm.py`)
- ✅ Initialization with default and custom configs
- ✅ Price history tracking with length limits
- ✅ Full PriceData history storage
- ✅ Multiple symbol handling
- ✅ Signal generation logic
- ✅ Config merging

#### Portfolio (`test_portfolio.py`)
- ✅ Portfolio base class initialization
- ✅ Order manager integration
- ✅ Processing filled buy/sell orders
- ✅ Order storage and pending order tracking
- ✅ Portfolio value calculation
- ✅ History tracking
- ✅ SingleSymbolPortfolio buy/sell logic
- ✅ Position creation and tracking

#### OrderManager (`test_order_manager.py`)
- ✅ PerfectOm buy/sell order creation
- ✅ Immediate order filling
- ✅ Order ID uniqueness
- ✅ Order status tracking
- ✅ Abstract base class enforcement
- ✅ Price and cash calculations

#### DataProvider (`test_data_provider.py`)
- ✅ TestDataProvider initialization and loading
- ✅ CSV file reading
- ✅ Iterator pattern
- ✅ Chronological ordering
- ✅ Timestamp grouping
- ✅ Data truncation
- ✅ Path resolution (absolute and relative)
- ✅ Multiple symbols per tick

#### Utilities (`test_utils.py`)
- ✅ Dynamic class instantiation
- ✅ Find PriceData in list
- ✅ Find MarketSignal in list
- ✅ Case sensitivity
- ✅ Error handling

### Integration Tests

#### Simulator (`test_simulator.py`)
- ✅ Initialization with objects and config
- ✅ Single symbol simulation
- ✅ Multiple tick processing
- ✅ Order execution
- ✅ Cash tracking
- ✅ Position tracking
- ✅ Portfolio value calculation
- ✅ Algorithm history population
- ✅ Data truncation
- ✅ End-to-end pipeline

## Fixtures

### Shared Fixtures (in `conftest.py`)

- `sample_timestamp` - Fixed datetime for testing
- `sample_pricedata` - Single PriceData object
- `sample_pricedata_list` - List of PriceData for multiple symbols
- `sample_position` - Position object
- `sample_market_signal` - Single MarketSignal
- `sample_market_signals` - List of MarketSignals
- `sample_order` - Filled order
- `sample_pending_order` - Pending order
- `sample_csv_data` - Temporary CSV file with test data
- `algorithm_config` - Algorithm configuration dict
- `portfolio_config` - Portfolio configuration dict
- `data_provider_config` - DataProvider configuration dict

### Test Helpers

#### MockBroker (`fixtures/mock_broker.py`)
Mock broker for testing real-time order management:
- Simulates order execution delays
- Configurable fill probability
- Tracks pending and filled orders
- Useful for testing without actual broker connection

## Writing New Tests

### Example Unit Test
```python
def test_my_feature(sample_pricedata):
    """Test description"""
    # Arrange
    algo = MyAlgorithm()

    # Act
    result = algo.process([sample_pricedata])

    # Assert
    assert result is not None
    assert len(result) > 0
```

### Example Integration Test
```python
def test_end_to_end_scenario(sample_csv_data):
    """Test complete workflow"""
    # Set up components
    om = PerfectOm()
    al = TestAlgorithm({"history_length": 10})
    pf = SingleSymbolPortfolio({
        'symbol': 'AAPL',
        'cash': 100000,
        'keep_history': True
    })
    pf.set_order_manager(om)

    # Run simulation
    cfg = {
        "data_provider": {
            "provider": "data_providers.test_data_provider.TestDataProvider",
            "path": str(sample_csv_data),
            "truncate": 0
        }
    }
    sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
    sim.run()

    # Verify results
    assert len(pf.orders) > 0
    assert pf.total_value > 0
```

## Best Practices

1. **Use fixtures** - Reuse test data via pytest fixtures
2. **Test one thing** - Each test should verify one specific behavior
3. **Use descriptive names** - Test names should explain what they test
4. **AAA pattern** - Arrange, Act, Assert structure
5. **Mock external dependencies** - Don't rely on external services
6. **Test edge cases** - Empty lists, None values, boundary conditions
7. **Keep tests fast** - Unit tests should run in milliseconds

## TODO: Future Test Coverage

- [ ] Real-time engine tests
- [ ] Bracket order tests
- [ ] Stop loss order tests
- [ ] Risk manager tests
- [ ] Multiple portfolio strategies
- [ ] Order cancellation tests
- [ ] Partial fill handling
- [ ] Transaction cost tests
- [ ] Performance benchmarks
- [ ] Multi-symbol portfolio tests

## Dependencies

Required packages for testing:
```bash
pip install pytest pytest-cov
```

## CI/CD Integration

To run tests in CI/CD pipeline:
```yaml
test:
  script:
    - pip install -e .
    - pip install pytest pytest-cov
    - pytest tests/ --cov=core --cov=data_providers --cov=engines --cov-report=xml
```
