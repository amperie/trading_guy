# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a **modular trading backtesting and algorithm framework** with support for both simulation and real-time trading. The framework uses an event-driven architecture where market data flows through configurable components.

### Core Components

**Data Flow Pipeline (Event-Driven):**
1. **DataProvider** → Streams `PriceData` objects from various sources (CSV, APIs, live feeds)
2. **Algorithm** → Analyzes market data and generates `MarketSignal` objects
3. **Portfolio** → Converts signals into `Order` objects based on cash, positions, and strategy
4. **OrderManager** → Executes orders (simulated for backtesting or live via broker APIs)

Each component is designed to be swappable via configuration, enabling easy testing and deployment of different strategies and data sources.

### Key Design Patterns

**Configuration System:**
- Singleton `ConfigManager` reads from `config.yaml` at the root
- Located in `utils/config_manager.py`
- Use `ConfigManager().get_as_object()` for dot-notation access (e.g., `cfg.simulator.path`)
- Use `ConfigManager().get(key)` for dict-style access with nested keys (e.g., `"simulator.data_provider.path"`)
- Also configures logging system (level, format, file/console output)

**Dynamic Class Loading:**
- Components are instantiated via config using `utils.utils.instantiate_from_string()`
- Config specifies full dotted paths (e.g., `"data_providers.test_data_provider.TestDataProvider"`)
- Enables swapping implementations without code changes
- All components receive their config dict on initialization

**Algorithm Base Class:**
- All trading algorithms inherit from `core.algorithm.Algorithm`
- Override `on_data_logic(data: list[PriceData]) -> list[MarketSignal]` to implement strategy
- Automatic history tracking via config: `history_length` and `full_history` options
- Access price history via `self.price_history[symbol]` (deque of closing prices)
- Access full PriceData history via `self.price_data_history[symbol]` (deque of PriceData objects)
- History is automatically managed with configurable maxlen for memory efficiency

**DataProvider Base Class:**
- All data providers inherit from `data_providers.data_provider.DataProvider`
- Override `load_data()` to populate `self.data` (pandas DataFrame with required columns)
- Use `iterate()` generator to yield `list[PriceData]` objects (one list per timestamp)
- Automatically handles forward/backward iteration based on timestamp sorting
- Groups multiple symbols by timestamp for multi-asset strategies

**Portfolio Base Class:**
- All portfolios inherit from `core.portfolio.Portfolio` (abstract base class)
- Override `process_tick_market_signals_logic(signals, tick) -> list[Order]` to implement strategy
- Tracks cash, positions, total value, and order history
- Automatically processes filled orders and updates positions/cash
- Manages pending orders and queries OrderManager for status updates
- Optional history tracking: `tick_history`, `cash_history`, `value_history` (enabled via `keep_history: true`)
- **Implementations:**
  - `SingleSymbolPortfolio` (core/pf/): Manages one symbol, buys with all available cash or sells all positions

**OrderManager Base Class:**
- All order managers inherit from `core.order_manager.OrderManager` (abstract base class)
- Must implement `buy()`, `sell()`, and `get_order_status()` methods
- Handles order execution logic (backtesting vs. live broker APIs)
- **Implementations:**
  - `BacktestingOM` (core/om/): Instantly fills market orders, handles bracket orders with stop-loss/profit-taker logic

### Directory Structure

```
core/                      # Core domain objects and base classes
  classes.py               # Enums and dataclasses (PriceData, MarketSignal, Order, Position)
  algorithm.py             # Algorithm base class with history management
  portfolio.py             # Portfolio base class (signals → orders)
  order_manager.py         # OrderManager base class (order execution interface)
  analysis_engine.py       # Analysis engine (stub for future features)
  pf/                      # Portfolio implementations
    single_symbol_portfolio.py  # Single-symbol portfolio strategy
  om/                      # OrderManager implementations
    backtesting_om.py      # Backtesting order manager (instant fills, bracket order support)

data_providers/            # Data source implementations
  data_provider.py         # Base class with iterate() generator
  test_data_provider.py    # CSV file reader

engines/                   # Execution engines
  simulator.py             # Backtesting engine (orchestrates data → algo → portfolio → OM)
  real_time.py             # Live trading engine (for production deployment)

algorithms/                # Trading algorithm implementations
  test_algorithm.py        # Example algorithm (random buy/sell signals)

utils/                     # Shared utilities
  config_manager.py        # Singleton config loader (reads config.yaml)
  logger.py                # Singleton logger (configured via config.yaml)
  utils.py                 # Helper functions (instantiate_from_string, find_pricedata_in_list, etc.)

tests/                     # Test suite (110/116 passing, 95% coverage)
  unit/                    # Unit tests for individual components
  integration/             # End-to-end simulation tests
  fixtures/                # Test helpers (MockBroker, etc.)
  conftest.py              # Shared pytest fixtures

data/                      # Sample CSV data files
```

### Order Types and Lifecycle

**Supported Order Types (OrderType enum):**
- `MARKET`: Immediate execution at current market price
- `BRACKET`: Entry order with attached stop-loss and profit-taker orders
- `STOP_LOSS`: Sells when price drops to specified level
- `PROFIT_TAKER`: Sells when price rises to specified level

**Order Status Lifecycle (OrderStatus enum):**
1. `PENDING`: Order created but not yet filled
2. `PENDING_SALE`: Bracket order filled, waiting for stop-loss or profit-taker to trigger
3. `FILLED`: Order successfully executed
4. `CANCELED`: Order canceled (e.g., profit-taker canceled when stop-loss triggers)

**Bracket Orders:**
- Create using `Order.create_bracket_order(symbol, price, high_sell_price, low_sell_price, quantity, tx_cost)`
- Returns list of 3 orders: `[main_order, stop_loss_order, profit_order]`
- Main order has `child_orders` list and `child_orders_dict` with keys "STOP" and "PROFIT_TAKER"
- Child orders have `parent_id` pointing to main order's `order_id`
- When one child triggers, the other is automatically canceled
- BacktestingOM handles bracket order logic in `get_order_status()` method

**Order Actions (OrderAction enum):**
- `BUY`: Purchase shares
- `SELL`: Sell shares

### Logging System

**Configuration via config.yaml:**
```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  console: true
  folder: "logs"
  filename: "trading.log"  # Set to null to disable file logging
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**Usage in code:**
```python
from utils.logger import Logger

logger = Logger().get_logger(__name__)
logger.info("Algorithm started")
logger.error("Order failed", exc_info=True)
```

### Running Tests

**Run all tests:**
```bash
pytest tests/
pytest tests/ -v  # Verbose output
```

**Run specific test categories:**
```bash
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/unit/test_portfolio.py  # Specific file
```

**With coverage:**
```bash
pytest tests/ --cov=core --cov=data_providers --cov=engines --cov-report=html
# View report: open htmlcov/index.html
```

**Test Status:** 110/116 tests passing (95% pass rate). See `tests/TEST_SUMMARY.md` for details.

### Important Implementation Details

**Timestamp Handling:**
- Use `pd.to_datetime()` for parsing timestamps
- Set timezone with `.dt.tz_localize()` or `.dt.tz_convert()`
- DataProvider automatically detects chronological order via `df['timestamp'].iat[0] > df['timestamp'].iat[1]`

**DataFrame Iteration:**
- Use `df.itertuples()` (fast) instead of `df.iterrows()` (slow)
- Reverse iteration: `df[::-1].itertuples()`
- Access scalar values: `df['col'].iat[index]` (fastest)

**Config File Format (config.yaml):**
```yaml
logging:
  level: INFO
  console: true
  folder: "logs"
  filename: "trading.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

data_provider:
  provider: data_providers.test_data_provider.TestDataProvider
  path: "data/SPXU.csv"
  truncate: 0  # Number of rows to skip (0 = use all data)

simulator:
  data_provider:
    provider: "data_providers.test_data_provider.TestDataProvider"
    path: "data/test_data.csv"
    truncate: 20
  algorithm:
    algorithm: "algorithms.test_algorithm.TestAlgorithm"
    history_length: 10  # Algorithm-specific config
    full_history: false
  order_manager:
    order_manager: "core.om.backtesting_om.BacktestingOM"
  portfolio:
    portfolio: "core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    symbol: "SPXU"
    cash: 100000
    keep_history: true
  alpaca:  # For live trading (real_time.py)
    api_key: ""
    secret_key: ""
```

### Usage Examples

**Creating a Simple Trading Algorithm:**
```python
from core.algorithm import Algorithm
from core.classes import MarketSignal, PriceData, SignalType

class MyAlgorithm(Algorithm):
    """Simple moving average crossover strategy"""

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []

        for pd in data:
            # Access price history (requires history_length > 0 in config)
            if len(self.price_history[pd.symbol]) < 20:
                continue  # Not enough history yet

            prices = list(self.price_history[pd.symbol])
            sma_short = sum(prices[-5:]) / 5
            sma_long = sum(prices[-20:]) / 20

            if sma_short > sma_long:
                signals.append(MarketSignal(
                    type=SignalType.BUY,
                    symbol=pd.symbol,
                    strength=75
                ))
            elif sma_short < sma_long:
                signals.append(MarketSignal(
                    type=SignalType.SELL,
                    symbol=pd.symbol,
                    strength=75
                ))

        return signals
```

**Running a Backtest:**
```python
from engines.simulator import Simulator
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from core.om.backtesting_om import BacktestingOM

# Option 1: Use config.yaml settings
sim = Simulator(cfg_section_to_use="simulator")
sim.run()

# Option 2: Provide components directly
om = BacktestingOM()
pf = SingleSymbolPortfolio({
    'symbol': 'AAPL',
    'cash': 100000,
    'keep_history': True
})
pf.set_order_manager(om)

algo = MyAlgorithm({'history_length': 20})

cfg = {
    "data_provider": {
        "provider": "data_providers.test_data_provider.TestDataProvider",
        "path": "data/AAPL.csv",
        "truncate": 0
    }
}
sim = Simulator(cfg=cfg, al=algo, om=om, pf=pf)
sim.run()

# Access results
print(f"Final portfolio value: ${pf.total_value:,.2f}")
print(f"Final cash: ${pf.cash:,.2f}")
print(f"Total orders: {len(pf.orders)}")
print(f"Positions: {pf.positions}")
```

**Creating a Custom Portfolio Strategy:**
```python
from core.portfolio import Portfolio
from core.classes import Order, MarketSignal, PriceData, SignalType

class RiskManagedPortfolio(Portfolio):
    """Portfolio with position sizing based on signal strength"""

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> list[Order]:
        orders = []

        for signal in signals:
            pd = find_pricedata_in_list(signal.symbol, tick)
            max_position_size = self.cash * 0.1  # Risk max 10% per trade
            quantity = int(max_position_size / pd.close)

            if signal.type == SignalType.BUY and quantity > 0:
                order = self.order_manager.buy(signal.symbol, quantity, tick)
                orders.append(order)
            elif signal.type == SignalType.SELL and signal.symbol in self.positions:
                qty = self.positions[signal.symbol].quantity
                order = self.order_manager.sell(signal.symbol, qty, tick)
                orders.append(order)

        return orders
```

**Using Bracket Orders:**
```python
from core.classes import Order

# Create bracket order: buy at $100, stop-loss at $95, take-profit at $110
orders = Order.create_bracket_order(
    symbol="AAPL",
    price=100.0,
    low_sell_price=95.0,   # Stop-loss
    high_sell_price=110.0,  # Take-profit
    quantity=100,
    tx_cost=1.0
)

main_order, stop_order, profit_order = orders
print(f"Main order ID: {main_order.order_id}")
print(f"Child orders: {main_order.child_orders}")
print(f"Stop-loss triggers at: ${stop_order.price}")
print(f"Profit-taker triggers at: ${profit_order.price}")
```

### Common Patterns and Best Practices

**1. Use helper functions from utils.utils:**
```python
from utils.utils import find_pricedata_in_list, find_marketsignal_in_list

# Find specific symbol in current tick
price_data = find_pricedata_in_list("AAPL", tick)
signal = find_marketsignal_in_list("AAPL", signals)
```

**2. Portfolio always processes pending orders before new signals:**
- The `process_tick_market_signals()` method automatically calls `_process_pending_orders()` first
- Bracket orders transition from PENDING → PENDING_SALE → FILLED
- Check order status via `OrderManager.get_order_status()`

**3. Algorithm history is automatically updated:**
- Override `on_data_logic()`, not `on_data()`
- `on_data()` is marked `@final` and handles history management
- Configure `history_length` to control deque size (0 = no history)

**4. All orders get unique IDs:**
- Generated automatically: `f"local-{uuid.uuid4()}"`
- Access via `order.order_id`
- Track via `portfolio.orders_by_id` or `portfolio.pending_orders_by_id`

**5. DataProvider CSV format:**
- Required columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- Optional columns: `trade_count`, `vwap`, `exchange`
- Timestamps will be parsed and sorted automatically