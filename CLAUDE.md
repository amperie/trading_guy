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
- Uses backend abstraction pattern with three required methods:
  - `_submit_order_to_backend(order, tick, positions, pf_cash)` → Submit order for execution
  - `_update_order_status_from_backend(order, tick, positions, pf_cash)` → Update single order status
  - `_update_orders_statuses_from_backend(orders, tick, positions, pf_cash)` → Batch update order statuses
- Public API: `submit_order()`, `update_order_status()`, `update_pending_orders()`
- Tracks orders in three dicts: `_all_orders`, `_pending_orders_by_id`, `_filled_orders_by_id`
- Handles order execution logic (backtesting vs. live broker APIs)
- **Implementations:**
  - `BacktestingOM` (core/om/): Instantly fills market orders, handles bracket orders with stop-loss/profit-taker logic

**AnalysisEngine:**
- Located in `engines/analysis_engine.py`
- Provides comprehensive analysis of backtesting results
- Requires Portfolio with `keep_history=True` to access tick/value/cash history
- **Core Features:**
  - **Trade Extraction:** Pairs buy/sell orders using FIFO matching, handles bracket orders
  - **Performance Metrics:** Calculates 30+ metrics (returns, Sharpe, Sortino, drawdown, win rate, etc.)
  - **Returns Analysis:** Get returns at tick, daily, or monthly granularity
  - **Bracket Analysis:** Analyze effectiveness of stop-loss vs profit-taker exits
  - **Visualizations:** Equity curve, drawdown, trade P&L, returns distribution, comprehensive dashboard
  - **Reports:** Generate detailed text reports with all metrics
- **Key Methods:**
  - `extract_trades()` → Returns list of `Trade` objects (entry/exit pairs)
  - `calculate_metrics()` → Returns `PerformanceMetrics` dataclass with 30+ metrics
  - `get_tick_returns()` → Returns pandas Series of tick-level returns
  - `get_daily_returns()` → Returns pandas Series of daily returns (resampled)
  - `get_monthly_returns()` → Returns pandas Series of monthly returns (resampled)
  - `analyze_bracket_effectiveness()` → Returns dict with bracket order statistics
  - `plot_equity_curve()` → Plot portfolio value over time
  - `plot_portfolio_with_trades()` → Plot portfolio value with green X (BUY) and red X (SELL) markers (shows only filled orders with quantity > 0)
  - `plot_drawdown()` → Plot drawdown chart
  - `plot_trade_pnl()` → Plot individual trade P&L
  - `plot_returns_distribution()` → Plot histogram of returns
  - `plot_stock_performance()` → Plot individual stock prices and returns (normalized to 100)
  - `plot_interactive_portfolio()` → **NEW** Interactive Plotly chart with portfolio value, cash, trades, and stock prices (zoomable, clickable legend, shows only quantity > 0)
  - `plot_comprehensive_dashboard()` → Multi-panel dashboard with key metrics
  - `generate_report()` → Returns formatted text report string
  - `log_to_mlflow(experiment_name, run_name, description, tags, parameters, ...)` → Log all analysis results to MLflow with optional experiment name override (includes interactive chart)
  - `run_full_analysis(log_to_mlflow=True, experiment_name, ...)` → Run complete analysis and optionally log to MLflow with custom experiment (includes interactive chart)
- **Example Usage:**
  ```python
  from engines.analysis_engine import AnalysisEngine

  # Option 1: Manual analysis steps
  engine = AnalysisEngine(portfolio, order_manager)
  trades = engine.extract_trades()
  metrics = engine.calculate_metrics()
  tick_returns = engine.get_tick_returns()
  daily_returns = engine.get_daily_returns()
  report = engine.generate_report()

  # Generate visualizations
  engine.plot_equity_curve(save_path="equity_curve.png")
  engine.plot_portfolio_with_trades(save_path="portfolio_trades.png")
  engine.plot_comprehensive_dashboard(save_path="dashboard.png")

  # Option 2: Complete analysis with MLflow (recommended)
  engine = AnalysisEngine(portfolio, order_manager)
  results = engine.run_full_analysis(
      experiment_name="SMA Strategy Tests",  # Optional: override config experiment name
      run_name="SMA Crossover Strategy",
      description="Testing 5/20 SMA crossover on AAPL",
      parameters={"sma_short": 5, "sma_long": 20, "symbol": "AAPL"},
      log_to_mlflow=True,  # Logs 7 PNG charts + 1 interactive HTML chart + metrics + trades
      save_charts_locally=True,
      save_report_locally=True
  )
  # Results dict contains: trades, metrics, tick_returns, daily_returns, monthly_returns, bracket_analysis, report
  # MLflow artifacts: 7 static charts, interactive_portfolio.html, trades.json, reports

  # Option 3: Manual MLflow logging for fine-grained control
  engine = AnalysisEngine(portfolio, order_manager)
  engine.extract_trades()
  engine.calculate_metrics()
  engine.log_to_mlflow(
      experiment_name="Custom Experiments",  # Optional: override config experiment name
      run_name="Custom Run",
      parameters={"custom_param": "value"},
      log_charts=True,  # Includes interactive_portfolio.html + 7 static charts
      log_trades=True,
      chart_dpi=200
  )
  # View interactive chart in MLflow UI: http://hp.lan:8899
  ```

**MLflowClient:**
- Located in `utils/mlflow_client.py`
- Provides experiment tracking and logging for trading backtests
- Reads configuration from `config.yaml` under `mlflow` section
- **Core Features:**
  - **Run Management:** Start/stop runs with context manager support
  - **Parameter Logging:** Log algorithm and strategy parameters
  - **Metric Logging:** Log performance metrics (returns, Sharpe, win rate, etc.)
  - **Artifact Logging:** Save charts, reports, JSON data, HTML, markdown, and text files
  - **System Info:** Automatically log system and environment details
  - **Remote Tracking:** Connect to remote MLflow server (configured at hp.lan:8899)
- **Configuration (config.yaml):**
  ```yaml
  mlflow:
    enabled: true
    tracking_uri: "http://hp.lan:8899"
    experiment_name: "Trading Backtest"
    artifact_location: null
    run_name_prefix: ""
    auto_log_system_info: true
  ```
- **Key Methods:**
  - `from_config()` → Create client from config.yaml settings
  - `start_run(run_name, description, tags)` → Start new run
  - `end_run(status)` → End current run
  - `log_param(key, value)` / `log_params(dict)` → Log parameters
  - `log_metric(key, value, step)` / `log_metrics(dict, step)` → Log metrics
  - `log_text(text, filename)` → Log text artifact
  - `log_json(data, filename)` → Log JSON artifact
  - `log_markdown(markdown, filename)` → Log markdown artifact
  - `log_html(html, filename)` → Log HTML artifact
  - `log_chart(figure, filename, format, dpi)` → Log matplotlib/plotly chart
  - `log_artifact(local_path)` → Log existing file
  - `set_tag(key, value)` / `set_tags(dict)` → Set run tags
  - `log_model_info(dict)` → Log model/strategy metadata
  - `get_run_url()` → Get MLflow UI URL for current run
- **Example Usage:**
  ```python
  from utils.mlflow_client import MLflowClient
  from engines.analysis_engine import AnalysisEngine

  # Create client from config
  mlflow = MLflowClient.from_config()

  # Use context manager for automatic run management
  with mlflow.start_run(
      run_name="SMA Crossover Strategy",
      description="Testing 5/20 SMA crossover on AAPL"
  ):
      # Log algorithm parameters
      mlflow.log_params({
          "symbol": "AAPL",
          "sma_short": 5,
          "sma_long": 20,
          "initial_capital": 100000
      })

      # Run backtest
      sim.run()

      # Calculate metrics
      engine = AnalysisEngine(portfolio, order_manager)
      metrics = engine.calculate_metrics()

      # Log performance metrics
      mlflow.log_metrics({
          "total_return_pct": metrics.total_return_pct,
          "sharpe_ratio": metrics.sharpe_ratio,
          "max_drawdown_pct": metrics.max_drawdown_pct,
          "win_rate": metrics.win_rate,
          "total_trades": metrics.total_trades
      })

      # Log visualizations
      fig = engine.plot_equity_curve(show=False)
      mlflow.log_chart(fig, "equity_curve", format="png")

      fig = engine.plot_portfolio_with_trades(show=False)
      mlflow.log_chart(fig, "portfolio_trades", format="png")

      # Log text report
      report = engine.generate_report()
      mlflow.log_text(report, "performance_report.txt")

      # Log trades as JSON
      trades = engine.extract_trades()
      trades_data = [{"symbol": t.symbol, "pnl": t.pnl, "pnl_pct": t.pnl_pct} for t in trades]
      mlflow.log_json(trades_data, "trades.json")

      # View run in MLflow UI
      print(f"View run: {mlflow.get_run_url()}")
  ```

### Directory Structure

```
core/                      # Core domain objects and base classes
  classes.py               # Enums and dataclasses (PriceData, MarketSignal, Order, Position)
  algorithm.py             # Algorithm base class with history management
  portfolio.py             # Portfolio base class (signals → orders)
  order_manager.py         # OrderManager base class (order execution interface)
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
  analysis_engine.py       # Comprehensive backtesting analysis engine (30+ metrics, visualizations, reports)

algorithms/                # Trading algorithm implementations
  test_algorithm.py        # Example algorithm (random buy/sell signals)

utils/                     # Shared utilities
  config_manager.py        # Singleton config loader (reads config.yaml)
  logger.py                # Singleton logger (configured via config.yaml)
  mlflow_client.py         # MLflow experiment tracking client (logs runs, metrics, artifacts)
  utils.py                 # Helper functions (instantiate_from_string, find_pricedata_in_list, aggregate_stock_data, etc.)

tests/                     # Test suite (123 tests, 91 passing)
  README.md                # Comprehensive test suite documentation
  unit/                    # Unit tests for individual components
    test_aggregate_stock_data.py          # Data aggregation tests (18/18 passing)
    test_indicators.py                    # Technical indicators (66/66 passing)
    test_technical_analyzer.py            # Analyzer API (20/20 passing)
    test_bracket_order_progression.py     # Bracket orders (6/6 passing)
    test_portfolio.py                     # Portfolio management (needs fixtures)
    test_analysis_engine.py               # Analysis engine (needs fixtures)

scratch/                   # Development notebooks and experiments
  data_ingest.ipynb        # Data ingestion experiments
  plots.ipynb              # Visualization experiments
  scratch.ipynb            # General scratch notebook
  testing.ipynb            # Interactive testing notebook
  om/                      # Order manager experiments

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
- Create using `BracketOrder.create_bracket_order(symbol, price, high_sell_price, low_sell_price, quantity, tx_cost, current_tick=None)`
- Returns a single `BracketOrder` object (not a list)
- Main order has child orders accessible via `get_child_order(name)` method:
  - `"STOP"` - Stop-loss order (triggers when price <= stop_price)
  - `"PROFIT"` - Profit-taker order (triggers when price >= profit_price)
  - `"MANUAL_ORDER"` - Manual exit order (created when MANUAL_SALE flag is set)
- BracketOrder properties:
  - `MANUAL_SALE` (bool) - Flag to trigger manual exit at market price
  - `SOLD_ORDER` (Order) - Reference to the child order that completed the sale
- Child orders have `parent_id` pointing to main order's `order_id`
- When one child triggers, the other is automatically canceled
- BacktestingOM handles bracket order logic in `_update_order_status_from_backend()` method

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
pytest tests/                    # All tests
pytest tests/ -v                 # Verbose output
pytest tests/unit/               # Unit tests only
```

**Run specific test files:**
```bash
pytest tests/unit/test_aggregate_stock_data.py -v     # Data aggregation (18 tests)
pytest tests/unit/test_indicators.py -v               # Technical indicators (66 tests)
pytest tests/unit/test_bracket_order_progression.py -v  # Bracket orders (6 tests)
```

**Run only passing tests:**
```bash
pytest tests/unit/test_aggregate_stock_data.py tests/unit/test_indicators.py tests/unit/test_technical_analyzer.py tests/unit/test_bracket_order_progression.py -v
```

**With coverage:**
```bash
pytest tests/ --cov=core --cov=utils --cov-report=html
# View report: open htmlcov/index.html
```

**Test Status:** 91/123 tests passing (74%). See `tests/README.md` for detailed test documentation.

**Passing Test Suites:**
- `test_aggregate_stock_data.py`: 18/18 - Multi-symbol OHLCV aggregation, timestamp handling, volume preservation
- `test_indicators.py`: 66/66 - EMA, MACD, RSI calculations with real data validation
- `test_technical_analyzer.py`: 20/20 - High-level API wrapper, backward compatibility
- `test_bracket_order_progression.py`: 6/6 - Order lifecycle, stop-loss/profit-taker triggers

**Needs Fixtures:**
- `test_portfolio.py`: 0/23 - Portfolio and order management
- `test_analysis_engine.py`: 1/10 - Backtesting performance analysis

### Important Implementation Details

**Avoiding Circular Imports:**
- `core/classes.py` contains only dataclasses and enums - NO imports from other core modules
- Utility functions that reference core classes should go in `utils/utils.py`
- If you need to find a PriceData in a list within classes.py, inline the logic:
  ```python
  pd = next((x for x in tick if x.symbol == symbol), None)
  ```
- This prevents circular dependencies: `core.classes` ← `utils.utils` ← `core.classes` ❌

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

mlflow:
  enabled: true
  tracking_uri: "http://hp.lan:8899"
  experiment_name: "Trading Backtest"
  artifact_location: null
  run_name_prefix: ""
  auto_log_system_info: true

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
from core.om.backtesting_om_old import BacktestingOM

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

### Data Aggregation Utilities

**Stock Data Aggregation (`utils.utils.aggregate_stock_data`):**
- Aggregates multi-symbol OHLCV data to different time granularities
- Keeps symbols separated (no data mixing across symbols)
- Timestamps represent the END of the interval (right edge)
  - Example: 5-minute interval from 9:26-9:30 has timestamp 9:30
- Proper OHLCV handling:
  - Open: first value in interval
  - High: max value in interval
  - Low: min value in interval
  - Close: last value in interval
  - Volume: sum of all values in interval
- Automatically removes empty intervals (gaps in data)
- **Usage:**
```python
from utils.utils import aggregate_stock_data

# Load 1-minute data
df_1min = pd.read_csv('data/SPXU_GDXU_UPRO_1min.csv')

# Aggregate to different timeframes
df_5min = aggregate_stock_data(df_1min, interval='5min')
df_15min = aggregate_stock_data(df_1min, interval='15min')
df_hourly = aggregate_stock_data(df_1min, interval='1h')
df_daily = aggregate_stock_data(df_1min, interval='1D')

# With custom aggregation rules
custom_rules = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'vwap': 'mean'  # Custom field aggregation
}
df_custom = aggregate_stock_data_custom(df_1min, interval='5min', agg_rules=custom_rules)
```

- **Supported intervals:** Any pandas resample frequency string
  - Minutes: `'5min'`, `'15min'`, `'30min'`
  - Hours: `'1h'`, `'4h'`
  - Days: `'1D'`
  - Weeks: `'1W'`
  - Months: `'1M'`

- **Data preservation:**
  - 100% volume preservation (no data loss)
  - All OHLC relationships maintained (High ≥ Low, etc.)
  - Symbol separation guaranteed
  - Test coverage: 18/18 tests passing (100%)

- **Performance:**
  - 1.5M rows → 437K rows (5min): 72.4% reduction
  - 1.5M rows → 165K rows (15min): 89.6% reduction
  - 1.5M rows → 45K rows (1hour): 97.2% reduction
  - 1.5M rows → 3K rows (1day): 99.8% reduction