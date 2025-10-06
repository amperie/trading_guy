# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a **trading backtesting and algorithm framework** with support for both simulation and real-time trading.

### Core Components

**Data Flow Pipeline:**
1. **DataProvider** → Yields `PriceData` objects from various sources (CSV, APIs, live streams)
2. **Algorithm** → Processes market data and generates `MarketSignal` objects
3. **Portfolio** → Converts signals into `Order` objects based on strategy/priorities
4. **OrderManager** → Executes orders (backtesting or live)

### Key Design Patterns

**Configuration System:**
- Singleton `ConfigManager` reads from `config.yaml` at the root
- Located in `utils/config_manager.py`
- Use `ConfigManager().get_as_object()` for dot-notation access (e.g., `cfg.simulator.path`)
- Use `ConfigManager().get(key)` for dict-style access with nested keys (e.g., `"simulator.data_provider.path"`)

**Dynamic Class Loading:**
- Components are instantiated via config using `utils.utils.instantiate_from_string()`
- Config specifies full dotted paths (e.g., `"data_providers.test_data_provider.TestDataProvider"`)
- Enables swapping implementations without code changes

**Algorithm Base Class:**
- All trading algorithms inherit from `core.algorithm.Algorithm`
- Override `on_data_logic(data: Dict[str, PriceData]) -> list[MarketSignal]`
- Automatic history tracking via config: `history_length` and `full_history` options
- Access price history via `self.price_history[symbol]` (deque of closing prices)
- Access full PriceData history via `self.price_data_history[symbol]`

**DataProvider Base Class:**
- All data providers inherit from `data_providers.data_provider.DataProvider`
- Override `load_data()` to populate `self.data` (pandas DataFrame)
- Use `iterate()` generator to yield `PriceData` objects
- Automatically handles forward/backward iteration based on timestamp sorting

### Directory Structure

```
core/               # Core domain objects and base classes
  classes.py        # Enums (SignalType, OrderType, OrderStatus) and dataclasses (PriceData, MarketSignal, Order)
  algorithm.py      # Algorithm base class with history management
  portfolio.py      # Portfolio management (signals → orders)
  order_manager.py  # Order execution (backtesting/live)

data_providers/     # Data source implementations
  data_provider.py  # Base class with iterate() generator
  test_data_provider.py  # CSV file reader

engines/            # Execution engines
  simulator.py      # Backtesting engine
  real_time.py      # Live trading engine

algorithms/         # Trading algorithm implementations
  test_algorithm.py # Example algorithm stub

utils/              # Shared utilities
  config_manager.py # Singleton config loader
  utils.py          # Dynamic class instantiation

tests/              # Test files and notebooks
```

### Running Tests

Run the main test file:
```bash
python tests/testing.py
```

For Jupyter notebook exploration:
```bash
jupyter notebook tests/data_ingest.ipynb
```

### Important Implementation Details

**Timestamp Handling:**
- Use `pd.to_datetime()` for parsing timestamps
- Set timezone with `.dt.tz_localize()` or `.dt.tz_convert()`
- DataProvider automatically detects chronological order via `df['timestamp'].iat[0] > df['timestamp'].iat[1]`

**DataFrame Iteration:**
- Use `df.itertuples()` (fast) instead of `df.iterrows()` (slow)
- Reverse iteration: `df[::-1].itertuples()`
- Access scalar values: `df['col'].iat[index]` (fastest)

**Config File Format:**
```yaml
data_provider:
  provider: data_providers.test_data_provider.TestDataProvider
  path: "../SPXU.csv"
  truncate: 0

simulator:
  data_provider:
    provider: "data_providers.test_data_provider.TestDataProvider"
    path: "../SPXU.csv"
    truncate: 20
```