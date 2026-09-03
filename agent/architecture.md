# Architecture

Modular trading backtesting and algorithm framework. Event-driven; supports simulation and live trading.

**Pipeline:** DataProvider → Algorithm → Portfolio → OrderManager
- **DataProvider** streams `PriceData` (CSV, APIs, live feeds)
- **Algorithm** emits `MarketSignal` objects
- **Portfolio** turns signals into `Order` objects from cash, positions, and strategy
- **OrderManager** executes orders (simulated or live)

Optional **TickAggregationPassthroughEngine** sits between the DataProvider and the downstream engine and folds 1-min ticks into N-min bars before the algorithm sees them.

Swap components in config with `utils.utils.instantiate_from_string()` (dotted paths).

**Config:** singleton `ConfigManager` in `utils/config_manager.py` reads `config.yaml`.
- `ConfigManager().get_as_object()` — dot-notation
- `.get(key)` — nested dict access
- Every component receives its config dict in `__init__`
