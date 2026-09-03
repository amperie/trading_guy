# Portfolio

Base: `core.portfolio.Portfolio`.

Override `process_tick_market_signals_logic(signals, tick) -> TickResults`.
`process_market_signals_for_tick()` is `@final` — it updates pending OM orders, syncs the broker, and refreshes portfolio value.

`get_price(symbol, tick) -> float | None` is `@final`: tick first, then `self.previous_price`. Required for live ticks that carry a single symbol.

Tracks: `cash`, `positions`, `total_value`, `previous_price`.
With `keep_history: true`: `tick_history`, `cash_history`, `value_history`, `signals_history`.

`reconfigure(new_params)` keeps cash, positions, and history.

Impl: `SingleSymbolPortfolio` (`core/pf/`) — all-in buy, sell all.
Impl: `DualSymbolSwitchPortfolio` (`core/pf/`) — two symbols (e.g. UPRO/SPXU), brackets, holding periods, manual sale switch.
