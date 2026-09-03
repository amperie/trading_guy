# MultiTimeframeAlgorithm

Base: `core.multi_timeframe_algorithm.MultiTimeframeAlgorithm`. Extends `Algorithm` (warmup, history, `reconfigure` inherited).

Override `on_mtf_data(tick, new_bars) -> list[MarketSignal]`. Do not override `on_data_logic()` (`@final` here).

Config: `timeframes` (required `list[int]` of minute periods) plus all `Algorithm` keys.

- `self.bar_history[period_minutes][symbol]` — deque of completed `PriceData` bars, maxlen=`history_length`
- `new_bars: dict[int, list[PriceData]]` — sparse; key present only when a bar completed this tick
- `required_warmup_bars` defaults to `history_length × max(timeframes)`; override to change it

Window alignment matches `TickAggregationPassthroughEngine`: anchors to market open; boundary ticks flush immediately.
