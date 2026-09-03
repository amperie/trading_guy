# Algorithm

Base: `core.algorithm.Algorithm`.

Override `on_data_logic(data: list[PriceData]) -> list[MarketSignal]`. Do not override `on_data()` (`@final`).

Config: `history_length` (deque maxlen), `full_history` (bool).
State: `self.price_history[symbol]` (closes), `self.price_data_history[symbol]` (`PriceData`).

Warmup: `on_data()` always calls `on_data_logic()` (builds state) but returns `[]` until `is_warmed_up`.
- `_ticks_seen` increments on every `on_data()` call
- `required_warmup_bars` defaults to `history_length`; override to raise the threshold
- `is_warmed_up` is True when `_ticks_seen >= required_warmup_bars`
- Session replay is one stream; signals are suppressed during warmup

`reconfigure(new_params)` updates params without dropping history. If `__init__` caches config:

    def reconfigure(self, new_params):
        super().reconfigure(new_params)
        for attr in ("my_param",):
            if attr in new_params:
                setattr(self, attr, new_params[attr])
