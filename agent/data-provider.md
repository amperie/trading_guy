# DataProvider

Base: `data_providers.data_provider.DataProvider`.

Override `load_data()` to fill `self.data` as a DataFrame with columns `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`. Optional: `trade_count`, `vwap`, `exchange`.

`iterate()` yields `list[PriceData]` per timestamp, auto-detects chronological order, and groups by timestamp.

**TestDataProvider:** `path` required; `truncate` optional (default 0 = all rows); `start_date` / `end_date` optional.
