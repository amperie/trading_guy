# Implementation constraints

**Circular imports:** `core/classes.py` must not import other core modules. Put helpers that reference core classes in `utils/utils.py`. Inline lookup in classes.py: `next((x for x in tick if x.symbol == symbol), None)`.

**Timestamps:** `pd.to_datetime()`, `.dt.tz_localize()` / `.dt.tz_convert()`. DataProvider order: `df['timestamp'].iat[0] > df['timestamp'].iat[1]`.

**DataFrames:** `df.itertuples()`, not `df.iterrows()`. Scalars: `df['col'].iat[index]`.

**Config shape:** `logging`, `mlflow`, then `simulator` with `data_provider`, `algorithm`, `order_manager`, `portfolio` dicts (each has a class dotted path). See example profiles under `configs/`.

**Aggregation** (optional, any profile): `enabled`, `aggregation_period_minutes`, `use_market_open`, `market_open_hour/minute`. Live-only: `batch_symbols`, `expected_symbols`, `batch_timeout_seconds`. Also `--agg-period N` on the CLI.
