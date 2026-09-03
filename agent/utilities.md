# Utilities

`utils.utils`: `instantiate_from_string()`, `find_pricedata_in_list()`, `find_marketsignal_in_list()`, `aggregate_stock_data(df, interval)`, `parse_search_space()`, `compute_warmup_start_date(warmup_bars, timeframe, reference_dt)`.

`utils/logger.py`: `Logger().get_logger(__name__)`. Config from `config.yaml` logging. Per-call color: `logger.info("msg", color="magenta")`. Colors: red, bold_red, green, yellow, blue, magenta, cyan, white.

`utils/mlflow_client.py`: `MLflowClient.from_config()` — `http://z440.lan:5000`. Context manager `start_run()`; log params, metrics, artifacts, charts, text, json, html.

`utils/performance_tracker.py`: rolling portfolio value; used by SelfOptimizingLiveEngine.

`aggregate_stock_data`: multi-symbol OHLCV to any pandas resample frequency. Timestamps = end of interval. Symbols stay separate. OHLCV: first/max/min/last/sum.
