# Analysis

**AnalysisEngine** (`trading/analysis/analysis_engine.py`): needs Portfolio `keep_history=True`.
- `extract_trades()` → FIFO `Trade` objects; `calculate_metrics()` → `PerformanceMetrics` (30+)
- Returns: `get_tick_returns()`, `get_daily_returns()`, `get_monthly_returns()`
- Plots: equity, portfolio+trades, drawdown, trade PnL, returns distribution, stock performance, interactive Plotly, dashboard
- Reports: `generate_report()`, `generate_signals_orders_report()`, `generate_signals_orders_dataframe()` (metadata exploded to dotted columns)
- `run_full_analysis(...)` — analysis + MLflow (7 PNGs, HTML, signals, CSV/Parquet)
- `log_to_mlflow(...)` — manual logging

**PortfolioAnalyzer** (`trading/analysis/portfolio_analyzer.py`): same interface, preferred for stored sessions.
- `PortfolioAnalyzer(portfolio)` — in-memory
- `from_mongodb(session_id, ...)` / `from_mongodb_multi(session_ids, ...)` — MongoDB; falls back to `state_store` config
- Session metadata (algo class, params) is logged to MLflow automatically
- `run_analysis(...)` / `run_full_analysis(...)` → `{"metrics", "trades", "files"}`
