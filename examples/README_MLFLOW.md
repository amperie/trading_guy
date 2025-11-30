# MLflow Integration with AnalysisEngine

## Overview

The `AnalysisEngine` class now includes comprehensive MLflow integration for tracking and logging backtesting results. This allows you to automatically log performance metrics, charts, trades, and reports to your MLflow tracking server.

## New Methods

### 1. `run_full_analysis()`

Runs a complete analysis (similar to `analysis_example.py`) and optionally logs everything to MLflow.

**Features:**
- Extracts all trades
- Calculates 30+ performance metrics
- Computes returns at tick, daily, and monthly granularity
- Analyzes bracket order effectiveness
- Generates text reports
- Creates all visualizations
- Logs everything to MLflow (by default)
- Optionally saves files locally

**Usage:**
```python
from engines.analysis_engine import AnalysisEngine

engine = AnalysisEngine(portfolio, order_manager)

# Run complete analysis with MLflow logging (default)
results = engine.run_full_analysis(
    run_name="My Strategy Test",
    description="Testing SMA crossover strategy",
    parameters={"sma_short": 5, "sma_long": 20, "symbol": "AAPL"},
    tags={"environment": "production", "version": "1.0"},
    log_to_mlflow=True,  # Default
    save_charts_locally=False,
    save_report_locally=False
)

# Access results
print(f"Total trades: {len(results['trades'])}")
print(f"Sharpe ratio: {results['metrics'].sharpe_ratio:.2f}")
print(f"Win rate: {results['metrics'].win_rate:.1f}%")
```

**Returns a dictionary with:**
- `trades`: List of Trade objects
- `metrics`: PerformanceMetrics dataclass
- `tick_returns`: pandas Series
- `daily_returns`: pandas Series
- `monthly_returns`: pandas Series
- `bracket_analysis`: dict (if applicable)
- `report`: str (full text report)

### 2. `log_to_mlflow()`

Manually logs analysis results to MLflow with fine-grained control.

**Usage:**
```python
# First, run analysis steps manually
engine = AnalysisEngine(portfolio, order_manager)
trades = engine.extract_trades()
metrics = engine.calculate_metrics()

# Then log to MLflow
engine.log_to_mlflow(
    run_name="Custom Run",
    description="Manual logging example",
    parameters={"strategy": "custom", "param1": 123},
    tags={"manual": "true"},
    log_charts=True,
    log_trades=True,
    log_report=True,
    chart_dpi=200  # High quality charts
)
```

## What Gets Logged to MLflow

### Metrics (40+ performance metrics)
- Returns: total_return, total_return_pct, annualized_return
- Risk: sharpe_ratio, sortino_ratio, max_drawdown, volatility, calmar_ratio, ulcer_index
- Trades: total_trades, winning_trades, losing_trades, win_rate, profit_factor
- P&L: avg_win, avg_loss, largest_win, largest_loss, avg_trade_pnl
- Bracket: bracket_trades, bracket_stop_rate, bracket_profit_rate
- Daily: best_day, worst_day, avg_daily_return
- Distribution: skewness, kurtosis
- Equity: initial_equity, final_equity, peak_equity
- Time: total_days, trading_days

### Parameters
- Strategy/algorithm parameters you provide
- Symbol, initial cash, etc.

### Artifacts
1. **Static Charts (7 PNG files):**
   - `equity_curve.png` - Portfolio value over time
   - `portfolio_with_trades.png` - Equity with BUY/SELL markers
   - `drawdown.png` - Drawdown chart
   - `trade_pnl.png` - Individual trade P&L
   - `returns_distribution.png` - Returns histogram
   - `stock_performance.png` - Stock prices and normalized returns
   - `dashboard.png` - Comprehensive multi-panel dashboard

2. **Interactive Chart (HTML):**
   - `interactive_portfolio.html` - **NEW!** Interactive Plotly chart with:
     - ✓ Zoomable timeline (click and drag)
     - ✓ Portfolio value and cash balance lines
     - ✓ BUY/SELL markers with transaction details
     - ✓ Individual stock price lines (hidden by default)
     - ✓ Clickable legend to show/hide any line
     - ✓ Range slider for date selection
     - ✓ Hover tooltips with detailed information
     - ✓ Dual y-axes (portfolio/cash on left, stock prices on right)
     - Open in MLflow UI to interact with the chart!

3. **Data Files:**
   - `trades.json` - All trades with entry/exit times, prices, P&L
   - `bracket_analysis.json` - Bracket order effectiveness analysis
   - `performance_report.txt` - Full text report
   - `summary.md` - Markdown summary

### Tags
- Custom tags you provide
- System info (if `auto_log_system_info=true` in config):
  - Python version
  - Platform (Windows/Linux/Mac)
  - Architecture
  - Hostname

## Configuration

MLflow settings are in `config.yaml`:

```yaml
mlflow:
  enabled: true  # Enable/disable MLflow tracking
  tracking_uri: "http://hp.lan:8899"  # MLflow server
  experiment_name: "Trading Backtest"  # Default experiment
  artifact_location: null  # Custom artifact storage (optional)
  run_name_prefix: ""  # Prefix for run names (optional)
  auto_log_system_info: true  # Auto-log system details
```

## Examples

See `examples/analysis_with_mlflow_example.py` for comprehensive examples:

1. **Simple analysis with MLflow** - Basic usage with automatic logging
2. **Analysis without MLflow** - Disable logging with `log_to_mlflow=False`
3. **Analysis with local files** - Save both to MLflow and local files
4. **Manual MLflow logging** - Fine-grained control over what gets logged
5. **Compare strategies** - Run multiple experiments and compare in MLflow UI

## Viewing Results

After running analysis, view results in the MLflow UI:
- **URL:** http://hp.lan:8899
- Compare multiple runs
- View charts and artifacts
- Track metrics over time
- Filter by tags and parameters

## Quick Start

```python
# 1. Run backtest
sim = BacktestingEngine(...)
sim.run()

# 2. Run full analysis with MLflow (one line!)
engine = AnalysisEngine(sim.pf, sim.om)
results = engine.run_full_analysis(
    run_name="My First Run",
    parameters={"symbol": "AAPL", "initial_cash": 10000}
)

# 3. View in MLflow UI
# http://hp.lan:8899
```

## Benefits

✅ **Automatic logging** - One method call logs everything
✅ **Experiment tracking** - Compare different strategies and parameters
✅ **Reproducibility** - All parameters and results are saved
✅ **Visualization** - All charts automatically uploaded
✅ **Collaboration** - Share results via MLflow UI URL
✅ **History** - Never lose a backtest result
✅ **Analysis** - Compare metrics across runs

## Disabling MLflow

If you don't want to use MLflow:

```python
# Option 1: Disable in config
# config.yaml: mlflow.enabled = false

# Option 2: Disable per run
results = engine.run_full_analysis(log_to_mlflow=False)
```

## Requirements

```bash
pip install mlflow
pip install plotly  # For interactive charts
```

**Notes:**
- MLflow tracking server must be running at the configured URI (http://hp.lan:8899)
- Plotly is optional - if not installed, the interactive chart will be skipped
- All static charts use matplotlib (already installed)
