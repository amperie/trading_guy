# MLflow Artifacts - Complete List

When you run `engine.run_full_analysis(log_to_mlflow=True)` or `engine.log_to_mlflow()`, the following artifacts are automatically logged to MLflow:

## 📊 Static Charts (PNG Format)

1. **equity_curve.png**
   - Portfolio value over time
   - Simple line chart
   - Static image

2. **portfolio_with_trades.png**
   - Portfolio value with BUY/SELL markers
   - Green X = BUY, Red X = SELL
   - Static image

3. **drawdown.png**
   - Drawdown chart over time
   - Shows peak-to-trough declines
   - Static image

4. **trade_pnl.png**
   - Individual trade P&L
   - Bar chart (green = profit, red = loss)
   - Static image

5. **returns_distribution.png**
   - Histogram of returns
   - Shows distribution shape
   - Static image

6. **stock_performance.png**
   - Individual stock prices over time
   - Stock returns normalized to 100
   - Static image

7. **dashboard.png**
   - Comprehensive multi-panel dashboard
   - Includes equity, drawdown, P&L, and metrics
   - Static image

## 🎯 Interactive Chart (HTML Format)

8. **interactive_portfolio.html** ⭐ NEW!
   - **Fully interactive Plotly chart**
   - Opens in browser with full interactivity
   - Features:
     - ✅ Zoom by clicking and dragging
     - ✅ Range slider for date selection
     - ✅ Click legend to hide/show lines
     - ✅ Hover for detailed tooltips
     - ✅ Portfolio value (blue line)
     - ✅ Cash balance (green dotted line)
     - ✅ BUY markers (green X)
     - ✅ SELL markers (red X)
     - ✅ Stock prices (dashed lines, hidden by default)
     - ✅ Dual y-axes (portfolio/cash on left, stocks on right)

## 📄 Data Files (JSON/Text)

9. **trades.json**
   - All executed trades
   - Entry/exit times, prices, P&L, duration
   - Bracket order information

10. **bracket_analysis.json**
    - Bracket order effectiveness analysis
    - Stop-loss vs profit-taker statistics
    - Win rates by exit type

11. **performance_report.txt**
    - Complete text report
    - 30+ performance metrics
    - Formatted for easy reading

12. **summary.md**
    - Markdown summary
    - Key metrics highlighted
    - Good for documentation

## 📈 Metrics (40+ Logged)

- **Returns**: total_return, total_return_pct, annualized_return
- **Risk**: sharpe_ratio, sortino_ratio, max_drawdown, volatility, calmar_ratio
- **Trades**: total_trades, winning_trades, losing_trades, win_rate
- **P&L**: avg_win, avg_loss, profit_factor
- **Bracket**: bracket_stop_rate, bracket_profit_rate
- **Daily**: best_day, worst_day, avg_daily_return
- **Distribution**: skewness, kurtosis
- **Equity**: initial_equity, final_equity, peak_equity
- **Time**: total_days, trading_days

## 🏷️ Tags

- Custom tags you provide
- System information (if enabled):
  - Python version
  - Platform (Windows/Linux/Mac)
  - Architecture
  - Hostname

## How to View in MLflow

1. **Go to MLflow UI**: http://z440.lan:5000
2. **Find your run** by name or date
3. **Click on the run** to open details
4. **Click "Artifacts" tab**
5. **See all files listed above**

### To View Interactive Chart:

1. In Artifacts tab, locate `interactive_portfolio.html`
2. Click on the filename
3. MLflow will display the interactive chart in the UI
4. You can zoom, pan, and interact directly in MLflow!
5. Or download and open in any web browser

## Example: What You'll See

```
Artifacts/
├── equity_curve.png
├── portfolio_with_trades.png
├── drawdown.png
├── trade_pnl.png
├── returns_distribution.png
├── stock_performance.png
├── dashboard.png
├── interactive_portfolio.html  ⭐ NEW INTERACTIVE!
├── trades.json
├── bracket_analysis.json
├── performance_report.txt
└── summary.md
```

## Key Benefits

✅ **Complete Record**: All analysis results in one place
✅ **Interactive Exploration**: Use interactive chart for deep analysis
✅ **Static Reports**: Use PNG charts for presentations/reports
✅ **Data Export**: JSON files for further analysis
✅ **Shareability**: Share MLflow run URL with team
✅ **History**: Never lose a backtest result
✅ **Comparison**: Compare multiple runs side-by-side

## File Sizes (Approximate)

| File | Size | Type |
|------|------|------|
| equity_curve.png | 50-150 KB | Image |
| portfolio_with_trades.png | 50-150 KB | Image |
| drawdown.png | 50-150 KB | Image |
| trade_pnl.png | 50-150 KB | Image |
| returns_distribution.png | 50-150 KB | Image |
| stock_performance.png | 50-150 KB | Image |
| dashboard.png | 150-300 KB | Image |
| **interactive_portfolio.html** | **500 KB - 2 MB** | **Interactive** |
| trades.json | 10-100 KB | Data |
| bracket_analysis.json | 5-20 KB | Data |
| performance_report.txt | 5-15 KB | Text |
| summary.md | 3-10 KB | Text |

**Total**: ~1-3 MB per run (most is the interactive HTML chart)

## Notes

- All files are automatically generated
- No manual intervention needed
- Gracefully handles missing dependencies (skips if Plotly not installed)
- Interactive chart works offline once downloaded
- All charts are high quality (150+ DPI for PNGs)

## Usage

```python
# Method 1: Automatic (recommended)
results = engine.run_full_analysis(
    run_name="My Strategy",
    log_to_mlflow=True  # All artifacts logged automatically
)

# Method 2: Manual control
engine.log_to_mlflow(
    run_name="My Strategy",
    log_charts=True,  # Include all charts (static + interactive)
    log_trades=True,  # Include trades.json
    log_report=True   # Include text report
)
```

Both methods log the **interactive_portfolio.html** chart automatically! 🎉
