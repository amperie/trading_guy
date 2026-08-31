# Interactive Portfolio Chart

## Overview

The AnalysisEngine now includes a **fully interactive Plotly chart** that provides a rich, zoomable visualization of your trading backtest results.

## Key Features

### 🔍 Interactive Zooming
- **Click and drag** on the chart to zoom into specific date ranges
- **Double-click** to reset zoom
- **Range slider** at the bottom for quick date selection
- **Pinch to zoom** on touch devices

### 📊 Multi-Layer Visualization

The chart displays multiple data layers on dual y-axes:

**Left Y-Axis (Portfolio/Cash):**
- **Portfolio Value** - Blue solid line showing total portfolio value over time
- **Cash Balance** - Green dotted line showing available cash
- **BUY Markers** - Green X marks where purchases occurred
- **SELL Markers** - Red X marks where sales occurred

**Right Y-Axis (Stock Prices):**
- **Individual Stock Prices** - Dashed lines for each stock in your data
- **Hidden by default** to avoid clutter
- **Click legend to show/hide** individual stocks

### 💡 Interactive Legend

Click on any legend item to hide/show that data series:
- Toggle portfolio value on/off
- Toggle cash balance on/off
- Show/hide individual stock prices
- Show/hide buy/sell markers
- Customize your view for focused analysis

### 🎯 Detailed Hover Information

Hover over any point to see:
- **Portfolio Value**: Exact date and dollar amount
- **Cash Balance**: Current cash available
- **BUY Transactions**: Symbol, quantity, price per share, total cost
- **SELL Transactions**: Symbol, quantity, price per share, total proceeds
- **Stock Prices**: Current price for each stock

### 📱 Responsive & Exportable

- Works on desktop and mobile browsers
- Save as standalone HTML file
- Share with team members
- Embed in reports
- View offline

## Usage

### Basic Usage

```python
from trading.analysis.analysis_engine import AnalysisEngine

# After running your backtest
engine = AnalysisEngine(portfolio, order_manager)

# Create and display interactive chart
fig = engine.plot_interactive_portfolio(
    show=True,  # Opens in browser
    save_path="my_chart.html"  # Saves to file
)
```

### Save Without Opening

```python
# Just save to file, don't open browser
fig = engine.plot_interactive_portfolio(
    show=False,
    save_path="portfolio_analysis.html"
)
```

### With Full Analysis

The interactive chart is **automatically included** when using `run_full_analysis()`:

```python
# Run complete analysis
results = engine.run_full_analysis(
    run_name="My Strategy",
    log_to_mlflow=True,  # Logs to MLflow (includes interactive chart)
    save_charts_locally=True  # Saves to local files
)

# Interactive chart saved to: ./interactive_portfolio.html
# Also logged to MLflow as HTML artifact
```

### MLflow Integration

When logging to MLflow, the interactive chart is automatically included:

```python
engine.log_to_mlflow(
    run_name="Strategy Test",
    parameters={"symbol": "AAPL"},
    log_charts=True  # Includes interactive chart
)

# View in MLflow UI:
# 1. Go to http://z440.lan:5000
# 2. Click on your run
# 3. Go to "Artifacts" tab
# 4. Click on "interactive_portfolio.html"
# 5. Interact with the chart in MLflow!
```

## Chart Components

### Portfolio Value Line
- **Color**: Blue
- **Style**: Solid line
- **Y-Axis**: Left (primary)
- **Shows**: Total portfolio value (cash + positions)

### Cash Balance Line
- **Color**: Green
- **Style**: Dotted line
- **Y-Axis**: Left (primary)
- **Shows**: Available cash for trading

### BUY Markers
- **Color**: Green
- **Style**: X marker
- **Size**: 15
- **Hover Info**: Symbol, quantity, price, total cost

### SELL Markers
- **Color**: Red
- **Style**: X marker
- **Size**: 15
- **Hover Info**: Symbol, quantity, price, total proceeds

### Stock Price Lines
- **Colors**: Purple, Orange, Brown, Pink, etc.
- **Style**: Dashed lines
- **Y-Axis**: Right (secondary)
- **Visibility**: Hidden by default (click legend to show)
- **Shows**: Individual stock closing prices

## Examples

See `examples/interactive_chart_example.py` for comprehensive examples:

1. **Basic Example** - Create and display the chart
2. **MLflow Integration** - Auto-log with full analysis
3. **Standalone Chart** - Save without opening browser

## Technical Details

### Dependencies
```bash
pip install plotly
```

### Chart Technology
- Built with **Plotly** - industry-standard interactive charting library
- Outputs as **standalone HTML** - no external dependencies
- **JavaScript-based** - works in any modern browser
- **Responsive** - adapts to screen size

### Performance
- Handles thousands of data points smoothly
- Lazy loading for stock price lines (hidden by default)
- Optimized hover tooltips
- Efficient zoom/pan operations

### Browser Compatibility
- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers

## Advantages Over Static Charts

| Feature | Static Chart | Interactive Chart |
|---------|-------------|-------------------|
| Zoom into dates | ❌ | ✅ |
| Hide/show lines | ❌ | ✅ |
| Hover for details | ❌ | ✅ |
| Date range slider | ❌ | ✅ |
| Multi-axis support | Limited | ✅ Full |
| Share as HTML | ❌ | ✅ |
| Works offline | ✅ | ✅ |
| Print quality | ✅ Better | ✅ Good |

## Tips & Tricks

### 💡 Focus on Specific Trades
1. Click to hide portfolio value and cash
2. Show only BUY and SELL markers
3. Zoom into the trade timeframe
4. Hover over markers for transaction details

### 💡 Analyze Stock Performance
1. Click on a stock price in the legend to show it
2. Compare stock price movements with portfolio value
3. Identify correlations between price changes and trades

### 💡 Find Optimal Entry/Exit Points
1. Show stock price for your traded symbol
2. Look at BUY markers relative to price lows
3. Look at SELL markers relative to price highs
4. Analyze timing effectiveness

### 💡 Share with Team
1. Save chart as HTML file
2. Send file via email or share drive
3. Recipients can open in any browser
4. No special software needed!

### 💡 Customize View
- Hide cash if focusing on total value
- Show only one stock at a time for clarity
- Hide markers when analyzing overall trends
- Show everything for comprehensive overview

## Troubleshooting

### Chart doesn't open in browser
- Check that `show=True` is set
- Verify your default browser is set correctly
- Try opening the saved HTML file manually

### Stock prices not visible
- Stock price lines are **hidden by default**
- Click on them in the legend to show them
- This is intentional to reduce visual clutter

### "Plotly not installed" error
```bash
pip install plotly
```

### Chart loads slowly
- This is normal for large datasets (>10,000 points)
- Consider filtering data or using date ranges
- The initial load may take a few seconds

## Future Enhancements

Potential future additions:
- Volume bars overlay
- Technical indicators (moving averages, RSI, etc.)
- Multiple portfolios comparison
- Drawdown shading
- Annotations for significant events
- Export to PNG/SVG from browser

## Questions?

See `examples/interactive_chart_example.py` for working examples or refer to the main documentation in `CLAUDE.md`.
