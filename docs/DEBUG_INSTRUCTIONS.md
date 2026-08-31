# Debugging Interactive Chart MLflow Integration

## Problem
The `interactive_portfolio.html` file is not appearing in MLflow artifacts.

## Solution Steps

### Step 1: Run the Debug Script

```bash
cd E:\Programming\trading_guy
python examples/debug_mlflow_html.py
```

This script will check:
- ✓ Plotly installation
- ✓ MLflow installation
- ✓ Configuration settings
- ✓ MLflow client creation
- ✓ Basic HTML logging
- ✓ Backtest execution
- ✓ Chart generation
- ✓ HTML conversion
- ✓ Manual HTML logging
- ✓ Full log_to_mlflow() method

### Step 2: Check the Output

The script will create **3 test runs** in MLflow:

1. **"DEBUG: Test HTML Logging"**
   - Should contain: `test.html`
   - This tests basic HTML logging

2. **"DEBUG: Manual HTML Logging"**
   - Should contain: `manual_interactive_chart.html`
   - This tests Plotly chart HTML logging

3. **"DEBUG: Full Analysis Logging"**
   - Should contain: `interactive_portfolio.html`
   - This tests the full integration

### Step 3: Verify in MLflow UI

1. Go to: **http://z440.lan:5000**
2. Look for runs starting with "DEBUG:"
3. Click on each run
4. Go to **"Artifacts"** tab
5. Check if the HTML files are there

### Expected Results

If everything is working, you should see:

```
Artifacts/
├── test.html  (in run 1)

Artifacts/
├── manual_interactive_chart.html  (in run 2)

Artifacts/
├── equity_curve.png
├── portfolio_with_trades.png
├── drawdown.png
├── trade_pnl.png
├── returns_distribution.png
├── stock_performance.png
├── dashboard.png
├── interactive_portfolio.html  ⭐ (in run 3)
├── summary.md
```

## Troubleshooting

### Issue: "Plotly is NOT installed"
**Solution:**
```bash
pip install plotly
```

### Issue: "MLflow is NOT installed"
**Solution:**
```bash
pip install mlflow
```

### Issue: "MLflow tracking is disabled in config"
**Solution:**
Check `config.yaml`:
```yaml
mlflow:
  enabled: true  # Make sure this is true
  tracking_uri: "http://z440.lan:5000"
```

### Issue: HTML files appear in some runs but not others
**Check the logs** for error messages. The debug script shows detailed logs.

Common errors:
- **ImportError**: Plotly not installed
- **ConnectionError**: MLflow server not reachable
- **FileNotFoundError**: Temp directory issues

### Issue: HTML files are there but empty or very small
**Check the HTML size** in the debug output. It should be >100KB for the interactive chart.

If it's too small:
- Chart generation failed
- Plotly conversion failed
- Check for JavaScript errors

### Issue: Can't connect to MLflow server
**Verify MLflow server is running:**
```bash
# Test connection
curl http://z440.lan:5000

# Or in Python
import requests
response = requests.get("http://z440.lan:5000")
print(response.status_code)  # Should be 200
```

## Manual Test

If the debug script doesn't help, try this minimal test:

```python
from utils.mlflow_client import MLflowClient

# Create client
client = MLflowClient.from_config()

# Start run
with client.start_run(run_name="MINIMAL TEST"):
    # Log simple HTML
    html = "<h1>Hello from MLflow!</h1>"
    client.log_html(html, "test_minimal")
    print(f"URL: {client.get_run_url()}")

# Check MLflow UI for "MINIMAL TEST" run
# Should have test_minimal.html artifact
```

## Verbose Logging

To see detailed logs, enable DEBUG level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your analysis
engine.run_full_analysis(log_to_mlflow=True)
```

This will show:
- ✓ "Generating interactive Plotly chart..."
- ✓ "Converting chart to HTML..."
- ✓ "HTML generated, size: XXX bytes"
- ✓ "Logging HTML to MLflow..."
- ✓ "Successfully logged interactive chart: interactive_portfolio.html"

If you don't see these messages, the chart isn't being generated or logged.

## Check Logs

Look for these specific log messages:

**Success:**
```
INFO - Generating interactive Plotly chart...
DEBUG - Converting chart to HTML...
DEBUG - HTML generated, size: 523,456 bytes
DEBUG - Logging HTML to MLflow...
INFO - ✅ Successfully logged interactive chart: interactive_portfolio.html
```

**Failure:**
```
WARNING - ⚠️  Plotly not installed, skipping interactive chart
```
or
```
ERROR - ❌ Failed to log interactive chart: [error details]
```

## Getting Help

If you've tried all the above and it still doesn't work:

1. **Run the debug script** and save the output
2. **Check MLflow UI** and take screenshots
3. **Check the logs** with DEBUG level
4. **Verify the local HTML file** was created (debug_chart.html)
5. **Open the local HTML** in a browser - does it work?

## Quick Verification

Run this one-liner to test everything:

```bash
python examples/debug_mlflow_html.py && echo "Check http://z440.lan:5000 for DEBUG runs"
```

If all checks pass ✓, the integration is working!
