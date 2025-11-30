"""
Debug script to diagnose why interactive_portfolio.html isn't showing in MLflow
"""
import sys
from pathlib import Path

print("="*80)
print("MLFLOW HTML LOGGING DEBUG SCRIPT")
print("="*80)

# Check 1: Verify Plotly is installed
print("\n[CHECK 1] Verifying Plotly installation...")
try:
    import plotly.graph_objects as go
    import plotly
    print(f"   ✓ Plotly is installed (version {plotly.__version__})")
except ImportError as e:
    print(f"   ✗ Plotly is NOT installed: {e}")
    print("   Install with: pip install plotly")
    sys.exit(1)

# Check 2: Verify MLflow is installed
print("\n[CHECK 2] Verifying MLflow installation...")
try:
    import mlflow
    print(f"   ✓ MLflow is installed (version {mlflow.__version__})")
except ImportError as e:
    print(f"   ✗ MLflow is NOT installed: {e}")
    print("   Install with: pip install mlflow")
    sys.exit(1)

# Check 3: Verify config
print("\n[CHECK 3] Checking MLflow configuration...")
try:
    from utils.config_manager import ConfigManager
    config = ConfigManager().get("mlflow", {})
    print(f"   Enabled: {config.get('enabled', 'NOT SET')}")
    print(f"   Tracking URI: {config.get('tracking_uri', 'NOT SET')}")
    print(f"   Experiment: {config.get('experiment_name', 'NOT SET')}")
except Exception as e:
    print(f"   ✗ Error reading config: {e}")

# Check 4: Create MLflow client
print("\n[CHECK 4] Creating MLflow client...")
try:
    from utils.mlflow_client import MLflowClient
    client = MLflowClient.from_config()
    print(f"   ✓ MLflow client created")
    print(f"   Enabled: {client.enabled}")
    print(f"   Tracking URI: {client.tracking_uri}")
except Exception as e:
    print(f"   ✗ Failed to create client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 5: Test basic HTML logging
print("\n[CHECK 5] Testing basic HTML logging...")
try:
    with client.start_run(run_name="DEBUG: Test HTML Logging", description="Testing HTML artifact logging"):
        test_html = "<html><body><h1>Test HTML</h1><p>If you see this, HTML logging works!</p></body></html>"
        client.log_html(test_html, "test")
        print("   ✓ HTML logged successfully")
        print(f"   Run URL: {client.get_run_url()}")
except Exception as e:
    print(f"   ✗ Failed to log HTML: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 6: Run a quick backtest
print("\n[CHECK 6] Running backtest...")
try:
    from algorithms.test_algorithm import TestAlgorithm
    from core.om.backtesting_om import BacktestingOM
    from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
    from data_providers.test_data_provider import TestDataProvider
    from engines.backtest_engine import BacktestingEngine

    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOM()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()
    print("   ✓ Backtest complete")
except Exception as e:
    print(f"   ✗ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 7: Create interactive chart
print("\n[CHECK 7] Creating interactive chart...")
try:
    from engines.analysis_engine import AnalysisEngine
    engine = AnalysisEngine(sim.pf, pf.om)
    fig = engine.plot_interactive_portfolio(show=False, save_path="debug_chart.html")
    print("   ✓ Interactive chart created")
    print("   ✓ Saved to: debug_chart.html")
except Exception as e:
    print(f"   ✗ Failed to create chart: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 8: Convert chart to HTML
print("\n[CHECK 8] Converting chart to HTML string...")
try:
    html_string = fig.to_html(include_plotlyjs='cdn')
    html_size = len(html_string)
    print(f"   ✓ HTML generated, size: {html_size:,} bytes")
    if html_size < 1000:
        print(f"   ⚠️  WARNING: HTML is very small, might be incomplete")
    else:
        print(f"   ✓ HTML size looks good")
except Exception as e:
    print(f"   ✗ Failed to convert to HTML: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 9: Log HTML to MLflow manually
print("\n[CHECK 9] Logging HTML to MLflow manually...")
try:
    with client.start_run(run_name="DEBUG: Manual HTML Logging", description="Manually logging Plotly chart HTML"):
        client.log_html(html_string, "manual_interactive_chart")
        print("   ✓ HTML logged successfully")
        print(f"   Run URL: {client.get_run_url()}")
except Exception as e:
    print(f"   ✗ Failed to log HTML: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 10: Use log_to_mlflow
print("\n[CHECK 10] Using engine.log_to_mlflow()...")
try:
    # Set logging level to DEBUG to see all messages
    import logging
    logging.basicConfig(level=logging.DEBUG)

    engine.log_to_mlflow(
        run_name="DEBUG: Full Analysis Logging",
        description="Testing full log_to_mlflow with interactive chart",
        parameters={"debug": True},
        log_charts=True,
        log_trades=False,
        log_report=False
    )
    print("   ✓ log_to_mlflow completed")
except Exception as e:
    print(f"   ✗ log_to_mlflow failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
print("\nResults:")
print("  ✓ Plotly installed and working")
print("  ✓ MLflow client created and working")
print("  ✓ Basic HTML logging works")
print("  ✓ Interactive chart created successfully")
print("  ✓ Chart converted to HTML")
print("  ✓ HTML logged to MLflow manually")
print("  ✓ log_to_mlflow executed")
print("\nCheck MLflow UI for the following runs:")
print("  1. 'DEBUG: Test HTML Logging' - should have test.html")
print("  2. 'DEBUG: Manual HTML Logging' - should have manual_interactive_chart.html")
print("  3. 'DEBUG: Full Analysis Logging' - should have interactive_portfolio.html")
print("\nMLflow UI: http://hp.lan:8899")
print("\nIf you see all three HTML files in their respective runs,")
print("then the integration is working correctly!")
print("="*80)
