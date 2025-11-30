"""
Quick test to verify interactive HTML chart is logged to MLflow
"""
from pathlib import Path
from algorithms.test_algorithm import TestAlgorithm
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from data_providers.test_data_provider import TestDataProvider
from engines.backtest_engine import BacktestingEngine
from engines.analysis_engine import AnalysisEngine

print("="*80)
print("TESTING INTERACTIVE HTML CHART LOGGING TO MLFLOW")
print("="*80)

# Run backtest
print("\n1. Running backtest...")
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

# Create analysis engine
print("\n2. Creating analysis engine...")
engine = AnalysisEngine(sim.pf, pf.om)
print("   ✓ Analysis engine created")

# Test 1: Try generating the chart standalone first
print("\n3. Testing standalone chart generation...")
try:
    fig = engine.plot_interactive_portfolio(show=False, save_path="test_chart.html")
    print("   ✓ Interactive chart generated successfully")
    print("   ✓ Saved to: test_chart.html")
except Exception as e:
    print(f"   ✗ Failed to generate chart: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Log to MLflow
print("\n4. Logging to MLflow...")
try:
    engine.log_to_mlflow(
        run_name="TEST: Interactive HTML Chart",
        description="Testing that interactive_portfolio.html gets logged",
        parameters={"test": "html_logging"},
        log_charts=True,
        log_trades=False,
        log_report=False
    )
    print("   ✓ Logged to MLflow")
except Exception as e:
    print(f"   ✗ Failed to log to MLflow: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
print("\nTo verify the interactive chart was logged:")
print("  1. Go to http://hp.lan:8899")
print("  2. Find run: 'TEST: Interactive HTML Chart'")
print("  3. Click 'Artifacts' tab")
print("  4. Look for 'interactive_portfolio.html'")
print("\nIf you see the file, the integration is working! ✓")
print("If not, check the logs above for errors.")
print("="*80)
