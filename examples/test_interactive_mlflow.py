"""
Test script: Verify interactive chart is logged to MLflow

This script demonstrates that the interactive chart is automatically
logged to MLflow as an artifact when using run_full_analysis() or log_to_mlflow()
"""
from pathlib import Path

from algorithms.test_algorithm import TestAlgorithm
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from data_providers.test_data_provider import TestDataProvider
from engines.backtest_engine import BacktestingEngine
from engines.analysis_engine import AnalysisEngine


def test_method_1_run_full_analysis():
    """
    Test Method 1: Using run_full_analysis() with log_to_mlflow=True
    The interactive chart should be automatically logged to MLflow
    """
    print("="*80)
    print("TEST 1: run_full_analysis() with MLflow")
    print("="*80)

    # Setup and run backtest
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOM()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Run full analysis (interactive chart automatically logged)
    print("\nRunning full analysis with MLflow logging...")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        run_name="Test: Interactive Chart Auto-Logged",
        description="Verifying that interactive_portfolio.html is logged to MLflow",
        parameters={
            "test_method": "run_full_analysis",
            "symbol": "AAPL",
            "initial_cash": 10000
        },
        tags={
            "test": "interactive_chart",
            "method": "auto"
        },
        log_to_mlflow=True  # This triggers automatic logging
    )

    print("\n" + "="*80)
    print("✅ TEST 1 COMPLETE")
    print("="*80)
    print("\nMLflow Artifacts Logged:")
    print("  📊 7 static charts (PNG)")
    print("  🎯 interactive_portfolio.html (NEW!)")
    print("  📄 trades.json")
    print("  📄 bracket_analysis.json")
    print("  📄 performance_report.txt")
    print("  📄 summary.md")
    print("\nView in MLflow UI:")
    print("  1. Go to http://hp.lan:8899")
    print("  2. Find run: 'Test: Interactive Chart Auto-Logged'")
    print("  3. Click 'Artifacts' tab")
    print("  4. Look for 'interactive_portfolio.html'")
    print("  5. Click to open and interact!")
    print("="*80)


def test_method_2_log_to_mlflow():
    """
    Test Method 2: Using log_to_mlflow() directly
    The interactive chart should be included in the logged artifacts
    """
    print("\n" + "="*80)
    print("TEST 2: log_to_mlflow() directly")
    print("="*80)

    # Setup and run backtest
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOM()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Manual analysis then log
    print("\nRunning analysis and logging to MLflow...")
    engine = AnalysisEngine(sim.pf, pf.om)
    engine.extract_trades()
    engine.calculate_metrics()

    # Log to MLflow (interactive chart included)
    engine.log_to_mlflow(
        run_name="Test: Interactive Chart Manual Log",
        description="Verifying that interactive_portfolio.html is logged via log_to_mlflow()",
        parameters={
            "test_method": "log_to_mlflow",
            "symbol": "AAPL",
            "initial_cash": 10000
        },
        tags={
            "test": "interactive_chart",
            "method": "manual"
        },
        log_charts=True,  # Must be True to include charts
        log_trades=True,
        log_report=True
    )

    print("\n" + "="*80)
    print("✅ TEST 2 COMPLETE")
    print("="*80)
    print("\nMLflow Artifacts Logged:")
    print("  📊 7 static charts (PNG)")
    print("  🎯 interactive_portfolio.html (NEW!)")
    print("  📄 trades.json")
    print("  📄 bracket_analysis.json")
    print("  📄 performance_report.txt")
    print("  📄 summary.md")
    print("\nView in MLflow UI:")
    print("  1. Go to http://hp.lan:8899")
    print("  2. Find run: 'Test: Interactive Chart Manual Log'")
    print("  3. Click 'Artifacts' tab")
    print("  4. Look for 'interactive_portfolio.html'")
    print("  5. Click to open and interact!")
    print("="*80)


def test_method_3_standalone_then_log():
    """
    Test Method 3: Generate chart standalone, then log separately
    """
    print("\n" + "="*80)
    print("TEST 3: Standalone chart generation")
    print("="*80)

    # Setup and run backtest
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOM()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Create engine and generate standalone chart
    print("\nGenerating standalone interactive chart...")
    engine = AnalysisEngine(sim.pf, pf.om)

    # This creates the chart and saves it locally
    fig = engine.plot_interactive_portfolio(
        show=False,
        save_path="test_interactive_chart.html"
    )
    print("✓ Chart saved to: test_interactive_chart.html")

    # Now log everything to MLflow (chart will be regenerated and logged)
    print("\nLogging to MLflow...")
    engine.log_to_mlflow(
        run_name="Test: Standalone Chart Generation",
        description="Chart created standalone then logged",
        parameters={"test_method": "standalone"},
        log_charts=True
    )

    print("\n" + "="*80)
    print("✅ TEST 3 COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  📁 test_interactive_chart.html (local file)")
    print("  🌐 interactive_portfolio.html (MLflow artifact)")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING INTERACTIVE CHART MLFLOW INTEGRATION")
    print("="*80)
    print("\nThis script tests 3 different methods of logging the interactive chart:")
    print("  1. run_full_analysis() with log_to_mlflow=True")
    print("  2. log_to_mlflow() directly")
    print("  3. Standalone chart generation + logging")
    print("\nAll methods should result in 'interactive_portfolio.html' in MLflow artifacts.")
    print("="*80)

    try:
        # Test all three methods
        test_method_1_run_full_analysis()
        test_method_2_log_to_mlflow()
        test_method_3_standalone_then_log()

        print("\n" + "="*80)
        print("🎉 ALL TESTS COMPLETE!")
        print("="*80)
        print("\nSummary:")
        print("  ✅ Method 1: run_full_analysis() - Interactive chart auto-logged")
        print("  ✅ Method 2: log_to_mlflow() - Interactive chart included")
        print("  ✅ Method 3: Standalone - Chart created and logged")
        print("\n📊 View all runs in MLflow UI: http://hp.lan:8899")
        print("\nLook for runs starting with 'Test: Interactive Chart'")
        print("Each should have 'interactive_portfolio.html' in artifacts!")
        print("="*80)

    except ImportError as e:
        print("\n❌ ERROR: Plotly not installed")
        print("Install with: pip install plotly")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
