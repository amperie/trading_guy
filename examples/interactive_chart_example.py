"""
Example: Using the Interactive Plotly Chart

This example demonstrates the new interactive portfolio visualization
with zoomable timeline, clickable legend, and detailed hover information.
"""
from pathlib import Path

from trading.core.algorithms.test_algorithm import TestAlgorithm
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.analysis.analysis_engine import AnalysisEngine


def main():
    """
    Run a backtest and create an interactive Plotly chart
    """
    print("="*80)
    print("INTERACTIVE PORTFOLIO CHART EXAMPLE")
    print("="*80)

    # Run a backtest
    print("\nRunning backtest...")
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOrderManager()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Create analysis engine
    print("\nGenerating interactive chart...")
    engine = AnalysisEngine(sim.pf, pf.om)

    # Create the interactive chart
    # This will open in your default browser
    fig = engine.plot_interactive_portfolio(
        show=True,  # Opens in browser
        save_path="interactive_portfolio.html"  # Also saves to file
    )

    print("\n" + "="*80)
    print("INTERACTIVE CHART FEATURES")
    print("="*80)
    print("""
    ✓ ZOOM: Click and drag on the chart to zoom into specific time periods
    ✓ RESET: Double-click to reset zoom
    ✓ RANGE SLIDER: Use the slider at the bottom to select date ranges
    ✓ HIDE/SHOW: Click on legend items to hide/show specific lines:
        - Portfolio Value (blue solid line)
        - Cash Balance (green dotted line)
        - Stock Prices (dashed lines, hidden by default)
        - BUY markers (green X)
        - SELL markers (red X)
    ✓ HOVER: Hover over any point to see detailed information
    ✓ PAN: Shift+drag to pan across the chart

    The chart displays:
    • Portfolio value over time (left y-axis)
    • Cash balance over time (left y-axis)
    • Individual stock prices (right y-axis, hidden by default)
    • Buy and sell transactions with details

    Stock price lines are HIDDEN by default to reduce clutter.
    Click on them in the legend to show them!
    """)
    print("="*80)
    print(f"\nChart saved to: interactive_portfolio.html")
    print("You can open this file in any web browser to view the interactive chart.")
    print("="*80)


def example_with_mlflow():
    """
    Example: The interactive chart is automatically logged to MLflow
    """
    print("\n" + "="*80)
    print("MLFLOW INTEGRATION EXAMPLE")
    print("="*80)

    # Run a backtest
    print("\nRunning backtest...")
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOrderManager()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Run full analysis with MLflow
    # The interactive chart is automatically included!
    print("\nRunning full analysis with MLflow...")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        run_name="Interactive Chart Demo",
        description="Demonstrates automatic logging of interactive Plotly chart",
        parameters={"symbol": "AAPL", "initial_cash": 10000},
        log_to_mlflow=True,
        save_charts_locally=True,  # Also saves locally
        output_dir="./analysis_output"
    )

    print("\n" + "="*80)
    print("The interactive chart has been:")
    print("  1. Logged to MLflow as an HTML artifact")
    print("  2. Saved locally to ./analysis_output/interactive_portfolio.html")
    print("\nView in MLflow UI: http://hp.lan:8899")
    print("="*80)


def example_standalone_chart():
    """
    Example: Create just the interactive chart without full analysis
    """
    print("\n" + "="*80)
    print("STANDALONE CHART EXAMPLE")
    print("="*80)

    # Run a backtest
    print("\nRunning backtest...")
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOrderManager()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Create just the interactive chart
    print("\nCreating standalone interactive chart...")
    engine = AnalysisEngine(sim.pf, pf.om)

    # Save to file without opening browser
    fig = engine.plot_interactive_portfolio(
        show=False,  # Don't open browser
        save_path="my_portfolio_chart.html"
    )

    print("\nChart saved to: my_portfolio_chart.html")
    print("Open this file in your browser to explore the interactive chart.")
    print("="*80)


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run additional examples:
    # example_with_mlflow()
    # example_standalone_chart()
