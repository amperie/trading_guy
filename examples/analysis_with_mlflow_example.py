"""
Example: Using AnalysisEngine.run_full_analysis() with MLflow integration

This example shows how to run a complete backtest analysis with automatic
MLflow logging using the new run_full_analysis() method.
"""
from pathlib import Path

from trading.core.algorithms.test_algorithm import TestAlgorithm
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.analysis.analysis_engine import AnalysisEngine


def example_1_simple_analysis_with_mlflow():
    """
    Example 1: Run analysis with MLflow logging (default behavior)
    """
    print("="*80)
    print("EXAMPLE 1: Simple Analysis with MLflow")
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

    # Run full analysis with MLflow logging (default)
    print("\nRunning full analysis with MLflow...")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        run_name="Simple Backtest Example",
        description="Testing TestAlgorithm with default parameters",
        parameters={
            "symbol": "AAPL",
            "initial_cash": 10000.0,
            "algorithm": "TestAlgorithm",
            "history_length": 10
        },
        tags={
            "environment": "example",
            "version": "1.0"
        }
    )

    print("\nAnalysis complete! Results logged to MLflow.")
    print(f"Total trades: {len(results['trades'])}")
    print(f"Final return: {results['metrics'].total_return_pct:.2f}%")


def example_2_analysis_without_mlflow():
    """
    Example 2: Run analysis without MLflow (disable logging)
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Analysis without MLflow")
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

    # Run analysis without MLflow
    print("\nRunning analysis (MLflow disabled)...")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        log_to_mlflow=False,  # Disable MLflow logging
        show_summary=True
    )

    print("\nAnalysis complete! No MLflow logging performed.")


def example_3_analysis_with_local_files():
    """
    Example 3: Run analysis with both MLflow and local file outputs
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Analysis with MLflow + Local Files")
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

    # Run analysis with both MLflow and local file outputs
    print("\nRunning analysis with MLflow and local file outputs...")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        run_name="Full Analysis with Local Files",
        description="Complete analysis with both MLflow and local file outputs",
        parameters={
            "symbol": "AAPL",
            "initial_cash": 10000.0,
            "algorithm": "TestAlgorithm"
        },
        log_to_mlflow=True,
        save_charts_locally=True,
        save_report_locally=True,
        output_dir="./analysis_output"
    )

    print("\nAnalysis complete!")
    print("- Results logged to MLflow")
    print("- Charts saved to ./analysis_output/")
    print("- Report saved to ./analysis_output/backtest_report.txt")


def example_4_manual_mlflow_logging():
    """
    Example 4: Use log_to_mlflow() method directly for fine-grained control
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Manual MLflow Logging")
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

    # Manually calculate metrics and then log
    print("\nManually running analysis steps...")
    engine = AnalysisEngine(sim.pf, pf.om)

    # Extract trades manually
    trades = engine.extract_trades()
    print(f"Extracted {len(trades)} trades")

    # Calculate metrics manually
    metrics = engine.calculate_metrics()
    print(f"Calculated metrics: Sharpe={metrics.sharpe_ratio:.2f}")

    # Now log to MLflow with fine-grained control
    print("\nLogging to MLflow with custom settings...")
    engine.log_to_mlflow(
        run_name="Manual MLflow Logging Example",
        description="Demonstrates fine-grained control over MLflow logging",
        parameters={
            "symbol": "AAPL",
            "strategy": "test",
            "custom_param": "custom_value"
        },
        tags={
            "manual": "true",
            "experiment_type": "demo"
        },
        log_charts=True,
        log_trades=True,
        log_report=True,
        chart_dpi=200  # Higher quality charts
    )

    print("\nManual logging complete!")


def example_5_compare_strategies():
    """
    Example 5: Compare multiple strategy runs in MLflow
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Multiple Strategies")
    print("="*80)

    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    # Test different initial capital amounts
    capital_amounts = [5000, 10000, 20000]

    for capital in capital_amounts:
        print(f"\nRunning backtest with ${capital} initial capital...")

        om = BacktestingOrderManager()
        al = TestAlgorithm({"history_length": 10, "full_history": True})
        dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
        dp = TestDataProvider(dp_cfg)
        pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, float(capital), {}, True)

        sim = BacktestingEngine({}, dp, al, om, pf)
        sim.run()

        # Run analysis and log to MLflow
        engine = AnalysisEngine(sim.pf, pf.om)
        results = engine.run_full_analysis(
            run_name=f"Capital Comparison ${capital}",
            description=f"Testing with ${capital} initial capital",
            parameters={
                "initial_capital": capital,
                "symbol": "AAPL",
                "algorithm": "TestAlgorithm"
            },
            tags={
                "comparison": "capital_test",
                "capital_amount": str(capital)
            },
            show_summary=False  # Suppress console output
        )

        print(f"  - Final equity: ${results['metrics'].final_equity:,.2f}")
        print(f"  - Total return: {results['metrics'].total_return_pct:.2f}%")

    print("\nAll runs logged to MLflow! Compare them in the UI.")


if __name__ == "__main__":
    # Run all examples
    example_1_simple_analysis_with_mlflow()

    example_2_analysis_without_mlflow()

    example_3_analysis_with_local_files()

    example_4_manual_mlflow_logging()

    example_5_compare_strategies()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nCheck MLflow UI to view logged experiments:")
    print("http://hp.lan:8899")
    print("="*80)
