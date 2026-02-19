"""
Parameterized backtesting with Ray for parallel execution

Each variation runs independently and logs results to MLflow.

Usage:
    pip install ray
    python examples/parallel_analysis_ray.py
"""
from pathlib import Path
from typing import Dict, Any, List
import ray

from trading.core.algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.analysis.analysis_engine import AnalysisEngine
from utils.logger import Logger


@ray.remote
def run_single_backtest(params: Dict[str, Any]):
    """
    Run a single backtest and log results to MLflow.

    Args:
        params: Dictionary with:
            - symbol, initial_cash, data_path
            - macd_fastperiod, macd_slowperiod, macd_signalperiod
            - rsi_period, history_length
            - run_name, description
    """
    logger = Logger().get_logger(__name__)
    logger.info(f"Starting: {params['run_name']}")

    try:
        # Setup components
        om = BacktestingOrderManager()

        alg_cfg = {
            "macd_fastperiod": params["macd_fastperiod"],
            "macd_slowperiod": params["macd_slowperiod"],
            "macd_signalperiod": params["macd_signalperiod"],
            "rsi_period": params["rsi_period"],
        }
        al = MacdRsiAlgorithm(alg_cfg, history_length=params["history_length"])

        dp_cfg = {
            "path": params["data_path"],
            "provider": "data_providers.test_data_provider.TestDataProvider"
        }
        dp = TestDataProvider(dp_cfg)

        pf = SingleSymbolPortfolio(
            {
                "symbol": params["symbol"],
                "stop_pct": params["stop_pct"],
                "profit_pct": params["profit_pct"],
            },
            om,
            params["initial_cash"],
            {},
            keep_history=True
        )

        # Run backtest
        sim = BacktestingEngine({}, dp, al, om, pf)
        sim.run()

        # Analyze and log to MLflow
        engine = AnalysisEngine(sim.pf, pf.om)
        results = engine.run_full_analysis(
            run_name=params["run_name"],
            description=params["description"],
            parameters=params,
            log_to_mlflow=True,
            save_charts_locally=False,
            save_report_locally=False
        )

        logger.info(f"Completed: {params['run_name']} - Return: {results['metrics'].total_return_pct:.2f}%")

    except Exception as e:
        logger.error(f"Failed: {params['run_name']} - {str(e)}", exc_info=True)


def generate_variations() -> List[Dict[str, Any]]:
    """Generate parameter variations to test."""
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "SPXU_GDXU_UPRO_1min.csv")

    variations = []

    # Test different symbols and MACD configurations
    symbols = ["UPRO", "SPXU", "GDXU"]
    macd_configs = [
        {"fast": 800, "slow": 2100, "signal": 700, "rsi": 1000},   # Fast
        {"fast": 1200, "slow": 2600, "signal": 900, "rsi": 1400},  # Standard
        {"fast": 1600, "slow": 3100, "signal": 1100, "rsi": 1800}, # Slow
    ]
    pf_configs = [
        {"stop_pct": 5, "profit_pct": 5},
        {"stop_pct": 5, "profit_pct": 10},
        {"stop_pct": 2, "profit_pct": 5},
        {"stop_pct": 2, "profit_pct": 10},
    ]

    for symbol in symbols:
        for config in macd_configs:
            for pf_config in pf_configs:
                variations.append({
                    "run_name": f"{symbol}_MACD{config['fast']}-{config['slow']}_RSI{config['rsi']}",
                    "description": f"MACD({config['fast']},{config['slow']},{config['signal']}) RSI({config['rsi']}) on {symbol}",
                    "symbol": symbol,
                    "initial_cash": 1000.0,
                    "data_path": data_path,
                    "macd_fastperiod": config["fast"],
                    "macd_slowperiod": config["slow"],
                    "macd_signalperiod": config["signal"],
                    "rsi_period": config["rsi"],
                    "history_length": 3000,
                    "stop_pct": pf_config["profit_pct"],
                    "profit_pct": pf_config["profit_pct"],
                })

    return variations


def main():
    """Submit all backtest variations to Ray and wait for completion."""
    logger = Logger().get_logger(__name__)

    # Initialize Ray
    # local
    ray.init(ignore_reinit_error=True)
    # server
    # ray.init(address="ray://192.168.89.173:8265", ignore_reinit_error=True)
    logger.info(f"Ray initialized: {ray.cluster_resources()}")

    # Generate and submit all variations
    variations = generate_variations()
    logger.info(f"Submitting {len(variations)} backtests to Ray...")

    # Store futures to wait for completion
    futures = [run_single_backtest.remote(params) for params in variations]

    logger.info(f"All {len(variations)} jobs submitted.")
    logger.info("MLflow UI: http://hp.lan:8899")
    logger.info("Waiting for all jobs to complete...")

    # Wait for all jobs to finish
    ray.get(futures)

    logger.info("All jobs completed!")


if __name__ == "__main__":
    main()
