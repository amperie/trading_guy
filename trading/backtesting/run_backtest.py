
from pathlib import Path

from trading.core.algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
from trading.core.algorithms.test_algorithm import ReadSignalsFromFile
from trading.core.om.backtesting_om import BacktestingOM
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.engines.analysis_engine import AnalysisEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def run_backtest_local(backtest_cfg: dict, alg_cfg: dict, pf_cfg: dict, dp_cfg: dict):
    # Run a backtest using the simulator
    experiment_name = backtest_cfg['experiment_name']
    symbol = backtest_cfg['symbol']
    starting_cash = backtest_cfg['starting_cash']
    run_name = backtest_cfg['run_name']
    desc = backtest_cfg['description']
    params = backtest_cfg | alg_cfg | pf_cfg | dp_cfg

    print(f"Running backtest for {backtest_cfg}")
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / f"{symbol}_5min.csv")
    om = BacktestingOM()

    al = MacdRsiAlgorithm(alg_cfg)
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio(pf_cfg, om, starting_cash, {}, True)
    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Create the analysis engine
    print(f"\nAnalyzing results for run {run_name}")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        experiment_name=experiment_name,
        run_name=run_name,
        description=desc,
        parameters=params,
        log_to_mlflow=True,
        save_charts_locally=False,
        save_report_locally=False
    )
    return results

def example_usage():
    project_root = Path(__file__).parent.parent
    symbol = "UPRO"
    data_path = str(project_root / "data" / f"{symbol}_5min.csv")
    dp_cfg = {
        "path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider",
        "truncate": 2000000,
        # "start_date": "08/01/2024"
    }
    backtest_cfg = {
        "symbol": symbol,"run_name": f"{symbol}_test", "description": "Backtest", "starting_cash": 1000.0,
        "experiment_name": "Tests"
    }

    alg_cfg = {
        "macd_fastperiod": 54,
        "macd_slowperiod": 117,
        "macd_signalperiod": 40,
        "rsi_period": 14,
        "extra_history_period": 50
    }
    """
    alg_cfg = {
        "threshold": .15,
        "csv_path": str(project_root / "data" / f"dr_signals_{symbol}.csv"),
    }"""
    pf_cfg = {"symbol": symbol, "stop_pct": 2, "profit_pct": 5}
    run_backtest_local(backtest_cfg, alg_cfg, pf_cfg, dp_cfg)

def example_usage2():
    project_root = Path(__file__).parent.parent
    symbol = "SPY"
    data_path = str(project_root / ".." / "data" / f"{symbol}_5min.csv")
    dp_cfg = {
        "path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider",
        "truncate": 2000000,
        # "start_date": "08/01/2024"
    }
    backtest_cfg = {
        "symbol": symbol,"run_name": f"{symbol}_test", "description": "Backtest", "starting_cash": 1000.0,
        "experiment_name": "Tests"
    }

    alg_cfg = {
        "macd_fastperiod": 54,
        "macd_slowperiod": 117,
        "macd_signalperiod": 40,
        "rsi_period": 14,
        "extra_history_period": 50
    }
    """
    alg_cfg = {
        "threshold": .15,
        "csv_path": str(project_root / "data" / f"dr_signals_{symbol}.csv"),
    }"""
    pf_cfg = {"symbol": symbol, "stop_pct": 2, "profit_pct": 5}
    run_backtest_local(backtest_cfg, alg_cfg, pf_cfg, dp_cfg)

if __name__ == "__main__":
    #tune_backtest_hyperparameters()
    example_usage2()
    #run_multiple_backtests()
    #run_parallel_backtests()
