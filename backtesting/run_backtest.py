
from pathlib import Path

from scripts.regsetup import description

from algorithms.test_algorithm import TestAlgorithm
from algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from data_providers.test_data_provider import TestDataProvider
from engines.backtest_engine import BacktestingEngine
from engines.analysis_engine import AnalysisEngine
from utils.logger import Logger
import ray
from ray import tune

logger = Logger().get_logger(__name__)

@ray.remote
def run_backtest(backtest_cfg: dict, alg_cfg: dict, pf_cfg: dict, dp_cfg: dict):
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
    data_path = str(project_root / "data" / "GDXU_5min.csv")
    dp_cfg = {
        "path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider",
        "truncate": 10000
    }
    backtest_cfg = {
        "symbol": "GDXU","run_name": "GDXU_Recreation", "description": "Backtest", "starting_cash": 1000.0,
        "experiment_name": "Backtesting"
    }
    alg_cfg = {
        "macd_fastperiod": 540,
        "macd_slowperiod": 1170,
        "macd_signalperiod": 405,
        "rsi_period": 14,
        "extra_history_period": 10000
    }
    pf_cfg = {"symbol": "GDXU", "stop_pct": 5, "profit_pct": 10}
    run_backtest_local(backtest_cfg, alg_cfg, pf_cfg, dp_cfg)

def create_multiple_backtests() -> list[dict]:

    # Parameters to test
    tickers = ["GDXU", "SPXU", "UPRO"]
    periods = [
        {"macd_fastperiod": 12 * mult, "macd_slowperiod": 26 * mult, "macd_signalperiod": 9 * mult, "rsi_period": 14 * mult}
        for mult in [30, 35, 40, 45, 50]]
    profit_capture = [(10, 5), (10, 10), (15, 5), (5, 15), (5, 20), (5, 10), (5, 5)]
    ex = "Backtesting"

    project_root = Path(__file__).parent.parent

    ret_val = []

    for ticker in tickers:
        for period in periods:
            for pc in profit_capture:

                backtest_cfg = {
                    "symbol": ticker, "run_name": f"{ticker}_MACD_RSI", "experiment_name": ex,
                    "starting_cash": 1000.0, "description": f"{ticker}_MACD_RSI"}

                alg_cfg = {
                    "macd_fastperiod": period["macd_fastperiod"],
                    "macd_slowperiod": period["macd_slowperiod"],
                    "macd_signalperiod": period["macd_signalperiod"],
                    "rsi_period": period["rsi_period"],
                    "extra_history_period": 1000
                }

                pf_cfg = {
                    "symbol": ticker,
                    "stop_pct": pc[0],
                    "profit_pct": pc[1]
                }

                data_path = str(project_root / "data" / f"{ticker}_5min.csv")
                dp_cfg = {"path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider"}
                cfg = {
                    "backtest_cfg": backtest_cfg,
                    "alg_cfg": alg_cfg,
                    "pf_cfg": pf_cfg,
                    "dp_cfg": dp_cfg
                }
                ret_val.append(cfg)
    return ret_val

def run_multiple_backtests():
    bts = create_multiple_backtests()

    for bt in bts:
        logger.info(f"Running backtest: {bt}")
        run_backtest(
            bt['backtest_cfg'],
            bt['alg_cfg'],
            bt['pf_cfg'],
            bt['dp_cfg'],
        )

def run_parallel_backtests():
    bts = create_multiple_backtests()
    logger.info(f"Submitting {len(bts)} backtests to Ray")
    futures = []
    ray.init(ignore_reinit_error=True)

    for bt in bts:
        logger.info(f"Submitting backtests to Ray: {bt}")

        btc = bt['backtest_cfg']
        alc = bt['alg_cfg']
        pfc = bt['pf_cfg']
        dpc = bt['dp_cfg']
        futures.append(run_backtest.remote(btc, alc, pfc, dpc))

        logger.info("Waiting for all jobs to complete...")

        # Wait for all jobs to finish
        ray.get(futures)

        logger.info("All jobs completed!")

def backtest_objective_fn(config: dict) -> float:

    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "GDXU_5min.csv")
    dp_cfg = {
        "path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider",
        "truncate": 10000000
    }
    backtest_cfg = {
        "symbol": "GDXU","run_name": "GDXU_Recreation", "description": "Backtest", "starting_cash": 1000.0,
        "experiment_name": "Hyperparameter Optimization Runs"
    }
    alg_cfg = {
        "macd_fastperiod": int(config["macd_fastperiod"]),
        "macd_slowperiod": int(config["macd_slowperiod"]),
        "macd_signalperiod": int(config["macd_signalperiod"]),
        "rsi_period": int(config["rsi_period"]),
        "extra_history_period": int(config["extra_history_period"])
    }
    pf_cfg = {
        "symbol": "GDXU",
        "stop_pct": int(config["stop_pct"]),
        "profit_pct": int(config["profit_pct"])
    }
    result = run_backtest_local(backtest_cfg, alg_cfg, pf_cfg, dp_cfg)
    return result['metrics'].annualized_return * -1

def tune_backtest_hyperparameters():
    search_space = {
        "macd_fastperiod": tune.uniform(400, 2000),
        "macd_slowperiod": tune.uniform(1000, 5000),
        "macd_signalperiod": tune.uniform(100, 2000),
        "rsi_period": tune.uniform(10, 2000),
        "stop_pct": tune.uniform(0, 25),
        "profit_pct": tune.uniform(0, 25),
        "extra_history_period": tune.uniform(1000, 20000),
    }
    tuner = tune.Tuner(backtest_objective_fn, param_space=search_space)
    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min").config)

if __name__ == "__main__":
    tune_backtest_hyperparameters()
    example_usage()
    run_multiple_backtests()
    run_parallel_backtests()