from pathlib import Path
from typing import Type
import math

from trading.core.algorithm import Algorithm
from trading.algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.om.order_manager import OrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.core.portfolio import Portfolio
from trading.data_providers.data_provider import DataProvider
from trading.data_providers.test_data_provider import TestDataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.analysis.analysis_engine import AnalysisEngine
from utils.logger import Logger
from utils.utils import apply_tunable_config
import ray
from ray import air
from ray import tune
from ray.tune.search.optuna import OptunaSearch

logger = Logger().get_logger(__name__)


def _short_trial_dirname_creator(trial) -> str:
    """Keep Ray trial directory names short enough for Windows path limits."""
    return f"trial_{trial.trial_id}"


def _build_algorithm(algorithm_class: Type[Algorithm], alg_cfg: dict) -> Algorithm:
    alg_cfg_local = dict(alg_cfg)
    history_length = alg_cfg_local.pop("history_length", 0)
    return algorithm_class(alg_cfg_local, history_length=history_length)


def create_multiple_backtests(
    algorithm_class: Type[Algorithm] = MacdRsiAlgorithm,
    portfolio_class: Type[Portfolio] = SingleSymbolPortfolio,
    data_provider_class: Type[DataProvider] = TestDataProvider,
    order_manager_class: Type[OrderManager] = BacktestingOrderManager,
) -> list[dict]:
    """
    Create multiple backtest configurations.

    Args:
        algorithm_class: The algorithm class to use for backtesting
        portfolio_class: The portfolio class to use for backtesting
        data_provider_class: The data provider class to use for backtesting
        order_manager_class: The order manager class to use for backtesting

    Returns:
        List of backtest configuration dictionaries
    """
    # Parameters to test
    tickers = ["SPY"]
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
                dp_cfg = {"path": data_path}
                cfg = {
                    "backtest_cfg": backtest_cfg,
                    "alg_cfg": alg_cfg,
                    "pf_cfg": pf_cfg,
                    "dp_cfg": dp_cfg,
                    "algorithm_class": algorithm_class,
                    "portfolio_class": portfolio_class,
                    "data_provider_class": data_provider_class,
                    "order_manager_class": order_manager_class,
                }
                ret_val.append(cfg)
    return ret_val

def run_backtest_core(
    backtest_cfg: dict,
    alg_cfg: dict,
    pf_cfg: dict,
    dp_cfg: dict,
    algorithm_class: Type[Algorithm],
    portfolio_class: Type[Portfolio],
    data_provider_class: Type[DataProvider],
    order_manager_class: Type[OrderManager],
    log_to_mlflow: bool = True,
) -> dict:
    """
    Core backtest execution logic used by both local and Ray remote functions.

    Args:
        backtest_cfg: Backtest configuration (experiment_name, symbol, starting_cash, run_name, description)
        alg_cfg: Algorithm configuration
        pf_cfg: Portfolio configuration
        dp_cfg: Data provider configuration
        algorithm_class: The algorithm class to instantiate
        portfolio_class: The portfolio class to instantiate
        data_provider_class: The data provider class to instantiate
        order_manager_class: The order manager class to instantiate

    Returns:
        Results dictionary from AnalysisEngine.run_full_analysis()
    """
    experiment_name = backtest_cfg['experiment_name']
    starting_cash = backtest_cfg['starting_cash']
    run_name = backtest_cfg['run_name']
    desc = backtest_cfg['description']
    config_artifact_path = backtest_cfg.get('config_artifact_path')
    git_tags = backtest_cfg.get('git_tags') or {}
    benchmark_paths = backtest_cfg.get('benchmark_paths') or {}
    params = backtest_cfg | alg_cfg | pf_cfg | dp_cfg

    print(f"Running backtest for {backtest_cfg}")

    om = order_manager_class()
    al = _build_algorithm(algorithm_class, alg_cfg)
    dp = data_provider_class(dp_cfg)
    pf = portfolio_class(pf_cfg, om, starting_cash, {}, True)

    sim = BacktestingEngine({"state_store": {"enabled": False}}, dp, al, om, pf)
    sim.run()

    print(f"\nAnalyzing results for run {run_name}")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        experiment_name=experiment_name,
        run_name=run_name,
        description=desc,
        parameters=params,
        tracking_uri=backtest_cfg.get("tracking_uri"),
        log_to_mlflow=log_to_mlflow,
        save_charts_locally=False,
        save_report_locally=False,
        tags=git_tags if git_tags else None,
        artifact_paths=[config_artifact_path] if config_artifact_path else None,
        benchmark_paths=benchmark_paths if benchmark_paths else None,
    )
    return results


@ray.remote
def run_backtest(
    backtest_cfg: dict,
    alg_cfg: dict,
    pf_cfg: dict,
    dp_cfg: dict,
    algorithm_class: Type[Algorithm],
    portfolio_class: Type[Portfolio],
    data_provider_class: Type[DataProvider],
    order_manager_class: Type[OrderManager],
) -> dict:
    """
    Ray remote wrapper for running a backtest.

    Args:
        backtest_cfg: Backtest configuration
        alg_cfg: Algorithm configuration
        pf_cfg: Portfolio configuration
        dp_cfg: Data provider configuration
        algorithm_class: The algorithm class to instantiate
        portfolio_class: The portfolio class to instantiate
        data_provider_class: The data provider class to instantiate
        order_manager_class: The order manager class to instantiate

    Returns:
        Results dictionary from AnalysisEngine.run_full_analysis()
    """
    return run_backtest_core(
        backtest_cfg, alg_cfg, pf_cfg, dp_cfg,
        algorithm_class, portfolio_class, data_provider_class, order_manager_class
    )

def run_parallel_backtests(
    algorithm_class: Type[Algorithm] = MacdRsiAlgorithm,
    portfolio_class: Type[Portfolio] = SingleSymbolPortfolio,
    data_provider_class: Type[DataProvider] = TestDataProvider,
    order_manager_class: Type[OrderManager] = BacktestingOrderManager,
) -> list[dict]:
    """
    Run multiple backtests in parallel using Ray.

    Args:
        algorithm_class: The algorithm class to use for backtesting
        portfolio_class: The portfolio class to use for backtesting
        data_provider_class: The data provider class to use for backtesting
        order_manager_class: The order manager class to use for backtesting

    Returns:
        List of results from all backtests
    """
    bts = create_multiple_backtests(
        algorithm_class=algorithm_class,
        portfolio_class=portfolio_class,
        data_provider_class=data_provider_class,
        order_manager_class=order_manager_class,
    )
    logger.info(f"Submitting {len(bts)} backtests to Ray")

    ray.init(ignore_reinit_error=True)

    # Submit ALL jobs first
    futures = []
    for bt in bts:
        logger.info(f"Submitting backtest to Ray: {bt['backtest_cfg']}")
        futures.append(run_backtest.remote(
            bt['backtest_cfg'],
            bt['alg_cfg'],
            bt['pf_cfg'],
            bt['dp_cfg'],
            bt['algorithm_class'],
            bt['portfolio_class'],
            bt['data_provider_class'],
            bt['order_manager_class'],
        ))

    # Wait for all jobs AFTER submitting all (this is the fix for parallelism)
    logger.info("Waiting for all jobs to complete...")
    results = ray.get(futures)
    logger.info("All jobs completed!")

    return results

def backtest_objective_fn(
    config: dict,
    symbol: str,
    algorithm_class: Type[Algorithm],
    portfolio_class: Type[Portfolio],
    data_provider_class: Type[DataProvider],
    order_manager_class: Type[OrderManager],
    base_algorithm_config: dict,
    base_portfolio_config: dict,
    base_data_provider_config: dict,
    base_backtest_config: dict,
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
    log_to_mlflow: bool = False,
) -> dict:
    """
    Generic objective function for hyperparameter optimization with Ray Tune.

    This function is completely generic and works with any algorithm/portfolio combination
    without requiring code changes or conditional logic.

    Args:
        config: Hyperparameter configuration from Ray Tune (sampled values)
        symbol: Stock symbol to backtest
        algorithm_class: The algorithm class to use
        portfolio_class: The portfolio class to use
        data_provider_class: The data provider class to use
        order_manager_class: The order manager class to use
        base_algorithm_config: Static algorithm parameters
        base_portfolio_config: Static portfolio parameters
        base_data_provider_config: Data provider configuration
        base_backtest_config: Backtest configuration
        algorithm_param_keys: List of config keys that go to algorithm
        portfolio_param_keys: List of config keys that go to portfolio

    Returns:
        Dictionary with optimization metric
    """
    alg_cfg = apply_tunable_config(base_algorithm_config, config, algorithm_param_keys)
    pf_cfg = apply_tunable_config(base_portfolio_config, config, portfolio_param_keys)

    # Use base configs directly (no tuning for these components)
    dp_cfg = base_data_provider_config
    backtest_cfg = base_backtest_config

    # Run backtest with merged configurations
    result = run_backtest_local(
        backtest_cfg, alg_cfg, pf_cfg, dp_cfg,
        algorithm_class, portfolio_class, data_provider_class, order_manager_class,
        log_to_mlflow=log_to_mlflow,
    )

    return {"_metric": result['metrics'].annualized_return}


def run_backtest_local(
    backtest_cfg: dict,
    alg_cfg: dict,
    pf_cfg: dict,
    dp_cfg: dict,
    algorithm_class: Type[Algorithm] = MacdRsiAlgorithm,
    portfolio_class: Type[Portfolio] = SingleSymbolPortfolio,
    data_provider_class: Type[DataProvider] = TestDataProvider,
    order_manager_class: Type[OrderManager] = BacktestingOrderManager,
    log_to_mlflow: bool = True,
) -> dict:
    """
    Run a backtest locally (not using Ray).

    Args:
        backtest_cfg: Backtest configuration
        alg_cfg: Algorithm configuration
        pf_cfg: Portfolio configuration
        dp_cfg: Data provider configuration
        algorithm_class: The algorithm class to instantiate
        portfolio_class: The portfolio class to instantiate
        data_provider_class: The data provider class to instantiate
        order_manager_class: The order manager class to instantiate

    Returns:
        Results dictionary from AnalysisEngine.run_full_analysis()
    """
    return run_backtest_core(
        backtest_cfg, alg_cfg, pf_cfg, dp_cfg,
        algorithm_class, portfolio_class, data_provider_class, order_manager_class,
        log_to_mlflow=log_to_mlflow,
    )

def run_single_backtest(
    backtest_cfg: dict,
    alg: Algorithm,
    pf: Portfolio,
    dp: DataProvider,
) -> dict:
    """
    Run a single backtest with fully instantiated classes.

    This function mirrors run_backtest_local but takes instantiated objects
    instead of classes and configuration dictionaries. Useful when you need
    to manually configure components before running the backtest.

    Args:
        backtest_cfg: Backtest configuration containing:
            - experiment_name: Name of the MLflow experiment
            - run_name: Name for this specific run
            - description: Description of the backtest
            - starting_cash: Initial cash amount (optional, uses pf.cash if not provided)
        alg: Fully instantiated Algorithm object
        pf: Fully instantiated Portfolio object
        dp: Fully instantiated DataProvider object

    Returns:
        Results dictionary from AnalysisEngine.run_full_analysis() containing:
            - trades: List of Trade objects
            - metrics: PerformanceMetrics object
            - tick_returns: Pandas Series of tick-level returns
            - daily_returns: Pandas Series of daily returns
            - monthly_returns: Pandas Series of monthly returns
            - bracket_analysis: Dict with bracket order statistics (if applicable)
            - report: Formatted text report string

    Example:
        from trading.algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
        from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
        from trading.data_providers.test_data_provider import TestDataProvider
        from trading.core.om.backtesting_om import BacktestingOM

        # Instantiate components
        om = BacktestingOM()
        alg = MacdRsiAlgorithm({'macd_fastperiod': 400, 'rsi_period': 50})
        dp = TestDataProvider({'path': 'data/SPY_5min.csv', 'truncate': 10000})
        pf = SingleSymbolPortfolio({'symbol': 'SPY'}, om, 1000.0, {}, True)

        # Run backtest
        backtest_cfg = {
            'experiment_name': 'My Tests',
            'run_name': 'Custom MACD Test',
            'description': 'Testing custom MACD parameters',
            'starting_cash': 1000.0
        }
        results = run_single_backtest(backtest_cfg, alg, pf, dp)
        print(f"Total return: {results['metrics'].total_return_pct:.2f}%")
    """
    experiment_name = backtest_cfg.get('experiment_name', 'Single Backtest')
    run_name = backtest_cfg.get('run_name', 'Backtest Run')
    desc = backtest_cfg.get('description', '')
    starting_cash = backtest_cfg.get('starting_cash', pf.cash)

    # Create parameters dict for logging (merge all configs if available)
    params = dict(backtest_cfg)

    # Try to extract algorithm config if available
    if hasattr(alg, 'cfg') and alg.cfg:
        params.update({f'alg_{k}': v for k, v in alg.cfg.items()})

    # Try to extract portfolio config if available
    if hasattr(pf, 'cfg') and pf.cfg:
        params.update({f'pf_{k}': v for k, v in pf.cfg.items()})

    # Try to extract data provider config if available
    if hasattr(dp, 'cfg') and dp.cfg:
        params.update({f'dp_{k}': v for k, v in dp.cfg.items()})

    print(f"Running single backtest: {run_name}")

    # Get order manager from portfolio
    om = pf.om

    # Run backtest
    sim = BacktestingEngine({}, dp, alg, om, pf)
    sim.run()

    print(f"\nAnalyzing results for run {run_name}")
    engine = AnalysisEngine(sim.pf, pf.om)

    # Run analysis and log to MLflow
    results = engine.run_full_analysis(
        experiment_name=experiment_name,
        run_name=run_name,
        description=desc,
        parameters=params,
        tracking_uri=backtest_cfg.get("tracking_uri"),
        log_to_mlflow=True,
        save_charts_locally=False,
        save_report_locally=False
    )
    return results

def tune_backtest_hyperparameters(
    symbol: str,
    algorithm_class: Type[Algorithm],
    portfolio_class: Type[Portfolio],
    data_provider_class: Type[DataProvider],
    order_manager_class: Type[OrderManager],
    base_algorithm_config: dict,
    base_portfolio_config: dict,
    base_data_provider_config: dict,
    base_backtest_config: dict,
    search_space: dict,
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
    num_samples: int = 50,
    max_concurrent_trials: int = 8,
    log_to_mlflow: bool = False,
    log_ray_worker_output: bool = True,
    return_trial_summaries: bool = False,
) -> dict | tuple[dict, list[dict]]:
    """
    Generic hyperparameter optimization using Ray Tune with Optuna.

    This function is completely generic and works with any algorithm/portfolio combination
    without requiring code changes. All configuration is passed in by the caller.

    Args:
        symbol: Stock symbol to backtest
        algorithm_class: The algorithm class to use
        portfolio_class: The portfolio class to use
        data_provider_class: The data provider class to use
        order_manager_class: The order manager class to use
        base_algorithm_config: Static algorithm parameters (not tuned)
        base_portfolio_config: Static portfolio parameters (not tuned)
        base_data_provider_config: Data provider configuration (path, truncate, etc.)
        base_backtest_config: Backtest configuration (symbol, starting_cash, etc.)
        search_space: Hyperparameter search space with tune.* objects
        algorithm_param_keys: List of search_space keys that go to algorithm
        portfolio_param_keys: List of search_space keys that go to portfolio
        num_samples: Number of hyperparameter combinations to try
        max_concurrent_trials: Maximum number of concurrent trials

    Returns:
        Best hyperparameter configuration found.
        When return_trial_summaries=True, also returns a list of
        {"config": ..., "metric": ...} dictionaries for all completed trials.

    Example:
        search_space = {
            "fast_window": tune.randint(5, 25),
            "slow_window": tune.randint(30, 80),
            "tx_cost": tune.uniform(0.0, 10.0),
        }
        algorithm_param_keys = ["fast_window", "slow_window"]
        portfolio_param_keys = ["tx_cost"]
    """
    # Initialize Ray with dashboard accessible remotely
    ray.init(
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        log_to_driver=log_ray_worker_output,
    )

    # Bind all parameters to the objective function
    trainable_with_params = tune.with_parameters(
        backtest_objective_fn,
        symbol=symbol,
        algorithm_class=algorithm_class,
        portfolio_class=portfolio_class,
        data_provider_class=data_provider_class,
        order_manager_class=order_manager_class,
        base_algorithm_config=base_algorithm_config,
        base_portfolio_config=base_portfolio_config,
        base_data_provider_config=base_data_provider_config,
        base_backtest_config=base_backtest_config,
        algorithm_param_keys=algorithm_param_keys,
        portfolio_param_keys=portfolio_param_keys,
        log_to_mlflow=log_to_mlflow,
    )

    # Configure Bayesian Optimization with Optuna
    optuna_search = OptunaSearch(
        metric="_metric",
        mode="max",
    )

    tuner = tune.Tuner(
        trainable_with_params,
        param_space=search_space,
        run_config=air.RunConfig(
            name="hpo",
        ),
        tune_config=tune.TuneConfig(
            metric="_metric",
            mode="max",
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            search_alg=optuna_search,
            trial_dirname_creator=_short_trial_dirname_creator,
        )
    )
    results = tuner.fit()
    trial_summaries = []
    for result in results:
        metric = result.metrics.get("_metric")
        if metric is None:
            continue
        metric_value = float(metric)
        if not math.isfinite(metric_value):
            continue
        trial_summaries.append({"config": result.config, "metric": metric_value})
    if not trial_summaries:
        raise RuntimeError(
            f"All {num_samples} HPO trials failed or produced no optimization metric. "
            "Check Ray Tune logs for individual trial errors."
        )

    best_config = max(trial_summaries, key=lambda trial: trial["metric"])["config"]
    if return_trial_summaries:
        print(best_config)
        return best_config, trial_summaries
    print(best_config)
    return best_config
