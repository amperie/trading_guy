import math
import os
import signal
import time
from numbers import Integral, Real
from pathlib import Path
from typing import Type

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
from utils.status_line import StatusLine
from utils.utils import apply_tunable_config
import ray
from ray import air
from ray import tune
from ray.tune.search.optuna import OptunaSearch

logger = Logger().get_logger(__name__)
_MISSING = object()


def _safe_ray_shutdown() -> None:
    try:
        ray.shutdown()
    except Exception:
        logger.warning("Ray shutdown failed during cleanup; continuing.", exc_info=True)


def _set_interrupt_handlers(handler) -> dict[int, object]:
    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is None:
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        except Exception:
            continue
    return previous


def _restore_interrupt_handlers(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except Exception:
            continue


class _TuneStatusCallback(tune.Callback):
    """TTY status line for driver-side Ray Tune progress."""

    def __init__(self, total_trials: int, metric_name: str = "_metric", enabled: bool | None = None):
        self.total_trials = max(0, int(total_trials))
        self.metric_name = metric_name
        self.completed_trials = 0
        self.best_metric = None
        self.started_at = time.monotonic()
        self.status_line = StatusLine(enabled=enabled)

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        total = int(max(0, round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _maybe_update_best(self, trial) -> None:
        metric = getattr(trial, "last_result", {}).get(self.metric_name)
        if metric is None:
            return
        metric = float(metric)
        if not math.isfinite(metric):
            return
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric

    def _refresh(self, final: bool = False) -> None:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        eta = None
        if self.completed_trials > 0 and self.total_trials > self.completed_trials:
            eta = (elapsed / self.completed_trials) * (self.total_trials - self.completed_trials)
        text = (
            f"[HPO] trials={self.completed_trials}/{self.total_trials} "
            f"elapsed={self._format_duration(elapsed)} "
            f"eta={self._format_duration(eta)}"
        )
        if self.best_metric is not None:
            text += f" best={self.best_metric:.4f}"
        if final:
            text += " completed"
        self.status_line.update(text)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self._maybe_update_best(trial)
        self._refresh()

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.completed_trials += 1
        self._maybe_update_best(trial)
        self._refresh()

    def on_trial_error(self, iteration, trials, trial, **info):
        self.completed_trials += 1
        self._refresh()

    def close(self) -> None:
        self._refresh(final=True)
        self.status_line.close()


def _short_trial_dirname_creator(trial) -> str:
    """Keep Ray trial directory names short enough for Windows path limits."""
    return f"trial_{trial.trial_id}"


def _get_nested_value(cfg: dict, dotted_key: str):
    current = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _validate_seed_value(key: str, value, domain) -> str | None:
    if hasattr(domain, "categories"):
        if value not in domain.categories:
            return f"not in search space choices {list(domain.categories)!r}"
        return None

    lower = getattr(domain, "lower", None)
    upper = getattr(domain, "upper", None)
    domain_name = domain.__class__.__name__.lower()
    if lower is None or upper is None:
        return None
    if "integer" in domain_name:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < lower or int(value) >= upper:
            return f"outside randint range [{lower}, {upper})"
        return None
    if "float" in domain_name:
        if isinstance(value, bool) or not isinstance(value, Real) or float(value) < lower or float(value) > upper:
            return f"outside float range [{lower}, {upper}]"
    return None


def _add_seed_value(seeded: dict[str, object], key: str, value, domain) -> None:
    invalid_reason = _validate_seed_value(key, value, domain)
    if invalid_reason:
        logger.warning("Skipping seeded HPO value %s=%r because it is %s", key, value, invalid_reason)
        return
    seeded[key] = value


def _build_seeded_trial_config(
    search_space: dict,
    base_algorithm_config: dict,
    base_portfolio_config: dict,
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
) -> dict[str, object]:
    seeded: dict[str, object] = {}
    for key in algorithm_param_keys:
        if key not in search_space:
            continue
        value = _get_nested_value(base_algorithm_config, key)
        if value is _MISSING:
            continue
        _add_seed_value(seeded, key, value, search_space[key])
    for key in portfolio_param_keys:
        if key not in search_space:
            continue
        value = _get_nested_value(base_portfolio_config, key)
        if value is _MISSING:
            continue
        _add_seed_value(seeded, key, value, search_space[key])
    missing_seed_keys = set(search_space) - set(seeded)
    if missing_seed_keys:
        logger.warning(
            "Skipping HPO warm-start seed because it does not cover all search-space keys: missing=%s",
            sorted(missing_seed_keys),
        )
        return {}
    return seeded


def _restore_env_var(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


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
    warmup_dp_cfg: dict | None = None,
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
    if warmup_dp_cfg is not None and al.required_warmup_bars > 0:
        warmup_dp = data_provider_class(warmup_dp_cfg)
        warmup_ticks = list(warmup_dp.iterate())
        if warmup_ticks:
            al.warm_up(warmup_ticks)
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
        backtest_cfg,
        alg_cfg,
        pf_cfg,
        dp_cfg,
        algorithm_class,
        portfolio_class,
        data_provider_class,
        order_manager_class,
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
    warmup_data_provider_config: dict | None = None,
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
        backtest_cfg, alg_cfg, pf_cfg, dp_cfg, warmup_data_provider_config,
        algorithm_class, portfolio_class, data_provider_class, order_manager_class,
        log_to_mlflow=log_to_mlflow,
    )

    score, details = objective_score(result["metrics"], backtest_cfg.get("objective"))
    return {"_metric": score, **details}


def objective_score(metrics, objective: dict | str | None = None) -> tuple[float, dict[str, float]]:
    if objective is None:
        objective = {"metric": "annualized_return"}
    if isinstance(objective, str):
        objective = {"metric": objective}
    metric = str(objective.get("metric", "annualized_return"))
    if metric != "composite_v1":
        value = _metric_value(metrics, metric)
        return value, {f"_objective_{metric}": value}

    weights = {
        "annualized_return": 1.0,
        "max_drawdown_pct": -1.5,
        "volatility": -0.25,
        "sortino_ratio": 5.0,
        "calmar_ratio": 2.0,
        **(objective.get("weights") or {}),
    }
    gates = objective.get("gates") or {}
    min_trades = int(gates.get("min_trades", 0) or 0)
    max_drawdown = gates.get("max_drawdown_pct")
    total_trades = int(_metric_value(metrics, "total_trades", 0.0) or 0)
    drawdown = abs(_metric_value(metrics, "max_drawdown_pct", 0.0))
    penalty = 0.0
    if total_trades < min_trades:
        penalty += float(objective.get("low_trade_penalty", 1000.0)) * (min_trades - total_trades)
    if max_drawdown is not None and drawdown > abs(float(max_drawdown)):
        penalty += float(objective.get("drawdown_gate_penalty", 100.0)) * (drawdown - abs(float(max_drawdown)))

    components = {key: _objective_component(metrics, key) for key in weights}
    score = sum(float(weight) * components[key] for key, weight in weights.items()) - penalty
    details = {f"_objective_{key}": float(value) for key, value in components.items()}
    details["_objective_penalty"] = float(penalty)
    details["_objective_total_trades"] = float(total_trades)
    return float(score), details


def _metric_value(metrics, key: str, default: float | None = None) -> float:
    value = getattr(metrics, key, default)
    if value is None:
        value = default
    if value is None:
        raise ValueError(f"Unknown objective metric '{key}'")
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _objective_component(metrics, key: str) -> float:
    value = _metric_value(metrics, key, 0.0)
    if key in {"max_drawdown", "max_drawdown_pct", "ulcer_index"}:
        return abs(value)
    return value


def run_backtest_local(
    backtest_cfg: dict,
    alg_cfg: dict,
    pf_cfg: dict,
    dp_cfg: dict,
    warmup_dp_cfg: dict | None = None,
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
        backtest_cfg,
        alg_cfg,
        pf_cfg,
        dp_cfg,
        algorithm_class,
        portfolio_class,
        data_provider_class,
        order_manager_class,
        warmup_dp_cfg=warmup_dp_cfg,
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
    warmup_data_provider_config: dict | None = None,
    ray_storage_path: str | None = None,
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
    previous_sigint_setting = os.environ.get("TUNE_DISABLE_SIGINT_HANDLER")
    previous_signal_handlers = _set_interrupt_handlers(signal.default_int_handler)
    os.environ["TUNE_DISABLE_SIGINT_HANDLER"] = "1"
    status_callback = _TuneStatusCallback(num_samples)
    seeded_trial_config = _build_seeded_trial_config(
        search_space=search_space,
        base_algorithm_config=base_algorithm_config,
        base_portfolio_config=base_portfolio_config,
        algorithm_param_keys=algorithm_param_keys,
        portfolio_param_keys=portfolio_param_keys,
    )

    try:
        ray.init(
            ignore_reinit_error=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            log_to_driver=log_ray_worker_output,
        )

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
            warmup_data_provider_config=warmup_data_provider_config,
            base_backtest_config=base_backtest_config,
            algorithm_param_keys=algorithm_param_keys,
            portfolio_param_keys=portfolio_param_keys,
            log_to_mlflow=log_to_mlflow,
        )

        optuna_search_kwargs = {
            "metric": "_metric",
            "mode": "max",
        }
        if seeded_trial_config:
            logger.info("Seeding first HPO trial with current config values: %s", seeded_trial_config)
            optuna_search_kwargs["points_to_evaluate"] = [seeded_trial_config]
        optuna_search = OptunaSearch(**optuna_search_kwargs)

        tuner = tune.Tuner(
            trainable_with_params,
            param_space=search_space,
            run_config=air.RunConfig(
                name="hpo",
                storage_path=ray_storage_path,
                callbacks=[status_callback],
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
            objective_details = {}
            for key, value in result.metrics.items():
                if str(key).startswith("_objective_") and value is not None:
                    try:
                        objective_details[key] = float(value)
                    except (TypeError, ValueError):
                        continue
            trial_summaries.append({"config": result.config, "metric": metric_value, "objective_details": objective_details})
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
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received during Ray Tune HPO; shutting Ray down immediately.")
        raise
    finally:
        status_callback.close()
        _restore_interrupt_handlers(previous_signal_handlers)
        _restore_env_var("TUNE_DISABLE_SIGINT_HANDLER", previous_sigint_setting)
        if ray.is_initialized():
            _safe_ray_shutdown()
