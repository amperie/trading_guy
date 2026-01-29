"""
Run launchers configured for remote Ray cluster execution.

This module contains the same functions as run_launchers.py but configured
to connect to a remote Ray cluster instead of running locally.
"""
from pathlib import Path
import ray


def run_ray_spy_trend_macd_remote(ray_address: str = "ray://192.168.1.100:10001"):
    """
    Run Ray Tune hyperparameter optimization on a remote Ray cluster.

    Args:
        ray_address: Address of the remote Ray cluster head node
                    Format: "ray://hostname:port" or "ray://ip:port"
                    Default port is usually 10001
                    Get this from `ray start --head` output on remote server

    Example:
        # Remote server IP: 192.168.1.100
        # Ray head node started with: ray start --head
        run_ray_spy_trend_macd_remote("ray://192.168.1.100:10001")
    """
    from run_backtest_ray import tune_backtest_hyperparameters
    from ray import tune
    from trading.core.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
    from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOM

    # Connect to remote Ray cluster
    print(f"Connecting to remote Ray cluster at {ray_address}...")
    ray.init(address=ray_address)
    print("Connected successfully!")
    print(f"Available resources: {ray.available_resources()}")

    # Base configurations (static parameters that don't change)
    base_alg_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
    }

    base_pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "min_signal_strength": 0,       # Fixed: accept all signals (not tuned)
        "holding_period_hours": 2,      # Fixed holding period (not tuned)
    }

    base_dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 10000000,
        "start_date": "01/01/2022",
    }

    base_backtest_cfg = {
        "symbol": "SPY",
        "run_name": "SPY_MACD_HPO_Remote",
        "description": "SPY MACD Trend Switch Optimization with Bracket Orders (Remote)",
        "starting_cash": 1000.0,
        "experiment_name": "SPY_MACD_Hyperparameter_Optimization_Remote"
    }

    # Search space (parameters to optimize)
    search_space = {
        # MACD Algorithm Parameters
        "macd_fast_period": tune.randint(5, 2500),        # Fast EMA: 5-2500 periods (standard: 12)
        "macd_slow_period": tune.randint(20, 5000),       # Slow EMA: 20-5000 periods (standard: 26)
        "macd_signal_period": tune.randint(5, 2000),      # Signal line: 5-2000 periods (standard: 9)
        "strength_scale": tune.uniform(5.0, 5.0),         # Signal strength multiplier: 5.0 (fixed)

        # Portfolio Parameters (Bracket Orders)
        "stop_pct": tune.uniform(1.0, 20.0),              # Stop-loss percentage: 1-20% (standard: 5%)
        "profit_pct": tune.uniform(1.0, 25.0),            # Profit-taker percentage: 1-25% (standard: 10%)
    }

    # Specify which hyperparameters go to which component
    algorithm_param_keys = ["macd_fast_period", "macd_slow_period", "macd_signal_period", "strength_scale"]
    portfolio_param_keys = ["stop_pct", "profit_pct"]

    # Run optimization on remote cluster
    print("Starting hyperparameter optimization on remote cluster...")
    best_config = tune_backtest_hyperparameters(
        symbol="SPY",
        algorithm_class=SpyTrendMACDAlgorithm,
        portfolio_class=DualSymbolSwitchPortfolio,
        data_provider_class=TestDataProvider,
        order_manager_class=BacktestingOM,

        base_algorithm_config=base_alg_cfg,
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=base_dp_cfg,
        base_backtest_config=base_backtest_cfg,

        search_space=search_space,
        algorithm_param_keys=algorithm_param_keys,
        portfolio_param_keys=portfolio_param_keys,

        num_samples=5000,
        max_concurrent_trials=8,
    )

    print(f"Best hyperparameters: {best_config}")

    # Shutdown Ray connection
    ray.shutdown()

    return best_config


if __name__ == "__main__":
    # Replace with your actual remote Ray cluster address
    RAY_CLUSTER_ADDRESS = "ray://192.168.1.100:10001"

    run_ray_spy_trend_macd_remote(RAY_CLUSTER_ADDRESS)
