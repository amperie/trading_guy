from pathlib import Path


def run_ray_macd_rsi():
    """
    Run Ray Tune hyperparameter optimization for MacdRsiAlgorithm + SingleSymbolPortfolio.

    Optimizes MACD/RSI indicator parameters and stop-loss/profit-taker percentages.
    """
    from run_backtest_ray import tune_backtest_hyperparameters
    from ray import tune
    from trading.core.algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
    from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOM

    # Base configurations (static parameters that don't change)
    base_alg_cfg = {}  # MacdRsi has no static params - all are tuned

    base_pf_cfg = {
        "symbol": "SPY",
    }

    base_dp_cfg = {
        "path": "data/SPY_5min.csv",
        "truncate": 10000000
    }

    base_backtest_cfg = {
        "symbol": "SPY",
        "run_name": "SPY_MacdRsi_HPO",
        "description": "MACD RSI Optimization",
        "starting_cash": 1000.0,
        "experiment_name": "Hyperparameter Optimization Runs"
    }

    # Search space (parameters to optimize)
    search_space = {
        "macd_fastperiod": tune.uniform(400, 2000),
        "macd_slowperiod": tune.uniform(1000, 5000),
        "macd_signalperiod": tune.uniform(100, 2000),
        "rsi_period": tune.uniform(10, 2000),
        "extra_history_period": tune.uniform(1000, 2000),
        "stop_pct": tune.uniform(1, 25),
        "profit_pct": tune.uniform(1, 25),
    }

    # Specify which hyperparameters go to which component
    algorithm_param_keys = ["macd_fastperiod", "macd_slowperiod", "macd_signalperiod", "rsi_period", "extra_history_period"]
    portfolio_param_keys = ["stop_pct", "profit_pct"]

    # Run optimization
    best_config = tune_backtest_hyperparameters(
        symbol="SPY",
        algorithm_class=MacdRsiAlgorithm,
        portfolio_class=SingleSymbolPortfolio,
        data_provider_class=TestDataProvider,
        order_manager_class=BacktestingOM,

        base_algorithm_config=base_alg_cfg,
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=base_dp_cfg,
        base_backtest_config=base_backtest_cfg,

        search_space=search_space,
        algorithm_param_keys=algorithm_param_keys,
        portfolio_param_keys=portfolio_param_keys,

        num_samples=50,
        max_concurrent_trials=8,
    )

    print(f"Best hyperparameters: {best_config}")
    return best_config

def run_ray_spy_trend_switch():
    """
    Run Ray Tune hyperparameter optimization for SpyTrendSwitchAlgorithm + DualSymbolSwitchPortfolio.

    Optimizes trend detection windows, signal strength scaling, transaction costs, and signal thresholds.
    Tests strategy that switches between UPRO (3x leveraged long) and SPXU (3x leveraged short) based on SPY trend.
    """
    from run_backtest_ray import tune_backtest_hyperparameters
    from ray import tune
    from trading.core.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm
    from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOM

    # Base configurations (static parameters that don't change)
    base_alg_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
    }

    base_pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
    }

    base_dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 10000000
    }

    base_backtest_cfg = {
        "symbol": "SPY",
        "run_name": "SPY_TrendSwitch_HPO",
        "description": "SPY Trend Switch Optimization",
        "starting_cash": 1000.0,
        "experiment_name": "Hyperparameter Optimization Runs"
    }

    # Search space (parameters to optimize)
    search_space = {
        "fast_window": tune.randint(5, 250),          # Short SMA: 5-25 bars
        "slow_window": tune.randint(30, 800),         # Long SMA: 30-80 bars
        "strength_scale": tune.uniform(5.0, 50.0),   # Signal strength multiplier: 5-50x
        "min_signal_strength": tune.randint(0, 1),  # Minimum signal threshold: 0-50
    }

    # Specify which hyperparameters go to which component
    algorithm_param_keys = ["fast_window", "slow_window", "strength_scale"]
    portfolio_param_keys = ["min_signal_strength"]

    # Run optimization
    best_config = tune_backtest_hyperparameters(
        symbol="SPY",
        algorithm_class=SpyTrendSwitchAlgorithm,
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

        num_samples=50,
        max_concurrent_trials=8,
    )

    print(f"Best hyperparameters: {best_config}")
    return best_config


def run_single_spy_trend_switch():
    from trading.core.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm
    from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOM
    from trading.backtesting.run_backtest_ray import run_single_backtest
    # Base configurations (static parameters that don't change)
    
    alg_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "fast_window": 22,
        "slow_window": 220,
        "strength_scale": 44,
    }

    pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "min_signal_strength": 0,
    }

    dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 10000000
    }

    backtest_cfg = {
        "symbol": "SPY",
        "run_name": "SPY_TrendSwitch_Manual",
        "description": "SPY Trend Switch Optimization",
        "starting_cash": 1000.0,
        "experiment_name": "Hyperparameter Optimization Runs"
    }
    # Instantiate components
    om = BacktestingOM()
    alg = SpyTrendSwitchAlgorithm(alg_cfg)
    dp = TestDataProvider(dp_cfg)
    pf = DualSymbolSwitchPortfolio(pf_cfg, om, 1000.0, {}, True)

    results = run_single_backtest(backtest_cfg, alg, pf, dp)
    print(f"Total return: {results['metrics'].total_return_pct:.2f}%")


if __name__ == "__main__":
    run_ray_spy_trend_switch()
    # run_single_spy_trend_switch()