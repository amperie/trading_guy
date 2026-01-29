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
        "spy_symbol": "UPRO",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
    }

    base_pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
    }

    base_dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 10000000,
        "start_date": "01/01/2022"
    }

    base_backtest_cfg = {
        "symbol": "UPRO",
        "run_name": "UPRO_TrendSwitch_HPO",
        "description": "SPY Trend Switch Optimization",
        "starting_cash": 1000.0,
        "experiment_name": "UPRO_switching_HyperspaceContour"
    }

    # Search space (parameters to optimize)
    search_space = {
        "fast_window": tune.randint(800, 900),          # Short SMA: 5-25 bars
        "slow_window": tune.randint(500, 2000),         # Long SMA: 30-80 bars
        "strength_scale": tune.uniform(5.0, 5.0),   # Signal strength multiplier: 5-50x
        "min_signal_strength": tune.randint(0, 1),  # Minimum signal threshold: 0-50
    }

    # Specify which hyperparameters go to which component
    algorithm_param_keys = ["fast_window", "slow_window", "strength_scale"]
    portfolio_param_keys = ["min_signal_strength"]

    # Run optimization
    best_config = tune_backtest_hyperparameters(
        symbol="UPRO",
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

        num_samples=5000,
        max_concurrent_trials=8,
    )

    print(f"Best hyperparameters: {best_config}")
    return best_config


def run_ray_spy_trend_macd():
    """
    Run Ray Tune hyperparameter optimization for SpyTrendMACDAlgorithm + DualSymbolSwitchPortfolio.

    Optimizes MACD parameters (fast/slow/signal periods), signal strength scaling, and portfolio thresholds.
    Tests strategy that switches between UPRO (3x leveraged long) and SPXU (3x leveraged short)
    based on SPY MACD trend signals.

    MACD Parameters:
    - macd_fast_period: Fast EMA period (standard: 12)
    - macd_slow_period: Slow EMA period (standard: 26)
    - macd_signal_period: Signal line EMA period (standard: 9)
    - strength_scale: Signal strength multiplier

    The algorithm generates BUY signals when MACD crosses above/below signal line.
    """
    from run_backtest_ray import tune_backtest_hyperparameters
    from ray import tune
    from trading.core.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
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
        "truncate": 10000000,
        "start_date": "01/01/2022"
    }

    base_backtest_cfg = {
        "symbol": "SPY",
        "run_name": "SPY_MACD_HPO",
        "description": "SPY MACD Trend Switch Optimization",
        "starting_cash": 1000.0,
        "experiment_name": "SPY_MACD_Hyperparameter_Optimization"
    }

    # Search space (parameters to optimize)
    search_space = {
        # MACD Algorithm Parameters
        "macd_fast_period": tune.randint(5, 25),        # Fast EMA: 5-25 periods (standard: 12)
        "macd_slow_period": tune.randint(20, 50),       # Slow EMA: 20-50 periods (standard: 26)
        "macd_signal_period": tune.randint(5, 20),      # Signal line: 5-20 periods (standard: 9)
        "strength_scale": tune.uniform(5.0, 50.0),      # Signal strength multiplier: 5-50x

        # Portfolio Parameters
        "min_signal_strength": tune.randint(0, 50),     # Minimum signal threshold: 0-50
    }

    # Specify which hyperparameters go to which component
    algorithm_param_keys = ["macd_fast_period", "macd_slow_period", "macd_signal_period", "strength_scale"]
    portfolio_param_keys = ["min_signal_strength"]

    # Run optimization
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


def run_split_period_spy_trend_switch():
    """
    Run split-period backtest (training + validation) for SpyTrendSwitchAlgorithm.

    This function uses SplitPeriodBacktestEngine to run two sequential backtests:
    - Training period: Uses data from start_date to cutoff_date
    - Validation period: Uses data from cutoff_date to end_date

    Both periods start fresh with the same initial cash and use identical parameters.
    Results are logged to MLflow with validation run including training metrics for comparison.
    """
    from trading.core.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm
    from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOM
    from trading.engines.split_period_backtest_engine import SplitPeriodBacktestEngine

    # Algorithm configuration
    alg_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "fast_window": 825,
        "slow_window": 958,
        "strength_scale": 43,
    }

    # Portfolio configuration
    pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "min_signal_strength": 0,
        "stop_pct": 5.0,
        "profit_pct": 10.0,
        "holding_period_hours": 2,
    }

    # Data provider configuration
    dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 0  # Use all available data
    }

    # Backtest configuration
    backtest_cfg = {
        "experiment_name": "SPY Trend Switch Validation",
        "run_name": "SPY_TrendSwitch_TrainVal",
        "description": "Split-period validation test for SPY trend switching strategy",
    }

    # Instantiate components
    om = BacktestingOM()
    alg = SpyTrendSwitchAlgorithm(alg_cfg)
    dp = TestDataProvider(dp_cfg)
    pf = DualSymbolSwitchPortfolio(pf_cfg, om, 1000.0, {}, True)

    # Create and run split-period engine
    engine = SplitPeriodBacktestEngine(
        cfg=backtest_cfg,
        dp=dp,
        al=alg,
        om=om,
        pf=pf,
        start_date='2022-01-03',     # Training start
        cutoff_date='2024-12-31',    # Training end / Validation start
        end_date='2026-12-31'        # Validation end
    )

    print("="*80)
    print("RUNNING SPLIT-PERIOD BACKTEST")
    print("="*80)
    print()

    results = engine.run()

    # Print summary
    print()
    print("="*80)
    print("SPLIT-PERIOD BACKTEST SUMMARY")
    print("="*80)
    print()

    train_metrics = results['training']['metrics']
    val_metrics = results['validation']['metrics']

    print("TRAINING PERIOD (2022-01-03 to 2022-07-01):")
    print(f"  Total Return:     {train_metrics.total_return_pct:>10.2f}%")
    print(f"  Sharpe Ratio:     {train_metrics.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {train_metrics.max_drawdown_pct:>10.2f}%")
    print(f"  Total Trades:     {train_metrics.total_trades:>10}")
    print(f"  Win Rate:         {train_metrics.win_rate:>10.2f}%")
    print()

    print("VALIDATION PERIOD (2022-07-01 to 2022-12-31):")
    print(f"  Total Return:     {val_metrics.total_return_pct:>10.2f}%")
    print(f"  Sharpe Ratio:     {val_metrics.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {val_metrics.max_drawdown_pct:>10.2f}%")
    print(f"  Total Trades:     {val_metrics.total_trades:>10}")
    print(f"  Win Rate:         {val_metrics.win_rate:>10.2f}%")
    print()

    print("PERFORMANCE DELTA (Validation - Training):")
    print(f"  Return Difference:  {val_metrics.total_return_pct - train_metrics.total_return_pct:>10.2f}%")
    print(f"  Sharpe Difference:  {val_metrics.sharpe_ratio - train_metrics.sharpe_ratio:>10.2f}")
    print(f"  Drawdown Change:    {val_metrics.max_drawdown_pct - train_metrics.max_drawdown_pct:>10.2f}%")
    print()

    print("="*80)
    print("MLflow UI: http://hp.lan:8899")
    print(f"Experiment: {backtest_cfg['experiment_name']}")
    print(f"Training Run: {backtest_cfg['run_name']}")
    print(f"Validation Run: {backtest_cfg['run_name']}_validation")
    print("="*80)

    return results


def run_split_period_spy_trend_macd():
    """
    Run split-period backtest (training + validation) for SpyTrendMACDAlgorithm.

    This function uses SplitPeriodBacktestEngine to run two sequential backtests
    using MACD-based trend detection instead of moving average crossovers.

    MACD Components:
    - Fast EMA - Slow EMA = MACD Line
    - EMA of MACD Line = Signal Line
    - MACD > Signal = Bullish -> UPRO
    - MACD < Signal = Bearish -> SPXU
    """
    from trading.core.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
    from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOM
    from trading.engines.split_period_backtest_engine import SplitPeriodBacktestEngine

    # Algorithm configuration (MACD parameters)
    alg_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "macd_fast_period": 12,      # Fast EMA (standard: 12)
        "macd_slow_period": 26,      # Slow EMA (standard: 26)
        "macd_signal_period": 9,     # Signal line (standard: 9)
        "strength_scale": 20.0,      # Signal strength multiplier
    }

    # Portfolio configuration
    pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "min_signal_strength": 0,
        "stop_pct": 5.0,
        "profit_pct": 10.0,
        "holding_period_hours": 2,
    }

    # Data provider configuration
    dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 0  # Use all available data
    }

    # Backtest configuration
    backtest_cfg = {
        "experiment_name": "SPY MACD Trend Switch Validation",
        "run_name": "SPY_MACD_TrainVal",
        "description": "Split-period validation test for SPY MACD trend switching strategy",
    }

    # Instantiate components
    om = BacktestingOM()
    alg = SpyTrendMACDAlgorithm(alg_cfg)
    dp = TestDataProvider(dp_cfg)
    pf = DualSymbolSwitchPortfolio(pf_cfg, om, 1000.0, {}, True)

    # Create and run split-period engine
    engine = SplitPeriodBacktestEngine(
        cfg=backtest_cfg,
        dp=dp,
        al=alg,
        om=om,
        pf=pf,
        start_date='2022-01-03',     # Training start
        cutoff_date='2022-07-01',    # Training end / Validation start
        end_date='2022-12-31'        # Validation end
    )

    print("="*80)
    print("RUNNING SPLIT-PERIOD BACKTEST (MACD-BASED)")
    print("="*80)
    print()
    print("MACD Parameters:")
    print(f"  Fast EMA Period:    {alg_cfg['macd_fast_period']}")
    print(f"  Slow EMA Period:    {alg_cfg['macd_slow_period']}")
    print(f"  Signal Period:      {alg_cfg['macd_signal_period']}")
    print(f"  Strength Scale:     {alg_cfg['strength_scale']}")
    print()

    results = engine.run()

    # Print summary
    print()
    print("="*80)
    print("SPLIT-PERIOD BACKTEST SUMMARY (MACD)")
    print("="*80)
    print()

    train_metrics = results['training']['metrics']
    val_metrics = results['validation']['metrics']

    print("TRAINING PERIOD (2022-01-03 to 2022-07-01):")
    print(f"  Total Return:     {train_metrics.total_return_pct:>10.2f}%")
    print(f"  Sharpe Ratio:     {train_metrics.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {train_metrics.max_drawdown_pct:>10.2f}%")
    print(f"  Total Trades:     {train_metrics.total_trades:>10}")
    print(f"  Win Rate:         {train_metrics.win_rate:>10.2f}%")
    print()

    print("VALIDATION PERIOD (2022-07-01 to 2022-12-31):")
    print(f"  Total Return:     {val_metrics.total_return_pct:>10.2f}%")
    print(f"  Sharpe Ratio:     {val_metrics.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {val_metrics.max_drawdown_pct:>10.2f}%")
    print(f"  Total Trades:     {val_metrics.total_trades:>10}")
    print(f"  Win Rate:         {val_metrics.win_rate:>10.2f}%")
    print()

    print("PERFORMANCE DELTA (Validation - Training):")
    print(f"  Return Difference:  {val_metrics.total_return_pct - train_metrics.total_return_pct:>10.2f}%")
    print(f"  Sharpe Difference:  {val_metrics.sharpe_ratio - train_metrics.sharpe_ratio:>10.2f}")
    print(f"  Drawdown Change:    {val_metrics.max_drawdown_pct - train_metrics.max_drawdown_pct:>10.2f}%")
    print()

    print("="*80)
    print("MLflow UI: http://hp.lan:8899")
    print(f"Experiment: {backtest_cfg['experiment_name']}")
    print(f"Training Run: {backtest_cfg['run_name']}")
    print(f"Validation Run: {backtest_cfg['run_name']}_validation")
    print("="*80)

    return results


if __name__ == "__main__":
    run_split_period_spy_trend_macd()
    # run_ray_spy_trend_switch()
    # run_ray_spy_trend_macd()
    # run_single_spy_trend_switch()
    # run_split_period_spy_trend_switch()
    # run_split_period_spy_trend_macd()