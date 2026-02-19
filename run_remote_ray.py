#!/usr/bin/env python
"""
Quick-start script for running hyperparameter optimization on remote Ray cluster.

Usage:
    python run_remote_ray.py --ray-address ray://192.168.1.100:10001
    python run_remote_ray.py --ray-address ray://192.168.1.100:10001 --samples 1000
    python run_remote_ray.py --check-connection ray://192.168.1.100:10001
"""
import argparse
import sys
import ray


def check_connection(ray_address: str):
    """Check if we can connect to the remote Ray cluster."""
    print(f"Checking connection to {ray_address}...")

    try:
        ray.init(address=ray_address)
        resources = ray.available_resources()

        print("\n" + "="*80)
        print("CONNECTION SUCCESSFUL")
        print("="*80)
        print(f"\nRay Cluster Address: {ray_address}")
        print(f"\nAvailable Resources:")
        for resource, amount in resources.items():
            print(f"  {resource}: {amount}")

        # Get cluster info
        nodes = ray.nodes()
        print(f"\nCluster Nodes: {len(nodes)}")
        for i, node in enumerate(nodes):
            print(f"\n  Node {i+1}:")
            print(f"    Alive: {node['Alive']}")
            print(f"    Resources: {node.get('Resources', {})}")

        print("\n" + "="*80)
        print("Ready to run hyperparameter optimization!")
        print("="*80)

        ray.shutdown()
        return True

    except Exception as e:
        print("\n" + "="*80)
        print("CONNECTION FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Ray head node is running on remote server:")
        print("     ray start --head --dashboard-host=0.0.0.0")
        print("  2. Check firewall allows connections on port 10001")
        print("  3. Verify ray address format: ray://hostname:10001")
        print("  4. Test network connectivity: ping hostname")
        print("="*80)
        return False


def run_optimization(ray_address: str, num_samples: int = 5000, max_concurrent: int = 8):
    """Run hyperparameter optimization on remote cluster."""
    print(f"\nConnecting to Ray cluster at {ray_address}...")

    try:
        ray.init(address=ray_address)
        print("Connected successfully!")
        print(f"Available resources: {ray.available_resources()}")

    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nRun with --check-connection first to diagnose the issue.")
        sys.exit(1)

    print(f"\nStarting hyperparameter optimization:")
    print(f"  Samples: {num_samples}")
    print(f"  Max concurrent trials: {max_concurrent}")
    print()

    # Import here to ensure Ray is initialized first
    from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters
    from ray import tune
    from trading.core.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
    from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.core.om.backtesting_om import BacktestingOrderManager

    # Configuration
    base_alg_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
    }

    base_pf_cfg = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "min_signal_strength": 0,
        "holding_period_hours": 2,
    }

    base_dp_cfg = {
        "path": "../data/SPY_UPRO_SPXU_5min.csv",
        "truncate": 10000000,
        "start_date": "01/01/2022",
    }

    base_backtest_cfg = {
        "symbol": "SPY",
        "run_name": "SPY_MACD_HPO_Remote",
        "description": "SPY MACD Optimization (Remote Cluster)",
        "starting_cash": 1000.0,
        "experiment_name": "SPY_MACD_Remote_Optimization"
    }

    search_space = {
        "macd_fast_period": tune.randint(5, 2500),
        "macd_slow_period": tune.randint(20, 5000),
        "macd_signal_period": tune.randint(5, 2000),
        "strength_scale": tune.uniform(5.0, 5.0),
        "stop_pct": tune.uniform(1.0, 20.0),
        "profit_pct": tune.uniform(1.0, 25.0),
    }

    algorithm_param_keys = ["macd_fast_period", "macd_slow_period", "macd_signal_period", "strength_scale"]
    portfolio_param_keys = ["stop_pct", "profit_pct"]

    # Run optimization
    try:
        best_config = tune_backtest_hyperparameters(
            symbol="SPY",
            algorithm_class=SpyTrendMACDAlgorithm,
            portfolio_class=DualSymbolSwitchPortfolio,
            data_provider_class=TestDataProvider,
            order_manager_class=BacktestingOrderManager,
            base_algorithm_config=base_alg_cfg,
            base_portfolio_config=base_pf_cfg,
            base_data_provider_config=base_dp_cfg,
            base_backtest_config=base_backtest_cfg,
            search_space=search_space,
            algorithm_param_keys=algorithm_param_keys,
            portfolio_param_keys=portfolio_param_keys,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
        )

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nBest hyperparameters:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        print("\n" + "="*80)

        return best_config

    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization on remote Ray cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check connection to remote cluster
  python run_remote_ray.py --check-connection ray://192.168.1.100:10001

  # Run optimization with default settings (5000 samples)
  python run_remote_ray.py --ray-address ray://192.168.1.100:10001

  # Run with custom sample count
  python run_remote_ray.py --ray-address ray://192.168.1.100:10001 --samples 1000

  # Run with more concurrent trials (if you have more CPUs)
  python run_remote_ray.py --ray-address ray://192.168.1.100:10001 --max-concurrent 16
        """
    )

    parser.add_argument(
        "--ray-address",
        type=str,
        required=True,
        help="Ray cluster address (e.g., ray://192.168.1.100:10001)"
    )

    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Only check connection, don't run optimization"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of hyperparameter combinations to try (default: 5000)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum number of concurrent trials (default: 8)"
    )

    args = parser.parse_args()

    if args.check_connection:
        success = check_connection(args.ray_address)
        sys.exit(0 if success else 1)
    else:
        run_optimization(args.ray_address, args.samples, args.max_concurrent)


if __name__ == "__main__":
    main()
