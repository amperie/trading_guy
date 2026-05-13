from __future__ import annotations

from pathlib import Path

from trading.commands.analysis import _collect_config_artifact_paths


def test_collect_config_artifact_paths_includes_runtime_and_component_sources():
    cfg = {
        "mode": "backtest",
        "algorithm": {
            "algorithm": "trading.algorithms.test_algorithm.TestAlgorithm",
            "history_length": 10,
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "symbol": "AAPL",
            "cash": 1000,
        },
        "order_manager": {
            "order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager",
        },
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": "data/SPY_5min_MarketHours.csv",
        },
        "analysis": {
            "enabled": True,
            "log_to_mlflow": True,
        },
    }

    artifacts = _collect_config_artifact_paths(cfg, config_path="configs/example_backtest.yaml")

    names = {Path(path).name for path in artifacts}
    assert "example_backtest.yaml" in names
    assert "runtime_config.yaml" in names
    assert "test_algorithm.py" in names
    assert "single_symbol_portfolio.py" in names
