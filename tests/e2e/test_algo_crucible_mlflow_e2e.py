from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from algo_crucible.orchestrator import CrucibleOrchestrator
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


pytestmark = pytest.mark.e2e


def _tracking_uri() -> str:
    if os.environ.get("CRUCIBLE_E2E_MLFLOW_URI"):
        return os.environ["CRUCIBLE_E2E_MLFLOW_URI"]
    with Path("config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["mlflow"]["tracking_uri"]


def test_crucible_populates_representative_mlflow_parent_run(tmp_path: Path):
    if os.environ.get("RUN_CRUCIBLE_E2E_MLFLOW") != "1":
        pytest.skip("Set RUN_CRUCIBLE_E2E_MLFLOW=1 to write a representative run to MLflow")

    run_name = os.environ.get("CRUCIBLE_E2E_RUN_NAME") or f"e2e_crucible_{datetime.utcnow():%Y%m%d_%H%M%S}"
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, run_name, _tracking_uri())

    oos_result = CrucibleOrchestrator(platform_path, workload_path).run_walk_forward_oos(use_ray=False)
    gate_result = CrucibleOrchestrator(platform_path, workload_path).run_regime_gate_stage(rerun=True)

    client = MlflowClient(tracking_uri=_tracking_uri())
    experiment = client.get_experiment_by_name("e2e-crucible")
    run = client.get_run(oos_result["mlflow_run_id"])
    summary_artifacts = {item.path for item in client.list_artifacts(oos_result["mlflow_run_id"], "summaries")}
    chart_artifacts = {item.path for item in client.list_artifacts(oos_result["mlflow_run_id"], "charts")}

    assert experiment is not None
    assert gate_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert run.info.experiment_id == experiment.experiment_id
    assert run.data.tags["crucible.run_name"] == run_name
    assert run.data.tags["crucible.status"] == "complete"
    assert "walk_forward_oos.jobs_complete" in run.data.metrics
    assert "walk_forward_oos.total_return_pct_std_dev" in run.data.metrics
    assert "walk_forward_oos.total_return_pct_min" in run.data.metrics
    assert "walk_forward_oos.total_return_pct_max" in run.data.metrics
    assert "walk_forward_oos.max_drawdown_pct_std_dev" in run.data.metrics
    assert "walk_forward_oos.volatility_std_dev" in run.data.metrics
    assert "regime_gate.passed" in run.data.metrics
    assert "summaries/oos_summary.csv" in summary_artifacts
    assert "summaries/validation_regime_summary.csv" in summary_artifacts
    assert "summaries/regime_gate_summary.csv" in summary_artifacts
    assert "charts/walk_forward_oos_distributions.svg" in chart_artifacts


def _configs(tmp_path: Path, data_path: Path, run_name: str, tracking_uri: str) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "e2e", "run_name": run_name},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "mlflow"},
        "mlflow": {"tracking_uri": tracking_uri, "parent_experiment_name": "e2e-crucible"},
        "ray": {"enabled": False, "max_concurrent_trials": 2},
        "walk_forward": {
            "optimization_window_days": 20,
            "validation_window_days": 10,
            "embargo_days": 3,
            "step_days": 15,
            "min_windows": 3,
        },
        "gates": {
            "generalist": {
                "min_trades": 0,
                "min_median_oos_return": 0.0,
                "min_profitable_windows_pct": 0.50,
                "max_drawdown": 0.25,
            },
            "specialist": {
                "min_regime_bars": 1,
                "min_regime_windows": 2,
                "min_regime_median_oos_return": 0.0,
                "min_regime_profitable_windows_pct": 0.60,
                "max_regime_drawdown": 0.20,
            },
        },
    }
    workload = {
        "workload": {"name": "e2e", "run_name": run_name},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "params": {
                "history_length": 1,
                "market_regime": {
                    "enabled": True,
                    "trend_lookback_days": 2,
                    "baseline_ma_window_days": 2,
                    "volatility_lookback_days": 2,
                    "volatility_percentile_window_days": 4,
                    "drawdown_lookback_days": 4,
                    "require_full_windows": False,
                },
            },
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {
                "cash": 100000,
                "keep_history": True,
                "symbol": "SPY",
                "stop_pct": 50.0,
                "profit_pct": 50.0,
            },
        },
        "fixed_assumptions": {"starting_cash": 100000},
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def teardown_module():
    mlflow.end_run()
