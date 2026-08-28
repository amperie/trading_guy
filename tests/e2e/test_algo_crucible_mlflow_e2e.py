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
from algo_crucible.paper_replay import load_frozen_candidate, replay_trace, trace_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


pytestmark = pytest.mark.e2e


def _tracking_uri() -> str:
    if os.environ.get("CRUCIBLE_E2E_MLFLOW_URI"):
        return os.environ["CRUCIBLE_E2E_MLFLOW_URI"]
    with Path("config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["mlflow"]["tracking_uri"]


def test_crucible_populates_representative_mlflow_parent_run(monkeypatch, tmp_path: Path):
    if os.environ.get("RUN_CRUCIBLE_E2E_MLFLOW") != "1":
        pytest.skip("Set RUN_CRUCIBLE_E2E_MLFLOW=1 to write a representative run to MLflow")

    run_name = os.environ.get("CRUCIBLE_E2E_RUN_NAME") or f"e2e_crucible_{datetime.utcnow():%Y%m%d_%H%M%S}"
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, run_name, _tracking_uri())
    monkeypatch.setattr("trading.launchers.run_backtest_ray.tune_backtest_hyperparameters", _fake_tune)
    monkeypatch.setattr("algo_crucible.orchestrator.run_validation_backtest", _stable_validation_worker)

    hpo_result = CrucibleOrchestrator(platform_path, workload_path).run_hpo_stage()
    oos_result = CrucibleOrchestrator(platform_path, workload_path).run_walk_forward_oos(use_ray=False)
    gate_result = CrucibleOrchestrator(platform_path, workload_path).run_regime_gate_stage(rerun=True)
    plateau_result = CrucibleOrchestrator(platform_path, workload_path).run_plateau_stage(rerun=True, use_ray=False)
    perturbation_result = CrucibleOrchestrator(platform_path, workload_path).run_perturbation_stage(rerun=True, use_ray=False)
    confirmation_result = CrucibleOrchestrator(platform_path, workload_path).run_confirmation_stage(rerun=True, use_ray=False)
    candidate = load_frozen_candidate(confirmation_result["run_dir"])
    (tmp_path / "paper_trace.csv").write_text(
        trace_to_csv(replay_trace(CrucibleOrchestrator(platform_path, workload_path).resolved_cfg, candidate, data_path)),
        encoding="utf-8",
    )
    paper_replay_result = CrucibleOrchestrator(platform_path, workload_path).run_paper_replay_stage(rerun=True)

    client = MlflowClient(tracking_uri=_tracking_uri())
    experiment = client.get_experiment_by_name("e2e-crucible")
    run = client.get_run(paper_replay_result["mlflow_run_id"])
    artifacts = _artifact_paths(client, oos_result["mlflow_run_id"])

    assert experiment is not None
    assert hpo_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert gate_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert plateau_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert perturbation_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert confirmation_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert paper_replay_result["mlflow_run_id"] == oos_result["mlflow_run_id"]
    assert run.info.experiment_id == experiment.experiment_id
    assert run.data.tags["crucible.run_name"] == run_name
    assert run.data.tags["crucible.status"] == "paper_replay_passed"
    assert run.data.params["algorithm.evaluation_symbols"] == "SPY"
    assert run.data.params["hpo.objective_metric"] == "composite_v1"
    assert "hpo.trials_complete" in run.data.metrics
    assert "walk_forward_oos.jobs_complete" in run.data.metrics
    assert "walk_forward_oos.total_return_pct_std_dev" in run.data.metrics
    assert "walk_forward_oos.total_return_pct_min" in run.data.metrics
    assert "walk_forward_oos.total_return_pct_max" in run.data.metrics
    assert "walk_forward_oos.max_drawdown_pct_std_dev" in run.data.metrics
    assert "walk_forward_oos.volatility_std_dev" in run.data.metrics
    assert "regime_gate.passed" in run.data.metrics
    assert "plateau.accepted_plateaus" in run.data.metrics
    assert "perturbation.accepted_candidates" in run.data.metrics
    assert "confirmation.promoted_candidates" in run.data.metrics
    assert "paper_replay.passed" in run.data.metrics
    assert "configs/resolved_config.yaml" in artifacts
    assert "stages/04_hpo/summaries/hpo_trial_summary.csv" in artifacts
    assert "stages/03_walk_forward_oos/summaries/oos_summary.csv" in artifacts
    assert "stages/03_walk_forward_oos/summaries/validation_regime_summary.csv" in artifacts
    assert "stages/05_regime_gate/summaries/regime_gate_summary.csv" in artifacts
    assert "stages/06_plateau/summaries/plateau_summary.csv" in artifacts
    assert "stages/07_perturbation/summaries/perturbation_summary.csv" in artifacts
    assert "stages/08_confirmation/summaries/confirmation_summary.csv" in artifacts
    assert "stages/09_paper_replay/summaries/paper_replay_trace.csv" in artifacts
    assert "stages/09_paper_replay/summaries/paper_replay_mismatches.csv" in artifacts
    assert "stages/03_walk_forward_oos/charts/walk_forward_oos_distributions.svg" in artifacts
    assert "stages/08_confirmation/promotion/promotion_packet.json" in artifacts
    assert "stages/08_confirmation/promotion/promotion_packet.yaml" in artifacts
    assert "stages/08_confirmation/promotion/promotion_packet.md" in artifacts


def _artifact_paths(client: MlflowClient, run_id: str, path: str | None = None) -> set[str]:
    paths = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            paths.update(_artifact_paths(client, run_id, item.path))
        else:
            paths.add(item.path)
    return paths


def _configs(tmp_path: Path, data_path: Path, run_name: str, tracking_uri: str) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "e2e", "run_name": run_name},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "mlflow"},
        "mlflow": {"tracking_uri": tracking_uri, "parent_experiment_name": "e2e-crucible"},
        "ray": {"enabled": False, "max_concurrent_trials": 2},
        "hpo": {
            "num_samples": 2,
            "objective": {
                "metric": "composite_v1",
                "weights": {
                    "annualized_return": 1.0,
                    "max_drawdown_pct": -1.5,
                    "volatility": -0.25,
                },
            },
        },
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
        "plateau": {
            "max_seeds": 1,
            "min_seed_distance": 0.10,
            "neighborhood_radius_pct": 0.08,
            "max_neighbors_per_seed": 3,
            "min_neighbor_trials": 1,
            "min_neighbor_pass_rate": 50,
            "max_peak_to_median_degradation": 500.0,
        },
        "perturbations": {
            "max_required_failures": 0,
            "min_scenario_pass_rate": 50,
            "scenarios": [
                {"name": "baseline", "required": True, "patch": {}},
                {
                    "name": "cost_slippage_2x",
                    "required": True,
                    "portfolio_param_multipliers": {"tx_cost": 2.0},
                },
            ],
        },
        "confirmation": {"start_date": "2024-04-01", "end_date": "2024-04-30", "min_return_pct": 0.0},
        "promotion": {"create_promoted_folder": False, "output_dir": str(tmp_path / "promoted")},
        "paper_replay": {"observed_trace_path": str(tmp_path / "paper_trace.csv"), "data_path": str(data_path)},
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
            "evaluation_symbols": ["SPY"],
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
                "tx_cost": 1.0,
            },
        },
        "fixed_assumptions": {"starting_cash": 100000},
        "search_space": {
            "space": {"stop_pct": {"type": "uniform", "low": 5.0, "high": 15.0}},
            "algorithm_param_keys": [],
            "portfolio_param_keys": ["stop_pct"],
        },
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def _fake_tune(**kwargs):
    objective = kwargs["base_backtest_config"].get("objective")
    details = {"_objective_annualized_return": 6.0}
    if objective:
        details.update({"_objective_max_drawdown_pct": 2.0, "_objective_volatility": 1.0})
    return (
        {"stop_pct": 8.0},
        [
            {"config": {"stop_pct": 8.0}, "metric": 6.0, "objective_details": details},
            {"config": {"stop_pct": 12.0}, "metric": 4.0, "objective_details": details},
        ],
    )


def _stable_validation_worker(payload):
    candidate = payload["candidate"]
    stop_pct = float(candidate["portfolio_params"].get("stop_pct", 10.0))
    ret = 4.0 if 7.0 <= stop_pct <= 13.0 else 1.0
    if "scenario_id" in payload and payload["scenario_id"].endswith("cost_slippage_2x"):
        ret = 2.0
    return {
        "seed_id": payload.get("seed_id"),
        "neighbor_id": payload.get("neighbor_id"),
        "scenario_id": payload.get("scenario_id"),
        "source_candidate_id": payload.get("source_candidate_id"),
        "candidate_id": candidate["candidate_id"],
        "window_id": payload["window"]["window_id"],
        "window": payload["window"],
        "overall_scorecard": {
            "total_return_pct": ret,
            "annualized_return": ret,
            "sharpe_ratio": ret,
            "sortino_ratio": ret,
            "max_drawdown_pct": -2.0,
            "win_rate": 100.0,
            "profit_factor": 1.0,
            "total_trades": 1,
            "final_equity": 100000 + ret,
            "initial_equity": 100000,
            "trading_days": 10,
            "volatility": 1.0,
        },
        "regime_scorecard": [
            {
                "regime": "RANGE_LOW_VOL",
                "bars": 10,
                "total_return_pct": ret,
                "annualized_return": ret,
                "max_drawdown_pct": -1.0,
                "total_trades": 1,
            }
        ],
    }


def teardown_module():
    mlflow.end_run()
