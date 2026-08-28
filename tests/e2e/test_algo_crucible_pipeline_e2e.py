from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from algo_crucible.jobs import CrucibleJob, RayJobRunner
from algo_crucible.orchestrator import CrucibleOrchestrator
from trading.launchers.run_backtest_ray import objective_score
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


pytestmark = pytest.mark.e2e


def test_full_local_crucible_pipeline_runs_through_perturbations(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, "pipeline_local_v1")
    _patch_hpo(monkeypatch)
    monkeypatch.setattr("algo_crucible.orchestrator.run_validation_backtest", _stable_validation_worker)

    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    hpo = orchestrator.run_hpo_stage()
    oos = CrucibleOrchestrator(platform_path, workload_path).run_walk_forward_oos(use_ray=False)
    gate = CrucibleOrchestrator(platform_path, workload_path).run_regime_gate_stage()
    plateau = CrucibleOrchestrator(platform_path, workload_path).run_plateau_stage(use_ray=False)
    perturbation = CrucibleOrchestrator(platform_path, workload_path).run_perturbation_stage(use_ray=False)
    confirmation = CrucibleOrchestrator(platform_path, workload_path).run_confirmation_stage(use_ray=False)

    run_dir = Path(confirmation["run_dir"])
    assert hpo["metrics"]["hpo.trials_complete"] == 2
    assert oos["metrics"]["walk_forward_oos.jobs_failed"] == 0
    assert gate["metrics"]["regime_gate.passed"] == 1.0
    assert plateau["metrics"]["plateau.accepted_plateaus"] >= 1.0
    assert perturbation["metrics"]["perturbation.accepted_candidates"] >= 1.0
    assert confirmation["metrics"]["confirmation.promoted_candidates"] >= 1.0
    assert confirmation["summary"]["paper_trading_started"] is False
    for artifact in (
        "summaries/hpo_trial_summary.csv",
        "summaries/oos_summary.csv",
        "summaries/regime_gate_summary.csv",
        "summaries/plateau_summary.csv",
        "summaries/perturbation_summary.csv",
        "summaries/confirmation_summary.csv",
        "charts/walk_forward_oos_distributions.svg",
        "promotion/promotion_packet.json",
        "promotion/promotion_packet.yaml",
        "promotion/promotion_packet.md",
    ):
        assert (run_dir / artifact).exists(), artifact
    assert not (tmp_path / "promoted").exists()


def test_oos_resume_reuses_complete_jobs_and_reruns_failed(tmp_path: Path, monkeypatch):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, "resume_local_v1")
    failed_once = set()

    def flaky_worker(payload):
        window_id = payload["window"]["window_id"]
        if window_id.endswith("000") and window_id not in failed_once:
            failed_once.add(window_id)
            raise RuntimeError("synthetic interruption")
        return _stable_validation_worker(payload)

    monkeypatch.setattr("algo_crucible.orchestrator.run_validation_backtest", flaky_worker)
    first = CrucibleOrchestrator(platform_path, workload_path).run_walk_forward_oos(use_ray=False)
    monkeypatch.setattr("algo_crucible.orchestrator.run_validation_backtest", _stable_validation_worker)
    second = CrucibleOrchestrator(platform_path, workload_path).run_walk_forward_oos(rerun=True, use_ray=False)

    assert first["summary"]["jobs_failed"] == 1
    assert second["summary"]["jobs_failed"] == 0
    assert second["summary"]["jobs_complete"] == second["summary"]["jobs_total"]
    assert second["summary"]["jobs_reused"] == first["summary"]["jobs_complete"]


def test_hpo_objective_compatibility_stage_outputs(tmp_path: Path, monkeypatch):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    captured = []

    def fake_tune(**kwargs):
        captured.append(kwargs["base_backtest_config"].get("objective"))
        details = {"_objective_annualized_return": 5.0}
        if captured[-1]:
            details["_objective_max_drawdown_pct"] = 2.0
        return (
            {"stop_pct": 8.0},
            [{"config": {"stop_pct": 8.0}, "metric": 5.0, "objective_details": details}],
        )

    monkeypatch.setattr("trading.launchers.run_backtest_ray.tune_backtest_hyperparameters", fake_tune)
    no_objective_platform, no_objective_workload = _configs(tmp_path, data_path, "objective_default_v1", include_objective=False)
    composite_platform, composite_workload = _configs(tmp_path, data_path, "objective_composite_v1")

    default = CrucibleOrchestrator(no_objective_platform, no_objective_workload).run_hpo_stage()
    composite = CrucibleOrchestrator(composite_platform, composite_workload).run_hpo_stage()
    default_trials = pd.read_csv(Path(default["run_dir"]) / "summaries" / "hpo_trial_summary.csv")
    composite_trials = pd.read_csv(Path(composite["run_dir"]) / "summaries" / "hpo_trial_summary.csv")

    assert captured[0] is None
    assert captured[1]["metric"] == "composite_v1"
    assert "_objective_annualized_return" in default_trials["objective_details"].iloc[0]
    assert "_objective_max_drawdown_pct" in composite_trials["objective_details"].iloc[0]
    assert objective_score(SimpleNamespace(annualized_return=3.0))[0] == 3.0


def test_ray_job_runner_smoke_with_local_state_store(tmp_path: Path):
    if os.environ.get("RUN_CRUCIBLE_RAY_E2E") != "1":
        pytest.skip("Set RUN_CRUCIBLE_RAY_E2E=1 to run Ray e2e smoke coverage")

    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, "ray_smoke_v1")
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    jobs = [CrucibleJob("e2e_ray", "echo", {"value": idx}) for idx in range(3)]

    result = RayJobRunner(use_ray=True, max_concurrent_jobs=2).run_jobs(
        run_id=orchestrator.resolved_cfg.crucible_run_id,
        jobs=jobs,
        worker=_ray_echo_worker,
        state_store=orchestrator.state_store,
    )

    assert result.jobs_complete == 3
    assert result.jobs_failed == 0
    assert len(list((Path(run["run_dir"]) / "stages" / "e2e_ray" / "results").glob("*.json"))) == 3


def _configs(
    tmp_path: Path,
    data_path: Path,
    run_name: str,
    *,
    include_objective: bool = True,
) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "e2e", "run_name": run_name},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "local"},
        "ray": {"enabled": False, "max_concurrent_trials": 2, "log_worker_output": False},
        "hpo": {"num_samples": 2},
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
                "min_regime_windows": 1,
                "min_regime_median_oos_return": 0.0,
                "min_regime_profitable_windows_pct": 0.50,
                "max_regime_drawdown": 0.25,
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
                {"name": "cost_slippage_2x", "required": True, "portfolio_param_multipliers": {"tx_cost": 2.0}},
            ],
        },
        "confirmation": {"start_date": "2024-04-01", "end_date": "2024-04-30", "min_return_pct": 0.0},
        "promotion": {"create_promoted_folder": False, "output_dir": str(tmp_path / "promoted")},
    }
    if include_objective:
        platform["hpo"]["objective"] = {
            "metric": "composite_v1",
            "weights": {"annualized_return": 1.0, "max_drawdown_pct": -1.5, "volatility": -0.25},
        }
    workload = {
        "workload": {"name": "e2e", "run_name": run_name},
        "data_provider": {"provider": "trading.data_providers.test_data_provider.TestDataProvider", "path": str(data_path)},
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
                "stop_pct": 10.0,
                "profit_pct": 10.0,
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
    platform_path = tmp_path / f"{run_name}_platform.yaml"
    workload_path = tmp_path / f"{run_name}_workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def _patch_hpo(monkeypatch):
    monkeypatch.setattr("trading.launchers.run_backtest_ray.tune_backtest_hyperparameters", _fake_tune)


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


def _ray_echo_worker(payload):
    return {"value": payload["value"], "doubled": payload["value"] * 2}
