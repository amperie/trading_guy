from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import algo_crucible.orchestrator as orchestrator_module
from algo_crucible.builders import build_candidate_from_params
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.perturbations import apply_scenario
from algo_crucible.perturbations import load_perturbation_candidates
from algo_crucible.scoring import rows_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def _configs(tmp_path: Path, data_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "perturbation_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "local"},
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
        },
        "perturbations": {
            "max_required_failures": 0,
            "min_scenario_pass_rate": 80,
            "scenarios": [
                {"name": "baseline", "required": True, "patch": {}},
                {
                    "name": "cost_slippage_2x",
                    "required": True,
                    "portfolio_param_multipliers": {"tx_cost": 2.0},
                },
            ],
        },
    }
    workload = {
        "workload": {"name": "perturbation", "run_name": "perturbation_v1"},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "evaluation_symbols": ["SPY"],
            "params": {"history_length": 1, "market_regime": {"enabled": True, "require_full_windows": False}},
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
            "space": {"stop_pct": {"type": "uniform", "low": 0.0, "high": 20.0}},
            "algorithm_param_keys": [],
            "portfolio_param_keys": ["stop_pct"],
        },
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def test_perturbation_stage_rejects_candidate_that_fails_cost_scenario(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    candidate = _write_inputs(orchestrator, Path(run["run_dir"]))

    def fake_validation(payload):
        ret = -5.0 if payload["scenario_id"].endswith("cost_slippage_2x") else 5.0
        return {
            "scenario_id": payload["scenario_id"],
            "source_candidate_id": payload["source_candidate_id"],
            "candidate_id": payload["candidate"]["candidate_id"],
            "window_id": payload["window"]["window_id"],
            "window": payload["window"],
            "overall_scorecard": {
                "total_return_pct": ret,
                "annualized_return": ret,
                "sharpe_ratio": ret,
                "sortino_ratio": ret,
                "max_drawdown_pct": -2.0,
                "win_rate": 100.0 if ret > 0 else 0.0,
                "profit_factor": 1.0,
                "total_trades": 1,
                "final_equity": 100000 + ret,
                "initial_equity": 100000,
                "trading_days": 10,
                "volatility": 1.0,
            },
            "regime_scorecard": [],
        }

    monkeypatch.setattr(orchestrator_module, "run_validation_backtest", fake_validation)
    result = CrucibleOrchestrator(platform_path, workload_path).run_perturbation_stage(rerun=True, use_ray=False)
    run_dir = Path(result["run_dir"])
    stage_dir = run_dir / "stages" / "07_perturbation"
    summary = pd.read_csv(stage_dir / "summaries" / "perturbation_summary.csv")
    scenarios = pd.read_csv(stage_dir / "summaries" / "perturbation_scenario_summary.csv")

    assert summary.iloc[0]["candidate_id"] == candidate.candidate_id
    assert summary.iloc[0]["accepted"] == False
    assert summary.iloc[0]["required_failure_count"] == 1
    assert summary.iloc[0]["failure_reason"] == "cost_slippage_fragile"
    assert "cost_slippage_fragile" in set(scenarios["failure_reason"].dropna())
    assert result["metrics"]["perturbation.rejected_candidates"] == 1.0


def test_scenario_patch_updates_executed_candidate_params(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    candidate = build_candidate_from_params(
        orchestrator.resolved_cfg,
        orchestrator.resolved_cfg.workload["algorithm"]["params"],
        orchestrator.resolved_cfg.workload["portfolio"]["params"],
    )

    _, perturbed = apply_scenario(
        orchestrator.resolved_cfg,
        candidate,
        {"patch": {"portfolio": {"params": {"tx_cost": 7.0}}}},
    )

    assert perturbed.portfolio_params["tx_cost"] == 7.0


def test_load_perturbation_candidates_handles_empty_plateau_summary(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    run_dir = Path(run["run_dir"])
    (run_dir / "summaries").mkdir(exist_ok=True)
    (run_dir / "summaries" / "plateau_summary.csv").write_text("", encoding="utf-8")
    (run_dir / "summaries" / "hpo_trial_summary.csv").write_text("", encoding="utf-8")

    assert load_perturbation_candidates(run_dir, orchestrator.resolved_cfg) == []


def _write_inputs(orchestrator: CrucibleOrchestrator, run_dir: Path):
    algorithm_params = dict(orchestrator.resolved_cfg.workload["algorithm"]["params"])
    portfolio_params = dict(orchestrator.resolved_cfg.workload["portfolio"]["params"])
    candidate = build_candidate_from_params(orchestrator.resolved_cfg, algorithm_params, portfolio_params)
    (run_dir / "summaries").mkdir(exist_ok=True)
    hpo_rows = [{
        "trial_id": "trial_0000",
        "candidate_id": candidate.candidate_id,
        "metric": 5.0,
        "config": json.dumps({"stop_pct": portfolio_params["stop_pct"]}),
        "algorithm_params": json.dumps(algorithm_params, sort_keys=True),
        "portfolio_params": json.dumps(portfolio_params, sort_keys=True),
        "status": "complete",
    }]
    plateau_rows = [{
        "seed_id": "seed_001",
        "candidate_id": candidate.candidate_id,
        "candidate_type": "generalist",
        "specialist_regimes": "",
        "accepted": True,
        "plateau_score": 1.0,
    }]
    (run_dir / "summaries" / "hpo_trial_summary.csv").write_text(rows_to_csv(hpo_rows), encoding="utf-8")
    (run_dir / "summaries" / "plateau_summary.csv").write_text(rows_to_csv(plateau_rows), encoding="utf-8")
    return candidate
