from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import algo_crucible.orchestrator as orchestrator_module
from algo_crucible.builders import build_candidate_from_params
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.scoring import rows_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def _configs(tmp_path: Path, data_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "plateau_v1"},
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
        "plateau": {
            "max_seeds": 2,
            "min_seed_distance": 0.10,
            "neighborhood_radius_pct": 0.10,
            "min_neighbor_trials": 3,
            "min_neighbor_pass_rate": 60,
            "max_peak_to_median_degradation": 500.0,
        },
    }
    workload = {
        "workload": {"name": "plateau", "run_name": "plateau_v1"},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "params": {"history_length": 1, "market_regime": {"enabled": True, "require_full_windows": False}},
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {"cash": 100000, "keep_history": True, "symbol": "SPY", "stop_pct": 10.0, "profit_pct": 10.0},
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


def test_plateau_stage_rejects_spike_and_accepts_broad_plateau(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    _write_hpo_trials(orchestrator, Path(run["run_dir"]))

    def fake_validation(payload):
        stop_pct = float(payload["candidate"]["portfolio_params"]["stop_pct"])
        if abs(stop_pct - 6.0) < 0.01:
            ret = 100.0
        elif 12.0 <= stop_pct <= 16.0:
            ret = 5.0
        else:
            ret = -5.0
        return {
            "seed_id": payload["seed_id"],
            "neighbor_id": payload["neighbor_id"],
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
    result = CrucibleOrchestrator(platform_path, workload_path).run_plateau_stage(rerun=True, use_ray=False)
    run_dir = Path(result["run_dir"])
    summary = pd.read_csv(run_dir / "summaries" / "plateau_summary.csv")
    accepted = summary[summary["accepted"] == True]
    rejected = summary[summary["accepted"] == False]

    assert len(summary) == 2
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert "plateau_pass_rate_too_low" in rejected.iloc[0]["failure_reason"]
    assert (run_dir / "summaries" / "plateau_seed_summary.csv").exists()
    assert (run_dir / "summaries" / "plateau_neighbor_summary.csv").exists()
    assert (run_dir / "summaries" / "plateau_artifact_index.json").exists()
    assert (run_dir / "plots" / "plateau_distance_decay_seed_001.svg").exists()
    assert "gate" in (run_dir / "plots" / "plateau_distance_decay_seed_001.svg").read_text(encoding="utf-8")
    assert result["metrics"]["plateau.accepted_plateaus"] == 1.0


def _write_hpo_trials(orchestrator: CrucibleOrchestrator, run_dir: Path) -> None:
    rows = []
    for idx, (stop_pct, metric) in enumerate([(6.0, 100.0), (14.0, 5.0)]):
        algorithm_params = dict(orchestrator.resolved_cfg.workload["algorithm"]["params"])
        portfolio_params = dict(orchestrator.resolved_cfg.workload["portfolio"]["params"])
        portfolio_params["stop_pct"] = stop_pct
        candidate = build_candidate_from_params(orchestrator.resolved_cfg, algorithm_params, portfolio_params)
        rows.append({
            "trial_id": f"trial_{idx:04d}",
            "candidate_id": candidate.candidate_id,
            "metric": metric,
            "config": json.dumps({"stop_pct": stop_pct}),
            "algorithm_params": json.dumps(algorithm_params, sort_keys=True),
            "portfolio_params": json.dumps(portfolio_params, sort_keys=True),
            "status": "complete",
        })
    (run_dir / "summaries").mkdir(exist_ok=True)
    (run_dir / "summaries" / "hpo_trial_summary.csv").write_text(rows_to_csv(rows), encoding="utf-8")
