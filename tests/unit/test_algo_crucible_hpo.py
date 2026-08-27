from __future__ import annotations

from pathlib import Path

import pandas as pd

import algo_crucible.hpo as hpo_module
from algo_crucible.config import resolve_config_dicts
from algo_crucible.hpo import build_hpo_summary
from algo_crucible.orchestrator import CrucibleOrchestrator
from tests.unit.test_algo_crucible_milestone1 import _write_data, _write_yaml


def _configs(tmp_path: Path, data_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "hpo_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs")},
        "state_store": {"backend": "local"},
        "ray": {"max_concurrent_trials": 2, "log_worker_output": False},
        "hpo": {"num_samples": 3},
    }
    workload = {
        "workload": {"name": "hpo", "run_name": "hpo_v1"},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "params": {
                "history_length": 1,
                "market_regime": {"enabled": True, "require_full_windows": False},
            },
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {"cash": 100000, "keep_history": True, "symbol": "SPY", "stop_pct": 10.0, "profit_pct": 10.0},
        },
        "fixed_assumptions": {"starting_cash": 100000},
        "search_space": {
            "space": {
                "history_length": {"type": "choice", "values": [1, 2]},
                "stop_pct": {"type": "uniform", "low": 5.0, "high": 20.0},
            },
            "algorithm_param_keys": ["history_length"],
            "portfolio_param_keys": ["stop_pct"],
        },
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def test_build_hpo_summary_creates_candidates_and_failed_trials():
    resolved = resolve_config_dicts(
        {"crucible": {"run_name": "hpo_summary"}},
        {
            "workload": {"run_name": "hpo_summary"},
            "algorithm": {"algorithm": "algo.A", "params": {"history_length": 1}},
            "portfolio": {"portfolio": "pf.P", "params": {"cash": 100}},
        },
    )

    result = build_hpo_summary(
        resolved_cfg=resolved,
        best_config={"history_length": 2},
        trial_summaries=[
            {"config": {"history_length": 2}, "metric": 2.0},
            {"config": {"history_length": 1}, "metric": 1.0},
            {"config": {"history_length": 3}, "metric": float("nan")},
        ],
        algorithm_param_keys=["history_length"],
        portfolio_param_keys=[],
        base_algorithm_config={"history_length": 1},
        base_portfolio_config={"cash": 100},
    )

    assert result["metrics"]["hpo.trials_total"] == 3
    assert result["metrics"]["hpo.trials_complete"] == 2
    assert result["metrics"]["hpo.trials_failed"] == 1
    assert result["trial_rows"][0]["metric"] == 2.0
    assert result["candidates"][0].candidate_id.startswith("candidate_")


def test_hpo_stage_writes_full_trial_and_candidate_summaries(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    captured = {}

    def fake_tune(**kwargs):
        captured.update(kwargs)
        return (
            {"history_length": 2, "stop_pct": 8.0},
            [
                {"config": {"history_length": 2, "stop_pct": 8.0}, "metric": 10.0},
                {"config": {"history_length": 1, "stop_pct": 12.0}, "metric": 5.0},
            ],
        )

    monkeypatch.setattr(hpo_module, "parse_search_space", lambda cfg: cfg)
    monkeypatch.setattr("trading.launchers.run_backtest_ray.tune_backtest_hyperparameters", fake_tune)

    result = CrucibleOrchestrator(platform_path, workload_path).run_hpo_stage()
    run_dir = Path(result["run_dir"])
    trials = pd.read_csv(run_dir / "summaries" / "hpo_trial_summary.csv")
    candidates = pd.read_csv(run_dir / "summaries" / "hpo_candidate_summary.csv")

    assert result["summary"]["hpo.trials_total"] == 2
    assert result["summary"]["hpo.best_metric"] == 10.0
    assert len(trials) == 2
    assert len(candidates) == 2
    assert captured["return_trial_summaries"] is True
    assert captured["algorithm_param_keys"] == ["history_length"]
    assert captured["portfolio_param_keys"] == ["stop_pct"]
