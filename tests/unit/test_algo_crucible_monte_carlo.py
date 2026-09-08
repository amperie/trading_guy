from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_crucible.monte_carlo import simulate_monte_carlo
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.scoring import rows_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def _configs(tmp_path: Path, data_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "monte_carlo_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "local"},
        "monte_carlo": {
            "num_paths": 50,
            "max_path_points": 12,
            "min_observations": 3,
            "seed": 7,
            "ruin_threshold_pct": -50.0,
            "max_ruin_probability": 1.0,
            "max_loss_probability": 1.0,
            "min_p5_terminal_return_pct": -100.0,
            "max_p95_drawdown_pct": 100.0,
        },
    }
    workload = {
        "workload": {"name": "monte_carlo", "run_name": "monte_carlo_v1"},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "evaluation_symbols": ["SPY"],
            "params": {"history_length": 1},
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {"cash": 100000, "keep_history": True, "symbol": "SPY"},
        },
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def test_simulate_monte_carlo_is_deterministic():
    rows = [{"candidate_id": "candidate_1", "return_pct": value} for value in [1.0, -0.5, 0.7, 1.2, -0.2]]
    platform = {"monte_carlo": {"num_paths": 20, "max_path_points": 5, "min_observations": 3, "seed": 123}}

    first = simulate_monte_carlo(rows, platform)
    second = simulate_monte_carlo(rows, platform)

    assert first == second
    assert first["summary_rows"][0]["candidate_id"] == "candidate_1"
    assert first["summary_rows"][0]["path_count"] == 20
    assert len(first["band_rows"]) == 5
    assert len(first["terminal_rows"]) == 20


def test_monte_carlo_stage_writes_summary_artifacts(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    run_dir = Path(run["run_dir"])
    stream_rows = [
        {"candidate_id": "candidate_1", "scenario_id": "baseline", "window_id": "w1", "step": idx, "timestamp": f"2024-01-{idx + 1:02d}", "return_pct": ret, "equity": 100 + idx}
        for idx, ret in enumerate([1.0, -0.5, 0.8, 0.2, -0.1], start=1)
    ]
    target = run_dir / "stages/07_perturbation/summaries"
    target.mkdir(parents=True, exist_ok=True)
    (target / "perturbation_return_stream.csv").write_text(rows_to_csv(stream_rows), encoding="utf-8")

    result = CrucibleOrchestrator(platform_path, workload_path).run_monte_carlo_stage(rerun=True)
    stage_dir = Path(result["run_dir"]) / "stages/08_monte_carlo/summaries"
    summary = pd.read_csv(stage_dir / "monte_carlo_summary.csv")
    bands = pd.read_csv(stage_dir / "monte_carlo_path_bands.csv")
    terminals = pd.read_csv(stage_dir / "monte_carlo_terminal_distribution.csv")

    assert summary.iloc[0]["candidate_id"] == "candidate_1"
    assert summary.iloc[0]["path_count"] == 50
    assert len(bands) == 5
    assert len(terminals) == 50
    assert result["metrics"]["monte_carlo.candidate_count"] == 1.0
    assert (stage_dir / "stage_summary.json").exists()
