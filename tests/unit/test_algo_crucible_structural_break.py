from __future__ import annotations

import pandas as pd

from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.scoring import rows_to_csv
from algo_crucible.structural_break import analyze_structural_breaks, structural_break_metrics
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def test_structural_break_rejects_concentrated_edge():
    rows = [
        {"candidate_id": "candidate_1", "return_pct": value}
        for value in [5, 5, 5, 5, 5, -1, -1, -1, -1, -1]
    ]

    analyzed = analyze_structural_breaks(
        rows,
        {
            "structural_break_stability": {
                "min_observations": 10,
                "rolling_window": 5,
                "max_segment_return_concentration": 0.60,
                "max_post_break_mean_degradation_pct": 3.0,
            }
        },
    )
    summary = analyzed["summary_rows"][0]

    assert summary["accepted"] is False
    assert summary["failure_reason"] == "edge_concentrated_in_one_period"
    assert structural_break_metrics(analyzed["summary_rows"])["structural_break.rejected_candidates"] == 1.0


def test_structural_break_uses_rolling_ic_when_signal_forward_returns_exist():
    rows = [
        {"candidate_id": "candidate_1", "timestamp": f"2024-01-{idx + 1:02d}", "signal": signal, "forward_return_pct": signal}
        for idx, signal in enumerate([1, 2, 3, 4, 1, 2, 3, 4])
    ]

    analyzed = analyze_structural_breaks(
        rows,
        {
            "structural_break_stability": {
                "min_observations": 8,
                "min_ic_observations": 3,
                "ic_windows": [4],
                "min_ic_mean": 0.5,
                "min_ic_ir": 0.0,
                "min_positive_ic_window_pct": 1.0,
            }
        },
    )
    summary = analyzed["summary_rows"][0]

    assert summary["analysis_mode"] == "rolling_ic"
    assert summary["accepted"] is True
    assert summary["ic_mean"] > 0.99


def test_structural_break_rejects_recent_ic_flip():
    signals = [1, 2, 3, 4, 1, 2, 3, 4]
    returns = [1, 2, 3, 4, -1, -2, -3, -4]
    rows = [
        {"candidate_id": "candidate_1", "timestamp": f"2024-01-{idx + 1:02d}", "signal": signal, "forward_return_pct": ret}
        for idx, (signal, ret) in enumerate(zip(signals, returns))
    ]

    analyzed = analyze_structural_breaks(
        rows,
        {
            "structural_break_stability": {
                "min_observations": 8,
                "min_ic_observations": 3,
                "ic_windows": [4],
                "recent_window_count": 2,
                "recent_min_ic_mean": 0.0,
                "min_post_break_ic_mean": -1.0,
                "max_post_break_ic_degradation": 3.0,
                "min_positive_ic_window_pct": 0.0,
            }
        },
    )
    summary = analyzed["summary_rows"][0]

    assert summary["accepted"] is False
    assert "recent_ic_sign_flip" in summary["failure_reason"]


def test_structural_break_stage_writes_summary_artifacts(tmp_path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    stage_dir = tmp_path / "runs" / orchestrator.resolved_cfg.crucible_run_id / "stages/07_perturbation/summaries"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.joinpath("perturbation_return_stream.csv").write_text(
        rows_to_csv([
            {"candidate_id": "candidate_1", "return_pct": value}
            for value in [1, 1, 1, 1, 1, 1, 1, 1]
        ]),
        encoding="utf-8",
    )

    result = CrucibleOrchestrator(platform_path, workload_path).run_structural_break_stability_stage()
    summary_path = tmp_path / "runs" / orchestrator.resolved_cfg.crucible_run_id / "stages/08_structural_break_stability/summaries/structural_break_summary.csv"
    summary = pd.read_csv(summary_path)
    resumed = CrucibleOrchestrator(platform_path, workload_path).run_structural_break_stability_stage()

    assert run["crucible_run_id"] == result["crucible_run_id"]
    assert summary.iloc[0]["candidate_id"] == "candidate_1"
    assert summary.iloc[0]["accepted"] == True
    assert result["summary"]["accepted_candidates"] == 1
    assert resumed["crucible_run_id"] == result["crucible_run_id"]


def _configs(tmp_path, data_path):
    platform = {
        "crucible": {"name": "test", "run_name": "structural_break_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "local"},
        "structural_break_stability": {
            "min_observations": 8,
            "rolling_window": 4,
            "max_segment_return_concentration": 0.75,
            "max_post_break_mean_degradation_pct": 10.0,
        },
    }
    workload = {
        "workload": {"name": "structural_break", "run_name": "structural_break_v1"},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
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
