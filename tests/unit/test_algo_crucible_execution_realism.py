from __future__ import annotations

import pandas as pd

from algo_crucible.execution_realism import analyze_execution_realism, execution_realism_metrics
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.scoring import rows_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def test_execution_realism_rejects_required_delay_failure():
    rows = [{"candidate_id": "candidate_1", "return_pct": value} for value in [0, 0, 10]]
    analyzed = analyze_execution_realism(
        rows,
        {
            "execution_realism_stress": {
                "min_scenario_pass_rate": 1.0,
                "max_required_failures": 0,
                "min_total_return_pct": 5.0,
                "max_total_return_degradation_pct": 4.0,
                "scenarios": [
                    {"name": "baseline", "required": True},
                    {"name": "one_bar_delay", "required": True, "delay_bars": 1},
                ],
            }
        },
    )
    summary = analyzed["summary_rows"][0]

    assert summary["accepted"] is False
    assert summary["failure_reason"] == "fill_delay_fragile"
    assert analyzed["scenario_rows"][1]["stress_type"] == "latency_delay"
    assert analyzed["scenario_rows"][1]["severity"] == "1bar"
    assert analyzed["scenario_rows"][1]["analysis_mode"] == "return_stream_proxy"
    assert "stressed_sharpe_like" in analyzed["scenario_rows"][1]
    assert execution_realism_metrics(analyzed["summary_rows"])["execution_realism.rejected_candidates"] == 1.0


def test_execution_realism_stage_writes_summary_artifacts(tmp_path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    source = tmp_path / "runs" / orchestrator.resolved_cfg.crucible_run_id / "stages/07_perturbation/summaries"
    source.mkdir(parents=True, exist_ok=True)
    source.joinpath("perturbation_return_stream.csv").write_text(
        rows_to_csv([{"candidate_id": "candidate_1", "return_pct": 1.0} for _ in range(8)]),
        encoding="utf-8",
    )

    result = CrucibleOrchestrator(platform_path, workload_path).run_execution_realism_stress_stage()
    summary_path = tmp_path / "runs" / orchestrator.resolved_cfg.crucible_run_id / "stages/08_execution_realism_stress/summaries/execution_realism_summary.csv"
    scenarios_path = tmp_path / "runs" / orchestrator.resolved_cfg.crucible_run_id / "stages/08_execution_realism_stress/summaries/execution_realism_scenarios.csv"
    summary = pd.read_csv(summary_path)
    scenarios = pd.read_csv(scenarios_path)

    assert run["crucible_run_id"] == result["crucible_run_id"]
    assert summary.iloc[0]["accepted"] == True
    assert len(scenarios) == 2
    assert result["summary"]["accepted_candidates"] == 1


def _configs(tmp_path, data_path):
    platform = {
        "crucible": {"name": "test", "run_name": "execution_realism_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "local"},
        "execution_realism_stress": {
            "min_scenario_pass_rate": 1.0,
            "max_required_failures": 0,
            "min_total_return_pct": -100.0,
            "max_total_return_degradation_pct": 100.0,
            "scenarios": [
                {"name": "baseline", "required": True},
                {"name": "one_bar_delay", "required": True, "delay_bars": 1},
            ],
        },
    }
    workload = {
        "workload": {"name": "execution_realism", "run_name": "execution_realism_v1"},
        "data_provider": {"provider": "trading.data_providers.test_data_provider.TestDataProvider", "path": str(data_path)},
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
