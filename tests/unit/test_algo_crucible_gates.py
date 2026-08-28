from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_crucible.gates import evaluate_regime_aware_gates, gate_summary_metrics
from algo_crucible.orchestrator import CrucibleOrchestrator
from tests.unit.test_algo_crucible_walk_forward_oos import _configs, _write_daily_data


PLATFORM = {
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
    }
}


def test_regime_gates_classify_generalist_specialist_and_reject():
    overall_rows = [
        {"candidate_id": "A", "total_return_pct": 2.0, "max_drawdown_pct": -5.0, "total_trades": 10},
        {"candidate_id": "A", "total_return_pct": -1.0, "max_drawdown_pct": -4.0, "total_trades": 10},
        {"candidate_id": "A", "total_return_pct": 3.0, "max_drawdown_pct": -3.0, "total_trades": 10},
        {"candidate_id": "B", "total_return_pct": -4.0, "max_drawdown_pct": -6.0, "total_trades": 5},
        {"candidate_id": "B", "total_return_pct": -3.0, "max_drawdown_pct": -5.0, "total_trades": 5},
        {"candidate_id": "C", "total_return_pct": -2.0, "max_drawdown_pct": -30.0, "total_trades": 0},
        {"candidate_id": "C", "total_return_pct": -1.0, "max_drawdown_pct": -20.0, "total_trades": 0},
    ]
    regime_rows = [
        {"candidate_id": "B", "regime": "RANGE_LOW_VOL", "bars": 10, "total_return_pct": 2.0, "max_drawdown_pct": -2.0},
        {"candidate_id": "B", "regime": "RANGE_LOW_VOL", "bars": 10, "total_return_pct": 1.0, "max_drawdown_pct": -3.0},
        {"candidate_id": "B", "regime": "UPTREND_HIGH_VOL", "bars": 10, "total_return_pct": -5.0, "max_drawdown_pct": -10.0},
    ]

    decisions = evaluate_regime_aware_gates(overall_rows, regime_rows, PLATFORM)
    by_id = {decision.candidate_id: decision for decision in decisions}
    metrics = gate_summary_metrics(decisions)

    assert by_id["A"].candidate_type == "generalist"
    assert by_id["B"].candidate_type == "specialist"
    assert by_id["B"].specialist_regimes == ["RANGE_LOW_VOL"]
    assert by_id["C"].candidate_type == "reject"
    assert "negative_median_oos_return" in by_id["C"].reason_codes
    assert metrics["regime_gate.generalists"] == 1.0
    assert metrics["regime_gate.specialists"] == 1.0
    assert metrics["regime_gate.rejected"] == 1.0


def test_regime_gate_stage_consumes_oos_outputs(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)

    orchestrator.run_walk_forward_oos(use_ray=False)
    result = CrucibleOrchestrator(platform_path, workload_path).run_regime_gate_stage()
    gate_summary = pd.read_csv(Path(result["run_dir"]) / "stages" / "05_regime_gate" / "summaries" / "regime_gate_summary.csv")

    assert result["summary"]["candidate_count"] == 1
    assert result["status"] == "running"
    assert len(gate_summary) == 1
    assert gate_summary["candidate_type"].iloc[0] in {"generalist", "specialist", "reject"}
