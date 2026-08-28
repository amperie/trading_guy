from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from algo_crucible.builders import build_candidate_from_params
from algo_crucible.confirmation import freeze_candidate
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.paper_replay import compare_traces, load_frozen_candidate, replay_trace, trace_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def test_compare_traces_passes_exact_match():
    trace = [{
        "timestamp": "2024-01-01T10:00:00",
        "regimes": [{"symbol": "SPY", "regime": "RANGE_LOW_VOL"}],
        "signals": [{"type": "BUY", "symbol": "SPY", "strength": 100, "metadata": {}}],
        "orders": [{"action": "BUY", "type": "MARKET", "symbol": "SPY", "quantity": 10, "price": 0.0}],
    }]

    result = compare_traces(trace, trace, {"paper_replay": {}})

    assert result["passed"] is True
    assert result["metrics"]["paper_replay.passed"] == 1.0


def test_compare_traces_fails_signal_mismatch():
    replay = [{"timestamp": "2024-01-01T10:00:00", "regimes": [], "signals": [], "orders": []}]
    observed = [{
        "timestamp": "2024-01-01T10:00:00",
        "regimes": [],
        "signals": [{"type": "BUY", "symbol": "SPY", "strength": 100, "metadata": {}}],
        "orders": [],
    }]

    result = compare_traces(replay, observed, {"paper_replay": {}})

    assert result["passed"] is False
    assert result["failure_reason"] == "paper_replay_mismatch"
    assert result["metrics"]["paper_replay.signal_mismatches"] == 1.0


def test_compare_traces_applies_fill_price_tolerance():
    replay = [{
        "timestamp": "2024-01-01T10:00:00",
        "regimes": [],
        "signals": [],
        "orders": [],
        "fills": [{"symbol": "SPY", "action": "BUY", "type": "MARKET", "quantity": 10, "price": 100.005}],
    }]
    observed = [{
        "timestamp": "2024-01-01T10:00:00",
        "regimes": [],
        "signals": [],
        "orders": [],
        "fills": [{"symbol": "SPY", "action": "BUY", "type": "MARKET", "quantity": 10, "price": 100.0}],
    }]

    assert compare_traces(replay, observed, {"paper_replay": {"fill_price_tolerance_pct": 0.01}})["passed"] is True
    assert compare_traces(replay, observed, {"paper_replay": {"fill_price_tolerance_pct": 0.001}})["passed"] is False


def test_paper_replay_stage_passes_matching_trace(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    observed_path = tmp_path / "paper_trace.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, observed_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    candidate = _freeze_candidate(orchestrator, Path(run["run_dir"]))
    observed_path.write_text(trace_to_csv(replay_trace(orchestrator.resolved_cfg, candidate, data_path)), encoding="utf-8")

    result = CrucibleOrchestrator(platform_path, workload_path).run_paper_replay_stage()

    assert result["status"] == "paper_replay_passed"
    assert result["metrics"]["paper_replay.passed"] == 1.0
    stage_dir = Path(result["run_dir"]) / "stages" / "09_paper_replay"
    assert (stage_dir / "summaries" / "paper_replay_trace.csv").exists()
    assert (stage_dir / "summaries" / "paper_replay_mismatches.csv").exists()


def test_paper_replay_stage_fails_changed_signal(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    observed_path = tmp_path / "paper_trace.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, observed_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    candidate = _freeze_candidate(orchestrator, Path(run["run_dir"]))
    trace = replay_trace(orchestrator.resolved_cfg, candidate, data_path)
    trace[0]["signals"] = [{"type": "SELL", "symbol": "SPY", "strength": 100, "metadata": {}}]
    observed_path.write_text(trace_to_csv(trace), encoding="utf-8")

    result = CrucibleOrchestrator(platform_path, workload_path).run_paper_replay_stage()
    mismatches = pd.read_csv(Path(result["run_dir"]) / "stages" / "09_paper_replay" / "summaries" / "paper_replay_mismatches.csv")

    assert result["status"] == "paper_replay_failed"
    assert result["summary"]["failure_reason"] == "paper_replay_mismatch"
    assert "signals_mismatch" in set(mismatches["type"])


def test_load_frozen_candidate_requires_tuning_lock(tmp_path: Path):
    run_dir = tmp_path / "run"
    frozen = run_dir / "frozen_candidates"
    frozen.mkdir(parents=True)
    candidate = {
        "candidate_id": "candidate_1",
        "algorithm_class": "a.A",
        "portfolio_class": "p.P",
        "algorithm_params": {},
        "portfolio_params": {},
    }
    (frozen / "candidate_1.json").write_text(json.dumps({"candidate": candidate, "tuning_locked": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="tuning_locked"):
        load_frozen_candidate(run_dir)


def _configs(tmp_path: Path, data_path: Path, observed_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "paper_replay_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs")},
        "state_store": {"backend": "local"},
        "paper_replay": {
            "observed_trace_path": str(observed_path),
            "data_path": str(data_path),
            "max_signal_mismatches": 0,
            "max_order_mismatches": 0,
            "max_regime_mismatches": 0,
        },
    }
    workload = {
        "workload": {"name": "paper_replay", "run_name": "paper_replay_v1"},
        "data_provider": {"provider": "trading.data_providers.test_data_provider.TestDataProvider", "path": str(data_path)},
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "evaluation_symbols": ["SPY"],
            "params": {"history_length": 1, "market_regime": {"enabled": True, "require_full_windows": False}},
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {"cash": 100000, "keep_history": True, "symbol": "SPY"},
        },
        "fixed_assumptions": {"starting_cash": 100000},
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def _freeze_candidate(orchestrator: CrucibleOrchestrator, run_dir: Path):
    candidate = build_candidate_from_params(
        orchestrator.resolved_cfg,
        orchestrator.resolved_cfg.workload["algorithm"]["params"],
        orchestrator.resolved_cfg.workload["portfolio"]["params"],
    )
    frozen = freeze_candidate(candidate, orchestrator.resolved_cfg)
    folder = run_dir / "frozen_candidates"
    folder.mkdir(exist_ok=True)
    (folder / f"{candidate.candidate_id}.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return candidate
