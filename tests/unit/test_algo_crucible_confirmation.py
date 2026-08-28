from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from algo_crucible.builders import build_candidate_from_params
from algo_crucible.confirmation import (
    build_promotion_packet,
    confirmation_window,
    freeze_candidate,
    load_confirmation_candidates,
    packet_markdown,
    summarize_confirmation,
)
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.scoring import rows_to_csv
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def test_confirmation_window_requires_untouched_dates():
    with pytest.raises(ValueError, match="confirmation.start_date"):
        confirmation_window({"confirmation": {"start_date": "2024-04-01"}})


def test_confirmation_stage_writes_frozen_candidate_and_packets(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    candidate = _write_inputs(orchestrator, Path(run["run_dir"]))
    captured = {}

    def fake_validation(payload):
        captured["window"] = payload["window"]
        captured["workload_dp"] = payload["workload"]["data_provider"]
        return {
            "candidate_id": payload["candidate"]["candidate_id"],
            "window_id": payload["window"]["window_id"],
            "window": payload["window"],
            "overall_scorecard": {
                "total_return_pct": 3.0,
                "annualized_return": 3.0,
                "max_drawdown_pct": -2.0,
                "volatility": 1.0,
                "sharpe_ratio": 1.0,
                "sortino_ratio": 1.0,
                "total_trades": 1,
            },
            "regime_scorecard": [],
        }

    monkeypatch.setattr("algo_crucible.orchestrator.run_validation_backtest", fake_validation)
    result = CrucibleOrchestrator(platform_path, workload_path).run_confirmation_stage(use_ray=False)
    run_dir = Path(result["run_dir"])
    packet = json.loads((run_dir / "promotion" / "promotion_packet.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "summaries" / "confirmation_summary.csv")

    assert captured["window"]["validation_start"] == "2024-04-01"
    assert captured["window"]["validation_end"] == "2024-04-30"
    assert captured["workload_dp"]["start_date"] == "2024-04-01"
    assert captured["workload_dp"]["end_date"] == "2024-04-30"
    assert result["status"] == "complete"
    assert result["summary"]["paper_trading_started"] is False
    assert result["metrics"]["confirmation.promoted_candidates"] == 1.0
    assert summary.iloc[0]["candidate_id"] == candidate.candidate_id
    assert (run_dir / "frozen_candidates" / f"{candidate.candidate_id}.json").exists()
    assert (run_dir / "promotion" / "promotion_packet.yaml").exists()
    assert (run_dir / "promotion" / "promotion_packet.md").exists()
    assert packet["decision"] == "promote_to_paper"
    assert packet["paper_trading_started"] is False
    assert packet["approval_required"] is True
    assert not (tmp_path / "promoted").exists()


def test_promotion_packet_has_required_sections(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    candidate = build_candidate_from_params(
        orchestrator.resolved_cfg,
        orchestrator.resolved_cfg.workload["algorithm"]["params"],
        orchestrator.resolved_cfg.workload["portfolio"]["params"],
    )
    frozen = freeze_candidate(candidate, orchestrator.resolved_cfg)
    packet = build_promotion_packet(
        resolved_cfg=orchestrator.resolved_cfg,
        confirmation_rows=[{"candidate_id": candidate.candidate_id, "confirmed": True}],
        frozen_candidates=[frozen],
        artifact_paths={"confirmation_summary": "summaries/confirmation_summary.csv"},
    )

    assert packet["decision"] == "promote_to_paper"
    assert packet["confirmed_candidate_ids"] == [candidate.candidate_id]
    assert packet["frozen_candidates"][0]["tuning_locked"] is True
    assert "## Confirmation Results" in packet_markdown(packet)


def test_confirmation_rejects_missing_return_metric(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    candidate = build_candidate_from_params(
        orchestrator.resolved_cfg,
        orchestrator.resolved_cfg.workload["algorithm"]["params"],
        orchestrator.resolved_cfg.workload["portfolio"]["params"],
    )

    rows = summarize_confirmation(
        [{"candidate": candidate, "candidate_type": "generalist"}],
        [{
            "status": "complete",
            "result": {
                "candidate_id": candidate.candidate_id,
                "overall_scorecard": {"max_drawdown_pct": -1.0, "total_trades": 1},
            },
        }],
        orchestrator.resolved_cfg.platform,
    )

    assert rows[0]["confirmed"] is False
    assert rows[0]["failure_reason"] == "confirmation_missing_return"


def test_confirmation_resume_requires_packet_artifact(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    _write_inputs(orchestrator, Path(run["run_dir"]))
    orchestrator.state_store.write_artifact_json(
        orchestrator.resolved_cfg.crucible_run_id,
        "summaries/stage_08_summary.json",
        {"partial": True},
    )

    monkeypatch.setattr("algo_crucible.orchestrator.run_validation_backtest", _successful_confirmation)
    result = CrucibleOrchestrator(platform_path, workload_path).run_confirmation_stage(use_ray=False)

    assert result["summary"]["confirmed_candidates"] == 1
    assert (Path(result["run_dir"]) / "promotion" / "promotion_packet.json").exists()


def test_confirmation_handles_no_accepted_perturbation_candidates(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    run_dir = Path(run["run_dir"])
    (run_dir / "summaries" / "hpo_trial_summary.csv").write_text("", encoding="utf-8")
    (run_dir / "summaries" / "perturbation_summary.csv").write_text("", encoding="utf-8")

    result = CrucibleOrchestrator(platform_path, workload_path).run_confirmation_stage(use_ray=False)
    packet = json.loads((run_dir / "promotion" / "promotion_packet.json").read_text(encoding="utf-8"))

    assert result["summary"]["confirmed_candidates"] == 0
    assert result["metrics"]["confirmation.candidates_total"] == 0.0
    assert packet["decision"] == "reject"


def _configs(tmp_path: Path, data_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "confirmation_v1"},
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
        "gates": {"generalist": {"min_trades": 0, "min_median_oos_return": 0.0, "max_drawdown": 0.25}},
        "confirmation": {"start_date": "2024-04-01", "end_date": "2024-04-30", "min_return_pct": 0.0},
        "promotion": {"create_promoted_folder": False, "output_dir": str(tmp_path / "promoted")},
    }
    workload = {
        "workload": {"name": "confirmation", "run_name": "confirmation_v1"},
        "data_provider": {"provider": "trading.data_providers.test_data_provider.TestDataProvider", "path": str(data_path)},
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {"algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm", "params": {"history_length": 1}},
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


def _write_inputs(orchestrator: CrucibleOrchestrator, run_dir: Path):
    algorithm_params = dict(orchestrator.resolved_cfg.workload["algorithm"]["params"])
    portfolio_params = dict(orchestrator.resolved_cfg.workload["portfolio"]["params"])
    candidate = build_candidate_from_params(orchestrator.resolved_cfg, algorithm_params, portfolio_params)
    (run_dir / "summaries").mkdir(exist_ok=True)
    (run_dir / "summaries" / "hpo_trial_summary.csv").write_text(rows_to_csv([{
        "trial_id": "trial_0000",
        "candidate_id": candidate.candidate_id,
        "metric": 5.0,
        "config": "{}",
        "algorithm_params": json.dumps(algorithm_params, sort_keys=True),
        "portfolio_params": json.dumps(portfolio_params, sort_keys=True),
        "status": "complete",
    }]), encoding="utf-8")
    (run_dir / "summaries" / "perturbation_summary.csv").write_text(rows_to_csv([{
        "candidate_id": candidate.candidate_id,
        "seed_id": "seed_001",
        "candidate_type": "generalist",
        "specialist_regimes": "",
        "accepted": True,
    }]), encoding="utf-8")
    return candidate


def _successful_confirmation(payload):
    return {
        "candidate_id": payload["candidate"]["candidate_id"],
        "window_id": payload["window"]["window_id"],
        "window": payload["window"],
        "overall_scorecard": {
            "total_return_pct": 3.0,
            "annualized_return": 3.0,
            "max_drawdown_pct": -2.0,
            "volatility": 1.0,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.0,
            "total_trades": 1,
        },
        "regime_scorecard": [],
    }
