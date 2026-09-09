from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import algo_crucible.orchestrator as orchestrator_module
from algo_crucible.builders import build_candidate_from_params
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.scoring import rows_to_csv
from algo_crucible.transfer import apply_transfer_instrument, summarize_transfer
from tests.unit.test_algo_crucible_milestone1 import _write_yaml
from tests.unit.test_algo_crucible_walk_forward_oos import _write_daily_data


def test_transfer_rejects_candidate_when_required_sibling_fails():
    analyzed = summarize_transfer(
        candidates=[{"seed_id": "seed_1", "candidate": type("CandidateStub", (), {"candidate_id": "candidate_1"})()}],
        instruments=[
            {"instrument_id": "instrument_001_spy", "name": "SPY", "symbol": "SPY", "required": True},
            {"instrument_id": "instrument_002_spxu", "name": "SPXU", "symbol": "SPXU", "required": False},
        ],
        job_results=[
            _job("candidate_1", "instrument_001_spy", -1.0),
            _job("candidate_1", "instrument_002_spxu", 2.0),
        ],
        platform={"cross_instrument_transfer": {"min_median_oos_return": 0.0, "min_profitable_windows_pct": 0.0}},
    )

    assert analyzed["summary_rows"][0]["accepted"] is False
    assert analyzed["summary_rows"][0]["required_failure_count"] == 1


def test_transfer_stage_writes_summary_artifacts(monkeypatch, tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    sibling_path = tmp_path / "sibling.csv"
    _write_daily_data(data_path)
    _write_daily_data(sibling_path)
    platform_path, workload_path = _configs(tmp_path, data_path, sibling_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    run = orchestrator.state_store.start_or_resume(orchestrator.resolved_cfg, rerun=True)
    candidate = _write_inputs(orchestrator, Path(run["run_dir"]))

    def fake_validation(payload):
        return {
            "instrument_id": payload["instrument_id"],
            "source_candidate_id": payload["source_candidate_id"],
            "candidate_id": payload["candidate"]["candidate_id"],
            "window_id": payload["window"]["window_id"],
            "window": payload["window"],
            "overall_scorecard": {
                "total_return_pct": 3.0,
                "annualized_return": 3.0,
                "sharpe_ratio": 1.0,
                "sortino_ratio": 1.0,
                "max_drawdown_pct": -2.0,
                "win_rate": 100.0,
                "profit_factor": 1.0,
                "total_trades": 2,
                "final_equity": 100003,
                "initial_equity": 100000,
                "trading_days": 10,
                "volatility": 1.0,
            },
            "regime_scorecard": [],
            "return_stream": [{"step": 1, "timestamp": "2024-01-01", "return_pct": 1.0, "equity": 101.0}],
        }

    monkeypatch.setattr(orchestrator_module, "run_validation_backtest", fake_validation)
    result = CrucibleOrchestrator(platform_path, workload_path).run_cross_instrument_transfer_stage(rerun=True, use_ray=False)
    stage_dir = Path(result["run_dir"]) / "stages" / "07_cross_instrument_transfer" / "summaries"
    summary = pd.read_csv(stage_dir / "transfer_summary.csv")
    instruments = pd.read_csv(stage_dir / "transfer_instrument_summary.csv")
    returns = pd.read_csv(stage_dir / "transfer_return_stream.csv")

    assert summary.iloc[0]["candidate_id"] == candidate.candidate_id
    assert summary.iloc[0]["accepted"] == True
    assert instruments.iloc[0]["symbol"] == "QQQ"
    assert returns.iloc[0]["instrument_id"] == "instrument_001_qqq"
    assert result["metrics"]["transfer.accepted_candidates"] == 1.0


def test_transfer_patch_updates_symbol_params(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    candidate = build_candidate_from_params(
        orchestrator.resolved_cfg,
        {"history_length": 1, "symbol": "SPY", "symbols": ["SPY"], "tradable_symbols": ["SPY"]},
        {"cash": 100000, "keep_history": True, "symbol": "SPY"},
    )

    workload, transferred = apply_transfer_instrument(
        orchestrator.resolved_cfg,
        candidate,
        {"symbol": "QQQ", "data_provider_patch": {"path": str(data_path), "symbols": ["QQQ"]}},
    )

    assert workload["data_provider"]["symbols"] == ["QQQ"]
    assert transferred.algorithm_params["symbol"] == "QQQ"
    assert transferred.portfolio_params["symbol"] == "QQQ"


def _job(candidate_id: str, instrument_id: str, total_return_pct: float) -> dict:
    return {
        "status": "complete",
        "result": {
            "source_candidate_id": candidate_id,
            "instrument_id": instrument_id,
            "overall_scorecard": {"total_return_pct": total_return_pct, "max_drawdown_pct": -1.0, "total_trades": 1},
        },
    }


def _configs(tmp_path: Path, data_path: Path, sibling_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "transfer_v1"},
        "resume": {"local_cache_dir": str(tmp_path / "runs"), "rerun_failed_jobs": True},
        "state_store": {"backend": "local"},
        "ray": {"enabled": False, "max_concurrent_trials": 2},
        "walk_forward": {"optimization_window_days": 20, "validation_window_days": 10, "embargo_days": 0, "step_days": 10, "min_windows": 1},
        "cross_instrument_transfer": {
            "min_windows": 1,
            "min_trades": 1,
            "min_median_oos_return": 0.0,
            "min_profitable_windows_pct": 0.50,
            "max_drawdown": 0.25,
            "instruments": [
                {
                    "name": "QQQ",
                    "symbol": "QQQ",
                    "required": True,
                    "data_provider_patch": {"path": str(sibling_path), "symbols": ["QQQ"]},
                    "algorithm_param_patch": {"symbol": "QQQ", "symbols": ["QQQ"], "tradable_symbols": ["QQQ"]},
                    "portfolio_param_patch": {"symbol": "QQQ"},
                }
            ],
        },
    }
    workload = {
        "workload": {"name": "transfer", "run_name": "transfer_v1"},
        "data_provider": {"provider": "trading.data_providers.test_data_provider.TestDataProvider", "path": str(data_path), "symbols": ["SPY"]},
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "evaluation_symbols": ["SPY"],
            "params": {"history_length": 1, "symbol": "SPY", "symbols": ["SPY"], "tradable_symbols": ["SPY"]},
        },
        "portfolio": {"portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio", "params": {"cash": 100000, "keep_history": True, "symbol": "SPY"}},
        "fixed_assumptions": {"starting_cash": 100000},
        "search_space": {"space": {}, "algorithm_param_keys": [], "portfolio_param_keys": []},
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
    (run_dir / "summaries" / "plateau_summary.csv").write_text(rows_to_csv([{
        "seed_id": "seed_001",
        "candidate_id": candidate.candidate_id,
        "accepted": True,
        "plateau_score": 1.0,
    }]), encoding="utf-8")
    return candidate
