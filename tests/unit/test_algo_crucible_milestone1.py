from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from algo_crucible.config import resolve_config_dicts
from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.state_store import ConfigChangedForRunName, LocalCrucibleStateStore, RunAlreadyComplete


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_data(path: Path, rows: int = 80) -> None:
    lines = ["timestamp,symbol,open,high,low,close,volume"]
    price = 100.0
    for idx in range(rows):
        price += 0.2 if idx < rows // 2 else -0.1
        ts = f"2024-01-{(idx // 20) + 2:02d} 09:{30 + (idx % 20):02d}:00"
        lines.append(f"{ts},SPY,{price:.2f},{price + 0.1:.2f},{price - 0.1:.2f},{price:.2f},1000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configs(tmp_path: Path, data_path: Path, run_name: str = "tiny_v1") -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": run_name},
        "resume": {"local_cache_dir": str(tmp_path / "runs")},
        "state_store": {"backend": "local"},
    }
    workload = {
        "workload": {"name": "tiny", "run_name": run_name},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "evaluation_symbols": ["SPY"],
            "params": {
                "history_length": 2,
                "market_regime": {
                    "enabled": True,
                    "trend_lookback_hours": 1,
                    "baseline_ma_window_hours": 1,
                    "volatility_lookback_hours": 1,
                    "volatility_percentile_window_hours": 2,
                    "drawdown_lookback_hours": 2,
                    "require_full_windows": False,
                },
            },
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {
                "cash": 100000,
                "keep_history": True,
                "symbol": "SPY",
                "stop_pct": 50.0,
                "profit_pct": 50.0,
            },
        },
        "fixed_assumptions": {"starting_cash": 100000},
    }
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    _write_yaml(platform_path, platform)
    _write_yaml(workload_path, workload)
    return platform_path, workload_path


def test_run_and_candidate_ids_are_deterministic():
    platform = {"crucible": {"run_name": "same"}}
    workload = {
        "workload": {"run_name": "same"},
        "algorithm": {"algorithm": "a.A", "evaluation_symbols": ["SPY"], "params": {"x": 1}},
        "portfolio": {"portfolio": "p.P", "params": {"y": 2}},
    }
    first = resolve_config_dicts(platform, workload)
    second = resolve_config_dicts(platform, workload)

    assert first.crucible_run_id == second.crucible_run_id
    assert first.resolved_config_hash == second.resolved_config_hash


def test_algorithm_evaluation_symbols_are_normalized_into_resolved_config():
    resolved = resolve_config_dicts(
        {"crucible": {"run_name": "symbols_v1"}},
        {
            "workload": {"run_name": "symbols_v1"},
            "data_provider": {"provider": "p.Provider", "symbols": ["SPY", "QQQ"]},
            "algorithm": {"algorithm": "a.A", "evaluation_symbols": ["qqq", "SPY", "spy"], "params": {}},
            "portfolio": {"portfolio": "p.P", "params": {"symbol": "SPY"}},
        },
    )

    assert resolved.workload["algorithm"]["evaluation_symbols"] == ["QQQ", "SPY"]
    assert resolved.workload["fixed_assumptions"]["evaluation_symbols"] == ["QQQ", "SPY"]


def test_data_provider_symbols_must_cover_algorithm_evaluation_symbols():
    with pytest.raises(ValueError, match="missing algorithm evaluation symbols"):
        resolve_config_dicts(
            {"crucible": {"run_name": "symbols_v1"}},
            {
                "workload": {"run_name": "symbols_v1"},
                "data_provider": {"provider": "p.Provider", "symbols": ["UPRO"]},
                "algorithm": {"algorithm": "a.A", "evaluation_symbols": ["SPY", "UPRO"], "params": {}},
                "portfolio": {"portfolio": "p.P", "params": {"symbol": "UPRO"}},
            },
        )


def test_algorithm_evaluation_symbols_are_required():
    with pytest.raises(ValueError, match="algorithm.evaluation_symbols is required"):
        resolve_config_dicts(
            {"crucible": {"run_name": "symbols_v1"}},
            {
                "workload": {"run_name": "symbols_v1"},
                "data_provider": {"provider": "p.Provider"},
                "algorithm": {"algorithm": "a.A", "params": {}},
                "portfolio": {"portfolio": "p.P", "params": {"symbol": "SPY"}},
            },
        )


def test_milestone1_single_candidate_outputs(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)

    result = CrucibleOrchestrator(platform_path, workload_path).run_milestone1()
    run_dir = Path(result["run_dir"])

    assert result["status"] == "complete"
    assert result["summary"]["backtest_count"] == 1
    assert (run_dir / "resolved_config.yaml").exists()
    stage_dir = run_dir / "stages" / "01_single_candidate"
    assert (stage_dir / "summaries" / "stage_summary.json").exists()
    assert (stage_dir / "summaries" / "candidate_summary.csv").read_text(encoding="utf-8").count("\n") == 2
    assert "regime" in (stage_dir / "summaries" / "regime_summary.csv").read_text(encoding="utf-8")
    assert result["metrics"]["milestone1.total_trades"] >= 0


def test_same_run_name_with_different_config_fails(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    CrucibleOrchestrator(platform_path, workload_path).run_milestone1()

    workload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    workload["portfolio"]["params"]["cash"] = 12345
    _write_yaml(workload_path, workload)

    with pytest.raises(ConfigChangedForRunName):
        CrucibleOrchestrator(platform_path, workload_path).run_milestone1()


def test_completed_duplicate_refuses_without_rerun(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    CrucibleOrchestrator(platform_path, workload_path).run_milestone1()

    with pytest.raises(RunAlreadyComplete):
        CrucibleOrchestrator(platform_path, workload_path).run_milestone1()


def test_incomplete_exact_run_resumes(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    resolved = CrucibleOrchestrator(platform_path, workload_path).resolved_cfg
    store = LocalCrucibleStateStore(tmp_path / "runs")
    manifest = store.start_or_resume(resolved)

    resumed = store.start_or_resume(resolved)

    assert resumed["crucible_run_id"] == manifest["crucible_run_id"]
    assert resumed["status"] == "running"
