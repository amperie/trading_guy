from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_crucible.orchestrator import CrucibleOrchestrator
from tests.unit.test_algo_crucible_milestone1 import _write_yaml


def _write_daily_data(path: Path, rows: int = 90) -> None:
    lines = ["timestamp,symbol,open,high,low,close,volume"]
    price = 100.0
    for idx, ts in enumerate(pd.date_range("2024-01-01 15:00:00", periods=rows, freq="D")):
        price += 0.5 if idx < rows // 2 else -0.2
        lines.append(f"{ts:%Y-%m-%d %H:%M:%S},SPY,{price:.2f},{price + 0.1:.2f},{price - 0.1:.2f},{price:.2f},1000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configs(tmp_path: Path, data_path: Path) -> tuple[Path, Path]:
    platform = {
        "crucible": {"name": "test", "run_name": "wf_oos_v1"},
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
    }
    workload = {
        "workload": {"name": "wf_oos", "run_name": "wf_oos_v1"},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": str(data_path),
        },
        "order_manager": {"order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager"},
        "algorithm": {
            "algorithm": "algo_crucible.testing.BuyAndHoldAlgorithm",
            "params": {
                "history_length": 1,
                "market_regime": {
                    "enabled": True,
                    "trend_lookback_days": 2,
                    "baseline_ma_window_days": 2,
                    "volatility_lookback_days": 2,
                    "volatility_percentile_window_days": 4,
                    "drawdown_lookback_days": 4,
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


def test_walk_forward_oos_runs_validation_windows_and_reuses_results(tmp_path: Path):
    data_path = tmp_path / "daily.csv"
    _write_daily_data(data_path)
    platform_path, workload_path = _configs(tmp_path, data_path)
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)

    first = orchestrator.run_walk_forward_oos(use_ray=False)
    second = CrucibleOrchestrator(platform_path, workload_path).run_walk_forward_oos(rerun=True, use_ray=False)
    run_dir = Path(first["run_dir"])
    windows = pd.read_csv(run_dir / "summaries" / "window_summary.csv")
    oos = pd.read_csv(run_dir / "summaries" / "oos_summary.csv")
    regimes = pd.read_csv(run_dir / "summaries" / "validation_regime_summary.csv")

    assert first["summary"]["window_count"] == 4
    assert len(oos) == 4
    assert second["summary"]["jobs_reused"] == 4
    assert not regimes.empty
    assert all(pd.to_datetime(windows["validation_start"]) >= pd.to_datetime(windows["train_end"]) + pd.Timedelta(days=3))
    assert set(oos["window_id"]) == set(windows["window_id"])
