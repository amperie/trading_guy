from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import trading.platform.runner as platform_runner
from trading.platform.runner import (
    _apply_tenant_mlflow_grouping,
    _equity_chart_points,
    _monthly_returns,
    _namespace,
    _platform_backtest_config,
    _csv_data_diagnostics,
    _emit_crucible_runtime_diagnostics,
    _crucible_config_paths,
    _return_histogram,
    _trade_rows,
    _write_backtest_evidence_artifact,
    _write_chart_artifacts,
    _write_crucible_evidence_artifact,
    _validate_crucible_data,
    tenant_mlflow_experiment_name,
)


def _args(**overrides):
    values = {
        "stage": "smoke",
        "run_id": "run-1",
        "strategy_id": "strategy-1",
        "tenant_id": "tenant-abc",
        "config": "platform:backtest",
        "account": "paper",
        "symbol": "SPY",
        "cash": 100000.0,
        "algorithm": None,
        "algorithm_url": None,
        "portfolio": None,
        "portfolio_url": None,
        "data": None,
        "no_mlflow": False,
        "run_name": None,
        "session_id": None,
        "agg_period": None,
        "experiment_name": "Ignored Platform Default",
        "workload_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tenant_mlflow_experiment_name_is_stable():
    assert tenant_mlflow_experiment_name("tenant-abc") == "QC_tenant_tenant-abc"


def test_platform_backtest_uses_tenant_mlflow_experiment():
    cfg = _platform_backtest_config(_args())

    assert cfg["analysis"]["experiment_name"] == "QC_tenant_tenant-abc"


def test_tenant_mlflow_grouping_overrides_loaded_config_and_tags():
    cfg = {
        "analysis": {
            "experiment_name": "Old Experiment",
            "mlflow_tags": {"source": "test"},
        },
        "mlflow": {"parent_experiment_name": "Old Parent"},
    }

    grouped = _apply_tenant_mlflow_grouping(cfg, _args())

    assert grouped["analysis"]["experiment_name"] == "QC_tenant_tenant-abc"
    assert grouped["mlflow"]["parent_experiment_name"] == "QC_tenant_tenant-abc"
    assert grouped["analysis"]["mlflow_tags"] == {
        "source": "test",
        "tenant_id": "tenant-abc",
        "platform.tenant_id": "tenant-abc",
    }


def test_platform_namespace_passes_tenant_experiment_override():
    args = _namespace(_args())

    assert args.mlflow_experiment_name_override == "QC_tenant_tenant-abc"


def test_crucible_config_uses_tenant_parent_experiment(monkeypatch):
    written: dict[str, dict] = {}
    monkeypatch.setattr(
        platform_runner,
        "_load_yaml",
        lambda path: (
            {
                "mlflow": {"parent_experiment_name": "Old Parent"},
                "paper_replay": {"data_path": "data/SPY_5min.csv"},
            }
            if "platform" in str(path)
            else {"data_provider": {"path": "../data/SPY_5min.csv"}}
        ),
    )
    monkeypatch.setattr(
        platform_runner,
        "_write_yaml",
        lambda path, payload: written.__setitem__(path.name, payload),
    )

    _crucible_config_paths(
        _args(
            stage="crucible",
            output_dir="output",
            workload_config="workload.yaml",
            hpo_samples=4,
            hpo_concurrency=1,
            validation_period_days=30,
            use_ray=False,
        )
    )

    assert written["crucible_platform.yaml"]["tenant_id"] == "tenant-abc"
    assert written["crucible_platform.yaml"]["mlflow"]["parent_experiment_name"] == "QC_tenant_tenant-abc"
    assert Path(written["crucible_workload.yaml"]["data_provider"]["path"]).is_absolute()
    assert Path(written["crucible_platform.yaml"]["paper_replay"]["data_path"]).is_absolute()


def test_crucible_config_uses_data_override_for_workload(monkeypatch):
    written: dict[str, dict] = {}
    data_path = str((Path.cwd() / "data" / "uploaded.csv").resolve())
    monkeypatch.setattr(
        platform_runner,
        "_load_yaml",
        lambda path: {"data_provider": {"path": "../data/SPY_5min.csv"}},
    )
    monkeypatch.setattr(
        platform_runner,
        "_write_yaml",
        lambda path, payload: written.__setitem__(path.name, payload),
    )

    _crucible_config_paths(
        _args(
            stage="crucible",
            output_dir="output",
            config="platform:crucible",
            data=data_path,
            hpo_samples=4,
            hpo_concurrency=1,
            validation_period_days=30,
            use_ray=False,
        )
    )

    assert written["crucible_workload.yaml"]["data_provider"]["path"] == data_path


def test_emit_crucible_runtime_diagnostics(tmp_path, monkeypatch):
    platform_path = tmp_path / "platform.yaml"
    workload_path = tmp_path / "workload.yaml"
    platform_path.write_text(
        """
crucible:
  run_name: run-1
hpo:
  num_samples: 4
""",
        encoding="utf-8",
    )
    workload_path.write_text(
        """
data_provider:
  path: /workspace/results/inputs/spy.csv
algorithm:
  algorithm: platform_assets.UploadedAlgorithm
  source_path: /workspace/results/components/algo.py
portfolio:
  portfolio: platform_assets.UploadedPortfolio
  source_path: /workspace/results/components/pf.py
platform_runtime_assets:
  - role: dataset
    uri: s3://bucket/tenant/uploads/datasets/spy.csv
    host_path: E:/runs/run-1/inputs/spy.csv
    container_path: /workspace/results/inputs/spy.csv
""",
        encoding="utf-8",
    )
    events = []
    monkeypatch.setattr(platform_runner, "emit", lambda *args, **kwargs: events.append((args, kwargs)))

    _emit_crucible_runtime_diagnostics(platform_path, workload_path)

    assert events[0][0] == (9, "Crucible runtime diagnostics")
    payload = events[0][1]
    assert payload["dataPath"] == "/workspace/results/inputs/spy.csv"
    assert payload["algorithmConfig"]["source_path"] == "/workspace/results/components/algo.py"
    assert payload["portfolioConfig"]["source_path"] == "/workspace/results/components/pf.py"
    assert payload["runtimeAssets"][0]["uri"] == "s3://bucket/tenant/uploads/datasets/spy.csv"


def test_validate_crucible_data_rejects_tiny_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2015-01-02T09:30:00,SPY,1,1,1,1,100\n"
        "2015-01-02T09:35:00,SPY,1,1,1,1,100\n",
        encoding="utf-8",
    )
    data_provider = {
        "path": str(path),
        "start_date": "2015-01-02",
        "end_date": "2022-12-30",
    }
    diagnostics = _csv_data_diagnostics(data_provider)

    assert diagnostics["totalRows"] == 2
    assert diagnostics["filteredRows"] == 2
    with pytest.raises(ValueError, match="Crucible dataset is too small"):
        _validate_crucible_data(data_provider, diagnostics)


def test_write_chart_artifacts_from_portfolio_value_history(monkeypatch):
    written: dict[str, str] = {}

    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, text, **kwargs: written.__setitem__(self.as_posix(), text),
    )

    start = datetime(2026, 1, 1)
    portfolio = SimpleNamespace(
        value_history={
            start: 100000.0,
            start + timedelta(days=1): 101000.0,
            start + timedelta(days=2): 99000.0,
        }
    )

    manifest = _write_chart_artifacts(Path("output"), "smoke", portfolio)

    assert manifest["charts"][0]["artifact"] == "charts/equity_curve.json"
    assert "output/chart_manifest.json" in written
    assert "output/charts/equity_curve.json" in written


def test_write_backtest_evidence_artifact(monkeypatch):
    written: dict[str, str] = {}
    start = datetime(2026, 1, 1)
    portfolio = SimpleNamespace(value_history={start: 100.0, start + timedelta(days=1): 110.0})
    trade = SimpleNamespace(
        entry_time=start,
        exit_time=start + timedelta(hours=1),
        quantity=7,
        pnl=42.5,
        pnl_pct=1.25,
        duration=3600,
    )
    analysis = {
        "daily_returns": {start: 0.01, start + timedelta(days=1): -0.005},
        "monthly_returns": {start: 0.035},
        "trades": [trade],
    }

    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, text, **kwargs: written.__setitem__(self.as_posix(), text),
    )

    evidence = _write_backtest_evidence_artifact(Path("output"), "research", portfolio, analysis, {"sharpe": 1.2})

    assert evidence["tradeCount"] == 1
    assert evidence["trades"][0]["bars"] == 12
    assert evidence["monthlyReturns"][0]["months"][0]["value"] == 3.5
    assert "output/backtest_evidence.json" in written


def test_equity_chart_points_include_drawdown():
    start = datetime(2026, 1, 1)
    portfolio = SimpleNamespace(
        value_history={
            start: 100.0,
            start + timedelta(days=1): 120.0,
            start + timedelta(days=2): 90.0,
        }
    )

    points = _equity_chart_points(portfolio)

    assert points[-1]["strategy"] == 90.0
    assert points[-1]["benchmark"] == 100.0
    assert points[-1]["drawdown"] == -25.0


def test_return_histogram_buckets_percent_returns():
    histogram = _return_histogram({"a": 0.011, "b": -0.004, "c": 0.002})

    assert histogram[0]["bucket"].endswith("%")
    assert sum(bucket["count"] for bucket in histogram) == 3


def test_monthly_returns_groups_by_year_and_month():
    rows = _monthly_returns({datetime(2026, 1, 1): 0.01, datetime(2026, 2, 1): -0.02})

    assert rows[0]["year"] == 2026
    assert rows[0]["months"][0] == {"month": "Jan", "value": 1.0}
    assert rows[0]["months"][1] == {"month": "Feb", "value": -2.0}


def test_trade_rows_are_capped_and_normalized():
    trades = [SimpleNamespace(entry_time="in", exit_time="out", side="buy", quantity=3, pnl=-5, pnl_pct=-1)]

    assert _trade_rows(trades)[0]["side"] == "long"


def test_write_crucible_evidence_aggregates_stage_outputs():
    root = Path(".pytest_tmp") / f"unit_crucible_evidence_{uuid4().hex}"
    run_dir = root / "crucible_runs" / "run-1"
    (run_dir / "stages/03_walk_forward_oos/summaries").mkdir(parents=True, exist_ok=True)
    (run_dir / "stages/05_regime_gate/summaries").mkdir(parents=True, exist_ok=True)
    (run_dir / "stages/08_confirmation/summaries").mkdir(parents=True, exist_ok=True)
    (run_dir / "stages/03_walk_forward_oos/summaries/stage_summary.json").write_text(
        '{"jobs_total": 2, "jobs_complete": 2, "profitable_windows_pct": 50}',
        encoding="utf-8",
    )
    (run_dir / "stages/03_walk_forward_oos/summaries/oos_summary.csv").write_text(
        "window_id,sharpe_ratio,total_return_pct,max_drawdown_pct\nw1,1.2,3.5,-4.1\n",
        encoding="utf-8",
    )
    (run_dir / "stages/05_regime_gate/summaries/stage_summary.json").write_text(
        '{"passed_candidate_count": 1, "reject_count": 0}',
        encoding="utf-8",
    )
    (run_dir / "stages/08_confirmation/summaries/stage_summary.json").write_text(
        '{"confirmed_candidates": 1, "rejected_candidates": 0}',
        encoding="utf-8",
    )

    evidence = _write_crucible_evidence_artifact(
        root / "platform_run",
        {"status": "complete", "run_dir": str(run_dir), "crucible_run_id": "run-1"},
        [("walk_forward_oos", "Running walk-forward OOS validation")],
        {"confirmation.promoted_candidates": 1.0, "walk_forward_oos.profitable_windows_pct": 50.0},
    )

    assert evidence["milestones"][1]["status"] == "complete"
    assert evidence["promotionCriteria"][-1]["status"] == "pass"
    assert evidence["walkForward"][0]["window"] == "w1"
    assert (root / "platform_run" / "crucible_evidence.json").exists()
