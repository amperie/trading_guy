from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient

from algo_crucible.orchestrator import CrucibleOrchestrator
from algo_crucible.state_store import ConfigChangedForRunName, RunAlreadyComplete
from tests.unit.test_algo_crucible_milestone1 import _configs, _write_data, _write_yaml


def _mlflow_platform(path: Path, tracking_dir: Path, cache_dir: Path, run_name: str = "mlflow_tiny_v1") -> Path:
    payload = {
        "crucible": {"name": "test", "run_name": run_name},
        "resume": {"local_cache_dir": str(cache_dir)},
        "state_store": {"backend": "mlflow"},
        "mlflow": {
            "tracking_uri": tracking_dir.as_uri(),
            "parent_experiment_name": "Algo Crucible Test",
        },
    }
    _write_yaml(path, payload)
    return path


def test_mlflow_state_store_logs_parent_run_and_artifacts(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    _, workload_path = _configs(tmp_path, data_path, run_name="mlflow_tiny_v1")
    platform_path = _mlflow_platform(tmp_path / "platform_mlflow.yaml", tmp_path / "mlruns", tmp_path / "runs")

    result = CrucibleOrchestrator(platform_path, workload_path).run_milestone1()

    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    run = client.get_run(result["mlflow_run_id"])
    artifacts = {item.path for item in client.list_artifacts(result["mlflow_run_id"], "summaries")}

    assert result["status"] == "complete"
    assert run.data.tags["crucible.run_id"] == result["crucible_run_id"]
    assert run.data.tags["crucible.status"] == "complete"
    assert "milestone1.total_return_pct" in run.data.metrics
    assert "summaries/candidate_summary.csv" in artifacts
    assert "summaries/regime_summary.csv" in artifacts


def test_mlflow_state_store_refuses_completed_duplicate(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    _, workload_path = _configs(tmp_path, data_path, run_name="mlflow_dup_v1")
    platform_path = _mlflow_platform(tmp_path / "platform_mlflow.yaml", tmp_path / "mlruns", tmp_path / "runs", "mlflow_dup_v1")
    CrucibleOrchestrator(platform_path, workload_path).run_milestone1()

    with pytest.raises(RunAlreadyComplete):
        CrucibleOrchestrator(platform_path, workload_path).run_milestone1()


def test_mlflow_state_store_rejects_changed_config_for_same_run_name(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_data(data_path)
    _, workload_path = _configs(tmp_path, data_path, run_name="mlflow_changed_v1")
    platform_path = _mlflow_platform(
        tmp_path / "platform_mlflow.yaml",
        tmp_path / "mlruns",
        tmp_path / "runs",
        "mlflow_changed_v1",
    )
    CrucibleOrchestrator(platform_path, workload_path).run_milestone1()

    workload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    workload["portfolio"]["params"]["cash"] = 55555
    _write_yaml(workload_path, workload)

    with pytest.raises(ConfigChangedForRunName):
        CrucibleOrchestrator(platform_path, workload_path).run_milestone1()


def teardown_module():
    mlflow.end_run()
