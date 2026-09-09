from __future__ import annotations

import json
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
    algo_path = tmp_path / "components" / "algo.py"
    portfolio_path = tmp_path / "components" / "portfolio.py"
    run_record_path = tmp_path / "platform_run_record.json"
    _write_data(data_path)
    algo_path.parent.mkdir()
    algo_path.write_text("class UploadedAlgorithm: pass\n", encoding="utf-8")
    portfolio_path.write_text("class UploadedPortfolio: pass\n", encoding="utf-8")
    run_record_path.write_text('{"run":{"id":"run_1"}}\n', encoding="utf-8")
    _, workload_path = _configs(tmp_path, data_path, run_name="mlflow_tiny_v1")
    workload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    workload["platform_runtime_assets"] = [
        {"role": "dataset", "container_path": str(data_path)},
        {"role": "algorithm", "container_path": str(algo_path)},
        {"role": "portfolio", "container_path": str(portfolio_path)},
        {"role": "platform_run_record", "container_path": str(run_record_path)},
    ]
    _write_yaml(workload_path, workload)
    platform_path = _mlflow_platform(tmp_path / "platform_mlflow.yaml", tmp_path / "mlruns", tmp_path / "runs")

    result = CrucibleOrchestrator(platform_path, workload_path).run_milestone1()

    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    run = client.get_run(result["mlflow_run_id"])
    artifacts = _artifact_paths(client, result["mlflow_run_id"])

    assert result["status"] == "complete"
    assert run.data.tags["crucible.run_id"] == result["crucible_run_id"]
    assert run.data.tags["crucible.status"] == "complete"
    assert run.data.params["algorithm.evaluation_symbols"] == "SPY"
    assert run.data.params["data_provider.class"] == "trading.data_providers.test_data_provider.TestDataProvider"
    assert "milestone1.total_return_pct" in run.data.metrics
    assert "configs/resolved_config.yaml" in artifacts
    assert "provenance/provenance_manifest.json" in artifacts
    assert "provenance/runtime_assets/dataset/data.csv" in artifacts
    assert "provenance/runtime_assets/algorithm/algo.py" in artifacts
    assert "provenance/runtime_assets/portfolio/portfolio.py" in artifacts
    assert (
        "provenance/runtime_assets/platform_run_record/platform_run_record.json"
        in artifacts
    )
    assert "stages/01_single_candidate/summaries/candidate_summary.csv" in artifacts
    assert "stages/01_single_candidate/summaries/regime_summary.csv" in artifacts
    provenance = json.loads(
        (Path(result["run_dir"]) / "provenance" / "provenance_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["dataset"]["exists"] is True
    assert provenance["dataset"]["sha256"]
    assert provenance["dataset"]["row_count"] > 0
    copied_assets = {item["role"]: item for item in provenance["copied_runtime_assets"]}
    assert copied_assets["dataset"]["file"]["sha256"]
    assert copied_assets["platform_run_record"]["file"]["exists"] is True


def _artifact_paths(client: MlflowClient, run_id: str, path: str | None = None) -> set[str]:
    paths = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            paths.update(_artifact_paths(client, run_id, item.path))
        else:
            paths.add(item.path)
    return paths


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
