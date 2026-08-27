from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


class CrucibleStateError(RuntimeError):
    pass


class ConfigChangedForRunName(CrucibleStateError):
    pass


class RunAlreadyComplete(CrucibleStateError):
    pass


class CrucibleStateStore:
    def start_or_resume(self, resolved_cfg, rerun: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def update_run(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def write_artifact_text(self, run_id: str, relative_path: str, text: str) -> str:
        raise NotImplementedError

    def write_artifact_json(self, run_id: str, relative_path: str, payload: Any) -> str:
        return self.write_artifact_text(run_id, relative_path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    def read_artifact_json(self, run_id: str, relative_path: str) -> Any | None:
        raise NotImplementedError


class LocalCrucibleStateStore:
    """Filesystem-backed state store for local development and tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_or_resume(self, resolved_cfg, rerun: bool = False) -> dict[str, Any]:
        exact = self._find_by(run_name=resolved_cfg.run_name, config_hash=resolved_cfg.resolved_config_hash)
        if exact:
            if exact["status"] == "complete" and not rerun:
                raise RunAlreadyComplete(f"{resolved_cfg.crucible_run_id} is already complete")
            return exact
        if self._find_by(run_name=resolved_cfg.run_name):
            raise ConfigChangedForRunName(f"run_name '{resolved_cfg.run_name}' already exists with a different config")

        run_dir = self.root / resolved_cfg.crucible_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "summaries").mkdir()
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_cfg.resolved, handle, sort_keys=True)
        manifest = {
            "run_name": resolved_cfg.run_name,
            "crucible_run_id": resolved_cfg.crucible_run_id,
            "resolved_config_hash": resolved_cfg.resolved_config_hash,
            "status": "running",
            "run_dir": str(run_dir),
        }
        self._write_json_atomic(run_dir / "manifest.json", manifest)
        return manifest

    def update_run(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        path = self.root / run_id / "manifest.json"
        manifest = self._read_json(path)
        manifest.update(patch)
        self._write_json_atomic(path, manifest)
        return manifest

    def write_artifact_text(self, run_id: str, relative_path: str, text: str) -> str:
        path = self.root / run_id / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(path, text)
        return str(path)

    def write_artifact_json(self, run_id: str, relative_path: str, payload: Any) -> str:
        return self.write_artifact_text(run_id, relative_path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    def read_artifact_json(self, run_id: str, relative_path: str) -> Any | None:
        path = self.root / run_id / relative_path
        if not path.exists():
            return None
        return self._read_json(path)

    def _find_by(self, *, run_name: str, config_hash: str | None = None) -> dict[str, Any] | None:
        for manifest_path in self.root.glob("*/manifest.json"):
            manifest = self._read_json(manifest_path)
            if manifest.get("run_name") != run_name:
                continue
            if config_hash is not None and manifest.get("resolved_config_hash") != config_hash:
                continue
            return manifest
        return None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        LocalCrucibleStateStore._write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
        os.replace(tmp, path)


class MLflowCrucibleStateStore(CrucibleStateStore):
    """MLflow-backed run registry with a local artifact staging cache."""

    tag_run_name = "crucible.run_name"
    tag_run_id = "crucible.run_id"
    tag_hash = "crucible.resolved_config_hash"
    tag_status = "crucible.status"

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str = "Algo Crucible",
        local_cache_dir: str | Path = "scratch/crucible_runs",
        artifact_location: str | None = None,
    ):
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise ImportError("MLflowCrucibleStateStore requires mlflow") from exc

        self.mlflow = mlflow
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.tracking_uri = tracking_uri
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        experiment = self.mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self.mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            experiment = self.mlflow.get_experiment(experiment_id)
        self.experiment = experiment
        self.experiment_id = experiment.experiment_id
        self.root = Path(local_cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_or_resume(self, resolved_cfg, rerun: bool = False) -> dict[str, Any]:
        exact = self._find_by(run_name=resolved_cfg.run_name, config_hash=resolved_cfg.resolved_config_hash)
        if exact:
            if exact["status"] == "complete" and not rerun:
                raise RunAlreadyComplete(f"{resolved_cfg.crucible_run_id} is already complete")
            return exact
        if self._find_by(run_name=resolved_cfg.run_name):
            raise ConfigChangedForRunName(f"run_name '{resolved_cfg.run_name}' already exists with a different config")

        run_dir = self._prepare_run_dir(resolved_cfg)
        tags = {
            self.tag_run_name: resolved_cfg.run_name,
            self.tag_run_id: resolved_cfg.crucible_run_id,
            self.tag_hash: resolved_cfg.resolved_config_hash,
            self.tag_status: "running",
            "crucible.platform_config_hash": resolved_cfg.platform_hash,
            "crucible.workload_config_hash": resolved_cfg.workload_hash,
        }
        with self.mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=resolved_cfg.crucible_run_id,
            tags=tags,
        ) as active:
            run_id = active.info.run_id
            self.mlflow.log_params({
                "crucible.run_name": resolved_cfg.run_name,
                "crucible.run_id": resolved_cfg.crucible_run_id,
                "crucible.resolved_config_hash": resolved_cfg.resolved_config_hash,
            })
            self.mlflow.log_artifact(str(run_dir / "resolved_config.yaml"))
        manifest = {
            "run_name": resolved_cfg.run_name,
            "crucible_run_id": resolved_cfg.crucible_run_id,
            "resolved_config_hash": resolved_cfg.resolved_config_hash,
            "status": "running",
            "run_dir": str(run_dir),
            "mlflow_run_id": run_id,
            "mlflow_run_url": self._run_url(run_id),
        }
        self._write_manifest(run_dir, manifest)
        return manifest

    def update_run(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        manifest = self._find_by_crucible_run_id(run_id)
        if manifest is None:
            raise CrucibleStateError(f"Unknown crucible run id: {run_id}")
        mlflow_run_id = manifest["mlflow_run_id"]
        status = patch.get("status")
        if status:
            self.client.set_tag(mlflow_run_id, self.tag_status, status)
        if "summary" in patch:
            self.client.log_dict(mlflow_run_id, patch["summary"], "summaries/stage_summary.json")
        for key, value in (patch.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and value is not None:
                self.client.log_metric(mlflow_run_id, key, float(value))
        manifest.update(patch)
        self._write_manifest(Path(manifest["run_dir"]), manifest)
        return manifest

    def write_artifact_text(self, run_id: str, relative_path: str, text: str) -> str:
        manifest = self._find_by_crucible_run_id(run_id)
        if manifest is None:
            raise CrucibleStateError(f"Unknown crucible run id: {run_id}")
        path = Path(manifest["run_dir"]) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        LocalCrucibleStateStore._write_text_atomic(path, text)
        artifact_path = str(Path(relative_path).parent).replace("\\", "/")
        artifact_path = None if artifact_path == "." else artifact_path
        self.client.log_artifact(manifest["mlflow_run_id"], str(path), artifact_path=artifact_path)
        return str(path)

    def read_artifact_json(self, run_id: str, relative_path: str) -> Any | None:
        manifest = self._find_by_crucible_run_id(run_id)
        if manifest is None:
            raise CrucibleStateError(f"Unknown crucible run id: {run_id}")
        local_path = Path(manifest["run_dir"]) / relative_path
        if not local_path.exists():
            try:
                downloaded = self.client.download_artifacts(manifest["mlflow_run_id"], relative_path)
                local_path = Path(downloaded)
            except Exception:
                return None
        if not local_path.exists():
            return None
        with local_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _prepare_run_dir(self, resolved_cfg) -> Path:
        run_dir = self.root / resolved_cfg.crucible_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summaries").mkdir(exist_ok=True)
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_cfg.resolved, handle, sort_keys=True)
        return run_dir

    def _find_by(self, *, run_name: str, config_hash: str | None = None) -> dict[str, Any] | None:
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.`{self.tag_run_name}` = '{_escape_mlflow_filter(run_name)}'",
            max_results=1000,
        )
        for run in runs:
            tags = run.data.tags
            if config_hash is not None and tags.get(self.tag_hash) != config_hash:
                continue
            return self._manifest_from_run(run)
        return None

    def _find_by_crucible_run_id(self, crucible_run_id: str) -> dict[str, Any] | None:
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.`{self.tag_run_id}` = '{_escape_mlflow_filter(crucible_run_id)}'",
            max_results=1,
        )
        return self._manifest_from_run(runs[0]) if runs else None

    def _manifest_from_run(self, run) -> dict[str, Any]:
        tags = run.data.tags
        run_id = tags.get(self.tag_run_id, run.info.run_name)
        run_dir = self.root / run_id
        return {
            "run_name": tags.get(self.tag_run_name),
            "crucible_run_id": run_id,
            "resolved_config_hash": tags.get(self.tag_hash),
            "status": tags.get(self.tag_status, "running"),
            "run_dir": str(run_dir),
            "mlflow_run_id": run.info.run_id,
            "mlflow_run_url": self._run_url(run.info.run_id),
        }

    def _run_url(self, run_id: str) -> str:
        uri = self.tracking_uri or self.mlflow.get_tracking_uri()
        if uri.startswith("file:"):
            return f"http://localhost:5000/#/experiments/{self.experiment_id}/runs/{run_id}"
        return f"{uri}/#/experiments/{self.experiment_id}/runs/{run_id}"

    @staticmethod
    def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        LocalCrucibleStateStore._write_json_atomic(run_dir / "manifest.json", manifest)


def create_state_store(platform: dict[str, Any]) -> CrucibleStateStore:
    backend = platform.get("state_store", {}).get("backend", "local")
    cache_dir = platform.get("resume", {}).get("local_cache_dir", "scratch/crucible_runs")
    if backend == "local":
        return LocalCrucibleStateStore(cache_dir)
    if backend == "mlflow":
        mlflow_cfg = platform.get("mlflow", {})
        return MLflowCrucibleStateStore(
            tracking_uri=mlflow_cfg.get("tracking_uri"),
            experiment_name=mlflow_cfg.get("parent_experiment_name", "Algo Crucible"),
            local_cache_dir=cache_dir,
            artifact_location=mlflow_cfg.get("artifact_location"),
        )
    raise ValueError(f"Unsupported crucible state_store backend: {backend}")


def _escape_mlflow_filter(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")
