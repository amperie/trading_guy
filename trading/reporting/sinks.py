from __future__ import annotations

import json
import os
import shutil
from abc import ABC
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trading.reporting.models import ReportArtifact, ReportResult
from utils.mlflow_client import MLflowClient


class AnalysisSink(ABC):
    name = "sink"

    def start_run(self, *, run_name=None, description=None, tags=None) -> dict[str, Any]:
        return {}

    def end_run(self, status: str = "FINISHED") -> None:
        return None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_text(self, text: str, filename: str) -> None:
        return None

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        return None

    def result(self) -> ReportResult:
        return ReportResult()


class MlflowAnalysisSink(AnalysisSink):
    name = "mlflow"

    def __init__(self, client: MLflowClient):
        self.client = client
        self.run_info: dict[str, Any] = {}

    @classmethod
    def from_report(cls, report) -> "MlflowAnalysisSink":
        if report.tracking_uri:
            client = MLflowClient(experiment_name=report.experiment_name, tracking_uri=report.tracking_uri)
        else:
            client = MLflowClient.from_config(experiment_name=report.experiment_name)
        return cls(client)

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def start_run(self, *, run_name=None, description=None, tags=None) -> dict[str, Any]:
        self.client.start_run(run_name=run_name, description=description, tags=tags)
        self.run_info = {"run_id": self.client.run_id or "", "run_url": self.client.get_run_url()}
        return self.run_info

    def end_run(self, status: str = "FINISHED") -> None:
        self.client.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        self.client.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.client.log_metrics(metrics, step=step)

    def log_text(self, text: str, filename: str) -> None:
        self.client.log_text(text, filename)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.client.log_artifact(local_path, artifact_path=artifact_path)

    def result(self) -> ReportResult:
        return ReportResult(sink_runs={self.name: self.run_info})


class LocalRunResultSink(AnalysisSink):
    name = "local_result"

    def __init__(self, output_dir: str | os.PathLike[str]):
        self.output_dir = Path(output_dir)
        self.artifact_dir = self.output_dir / "artifacts"
        self._result = ReportResult()

    def start_run(self, *, run_name=None, description=None, tags=None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if tags:
            self._result.tags.update(tags)
        info = {"result_dir": str(self.output_dir)}
        self._result.sink_runs[self.name] = info
        if run_name:
            self._result.tags["run_name"] = run_name
        if description:
            self._result.tags["description"] = description
        return info

    def end_run(self, status: str = "FINISHED") -> None:
        self._result.tags["status"] = status
        manifest = asdict(self._result)
        with (self.output_dir / "run_result.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    def log_params(self, params: dict[str, Any]) -> None:
        self._result.parameters.update(params)
        self._write_json("parameters.json", self._result.parameters)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        prefix = f"step_{step}." if step is not None else ""
        self._result.metrics.update({f"{prefix}{k}": float(v) for k, v in metrics.items()})
        self._write_json("metrics.json", self._result.metrics)

    def log_text(self, text: str, filename: str) -> None:
        path = self.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._record_artifact(path)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        src = Path(local_path)
        if not src.is_file():
            return
        dst_dir = self.artifact_dir / artifact_path if artifact_path else self.artifact_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        self._record_artifact(dst, artifact_path=artifact_path)

    def result(self) -> ReportResult:
        return self._result

    def _write_json(self, filename: str, payload: Any) -> None:
        with (self.output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _record_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        rel = path.relative_to(self.output_dir)
        self._result.artifacts.append(
            ReportArtifact(
                name=path.name,
                path=str(rel).replace("\\", "/"),
                artifact_path=artifact_path,
                kind=path.suffix.lstrip(".") or None,
                size_bytes=path.stat().st_size if path.exists() else None,
            )
        )


class CompositeAnalysisSink(AnalysisSink):
    name = "composite"

    def __init__(self, sinks: list[AnalysisSink]):
        self.sinks = sinks
        self._run_infos: dict[str, dict[str, Any]] = {}

    def start_run(self, *, run_name=None, description=None, tags=None) -> dict[str, Any]:
        for sink in self.sinks:
            self._run_infos[sink.name] = sink.start_run(
                run_name=run_name, description=description, tags=tags
            )
        return self._run_infos

    def end_run(self, status: str = "FINISHED") -> None:
        for sink in reversed(self.sinks):
            sink.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for sink in self.sinks:
            sink.log_metrics(metrics, step=step)

    def log_text(self, text: str, filename: str) -> None:
        for sink in self.sinks:
            sink.log_text(text, filename)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        for sink in self.sinks:
            sink.log_artifact(local_path, artifact_path=artifact_path)

    def result(self) -> ReportResult:
        result = ReportResult(sink_runs=dict(self._run_infos))
        for sink in self.sinks:
            child = sink.result()
            result.metrics.update(child.metrics)
            result.parameters.update(child.parameters)
            result.tags.update(child.tags)
            result.artifacts.extend(child.artifacts)
            result.sink_runs.update(child.sink_runs)
        return result


class SinkRun:
    def __init__(self, sink: AnalysisSink, *, run_name=None, description=None, tags=None):
        self.sink = sink
        self.kwargs = {"run_name": run_name, "description": description, "tags": tags}
        self.failed = False

    def __enter__(self):
        self.sink.start_run(**self.kwargs)
        return self.sink

    def __exit__(self, exc_type, exc, tb):
        self.sink.end_run(status="FAILED" if exc_type else "FINISHED")
        return False
