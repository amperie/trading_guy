from __future__ import annotations

import copy
import importlib
import inspect
import os
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

from trading.config.service import normalize_config_dict
from trading.commands.common import flatten_config
from trading.reporting import ExperimentReporter

SENSITIVE_KEYS = {"api_key", "secret_key", "password", "token"}


def _artifact_staging_root() -> Path:
    candidates = [
        Path(os.environ.get("TRADING_GUY_ARTIFACT_TMP", "E:/tmp")) / "trading_guy_codex_artifacts",
        Path.cwd() / ".tmp" / "codex_artifacts",
        Path.cwd(),
    ]
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    raise OSError("Could not create an artifact staging directory")


def get_git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return {}

    return {
        "git.commit": commit,
        "git.remote_url": remote_url,
    }


def _redact_sensitive(value):
    if isinstance(value, dict):
        return {
            key: ("" if key in SENSITIVE_KEYS else _redact_sensitive(inner))
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _stage_runtime_config_artifact(cfg: dict) -> str:
    tmpdir = _artifact_staging_root() / f"runtime_config_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    runtime_path = tmpdir / "runtime_config.yaml"
    runtime_path.write_text(
        yaml.safe_dump(_redact_sensitive(copy.deepcopy(cfg)), sort_keys=False),
        encoding="utf-8",
    )
    return str(runtime_path)


def _resolve_component_source_file(component: dict, role: str) -> tuple[Path | None, str | None]:
    implementation = component.get("implementation") or component.get(role)
    if not implementation:
        return None, None

    class_name = component.get("class_name") or implementation.rsplit(".", 1)[-1]
    source_path = component.get("source_path")
    if source_path:
        path = Path(str(source_path)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path.resolve(), None

    source_url = component.get("source_url")
    if source_url:
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported component source_url for {role}: {source_url}")
        tmpdir = _artifact_staging_root() / f"{role}_source_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        artifact_name = Path(parsed.path).name or f"{class_name}.py"
        target = tmpdir / artifact_name
        with urlopen(source_url, timeout=30) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            target.write_text(response.read().decode(encoding), encoding="utf-8")
        return target, None

    module_path, resolved_class_name = implementation.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, resolved_class_name)
    source_file = inspect.getsourcefile(cls)
    if not source_file:
        return None, f"{role} source file could not be resolved for {implementation}"
    return Path(source_file).resolve(), None


def _collect_config_artifact_paths(cfg: dict, config_path: str | None = None) -> list[str]:
    artifact_paths: list[str] = []
    if config_path:
        artifact_paths.append(config_path)

    artifact_paths.append(_stage_runtime_config_artifact(cfg))

    try:
        normalized = normalize_config_dict(copy.deepcopy(cfg))
    except Exception:
        return artifact_paths

    for role in ("algorithm", "portfolio"):
        component = getattr(normalized, role).model_dump(exclude_none=True)
        local_source, _ = _resolve_component_source_file(component, role)
        if local_source and local_source.is_file():
            artifact_paths.append(str(local_source))
    return artifact_paths


def run_analysis(cfg: dict, pf, om, config_path: str | None = None):
    analysis_cfg = cfg.get("analysis", {})
    if not analysis_cfg.get("enabled", False):
        return None

    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

    parameters = dict(analysis_cfg.get("parameters") or {})
    parameters.update(flatten_config(cfg))

    analyzer = PortfolioAnalyzer(pf, om)
    report, summary = ExperimentReporter.build_single_report(
        analyzer=analyzer,
        experiment_name=analysis_cfg.get("experiment_name"),
        run_name=analysis_cfg.get("run_name"),
        description=analysis_cfg.get("description"),
        tags=get_git_info() or None,
        benchmark_paths=analysis_cfg.get("benchmarks") or None,
        parameters=parameters,
        config_artifact_paths=_collect_config_artifact_paths(cfg, config_path=config_path),
    )
    ExperimentReporter.show_summary(summary)
    if analysis_cfg.get("log_to_mlflow", True):
        ExperimentReporter.log_to_mlflow(report)
    return ExperimentReporter.summary_to_legacy_dict(summary)
