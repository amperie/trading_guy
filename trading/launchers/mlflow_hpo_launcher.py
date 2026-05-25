from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from trading.commands.hpo import _fill_hpo_data_provider_creds, run_hpo_from_raw_config, run_hpo_split_from_raw_config
from trading.commands.common import load_account_creds, load_raw_config
from utils.logger import Logger
from utils.mlflow_client import MLFLOW_AVAILABLE, MLflowClient

logger = Logger().get_logger(__name__)
ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("TRADING_GUY_ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS", "10"))

RUN_URL_PATTERNS = [
    re.compile(r"/runs/([A-Za-z0-9-]+)"),
    re.compile(r"runs/([A-Za-z0-9-]+)"),
]
SOURCE_RUN_URL_PATTERN = re.compile(r"\((https?://[^)\s]+/runs/[A-Za-z0-9-]+[^)\s]*)\)")
NON_TUNABLE_COMPONENT_KEYS = {
    "symbol",
    "spy_symbol",
    "upro_symbol",
    "spxu_symbol",
    "cash",
    "keep_history",
    "history_length",
}
RUNTIME_CONFIG_KEYS = {
    "mode",
    "algorithm",
    "portfolio",
    "order_manager",
    "data_provider",
    "alpaca",
    "aggregation",
    "analysis",
    "optimization",
    "walk_forward",
    "hpo",
    "state_store",
    "mlflow",
    "logging",
}


@dataclass(slots=True)
class SourceRunContext:
    run_id: str
    run_name: str
    tracking_uri: str
    source_url: str
    raw_config: dict[str, Any]
    config_source: str


def extract_run_id(run_url: str) -> str:
    for pattern in RUN_URL_PATTERNS:
        match = pattern.search(run_url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract MLflow run ID from URL: {run_url}")


def infer_tracking_uri(run_url: str) -> str:
    parts = urlsplit(run_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Could not infer tracking URI from URL: {run_url}")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def coerce_string_value(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if text.startswith("0") and text != "0" and not text.startswith("0."):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _assign_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = target
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value


def reconstruct_config_from_params(params: dict[str, str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in params.items():
        if not key.startswith("config."):
            continue
        _assign_nested_value(config, key[len("config."):], coerce_string_value(value))
    return config


def _normalize_generated_component(
    raw_cfg: dict[str, Any],
    section_name: str,
    metadata_key: str,
) -> None:
    section = raw_cfg.get(section_name)
    metadata = raw_cfg.get(metadata_key)
    if not isinstance(section, dict) or not isinstance(metadata, dict):
        return

    selector_key = "implementation" if "implementation" in section else section_name
    selector_value = section.get(selector_key)
    if selector_value != "__generated__":
        return

    class_name = metadata.get("class_name")
    source_url = metadata.get("source_url")
    source_path = metadata.get("script_path")
    if class_name:
        section[selector_key] = class_name
        section.setdefault("class_name", class_name)
    if source_url:
        section.setdefault("source_url", source_url)
    if source_path:
        section.setdefault("source_path", source_path)


def _normalized_path_string(raw_path: str) -> str:
    return raw_path.replace("\\", "/").strip()


def sanitize_source_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(raw_cfg)

    if isinstance(cfg.get("execution_config"), dict):
        cfg = copy.deepcopy(cfg["execution_config"])

    _normalize_generated_component(cfg, "algorithm", "generated_algorithm")
    _normalize_generated_component(cfg, "portfolio", "generated_portfolio")

    sanitized = {key: value for key, value in cfg.items() if key in RUNTIME_CONFIG_KEYS}
    return sanitized


def _list_artifacts_recursive(client, run_id: str, path: str = "") -> list[Any]:
    items = client.list_artifacts(run_id, path) if path else client.list_artifacts(run_id)
    results: list[Any] = []
    for item in items:
        results.append(item)
        if item.is_dir:
            results.extend(_list_artifacts_recursive(client, run_id, item.path))
    return results


def _find_component_artifact_path(client, run_id: str, component_section: dict[str, Any]) -> str | None:
    source_path = component_section.get("source_path")
    class_name = component_section.get("class_name") or component_section.get("implementation")
    basename = Path(_normalized_path_string(source_path)).name if source_path else ""
    candidate_names = {name for name in (basename, f"{class_name}.py") if name}

    for artifact in _list_artifacts_recursive(client, run_id):
        if artifact.is_dir:
            continue
        artifact_name = Path(_normalized_path_string(artifact.path)).name
        if artifact_name in candidate_names:
            return artifact.path
    return None


def _resolve_component_sources_from_artifacts(
    cfg: dict[str, Any],
    client,
    tracking_uri: str,
    run_id: str,
) -> dict[str, Any]:
    resolved = copy.deepcopy(cfg)
    for section_name in ("algorithm", "portfolio"):
        section = resolved.get(section_name)
        if not isinstance(section, dict):
            continue
        source_path = section.get("source_path")
        if source_path:
            normalized = _normalized_path_string(source_path)
            candidate = Path(normalized)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                section["source_path"] = normalized
                continue

            artifact_path = _find_component_artifact_path(client, run_id, section)
            if artifact_path:
                downloaded = _download_artifact(tracking_uri, run_id, artifact_path)
                section["source_path"] = str(Path(downloaded).resolve())
    return resolved


def _find_config_artifact_path(client, run_id: str) -> str | None:
    candidates: list[str] = []
    for root in ("config", ""):
        artifacts = client.list_artifacts(run_id, root) if root else client.list_artifacts(run_id)
        for artifact in artifacts:
            if artifact.is_dir:
                continue
            lower = artifact.path.lower()
            if lower.endswith(".yaml") or lower.endswith(".yml"):
                candidates.append(artifact.path)
    return candidates[0] if candidates else None


def _download_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> str:
    import mlflow.artifacts

    logger.info(f"Downloading MLflow artifact run_id={run_id} path={artifact_path}")

    def _download() -> str:
        return mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_path}",
            tracking_uri=tracking_uri,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_download)
        try:
            return future.result(timeout=ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                "Timed out downloading MLflow artifact "
                f"run_id={run_id} path={artifact_path} after {ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS}s. "
                "If the artifact store is slow or unreachable, retry with "
                "TRADING_GUY_ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS set higher."
            ) from exc


def _artifact_exists(client, run_id: str, artifact_path: str) -> bool:
    normalized = artifact_path.strip("/")
    parent = str(Path(normalized).parent).replace("\\", "/")
    if parent == ".":
        parent = ""
    target_name = Path(normalized).name
    try:
        artifacts = client.list_artifacts(run_id, parent) if parent else client.list_artifacts(run_id)
    except Exception:
        return False
    for artifact in artifacts:
        if artifact.is_dir:
            continue
        if Path(_normalized_path_string(artifact.path)).name == target_name:
            return True
    return False


def _load_json_artifact(client, tracking_uri: str, run_id: str, artifact_path: str) -> dict[str, Any] | None:
    if not _artifact_exists(client, run_id, artifact_path):
        return None
    try:
        local_path = _download_artifact(tracking_uri, run_id, artifact_path)
    except Exception:
        return None
    try:
        with open(local_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _merge_hpo_artifact_config(raw_cfg: dict[str, Any], client, tracking_uri: str, run_id: str) -> dict[str, Any]:
    merged = copy.deepcopy(raw_cfg)
    artifact_cfg = _load_json_artifact(client, tracking_uri, run_id, "config/hpo_config.json")
    if artifact_cfg:
        merged.setdefault("hpo", {}).update(artifact_cfg)
    return merged


def _component_source_missing(cfg: dict[str, Any], section_name: str) -> bool:
    section = cfg.get(section_name)
    if not isinstance(section, dict):
        return False
    return not bool(section.get("source_path") or section.get("source_url"))


def _component_source_unusable(cfg: dict[str, Any], section_name: str) -> bool:
    section = cfg.get(section_name)
    if not isinstance(section, dict):
        return False

    source_url = section.get("source_url")
    if source_url:
        return False

    source_path = section.get("source_path")
    if not source_path:
        return True

    candidate = Path(_normalized_path_string(str(source_path))).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return not candidate.exists()


def _extract_source_run_url_from_description(description: str | None) -> str | None:
    if not description:
        return None
    match = SOURCE_RUN_URL_PATTERN.search(description)
    if match:
        return match.group(1)
    return None


def _merge_missing_component_sources(
    raw_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(raw_cfg)
    for section_name in ("algorithm", "portfolio"):
        section = merged.get(section_name)
        fallback_section = fallback_cfg.get(section_name)
        if not isinstance(section, dict) or not isinstance(fallback_section, dict):
            continue
        if section.get("implementation") != fallback_section.get("implementation"):
            continue
        if _component_source_unusable(merged, section_name):
            if fallback_section.get("source_path"):
                section["source_path"] = fallback_section["source_path"]
            if fallback_section.get("source_url"):
                section["source_url"] = fallback_section["source_url"]
            if fallback_section.get("class_name") and not section.get("class_name"):
                section["class_name"] = fallback_section["class_name"]
    return merged


def load_source_run_context(
    run_url: str,
    tracking_uri: str | None = None,
    *,
    _visited_run_ids: set[str] | None = None,
) -> SourceRunContext:
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow is required for the MLflow HPO launcher. Install with: pip install mlflow")

    import mlflow
    from mlflow.tracking import MlflowClient as BaseMlflowClient

    run_id = extract_run_id(run_url)
    visited = set(_visited_run_ids or set())
    if run_id in visited:
        raise ValueError(f"Detected recursive MLflow source-run reference while loading run {run_id}")
    visited.add(run_id)
    tracking = tracking_uri or infer_tracking_uri(run_url)
    mlflow.set_tracking_uri(tracking)
    client = BaseMlflowClient(tracking_uri=tracking)
    run = client.get_run(run_id)
    run_name = run.data.tags.get("mlflow.runName", f"run-{run_id[:8]}")
    description = run.data.tags.get("mlflow.note.content")

    artifact_path = _find_config_artifact_path(client, run_id)
    if artifact_path:
        local_path = _download_artifact(tracking, run_id, artifact_path)
        raw_cfg = sanitize_source_config(load_raw_config(local_path))
        source = f"artifact:{artifact_path}"
    else:
        raw_cfg = sanitize_source_config(reconstruct_config_from_params(run.data.params))
        if not raw_cfg:
            raise ValueError(
                "Could not reconstruct config from the MLflow run. "
                "The run has no YAML config artifact and no logged config.* params."
            )
        source = "params"

    raw_cfg = _resolve_component_sources_from_artifacts(raw_cfg, client, tracking, run_id)
    raw_cfg = _merge_hpo_artifact_config(raw_cfg, client, tracking, run_id)

    if _component_source_missing(raw_cfg, "algorithm") or _component_source_missing(raw_cfg, "portfolio"):
        fallback_url = _extract_source_run_url_from_description(description)
        if fallback_url:
            try:
                fallback_context = load_source_run_context(
                    fallback_url,
                    tracking_uri=tracking_uri,
                    _visited_run_ids=visited,
                )
            except Exception:
                fallback_context = None
            if fallback_context is not None:
                raw_cfg = _merge_missing_component_sources(raw_cfg, fallback_context.raw_config)

    return SourceRunContext(
        run_id=run_id,
        run_name=run_name,
        tracking_uri=tracking,
        source_url=run_url,
        raw_config=raw_cfg,
        config_source=source,
    )


def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer if answer else (default or "")


def prepare_hpo_config_from_source(
    source_context: SourceRunContext,
    experiment_name: str,
    *,
    split_validation: bool = False,
) -> dict[str, Any]:
    raw_cfg = copy.deepcopy(source_context.raw_config)
    raw_cfg["mode"] = "hpo"

    analysis_cfg = raw_cfg.setdefault("analysis", {})
    analysis_cfg["log_to_mlflow"] = True
    analysis_cfg["experiment_name"] = experiment_name
    suffix = "hpo_split" if split_validation else "hpo"
    analysis_cfg.setdefault("run_name", f"{source_context.run_name}_{suffix}")
    analysis_cfg["description"] = (
        f"Recreated {'split ' if split_validation else ''}HPO from source MLflow run {source_context.run_id} "
        f"({source_context.source_url})"
    )

    mlflow_cfg = raw_cfg.setdefault("mlflow", {})
    mlflow_cfg["enabled"] = True
    mlflow_cfg["tracking_uri"] = source_context.tracking_uri

    raw_cfg.setdefault("hpo", {})
    return raw_cfg


def _editor_command(editor: str | None = None) -> list[str]:
    system = platform.system().lower()
    if editor:
        return shlex.split(editor, posix=system != "windows")
    for env_key in ("CODEX_EDITOR", "VISUAL", "EDITOR"):
        value = os.environ.get(env_key)
        if value:
            return shlex.split(value, posix=system != "windows")
    if system == "windows":
        return ["notepad.exe"]
    if shutil.which("vim"):
        return ["vim"]
    raise FileNotFoundError(
        "vim was not found. Set CODEX_EDITOR, VISUAL, or EDITOR, install vim, or pass --editor."
    )


def edit_config_dict(
    prepared_cfg: dict[str, Any],
    editor: str | None = None,
    *,
    filename: str = "recreated_config.yaml",
    label: str = "config",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trading_config_edit_") as tmpdir:
        config_path = Path(tmpdir) / filename
        config_path.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False), encoding="utf-8")

        cmd = _editor_command(editor) + [str(config_path)]
        logger.info(f"Opening {label} in editor: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        with open(config_path, "r", encoding="utf-8") as handle:
            edited = yaml.safe_load(handle) or {}
        return edited


def edit_hpo_config(prepared_cfg: dict[str, Any], editor: str | None = None) -> dict[str, Any]:
    return edit_config_dict(
        prepared_cfg,
        editor=editor,
        filename="recreated_hpo_config.yaml",
        label="HPO config",
    )


def persist_edited_config(
    source_context: SourceRunContext,
    edited_cfg: dict[str, Any],
    *,
    output_dir_name: str,
    filename_prefix: str,
) -> str:
    out_dir = Path.cwd() / "scratch" / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_context.run_name).strip("_") or "mlflow_hpo"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{filename_prefix}_{safe_name}_{source_context.run_id[:8]}_{timestamp}.yaml"
    path.write_text(yaml.safe_dump(edited_cfg, sort_keys=False), encoding="utf-8")
    return str(path)


def persist_edited_hpo_config(source_context: SourceRunContext, edited_cfg: dict[str, Any]) -> str:
    return persist_edited_config(
        source_context,
        edited_cfg,
        output_dir_name="generated_hpo_configs",
        filename_prefix="hpo",
    )


def log_hpo_launcher_summary(
    source_context: SourceRunContext,
    prepared_cfg: dict[str, Any],
    best_config: dict[str, Any],
    edited_config_path: str | None = None,
    *,
    split_validation: bool = False,
) -> None:
    analysis_cfg = prepared_cfg.get("analysis", {})
    mlflow_cfg = prepared_cfg.get("mlflow", {})
    experiment_name = analysis_cfg.get("experiment_name", "HPO")
    tracking_uri = mlflow_cfg.get("tracking_uri")
    run_name = f"{analysis_cfg.get('run_name', source_context.run_name)}_launcher_summary"
    launcher_kind = "split_hpo" if split_validation else "hpo"

    client = MLflowClient(experiment_name=experiment_name, tracking_uri=tracking_uri)
    with client.start_run(
        run_name=run_name,
        description=f"{launcher_kind} launcher summary for source run {source_context.run_id}",
        tags={
            "source_run_id": source_context.run_id,
            "source_run_url": source_context.source_url,
            "launcher_kind": launcher_kind,
        },
    ):
        client.log_params({
            "source_run_id": source_context.run_id,
            "source_run_name": source_context.run_name,
            "config_source": source_context.config_source,
            "num_samples": prepared_cfg.get("hpo", {}).get("num_samples"),
            "max_concurrent_trials": prepared_cfg.get("hpo", {}).get("max_concurrent_trials"),
            "validation_period_days": prepared_cfg.get("hpo", {}).get("validation_period_days"),
        })
        client.log_json(prepared_cfg.get("hpo", {}), "hpo_launcher_config.json")
        client.log_json(best_config, "best_config.json")
        if edited_config_path and os.path.isfile(edited_config_path):
            client.log_artifact(edited_config_path, artifact_path="config")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "recreated_hpo_config.yaml"
                config_path.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False), encoding="utf-8")
                client.log_artifact(str(config_path), artifact_path="config")


def prompt_for_hpo_launch(source_context: SourceRunContext, *, split_validation: bool = False) -> str:
    default_experiment_name = "hpo_split_from_mlflow" if split_validation else "hpo_from_mlflow"
    return _prompt("MLflow experiment name for the recreated HPO", default_experiment_name)


def run_launcher(
    run_url: str,
    account: str,
    tracking_uri: str | None = None,
    editor: str | None = None,
    *,
    split_validation: bool = False,
) -> dict[str, Any]:
    source_context = load_source_run_context(run_url, tracking_uri=tracking_uri)
    experiment_name = prompt_for_hpo_launch(source_context, split_validation=split_validation)

    prepared_cfg = prepare_hpo_config_from_source(
        source_context=source_context,
        experiment_name=experiment_name,
        split_validation=split_validation,
    )

    edited_cfg = edit_hpo_config(prepared_cfg, editor=editor)
    edited_cfg = sanitize_source_config(edited_cfg) if "execution_config" in edited_cfg else edited_cfg
    edited_config_path = persist_edited_hpo_config(source_context, edited_cfg)

    creds = load_account_creds(account)
    _fill_hpo_data_provider_creds(edited_cfg, creds)
    final_hpo_cfg = edited_cfg.get("hpo", {})
    if not final_hpo_cfg.get("search_space"):
        raise ValueError(
            "The saved HPO config has an empty hpo.search_space. "
            "Add at least one tunable parameter in the editor before running."
        )
    if split_validation:
        best_config = run_hpo_split_from_raw_config(
            edited_cfg,
            config_artifact_path=edited_config_path,
            num_samples_override=final_hpo_cfg.get("num_samples"),
            max_concurrent_override=final_hpo_cfg.get("max_concurrent_trials"),
            validation_period_days_override=final_hpo_cfg.get("validation_period_days"),
        )
    else:
        best_config = run_hpo_from_raw_config(
            edited_cfg,
            config_artifact_path=edited_config_path,
            num_samples_override=final_hpo_cfg.get("num_samples"),
            max_concurrent_override=final_hpo_cfg.get("max_concurrent_trials"),
        )
    log_hpo_launcher_summary(
        source_context,
        edited_cfg,
        best_config,
        edited_config_path=edited_config_path,
        split_validation=split_validation,
    )
    return best_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch an HPO run from an existing MLflow backtest run.")
    parser.add_argument("--run-url", required=True, help="MLflow run URL to recreate")
    parser.add_argument("--account", required=True, help="Account name from accounts.yaml")
    parser.add_argument("--tracking-uri", help="Optional MLflow tracking URI override")
    parser.add_argument("--editor", help="Editor executable to open the generated HPO config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    best_config = run_launcher(args.run_url, args.account, tracking_uri=args.tracking_uri, editor=args.editor)
    logger.info("Recreated HPO complete. Best config:")
    for key, value in best_config.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
