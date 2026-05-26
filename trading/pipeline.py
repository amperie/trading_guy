from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from trading.commands.common import flatten_config
from trading.launchers.mlflow_promote_launcher import PromotionBundle, promote_run
from utils.mlflow_client import MLflowClient

DEFAULT_PIPELINE_EXPERIMENT = "Trading Pipeline Bundle Registry"


def _bundle_registry_experiment_name(raw_cfg: dict[str, Any]) -> str:
    pipeline_cfg = raw_cfg.get("pipeline", {}) or {}
    experiments = pipeline_cfg.get("experiments", {}) or {}
    if experiments.get("bundle_registry"):
        return str(experiments["bundle_registry"])
    if pipeline_cfg.get("experiment_name"):
        return str(pipeline_cfg["experiment_name"])
    try:
        from utils.config_manager import ConfigManager

        global_pipeline_cfg = ConfigManager().get("pipeline") or {}
        global_experiments = global_pipeline_cfg.get("experiments", {}) or {}
        if global_experiments.get("bundle_registry"):
            return str(global_experiments["bundle_registry"])
        if global_pipeline_cfg.get("experiment_name"):
            return str(global_pipeline_cfg["experiment_name"])
    except Exception:
        pass
    return DEFAULT_PIPELINE_EXPERIMENT


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    actual: float | None
    threshold: float | None
    comparator: str


@dataclass(slots=True)
class GateReport:
    stage: str
    passed: bool
    checks: list[GateCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _metric_value(container: Any, name: str) -> float | None:
    if container is None:
        return None
    if isinstance(container, dict):
        value = container.get(name)
    else:
        value = getattr(container, name, None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _gate_limit(
    checks: list[GateCheck],
    name: str,
    actual: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if minimum is not None:
        checks.append(
            GateCheck(
                name=name,
                passed=actual is not None and actual >= minimum,
                actual=actual,
                threshold=float(minimum),
                comparator=">=",
            )
        )
    if maximum is not None:
        checks.append(
            GateCheck(
                name=name,
                passed=actual is not None and actual <= maximum,
                actual=actual,
                threshold=float(maximum),
                comparator="<=",
            )
        )


def evaluate_research_gates(
    raw_cfg: dict[str, Any],
    backtest_result: dict[str, Any] | None,
    hpo_result: dict[str, Any],
    walk_forward_result: dict[str, Any] | None,
) -> GateReport:
    gates = (((raw_cfg.get("pipeline") or {}).get("gates") or {}).get("research") or {})
    checks: list[GateCheck] = []
    backtest_metrics = ((backtest_result or {}).get("analysis") or {}).get("metrics")
    val_metrics = (hpo_result.get("val_results") or {}).get("metrics")
    wf_agg = (walk_forward_result or {}).get("aggregate") or {}

    _gate_limit(
        checks,
        "backtest_annualized_return",
        _metric_value(backtest_metrics, "annualized_return"),
        minimum=gates.get("min_backtest_annualized_return"),
    )
    _gate_limit(
        checks,
        "backtest_max_drawdown_pct",
        _metric_value(backtest_metrics, "max_drawdown_pct"),
        maximum=gates.get("max_backtest_max_drawdown_pct"),
    )
    _gate_limit(
        checks,
        "val_annualized_return",
        _metric_value(val_metrics, "annualized_return"),
        minimum=gates.get("min_val_annualized_return"),
    )
    _gate_limit(
        checks,
        "val_sharpe_ratio",
        _metric_value(val_metrics, "sharpe_ratio"),
        minimum=gates.get("min_val_sharpe_ratio"),
    )
    _gate_limit(
        checks,
        "val_max_drawdown_pct",
        _metric_value(val_metrics, "max_drawdown_pct"),
        maximum=gates.get("max_val_max_drawdown_pct"),
    )
    _gate_limit(
        checks,
        "val_total_trades",
        _metric_value(val_metrics, "total_trades"),
        minimum=gates.get("min_val_total_trades"),
    )
    _gate_limit(
        checks,
        "wf_annualized_return",
        _metric_value(wf_agg, "wf_annualized_return"),
        minimum=gates.get("min_wf_annualized_return"),
    )
    _gate_limit(
        checks,
        "wf_sharpe_ratio",
        _metric_value(wf_agg, "wf_sharpe_ratio"),
        minimum=gates.get("min_wf_sharpe_ratio"),
    )
    _gate_limit(
        checks,
        "wf_max_drawdown_pct",
        _metric_value(wf_agg, "wf_max_drawdown_pct"),
        maximum=gates.get("max_wf_max_drawdown_pct"),
    )
    _gate_limit(
        checks,
        "wf_total_trades",
        _metric_value(wf_agg, "wf_total_trades"),
        minimum=gates.get("min_wf_total_trades"),
    )
    return GateReport(stage="research", passed=all(check.passed for check in checks), checks=checks)


def evaluate_review_gates(raw_cfg: dict[str, Any], review_result: dict[str, Any]) -> GateReport:
    gates = (((raw_cfg.get("pipeline") or {}).get("gates") or {}).get("review") or {})
    checks: list[GateCheck] = []
    _gate_limit(
        checks,
        "alpaca_live_equity_drift_pct",
        _metric_value(review_result, "alpaca_live_equity_drift_pct"),
        maximum=gates.get("max_alpaca_live_equity_drift_pct"),
    )
    _gate_limit(
        checks,
        "mongo_live_equity_drift_pct",
        _metric_value(review_result, "mongo_live_equity_drift_pct"),
        maximum=gates.get("max_mongo_live_equity_drift_pct"),
    )
    return GateReport(stage="review", passed=all(check.passed for check in checks), checks=checks)


def build_session_id(stage: str) -> str:
    return f"{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _bundle_from_paths(config_path: Path, manifest_path: Path, *, source_run_url: str = "") -> PromotionBundle:
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PromotionBundle(
        source_run_id=str(manifest.get("source_run_id", "")),
        source_run_name=str(manifest.get("source_run_name", config_path.stem)),
        source_run_url=source_run_url or str(manifest.get("source_run_url", "")),
        config_path=str(config_path),
        manifest_path=str(manifest_path),
        promoted_dir=str(config_path.parent),
        algorithm_path=manifest.get("components", {}).get("algorithm", {}).get("source_path"),
        portfolio_path=manifest.get("components", {}).get("portfolio", {}).get("source_path"),
    )


def _rewrite_bundle_manifest(bundle: PromotionBundle) -> None:
    manifest_path = Path(bundle.manifest_path)
    if not manifest_path.is_file():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["config_path"] = Path(bundle.config_path).relative_to(Path.cwd()).as_posix()
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_bundle_paper_mode(config_path: str, paper: bool) -> None:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    order_manager = cfg.setdefault("order_manager", {})
    target = order_manager.setdefault("params", {}) if "implementation" in order_manager else order_manager
    target["paper"] = bool(paper)
    cfg.setdefault("alpaca", {})["paper"] = bool(paper)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def clone_local_bundle(config_path: str, name: str | None = None) -> PromotionBundle:
    source_cfg = Path(config_path).resolve()
    source_dir = source_cfg.parent
    bundle_name = name or f"{source_dir.name}_clone"
    target_dir = Path.cwd() / "trading" / "promoted" / bundle_name
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing promoted bundle directory: {target_dir}")
    shutil.copytree(source_dir, target_dir)
    target_cfg = target_dir / f"{bundle_name}.yaml"
    copied_cfg = target_dir / source_cfg.name
    copied_cfg.rename(target_cfg)
    bundle = _bundle_from_paths(target_cfg, target_dir / "promotion_manifest.json")
    _rewrite_bundle_manifest(bundle)
    return bundle


def materialize_bundle(
    source: str,
    *,
    name: str | None = None,
    paper: bool | None = None,
    tracking_uri: str | None = None,
) -> PromotionBundle:
    if source.startswith(("http://", "https://")) and "/runs/" in source:
        bundle = promote_run(source, tracking_uri=tracking_uri, name=name)
    else:
        bundle = clone_local_bundle(source, name=name)
    if paper is not None:
        set_bundle_paper_mode(bundle.config_path, paper)
    return bundle


def log_registered_bundle(
    raw_cfg: dict[str, Any],
    bundle: PromotionBundle,
    *,
    stage: str,
    status: str,
    source_run_url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline_cfg = raw_cfg.get("pipeline", {}) or {}
    tracking_uri = (raw_cfg.get("mlflow") or {}).get("tracking_uri")
    bundle_name = Path(bundle.promoted_dir).name
    client = MLflowClient(
        experiment_name=_bundle_registry_experiment_name(raw_cfg),
        tracking_uri=tracking_uri,
        artifact_location=pipeline_cfg.get("artifact_location"),
    )
    manifest = {
        "stage": stage,
        "status": status,
        "registered_at": datetime.utcnow().isoformat() + "Z",
        "bundle_name": bundle_name,
        "bundle_config_path": bundle.config_path,
        "bundle_manifest_path": bundle.manifest_path,
        "source_run_id": bundle.source_run_id,
        "source_run_name": bundle.source_run_name,
        "source_run_url": source_run_url or bundle.source_run_url,
        "metadata": copy.deepcopy(metadata or {}),
    }
    local_manifest = Path(bundle.promoted_dir) / f"pipeline_{stage}_manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"bundle-registry {stage} {bundle_name} {timestamp}"
    description = (
        f"Pipeline bundle registration run for '{bundle_name}'. "
        f"Stage={stage}, status={status}. This run stores promoted bundle "
        f"artifacts and launch metadata only. Performance metrics live in the "
        f"source research, backtest, walk-forward, or replay runs."
    )

    with client.start_run(
        run_name=run_name,
        description=description,
        tags={
            "pipeline.stage": stage,
            "pipeline.status": status,
            "pipeline.run_type": "bundle_registry",
            "pipeline.bundle_name": bundle_name,
            "pipeline.source_run_id": bundle.source_run_id,
            "pipeline.source_run_url": source_run_url or bundle.source_run_url,
        },
    ):
        run_info = {"run_id": client.run_id or "", "run_url": client.get_run_url()}
        params = {
            "bundle_name": bundle_name,
            "bundle_source_run_id": bundle.source_run_id,
            "bundle_source_run_name": bundle.source_run_name,
            "bundle_source_run_url": source_run_url or bundle.source_run_url,
            "bundle_config_filename": Path(bundle.config_path).name,
            "bundle_stage": stage,
            "bundle_status": status,
            "bundle_registry_run": True,
        }
        params.update(flatten_config({"pipeline_bundle": metadata or {}}, prefix="meta"))
        client.log_params(params)
        client.log_json(manifest, f"pipeline_{stage}_manifest.json")
        client.log_artifact(bundle.config_path, artifact_path="config")
        if Path(bundle.manifest_path).is_file():
            client.log_artifact(bundle.manifest_path, artifact_path="bundle")
        client.log_artifact(str(local_manifest), artifact_path="bundle")
        client.log_artifacts(bundle.promoted_dir, artifact_path="bundle")
        client.log_text(
            "\n".join(
                [
                    f"Local config: {bundle.config_path}",
                    f"MLflow run URL: {run_info['run_url']}",
                    f"Run directly from MLflow: python run.py live --config {run_info['run_url']} --account <account> --session-id <session-id>",
                    f"Run from local bundle: python run.py live --config {bundle.config_path} --account <account> --session-id <session-id>",
                ]
            ),
            "launch_instructions.txt",
        )
    manifest.update(run_info)
    manifest["local_stage_manifest"] = str(local_manifest)
    return manifest
