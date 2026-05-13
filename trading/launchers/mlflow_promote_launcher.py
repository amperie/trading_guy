from __future__ import annotations

import copy
import inspect
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trading.launchers.mlflow_hpo_launcher import SourceRunContext, load_source_run_context
from utils.logger import Logger

logger = Logger().get_logger(__name__)


@dataclass(slots=True)
class PromotionBundle:
    source_run_id: str
    source_run_name: str
    source_run_url: str
    config_path: str
    manifest_path: str
    promoted_dir: str
    algorithm_path: str | None
    portfolio_path: str | None


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "promoted_run"


def _bundle_name(source_context: SourceRunContext, requested_name: str | None = None) -> str:
    if requested_name:
        return _slugify(requested_name)
    return _slugify(f"{source_context.run_name}_{source_context.run_id[:8]}")


def _component_class_name(component: dict[str, Any], role: str) -> str:
    implementation = component.get("implementation") or component.get(role) or role.title()
    return component.get("class_name") or implementation.rsplit(".", 1)[-1]


def _resolve_local_component_source(component: dict[str, Any], role: str) -> Path | None:
    source_path = component.get("source_path")
    if source_path:
        path = Path(str(source_path)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path.resolve()

    implementation = component.get("implementation") or component.get(role)
    if not implementation:
        return None

    try:
        module_path, class_name = implementation.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
    except Exception:
        return None

    source_file = inspect.getsourcefile(cls)
    return Path(source_file).resolve() if source_file else None


def _copy_component_source(
    component: dict[str, Any],
    role: str,
    target_dir: Path,
) -> str | None:
    source_path = _resolve_local_component_source(component, role)
    if source_path is None:
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    source_name = source_path.name
    target_name = source_name
    if role == "portfolio" and (target_dir / target_name).exists():
        target_name = f"portfolio_{source_name}"
    if role == "algorithm" and (target_dir / target_name).exists():
        target_name = f"algorithm_{source_name}"

    destination = target_dir / target_name
    shutil.copy2(source_path, destination)
    return destination.relative_to(Path.cwd()).as_posix()


def _infer_symbols(cfg: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    alpaca_symbols = cfg.get("alpaca", {}).get("symbols_to_subscribe", [])
    for symbol in alpaca_symbols:
        if isinstance(symbol, str) and symbol and symbol not in symbols:
            symbols.append(symbol)

    portfolio = cfg.get("portfolio", {})
    params = portfolio.get("params", portfolio)
    if isinstance(params, dict):
        for key, value in params.items():
            if key == "symbol" and isinstance(value, str) and value and value not in symbols:
                symbols.append(value)
            if key.endswith("_symbol") and isinstance(value, str) and value and value not in symbols:
                symbols.append(value)
    return symbols


def _build_live_config(source_context: SourceRunContext, bundle_name: str, promoted_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(source_context.raw_config)
    cfg["mode"] = "live"
    cfg.pop("data_provider", None)
    cfg.setdefault("optimization", {})["enabled"] = False

    analysis_cfg = cfg.setdefault("analysis", {})
    analysis_cfg["enabled"] = False
    analysis_cfg["log_to_mlflow"] = False
    analysis_cfg.setdefault("run_name", f"{bundle_name}_live")
    analysis_cfg.setdefault(
        "description",
        f"Promoted live config from MLflow run {source_context.run_id} ({source_context.source_url})",
    )

    state_store_cfg = cfg.setdefault("state_store", {})
    state_store_cfg["enabled"] = True
    state_store_cfg["session_id"] = ""

    alpaca_cfg = cfg.setdefault("alpaca", {})
    alpaca_cfg["api_key"] = ""
    alpaca_cfg["secret_key"] = ""
    alpaca_cfg["symbols_to_subscribe"] = _infer_symbols(cfg)
    alpaca_cfg.setdefault("subscribe_to_bars", True)
    alpaca_cfg.setdefault("subscribe_to_quotes", False)
    alpaca_cfg.setdefault("subscribe_to_trades", False)
    warmup_cfg = alpaca_cfg.setdefault("warmup", {})
    warmup_cfg.setdefault("provider", "trading.data_providers.alpaca_data_provider.AlpacaDataProvider")

    cfg["order_manager"] = {
        "implementation": "trading.core.om.alpaca_om.AlpacaOrderManager",
        "params": {
            "paper": True,
            "time_in_force": "day",
        },
    }

    for role in ("algorithm", "portfolio"):
        component = cfg.get(role)
        if not isinstance(component, dict):
            continue
        promoted_rel_path = _copy_component_source(component, role, promoted_dir)
        if promoted_rel_path:
            component["source_path"] = promoted_rel_path
            component["class_name"] = _component_class_name(component, role)

    return cfg


def _manifest_data(source_context: SourceRunContext, config_path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_run_id": source_context.run_id,
        "source_run_name": source_context.run_name,
        "source_run_url": source_context.source_url,
        "source_tracking_uri": source_context.tracking_uri,
        "config_path": config_path,
        "launch_example": (
            f"python run.py live --config {config_path} --account <account> --session-id <session-id>"
        ),
        "components": {
            role: {
                "implementation": cfg.get(role, {}).get("implementation") or cfg.get(role, {}).get(role),
                "class_name": cfg.get(role, {}).get("class_name"),
                "source_path": cfg.get(role, {}).get("source_path"),
            }
            for role in ("algorithm", "portfolio")
        },
    }


def validate_promoted_config(cfg: dict[str, Any]) -> None:
    from trading.config.service import normalize_config_dict

    normalize_config_dict(copy.deepcopy(cfg))


def promote_run(
    run_url: str,
    tracking_uri: str | None = None,
    name: str | None = None,
) -> PromotionBundle:
    source_context = load_source_run_context(run_url, tracking_uri=tracking_uri)
    bundle_name = _bundle_name(source_context, requested_name=name)

    promoted_dir = Path.cwd() / "trading" / "promoted" / bundle_name
    promoted_dir.mkdir(parents=True, exist_ok=True)

    live_cfg = _build_live_config(source_context, bundle_name, promoted_dir)
    validate_promoted_config(live_cfg)

    config_path = promoted_dir / f"{bundle_name}.yaml"
    config_path.write_text(yaml.safe_dump(live_cfg, sort_keys=False), encoding="utf-8")

    manifest_path = promoted_dir / "promotion_manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_data(source_context, config_path.relative_to(Path.cwd()).as_posix(), live_cfg), indent=2),
        encoding="utf-8",
    )

    return PromotionBundle(
        source_run_id=source_context.run_id,
        source_run_name=source_context.run_name,
        source_run_url=source_context.source_url,
        config_path=str(config_path),
        manifest_path=str(manifest_path),
        promoted_dir=str(promoted_dir),
        algorithm_path=live_cfg.get("algorithm", {}).get("source_path"),
        portfolio_path=live_cfg.get("portfolio", {}).get("source_path"),
    )
