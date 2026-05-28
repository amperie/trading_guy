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
from utils.config_manager import ConfigManager
from utils.logger import Logger

logger = Logger().get_logger(__name__)
REFERENCE_LIVE_CONFIG_PATH = Path("configs") / "example_live_spy_trend_macd.yaml"


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


def _load_reference_live_config() -> dict[str, Any]:
    with open(REFERENCE_LIVE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _legacy_component_section(component: dict[str, Any], role: str) -> dict[str, Any]:
    selector_key = {
        "algorithm": "algorithm",
        "portfolio": "portfolio",
        "order_manager": "order_manager",
    }[role]
    implementation = component.get("implementation") or component.get(role)
    section = {selector_key: implementation}
    params = component.get("params", {})
    if isinstance(params, dict):
        section.update(copy.deepcopy(params))
    if component.get("source_path"):
        section["source_path"] = component["source_path"]
    if component.get("class_name"):
        section["class_name"] = component["class_name"]
    return section


def _minimal_state_store_section(source_cfg: dict[str, Any]) -> dict[str, Any]:
    source_state = source_cfg.get("state_store", {})
    section = {"enabled": True, "session_id": ""}
    database = ConfigManager().get("state_store.default_live_database") or ConfigManager().get("state_store.database")
    if database:
        section["database"] = database
    if source_state.get("connection_uri"):
        section["connection_uri"] = source_state["connection_uri"]
    return section


def _minimal_aggregation_section(source_cfg: dict[str, Any]) -> dict[str, Any] | None:
    aggregation = source_cfg.get("aggregation", {})
    if not aggregation.get("enabled", False):
        return None
    section = {"enabled": True}
    for key in (
        "aggregation_period_minutes",
        "aggregation_start_minutes",
        "use_market_open",
        "market_open_hour",
        "market_open_minute",
        "batch_symbols",
        "expected_symbols",
        "batch_timeout_seconds",
    ):
        if key in aggregation:
            section[key] = copy.deepcopy(aggregation[key])
    return section


def _minimal_alpaca_section(source_cfg: dict[str, Any], symbols: list[str], reference_cfg: dict[str, Any]) -> dict[str, Any]:
    source_alpaca = source_cfg.get("alpaca", {})
    reference_alpaca = reference_cfg.get("alpaca", {})

    section = {
        "api_key": "",
        "secret_key": "",
        "symbols_to_subscribe": symbols,
        "subscribe_to_bars": source_alpaca.get(
            "subscribe_to_bars",
            reference_alpaca.get("subscribe_to_bars", True),
        ),
        "subscribe_to_quotes": source_alpaca.get(
            "subscribe_to_quotes",
            reference_alpaca.get("subscribe_to_quotes", False),
        ),
        "subscribe_to_trades": source_alpaca.get(
            "subscribe_to_trades",
            reference_alpaca.get("subscribe_to_trades", False),
        ),
    }
    if source_alpaca.get("override_url"):
        section["override_url"] = source_alpaca["override_url"]

    warmup_provider = source_alpaca.get("warmup", {}).get(
        "provider",
        reference_alpaca.get("warmup", {}).get(
            "provider",
            "trading.data_providers.alpaca_data_provider.AlpacaDataProvider",
        ),
    )
    section["warmup"] = {
        "provider": warmup_provider,
        "symbols": symbols,
    }

    timeframe = source_alpaca.get("warmup", {}).get("timeframe")
    if timeframe:
        section["warmup"]["timeframe"] = timeframe

    return section


def _build_live_config(source_context: SourceRunContext, bundle_name: str, promoted_dir: Path) -> dict[str, Any]:
    source_cfg = copy.deepcopy(source_context.raw_config)
    reference_cfg = _load_reference_live_config()
    symbols = _infer_symbols(source_cfg)

    cfg = {"mode": "live"}

    order_manager_cfg = source_cfg.get("order_manager", {})
    reference_order_manager = reference_cfg.get("order_manager", {})

    source_cfg["order_manager"] = {
        "implementation": "trading.core.om.alpaca_om.AlpacaOrderManager",
        "params": {
            "paper": order_manager_cfg.get("params", {}).get(
                "paper",
                reference_order_manager.get("paper", True),
            ),
            "time_in_force": order_manager_cfg.get("params", {}).get(
                "time_in_force",
                reference_order_manager.get("time_in_force", "day"),
            ),
        },
    }

    for role in ("algorithm", "portfolio"):
        component = source_cfg.get(role)
        if not isinstance(component, dict):
            continue
        promoted_rel_path = _copy_component_source(component, role, promoted_dir)
        if promoted_rel_path:
            component["source_path"] = promoted_rel_path
            component["class_name"] = _component_class_name(component, role)

    cfg["algorithm"] = _legacy_component_section(source_cfg["algorithm"], "algorithm")
    cfg["portfolio"] = _legacy_component_section(source_cfg["portfolio"], "portfolio")
    cfg["order_manager"] = _legacy_component_section(source_cfg["order_manager"], "order_manager")
    cfg["alpaca"] = _minimal_alpaca_section(source_cfg, symbols, reference_cfg)
    cfg["analysis"] = {
        "enabled": True,
        "log_to_mlflow": True,
    }
    cfg["state_store"] = _minimal_state_store_section(source_cfg)

    aggregation_section = _minimal_aggregation_section(source_cfg)
    if aggregation_section:
        cfg["aggregation"] = aggregation_section

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
