from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any

import yaml

from trading.config import ExperimentConfig, ExperimentService
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def apply_session_log_file(raw_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    run_name = getattr(args, "run_name", None) or raw_cfg.get("analysis", {}).get("run_name")
    if run_name:
        Logger().set_log_file(run_name)


def load_account_creds(account_name: str, path: str = "accounts.yaml") -> dict[str, str]:
    if not os.path.exists(path):
        logger.error(
            f"accounts.yaml not found at '{path}'. "
            "Copy accounts.yaml.example to accounts.yaml and fill in your credentials."
        )
        sys.exit(1)

    with open(path, "r") as handle:
        accounts = yaml.safe_load(handle) or {}

    if account_name not in accounts:
        logger.error(f"Account '{account_name}' not found in {path}. Available: {list(accounts.keys())}")
        sys.exit(1)

    entry = accounts[account_name]
    if not entry.get("api_key") or not entry.get("secret_key"):
        logger.error(f"Account '{account_name}' in {path} is missing api_key or secret_key.")
        sys.exit(1)
    return {"api_key": entry["api_key"], "secret_key": entry["secret_key"]}


def import_class(dotted_path: str):
    import importlib

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_component_param(cfg: dict[str, Any], section_name: str, legacy_selector: str, key: str, value: Any) -> None:
    if section_name not in cfg:
        return
    section = cfg[section_name]
    if key == legacy_selector and "implementation" in section:
        section["implementation"] = value
    elif "params" in section:
        section.setdefault("params", {})[key] = value
    else:
        section[key] = value


def _set_component_field(cfg: dict[str, Any], section_name: str, field: str, value: Any) -> None:
    if section_name not in cfg:
        return
    cfg[section_name][field] = value


def load_raw_config(config_path: str) -> dict[str, Any]:
    from utils.config_manager import ConfigManager

    root_cfg = ConfigManager().config
    with open(config_path, "r") as handle:
        profile = yaml.safe_load(handle) or {}
    return deep_merge(root_cfg, profile)


def apply_cli_overrides(raw_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(raw_cfg)

    if getattr(args, "symbol", None):
        _set_component_param(cfg, "portfolio", "portfolio", "symbol", args.symbol)
        if "alpaca" in cfg:
            cfg.setdefault("alpaca", {})["symbols_to_subscribe"] = [args.symbol]

    if getattr(args, "cash", None) is not None:
        _set_component_param(cfg, "portfolio", "portfolio", "cash", args.cash)

    if getattr(args, "algorithm", None):
        _set_component_param(cfg, "algorithm", "algorithm", "algorithm", args.algorithm)

    if getattr(args, "portfolio", None):
        _set_component_param(cfg, "portfolio", "portfolio", "portfolio", args.portfolio)

    if getattr(args, "algorithm_url", None):
        _set_component_field(cfg, "algorithm", "source_url", args.algorithm_url)

    if getattr(args, "portfolio_url", None):
        _set_component_field(cfg, "portfolio", "source_url", args.portfolio_url)

    if getattr(args, "data", None):
        _set_component_param(cfg, "data_provider", "provider", "path", args.data)

    if getattr(args, "no_mlflow", False):
        cfg.setdefault("analysis", {})["log_to_mlflow"] = False

    if getattr(args, "run_name", None):
        cfg.setdefault("analysis", {})["run_name"] = args.run_name

    if getattr(args, "alpaca_override_url", None):
        cfg.setdefault("alpaca", {})["override_url"] = args.alpaca_override_url

    if getattr(args, "session_id", None):
        cfg.setdefault("state_store", {})["session_id"] = args.session_id

    if getattr(args, "agg_period", None) is not None:
        cfg.setdefault("aggregation", {})["aggregation_period_minutes"] = args.agg_period
        cfg.setdefault("aggregation", {}).setdefault("enabled", True)

    return cfg


def fill_alpaca_creds(section: dict[str, Any], creds: dict[str, str]) -> None:
    section["api_key"] = creds.get("api_key", "")
    section["secret_key"] = creds.get("secret_key", "")


def flatten_config(cfg: dict[str, Any], prefix: str = "config") -> dict[str, Any]:
    from trading.config.service import _flatten_dict

    return _flatten_dict(cfg, prefix=prefix)


def resolve_alpaca_credentials(cfg: dict[str, Any], creds: dict[str, str]) -> dict[str, Any]:
    alpaca = cfg.setdefault("alpaca", {})
    fill_alpaca_creds(alpaca, creds)
    return cfg


def validate_session_id(cfg: dict[str, Any]) -> None:
    ss_cfg = cfg.get("state_store", {})
    if ss_cfg.get("enabled") and not ss_cfg.get("session_id"):
        logger.error(
            "state_store is enabled but session_id is not set. "
            "Pass --session-id <id> or set state_store.session_id explicitly."
        )
        sys.exit(1)


def adapt_live_config_to_mongo_backtest(raw_cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """
    Allow promoted live bundles to run through `backtest` unchanged.

    Promoted configs are intentionally written as `mode: live` bundles with an
    Alpaca order manager and no data_provider section. When a caller supplies a
    session_id, reinterpret that bundle as a backtest over the stored MongoDB
    ticks for that session.
    """
    cfg = copy.deepcopy(raw_cfg)
    ss_cfg = cfg.get("state_store", {})
    if not ss_cfg.get("session_id"):
        return cfg

    if not force and (cfg.get("mode") != "live" or cfg.get("data_provider")):
        return cfg

    cfg["mode"] = "backtest"
    cfg["order_manager"] = {
        "order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager",
        "market_hours_only": True,
    }
    cfg["data_provider"] = {
        "provider": "trading.data_providers.mongodb_data_provider.MongoDBDataProvider",
        "session_id": ss_cfg["session_id"],
        "connection_uri": ss_cfg.get("connection_uri"),
        "database": ss_cfg.get("database"),
    }
    return cfg


def build_experiment_config(raw_cfg: dict[str, Any]) -> ExperimentConfig:
    return ExperimentService.from_dict(raw_cfg)
