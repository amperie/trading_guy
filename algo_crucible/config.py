from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from algo_crucible.ids import hash16


@dataclass(frozen=True)
class ResolvedCrucibleConfig:
    platform: dict[str, Any]
    workload: dict[str, Any]
    resolved: dict[str, Any]
    platform_hash: str
    workload_hash: str
    resolved_config_hash: str
    run_name: str
    crucible_run_id: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_configs(platform_path: str | Path, workload_path: str | Path) -> ResolvedCrucibleConfig:
    platform = load_yaml(platform_path)
    workload = load_yaml(workload_path)
    return resolve_config_dicts(platform, workload)


def resolve_config_dicts(platform: dict[str, Any], workload: dict[str, Any]) -> ResolvedCrucibleConfig:
    platform = copy.deepcopy(platform)
    workload = copy.deepcopy(workload)
    _normalize_evaluation_symbols(workload)
    run_name = (
        workload.get("workload", {}).get("run_name")
        or platform.get("crucible", {}).get("run_name")
        or workload.get("workload", {}).get("name")
    )
    if not run_name:
        raise ValueError("workload.run_name or crucible.run_name is required")

    resolved = {
        "platform": platform,
        "workload": workload,
    }
    platform_hash = hash16(platform)
    workload_hash = hash16(workload)
    resolved_hash = hash16(resolved)
    crucible_run_id = f"{run_name}_{resolved_hash}"
    resolved["identity"] = {
        "run_name": run_name,
        "platform_config_hash": platform_hash,
        "workload_config_hash": workload_hash,
        "resolved_config_hash": resolved_hash,
        "crucible_run_id": crucible_run_id,
    }
    return ResolvedCrucibleConfig(
        platform=platform,
        workload=workload,
        resolved=resolved,
        platform_hash=platform_hash,
        workload_hash=workload_hash,
        resolved_config_hash=resolved_hash,
        run_name=run_name,
        crucible_run_id=crucible_run_id,
    )


def _normalize_evaluation_symbols(workload: dict[str, Any]) -> None:
    algorithm = workload.get("algorithm", {})
    symbols = _symbols(
        algorithm.get("evaluation_symbols")
        or algorithm.get("required_symbols")
        or algorithm.get("input_symbols")
    )
    if not symbols:
        raise ValueError("algorithm.evaluation_symbols is required")
    algorithm["evaluation_symbols"] = symbols
    workload.setdefault("fixed_assumptions", {})["evaluation_symbols"] = symbols
    data_symbols = _symbols(
        workload.get("data_provider", {}).get("symbols")
        or workload.get("data_provider", {}).get("symbols_to_subscribe")
    )
    missing = sorted(set(symbols) - set(data_symbols)) if data_symbols else []
    if missing:
        raise ValueError(f"data_provider.symbols missing algorithm evaluation symbols: {missing}")


def _symbols(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    else:
        raw = list(value)
    return sorted({str(symbol).strip().upper() for symbol in raw if str(symbol).strip()})
