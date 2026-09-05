from __future__ import annotations

import copy
from typing import Any

from trading.config.component_loader import import_component_class
from trading.config.models import ComponentConfig


_LEGACY_KEYS = {
    "algorithm": "algorithm",
    "portfolio": "portfolio",
    "data_provider": "provider",
    "order_manager": "order_manager",
}


def build_candidate(resolved_cfg) -> Any:
    from algo_crucible.ids import hash16
    from algo_crucible.models import Candidate

    workload = resolved_cfg.workload
    al_class = component_implementation(workload["algorithm"], "algorithm")
    pf_class = component_implementation(workload["portfolio"], "portfolio")
    al_params = copy.deepcopy(workload.get("algorithm", {}).get("params", {}))
    pf_params = copy.deepcopy(workload.get("portfolio", {}).get("params", {}))
    candidate_hash = hash16({
        "algorithm": component_identity(workload["algorithm"], "algorithm"),
        "portfolio": component_identity(workload["portfolio"], "portfolio"),
        "algorithm_params": al_params,
        "portfolio_params": pf_params,
        "workload_hash": resolved_cfg.workload_hash,
        "regime": al_params.get("market_regime", {}),
    })
    return Candidate(f"candidate_{candidate_hash}", al_class, pf_class, al_params, pf_params)


def build_candidate_from_params(resolved_cfg, algorithm_params: dict[str, Any], portfolio_params: dict[str, Any]) -> Any:
    from algo_crucible.ids import hash16
    from algo_crucible.models import Candidate

    workload = resolved_cfg.workload
    al_class = component_implementation(workload["algorithm"], "algorithm")
    pf_class = component_implementation(workload["portfolio"], "portfolio")
    candidate_hash = hash16({
        "algorithm": component_identity(workload["algorithm"], "algorithm"),
        "portfolio": component_identity(workload["portfolio"], "portfolio"),
        "algorithm_params": algorithm_params,
        "portfolio_params": portfolio_params,
        "workload_hash": resolved_cfg.workload_hash,
        "regime": algorithm_params.get("market_regime", {}),
    })
    return Candidate(f"candidate_{candidate_hash}", al_class, pf_class, algorithm_params, portfolio_params)


def build_components(workload: dict[str, Any], candidate):
    dp_cfg = component_params(workload["data_provider"], "data_provider")
    om_cfg = component_params(workload.get("order_manager", {}), "order_manager")
    dp_cls = component_class(workload["data_provider"], "data_provider")
    om_cls = component_class(workload.get("order_manager", {}), "order_manager")
    al_cls = component_class(workload["algorithm"], "algorithm", candidate.algorithm_class)
    pf_cls = component_class(workload["portfolio"], "portfolio", candidate.portfolio_class)

    dp = dp_cls(dp_cfg)
    om = om_cls(om_cfg)
    al = al_cls(
        copy.deepcopy(candidate.algorithm_params),
        int(candidate.algorithm_params.get("history_length", 0)),
    )
    starting_cash = float(candidate.portfolio_params.get("cash", workload.get("fixed_assumptions", {}).get("starting_cash", 0.0)))
    keep_history = bool(candidate.portfolio_params.get("keep_history", True))
    pf = pf_cls(
        copy.deepcopy(candidate.portfolio_params),
        om,
        starting_cash,
        {},
        keep_history,
    )
    return dp, al, om, pf


def component_implementation(section: dict[str, Any], role: str) -> str:
    legacy_key = _LEGACY_KEYS[role]
    value = section.get(legacy_key) or section.get("implementation")
    if not value:
        raise KeyError(f"{role}.{legacy_key} or {role}.implementation is required")
    return str(value)


def component_identity(section: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "implementation": component_implementation(section, role),
        "source_path": section.get("source_path"),
        "source_url": section.get("source_url"),
        "class_name": section.get("class_name"),
    }


def component_params(section: dict[str, Any], role: str) -> dict[str, Any]:
    reserved = {
        _LEGACY_KEYS[role],
        "implementation",
        "source_path",
        "source_url",
        "class_name",
        "params",
    }
    params = copy.deepcopy(section.get("params") or {})
    for key, value in section.items():
        if key not in reserved:
            params[key] = copy.deepcopy(value)
    return params


def component_class(section: dict[str, Any], role: str, implementation: str | None = None):
    return import_component_class(ComponentConfig(
        implementation=implementation or component_implementation(section, role),
        source_path=section.get("source_path"),
        source_url=section.get("source_url"),
        class_name=section.get("class_name"),
        params={},
    ))
