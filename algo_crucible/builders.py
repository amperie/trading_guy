from __future__ import annotations

import copy
from typing import Any

from utils.utils import instantiate_from_string


def build_candidate(resolved_cfg) -> Any:
    from algo_crucible.ids import hash16
    from algo_crucible.models import Candidate

    workload = resolved_cfg.workload
    al_class = workload["algorithm"]["algorithm"]
    pf_class = workload["portfolio"]["portfolio"]
    al_params = copy.deepcopy(workload.get("algorithm", {}).get("params", {}))
    pf_params = copy.deepcopy(workload.get("portfolio", {}).get("params", {}))
    candidate_hash = hash16({
        "algorithm_class": al_class,
        "portfolio_class": pf_class,
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
    al_class = workload["algorithm"]["algorithm"]
    pf_class = workload["portfolio"]["portfolio"]
    candidate_hash = hash16({
        "algorithm_class": al_class,
        "portfolio_class": pf_class,
        "algorithm_params": algorithm_params,
        "portfolio_params": portfolio_params,
        "workload_hash": resolved_cfg.workload_hash,
        "regime": algorithm_params.get("market_regime", {}),
    })
    return Candidate(f"candidate_{candidate_hash}", al_class, pf_class, algorithm_params, portfolio_params)


def build_components(workload: dict[str, Any], candidate):
    dp_cfg = copy.deepcopy(workload["data_provider"])
    om_cfg = copy.deepcopy(workload.get("order_manager", {}))
    dp_path = dp_cfg.pop("provider")
    om_path = om_cfg.pop("order_manager")

    dp = instantiate_from_string(dp_path, dp_cfg)
    om = instantiate_from_string(om_path, om_cfg)
    al = instantiate_from_string(
        candidate.algorithm_class,
        copy.deepcopy(candidate.algorithm_params),
        int(candidate.algorithm_params.get("history_length", 0)),
    )
    starting_cash = float(candidate.portfolio_params.get("cash", workload.get("fixed_assumptions", {}).get("starting_cash", 0.0)))
    keep_history = bool(candidate.portfolio_params.get("keep_history", True))
    pf = instantiate_from_string(
        candidate.portfolio_class,
        copy.deepcopy(candidate.portfolio_params),
        om,
        starting_cash,
        {},
        keep_history,
    )
    return dp, al, om, pf
