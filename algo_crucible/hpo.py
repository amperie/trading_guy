from __future__ import annotations

import copy
import importlib
import math
from typing import Any

from algo_crucible.builders import build_candidate_from_params
from utils.utils import apply_tunable_config, parse_search_space


def run_hpo_search(resolved_cfg) -> dict[str, Any]:
    from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters

    workload = resolved_cfg.workload
    hpo_cfg = _hpo_cfg(resolved_cfg)
    algorithm_param_keys = hpo_cfg.get("algorithm_param_keys", [])
    portfolio_param_keys = hpo_cfg.get("portfolio_param_keys", [])
    base_algorithm_config = copy.deepcopy(workload.get("algorithm", {}).get("params", {}))
    base_portfolio_config = copy.deepcopy(workload.get("portfolio", {}).get("params", {}))
    base_data_provider_config = copy.deepcopy(workload["data_provider"])
    base_data_provider_config.pop("provider", None)
    base_backtest_config = {
        "symbol": base_portfolio_config.get("symbol") or base_portfolio_config.get("upro_symbol", ""),
        "starting_cash": float(base_portfolio_config.get("cash", workload.get("fixed_assumptions", {}).get("starting_cash", 0.0))),
        "experiment_name": "Algo Crucible HPO",
        "run_name": resolved_cfg.crucible_run_id,
        "description": workload.get("workload", {}).get("description", ""),
    }
    best_config, trial_summaries = tune_backtest_hyperparameters(
        symbol=base_backtest_config["symbol"],
        algorithm_class=_import_class(workload["algorithm"]["algorithm"]),
        portfolio_class=_import_class(workload["portfolio"]["portfolio"]),
        data_provider_class=_import_class(workload["data_provider"]["provider"]),
        order_manager_class=_import_class(workload["order_manager"]["order_manager"]),
        base_algorithm_config=base_algorithm_config,
        base_portfolio_config=base_portfolio_config,
        base_data_provider_config=base_data_provider_config,
        base_backtest_config=base_backtest_config,
        search_space=parse_search_space(hpo_cfg.get("space") or hpo_cfg.get("search_space", {})),
        algorithm_param_keys=algorithm_param_keys,
        portfolio_param_keys=portfolio_param_keys,
        num_samples=int(hpo_cfg.get("num_samples", resolved_cfg.platform.get("hpo", {}).get("num_samples", 50))),
        max_concurrent_trials=int(hpo_cfg.get("max_concurrent_trials", resolved_cfg.platform.get("ray", {}).get("max_concurrent_trials", 8))),
        log_to_mlflow=bool(hpo_cfg.get("log_trials_to_mlflow", False)),
        log_ray_worker_output=bool(hpo_cfg.get("log_ray_worker_output", resolved_cfg.platform.get("ray", {}).get("log_worker_output", False))),
        return_trial_summaries=True,
    )
    return build_hpo_summary(
        resolved_cfg=resolved_cfg,
        best_config=best_config,
        trial_summaries=trial_summaries,
        algorithm_param_keys=algorithm_param_keys,
        portfolio_param_keys=portfolio_param_keys,
        base_algorithm_config=base_algorithm_config,
        base_portfolio_config=base_portfolio_config,
    )


def build_hpo_summary(
    *,
    resolved_cfg,
    best_config: dict[str, Any],
    trial_summaries: list[dict[str, Any]],
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
    base_algorithm_config: dict[str, Any],
    base_portfolio_config: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    candidates = []
    failed = []
    for idx, trial in enumerate(trial_summaries):
        status = trial.get("status", "complete")
        trial_config = trial.get("config", {})
        metric = trial.get("metric")
        if status != "complete" or metric is None or not math.isfinite(float(metric)):
            failed.append({
                "trial_id": trial.get("trial_id", f"trial_{idx:04d}"),
                "status": status,
                "failure_reason": trial.get("failure_reason", "missing_or_non_finite_metric"),
            })
            continue
        al_cfg = apply_tunable_config(base_algorithm_config, trial_config, algorithm_param_keys)
        pf_cfg = apply_tunable_config(base_portfolio_config, trial_config, portfolio_param_keys)
        candidate = build_candidate_from_params(resolved_cfg, al_cfg, pf_cfg)
        candidates.append(candidate)
        rows.append({
            "trial_id": trial.get("trial_id", f"trial_{idx:04d}"),
            "candidate_id": candidate.candidate_id,
            "metric": float(metric),
            "config": trial_config,
            "algorithm_params": al_cfg,
            "portfolio_params": pf_cfg,
            "status": "complete",
        })
    rows.sort(key=lambda row: row["metric"], reverse=True)
    return {
        "best_config": best_config,
        "trial_rows": rows,
        "failed_trials": failed,
        "candidates": candidates,
        "metrics": {
            "hpo.trials_total": len(trial_summaries),
            "hpo.trials_complete": len(rows),
            "hpo.trials_failed": len(failed),
            "hpo.best_metric": rows[0]["metric"] if rows else None,
            "hpo.median_metric": _median([row["metric"] for row in rows]),
            "hpo.candidates_created": len(candidates),
        },
    }


def _hpo_cfg(resolved_cfg) -> dict[str, Any]:
    workload_space = resolved_cfg.workload.get("search_space", {})
    platform_hpo = resolved_cfg.platform.get("hpo", {})
    return {**platform_hpo, **workload_space}


def _import_class(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)
