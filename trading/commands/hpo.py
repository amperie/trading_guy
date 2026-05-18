from __future__ import annotations

import argparse
from typing import Any

from trading.commands.analysis import get_git_info
from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    flatten_config,
    load_account_creds,
    load_raw_config,
)
from trading.config.component_loader import import_component_class
from utils.logger import Logger
from utils.utils import parse_search_space

logger = Logger().get_logger(__name__)


def _fill_hpo_data_provider_creds(raw_cfg: dict[str, Any], creds: dict[str, str]) -> None:
    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        target = dp_section.setdefault("params", {}) if "implementation" in dp_section else dp_section
        target["api_key"] = creds["api_key"]
        target["secret_key"] = creds["secret_key"]


def run_hpo_from_raw_config(
    raw_cfg: dict[str, Any],
    config_artifact_path: str | None = None,
    num_samples_override: int | None = None,
    max_concurrent_override: int | None = None,
) -> dict[str, Any]:
    hpo_cfg = raw_cfg.setdefault("hpo", {})
    if num_samples_override is not None:
        hpo_cfg["num_samples"] = num_samples_override
    if max_concurrent_override is not None:
        hpo_cfg["max_concurrent_trials"] = max_concurrent_override

    experiment = build_experiment_config(raw_cfg)
    config_dict = experiment.model_dump(exclude_none=True)

    logger.info("Starting HPO")

    if experiment.data_provider is None:
        raise ValueError("HPO requires a data_provider section.")

    al_class = import_component_class(experiment.algorithm)
    pf_class = import_component_class(experiment.portfolio)
    dp_class = import_component_class(experiment.data_provider)
    om_class = import_component_class(experiment.order_manager)

    base_al_cfg = dict(experiment.algorithm.params)
    history_length = base_al_cfg.pop("history_length", 0)
    if history_length:
        base_al_cfg["history_length"] = history_length
    base_pf_cfg = dict(experiment.portfolio.params)
    starting_cash = float(base_pf_cfg.get("cash", 10000.0))
    base_pf_cfg = {k: v for k, v in base_pf_cfg.items() if k not in ("cash", "keep_history")}
    base_dp_cfg = dict(experiment.data_provider.params)

    analysis_cfg = config_dict.get("analysis", {})
    base_backtest_cfg = {
        "starting_cash": starting_cash,
        "experiment_name": analysis_cfg.get("experiment_name", "HPO"),
        "run_name": analysis_cfg.get("run_name", "HPO_Run"),
        "description": analysis_cfg.get("description", ""),
        "symbol": base_pf_cfg.get("symbol") or base_pf_cfg.get("upro_symbol", ""),
        "config_artifact_path": config_artifact_path,
        "tracking_uri": config_dict.get("mlflow", {}).get("tracking_uri"),
        "git_tags": get_git_info(),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
    }
    base_backtest_cfg.update(flatten_config(config_dict))

    from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters

    best_config = tune_backtest_hyperparameters(
        symbol=base_backtest_cfg["symbol"],
        algorithm_class=al_class,
        portfolio_class=pf_class,
        data_provider_class=dp_class,
        order_manager_class=om_class,
        base_algorithm_config=base_al_cfg,
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=base_dp_cfg,
        base_backtest_config=base_backtest_cfg,
        search_space=parse_search_space(hpo_cfg.get("search_space", {})),
        algorithm_param_keys=hpo_cfg.get("algorithm_param_keys", []),
        portfolio_param_keys=hpo_cfg.get("portfolio_param_keys", []),
        num_samples=hpo_cfg.get("num_samples", 50),
        max_concurrent_trials=hpo_cfg.get("max_concurrent_trials", 8),
        log_to_mlflow=hpo_cfg.get("log_trials_to_mlflow", analysis_cfg.get("log_to_mlflow", True)),
        log_ray_worker_output=hpo_cfg.get("log_ray_worker_output", True),
    )
    return best_config


def cmd_hpo(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    creds = load_account_creds(args.account)
    _fill_hpo_data_provider_creds(raw_cfg, creds)

    logger.info(f"Starting HPO with profile: {args.config}")
    best_config = run_hpo_from_raw_config(
        raw_cfg,
        config_artifact_path=args.config,
        num_samples_override=getattr(args, "num_samples", None),
        max_concurrent_override=getattr(args, "max_concurrent_trials", None),
    )

    logger.info("HPO complete. Best config:")
    for key, val in best_config.items():
        logger.info(f"  {key}: {val}")
