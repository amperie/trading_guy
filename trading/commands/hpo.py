from __future__ import annotations

import argparse

from trading.commands.analysis import get_git_info
from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    flatten_config,
    import_class,
    load_account_creds,
    load_raw_config,
)
from utils.logger import Logger
from utils.utils import parse_search_space

logger = Logger().get_logger(__name__)


def _selector(section: dict, legacy_key: str) -> str:
    return section.get("implementation") or section[legacy_key]


def _params(section: dict, legacy_key: str) -> dict:
    if "params" in section:
        return dict(section.get("params") or {})
    return {k: v for k, v in section.items() if k != legacy_key}


def cmd_hpo(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    creds = load_account_creds(args.account)

    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        target = dp_section.setdefault("params", {}) if "implementation" in dp_section else dp_section
        target["api_key"] = creds["api_key"]
        target["secret_key"] = creds["secret_key"]

    hpo_cfg = raw_cfg.setdefault("hpo", {})
    if getattr(args, "num_samples", None) is not None:
        hpo_cfg["num_samples"] = args.num_samples
    if getattr(args, "max_concurrent_trials", None) is not None:
        hpo_cfg["max_concurrent_trials"] = args.max_concurrent_trials

    logger.info(f"Starting HPO with profile: {args.config}")

    al_section = raw_cfg["algorithm"]
    pf_section = raw_cfg["portfolio"]
    dp_section = raw_cfg["data_provider"]
    om_section = raw_cfg["order_manager"]

    al_class = import_class(_selector(al_section, "algorithm"))
    pf_class = import_class(_selector(pf_section, "portfolio"))
    dp_class = import_class(_selector(dp_section, "provider"))
    om_class = import_class(_selector(om_section, "order_manager"))

    base_al_cfg = _params(al_section, "algorithm")
    history_length = base_al_cfg.pop("history_length", 0)
    if history_length:
        base_al_cfg["history_length"] = history_length
    base_pf_cfg = _params(pf_section, "portfolio")
    starting_cash = float(base_pf_cfg.get("cash", 10000.0))
    base_pf_cfg = {k: v for k, v in base_pf_cfg.items() if k not in ("cash", "keep_history")}
    base_dp_cfg = _params(dp_section, "provider")

    analysis_cfg = raw_cfg.get("analysis", {})
    base_backtest_cfg = {
        "starting_cash": starting_cash,
        "experiment_name": analysis_cfg.get("experiment_name", "HPO"),
        "run_name": analysis_cfg.get("run_name", "HPO_Run"),
        "description": analysis_cfg.get("description", ""),
        "symbol": base_pf_cfg.get("symbol") or base_pf_cfg.get("upro_symbol", ""),
        "config_artifact_path": args.config,
        "git_tags": get_git_info(),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
    }
    base_backtest_cfg.update(flatten_config(raw_cfg))

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
    )

    logger.info("HPO complete. Best config:")
    for key, val in best_config.items():
        logger.info(f"  {key}: {val}")
