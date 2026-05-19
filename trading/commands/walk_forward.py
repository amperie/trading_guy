from __future__ import annotations

import argparse

from trading.commands.analysis import _collect_config_artifact_paths, get_git_info
from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    fill_alpaca_creds,
    flatten_config,
    load_account_creds,
    load_raw_config,
    validate_session_id,
)
from trading.config import ExperimentService
from trading.engines.walk_forward_engine import WalkForwardEngine
from trading.engines.walk_forward_window_hpo import WalkForwardWindowHPO
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def cmd_walk_forward(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    validate_session_id(raw_cfg)
    creds = load_account_creds(args.account)

    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        target = dp_section.setdefault("params", {}) if "implementation" in dp_section else dp_section
        fill_alpaca_creds(target, creds)

    experiment = build_experiment_config(raw_cfg)
    built = ExperimentService.build(experiment)

    logger.info(
        f"Starting walk-forward backtest with profile: {args.config} "
        f"(hash={ExperimentService.describe(experiment).config_hash})"
    )

    engine_cfg = {
        "walk_forward": raw_cfg.get("walk_forward", {}),
        "experiment_name": raw_cfg.get("analysis", {}).get("experiment_name", "Walk Forward Backtest"),
        "run_name": raw_cfg.get("analysis", {}).get("run_name", "WalkForward"),
        "description": raw_cfg.get("analysis", {}).get("description", ""),
        "log_to_mlflow": raw_cfg.get("analysis", {}).get("log_to_mlflow", True),
        "tracking_uri": raw_cfg.get("mlflow", {}).get("tracking_uri"),
        "state_store": raw_cfg.get("state_store", {}),
        "mlflow_parameters": flatten_config(raw_cfg),
        "mlflow_artifact_paths": _collect_config_artifact_paths(raw_cfg, config_path=args.config),
        "mlflow_tags": get_git_info(),
        "benchmark_paths": raw_cfg.get("analysis", {}).get("benchmarks") or {},
    }

    engine = WalkForwardEngine(
        cfg=engine_cfg,
        dp=built.data_provider,
        al=built.algorithm,
        om=built.order_manager,
        pf=built.portfolio,
    )
    results = engine.run()

    agg = results.get("aggregate", {})
    logger.info("Walk-forward complete:")
    for key, val in agg.items():
        logger.info(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")


def cmd_walk_forward_hpo(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    validate_session_id(raw_cfg)
    creds = load_account_creds(args.account)

    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        target = dp_section.setdefault("params", {}) if "implementation" in dp_section else dp_section
        fill_alpaca_creds(target, creds)

    experiment = build_experiment_config(raw_cfg)
    built = ExperimentService.build(experiment)

    logger.info(
        f"Starting walk-forward window HPO with profile: {args.config} "
        f"(hash={ExperimentService.describe(experiment).config_hash})"
    )

    analysis_cfg = raw_cfg.get("analysis", {})
    window_hpo_cfg = raw_cfg.get("walk_forward_window_hpo", {})
    engine_cfg = {
        "walk_forward": raw_cfg.get("walk_forward", {}),
        "walk_forward_window_hpo": window_hpo_cfg,
        "experiment_name": analysis_cfg.get("experiment_name", "Walk Forward Backtest"),
        "run_name": analysis_cfg.get("run_name", "WalkForwardWindowHPO"),
        "description": analysis_cfg.get("description", ""),
        "log_to_mlflow": analysis_cfg.get("log_to_mlflow", True),
        "tracking_uri": raw_cfg.get("mlflow", {}).get("tracking_uri"),
        "state_store": raw_cfg.get("state_store", {}),
        "mlflow_parameters": flatten_config(raw_cfg),
        "mlflow_artifact_paths": _collect_config_artifact_paths(raw_cfg, config_path=args.config),
        "mlflow_tags": get_git_info(),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
    }

    optimizer = WalkForwardWindowHPO(
        engine_cfg=engine_cfg,
        dp=built.data_provider,
        al=built.algorithm,
        om=built.order_manager,
        pf=built.portfolio,
    )
    results = optimizer.run()

    logger.info("Walk-forward window HPO complete:")
    logger.info(f"  best_windows: {results['best_windows']}")
    logger.info(f"  best_metric: {results['best_metric']:.6f}")
    final_agg = results.get("final_result", {}).get("aggregate", {})
    for key, val in final_agg.items():
        logger.info(f"  final_{key}: {val:.4f}" if isinstance(val, float) else f"  final_{key}: {val}")
