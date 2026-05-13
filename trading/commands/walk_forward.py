from __future__ import annotations

import argparse

from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    fill_alpaca_creds,
    load_account_creds,
    load_raw_config,
    validate_session_id,
)
from trading.config import ExperimentService
from trading.engines.walk_forward_engine import WalkForwardEngine
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
        "state_store": raw_cfg.get("state_store", {}),
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
