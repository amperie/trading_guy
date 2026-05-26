from __future__ import annotations

import argparse
import sys

from trading.commands.analysis import run_analysis
from trading.commands.common import (
    adapt_live_config_to_mongo_backtest,
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    fill_data_provider_creds,
    load_account_creds,
    load_raw_config,
    validate_session_id,
)
from trading.config import ExperimentService
from trading.engines.backtest_engine import BacktestingEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def cmd_backtest(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    experiment_name_override = getattr(args, "mlflow_experiment_name_override", None)
    if experiment_name_override:
        raw_cfg.setdefault("analysis", {})["experiment_name"] = experiment_name_override
    validate_session_id(raw_cfg)
    raw_cfg = adapt_live_config_to_mongo_backtest(
        raw_cfg,
        force=getattr(args, "mongo_backtest", False),
    )

    creds = load_account_creds(args.account)

    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        fill_data_provider_creds(raw_cfg, creds)
        target = dp_section.setdefault("params", {}) if "implementation" in dp_section else dp_section
        if not target.get("api_key") or not target.get("secret_key"):
            logger.error(
                "AlpacaDataProvider requires credentials. Set api_key / secret_key in the config file or in accounts.yaml."
            )
            sys.exit(1)

    experiment = build_experiment_config(raw_cfg)
    built = ExperimentService.build(experiment)
    if built.data_provider is None:
        logger.error(
            "Backtest requires a data_provider. "
            "Pass a backtest config, or use mongo-backtest with a promoted/live bundle plus --session-id."
        )
        sys.exit(1)

    logger.info(
        f"Starting backtest with profile: {args.config} "
        f"(hash={ExperimentService.describe(experiment).config_hash})"
    )

    engine = BacktestingEngine(
        cfg={"state_store": raw_cfg.get("state_store", {})},
        dp=built.data_provider,
        al=built.algorithm,
        om=built.order_manager,
        pf=built.portfolio,
    )

    agg_cfg = raw_cfg.get("aggregation", {})
    if agg_cfg.get("enabled", False):
        from trading.engines.tick_aggregation_passthrough_engine import TickAggregationPassthroughEngine

        agg_engine = TickAggregationPassthroughEngine(cfg={**agg_cfg, "downstream_engine": engine})
        agg_engine.dp = built.data_provider
        logger.info(f"Aggregation enabled: {agg_cfg.get('aggregation_period_minutes', 5)}-min bars")
        agg_engine.run()
    else:
        engine.run()

    logger.info(
        f"Backtest complete — Value: ${built.portfolio.total_value:,.2f}, "
        f"Cash: ${built.portfolio.cash:,.2f}, Positions: {list(built.portfolio.positions.keys())}"
    )
    analysis = run_analysis(raw_cfg, built.portfolio, built.order_manager, config_path=args.config)
    return {
        "analysis": analysis,
        "final_value": built.portfolio.total_value,
        "cash": built.portfolio.cash,
        "positions": list(built.portfolio.positions.keys()),
    }


def cmd_mongo_backtest(args: argparse.Namespace):
    if not getattr(args, "session_id", None):
        logger.error("mongo-backtest requires --session-id <id>.")
        sys.exit(1)
    args.mongo_backtest = True
    cmd_backtest(args)
