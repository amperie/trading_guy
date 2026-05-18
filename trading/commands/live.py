from __future__ import annotations

import argparse
import sys

from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    load_account_creds,
    load_raw_config,
    resolve_alpaca_credentials,
    validate_session_id,
)
from trading.config import ExperimentService
from trading.engines.alpaca_engine import AlpacaRealTimeEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def cmd_live(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)

    if not getattr(args, "session_id", None):
        logger.error(
            "Live mode requires --session-id <id> on the command line. "
            "Each live run must have an explicit session ID for MongoDB persistence."
        )
        sys.exit(1)

    raw_cfg.setdefault("state_store", {})["enabled"] = True
    validate_session_id(raw_cfg)

    creds = load_account_creds(args.account)
    raw_cfg = resolve_alpaca_credentials(raw_cfg, creds)

    alpaca_cfg = raw_cfg.get("alpaca", {})
    if not alpaca_cfg.get("api_key") or not alpaca_cfg.get("secret_key"):
        logger.error("Alpaca API credentials required. Set in config or in accounts.yaml.")
        sys.exit(1)

    om_section = raw_cfg.get("order_manager", {})
    om_target = om_section.setdefault("params", {}) if "implementation" in om_section else om_section
    om_target.setdefault("api_key", alpaca_cfg["api_key"])
    om_target.setdefault("secret_key", alpaca_cfg["secret_key"])
    raw_cfg["order_manager"] = om_section

    warmup = alpaca_cfg.get("warmup")
    if warmup:
        warmup.setdefault("api_key", alpaca_cfg["api_key"])
        warmup.setdefault("secret_key", alpaca_cfg["secret_key"])
        warmup.setdefault("symbols", alpaca_cfg.get("symbols_to_subscribe", []))

    raw_cfg.pop("data_provider", None)

    experiment = build_experiment_config(raw_cfg)
    built = ExperimentService.build(experiment)

    logger.info(
        f"Starting live trading with profile: {args.config} "
        f"(hash={ExperimentService.describe(experiment).config_hash})"
    )

    alpaca_cfg["state_store"] = raw_cfg.get("state_store", {})
    alpaca_engine = AlpacaRealTimeEngine(
        cfg=alpaca_cfg,
        dp=built.data_provider,
        al=built.algorithm,
        om=built.order_manager,
        pf=built.portfolio,
    )
    engine = alpaca_engine

    opt_cfg = raw_cfg.get("optimization", {})
    if opt_cfg.get("enabled", False):
        hist_dp = opt_cfg.get("historical_data_provider", {})
        if hist_dp:
            hist_dp.setdefault("api_key", alpaca_cfg["api_key"])
            hist_dp.setdefault("secret_key", alpaca_cfg["secret_key"])
        if opt_cfg.get("mode") == "walk_forward_live":
            from trading.engines.live_walk_forward_engine import LiveWalkForwardEngine

            logger.info(
                f"Live walk-forward enabled: schedule={opt_cfg.get('schedule', 'weekly')}, "
                f"optimization={opt_cfg.get('optimization_window_days', 90)}d, "
                f"validation={opt_cfg.get('validation_window_days', 20)}d"
            )
            engine = LiveWalkForwardEngine(alpaca_engine, opt_cfg)
        else:
            from trading.engines.self_optimizing_live_engine import SelfOptimizingLiveEngine

            logger.info(
                f"Self-optimization enabled: schedule={opt_cfg.get('schedule', 'daily')}, "
                f"window={opt_cfg.get('rolling_window_days', 90)}d"
            )
            engine = SelfOptimizingLiveEngine(alpaca_engine, opt_cfg)

    agg_cfg = raw_cfg.get("aggregation", {})
    if agg_cfg.get("enabled", False):
        import types
        from trading.engines.tick_aggregation_passthrough_engine import TickAggregationPassthroughEngine

        pipeline_shim = types.SimpleNamespace()
        pipeline_shim.on_tick = alpaca_engine._run_pipeline
        alpaca_engine._agg_engine = TickAggregationPassthroughEngine(
            cfg={**agg_cfg, "downstream_engine": pipeline_shim}
        )
        logger.info(f"Live aggregation enabled: {agg_cfg.get('aggregation_period_minutes', 5)}-min bars")

    engine.run()
