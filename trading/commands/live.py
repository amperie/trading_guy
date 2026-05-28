from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from utils.config_manager import ConfigManager
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def _infer_source_run_url(config_ref: str, explicit_source_run_url: str | None = None) -> str | None:
    if explicit_source_run_url:
        return explicit_source_run_url
    if config_ref.startswith(("http://", "https://")) and "/runs/" in config_ref:
        return config_ref
    config_path = Path(config_ref)
    if not config_path.is_file():
        return None
    manifest_candidates = [config_path.parent / "promotion_manifest.json", *config_path.parent.glob("pipeline_*_manifest.json")]
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_run_url = manifest.get("source_run_url")
        if source_run_url:
            return str(source_run_url)
    return None


def _is_local_bundle_config(config_ref: str) -> bool:
    config_path = Path(config_ref)
    if not config_path.is_file():
        return False
    return any(
        path.is_file()
        for path in [config_path.parent / "promotion_manifest.json", *config_path.parent.glob("pipeline_*_manifest.json")]
    )


def _apply_runtime_alpaca_endpoints(raw_cfg: dict, args: argparse.Namespace) -> None:
    if getattr(args, "alpaca_override_url", None):
        return
    root_alpaca = ConfigManager().get("alpaca") or {}
    alpaca = raw_cfg.setdefault("alpaca", {})
    for key in ("override_url", "historical_data_url", "data_url", "data_override_url"):
        if root_alpaca.get(key):
            alpaca[key] = root_alpaca[key]


def _set_order_manager_credentials(raw_cfg: dict, api_key: str, secret_key: str) -> None:
    om_section = raw_cfg.get("order_manager", {})
    om_section["api_key"] = api_key
    om_section["secret_key"] = secret_key
    if "implementation" in om_section or "params" in om_section:
        params = om_section.setdefault("params", {})
        params["api_key"] = api_key
        params["secret_key"] = secret_key
    raw_cfg["order_manager"] = om_section


def _live_state_store_database() -> str | None:
    config = ConfigManager()
    return config.get("state_store.default_live_database") or config.get("state_store.database")


def _force_live_state_store(raw_cfg: dict) -> None:
    state_store = raw_cfg.setdefault("state_store", {})
    state_store["enabled"] = True
    if database := _live_state_store_database():
        state_store["database"] = database


def cmd_live(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    _apply_runtime_alpaca_endpoints(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)

    if not getattr(args, "session_id", None):
        logger.error(
            "Live mode requires --session-id <id> on the command line. "
            "Each live run must have an explicit session ID for MongoDB persistence."
        )
        sys.exit(1)

    _force_live_state_store(raw_cfg)
    validate_session_id(raw_cfg)

    creds = load_account_creds(args.account)
    raw_cfg = resolve_alpaca_credentials(raw_cfg, creds)

    alpaca_cfg = raw_cfg.get("alpaca", {})
    if not alpaca_cfg.get("api_key") or not alpaca_cfg.get("secret_key"):
        logger.error("Alpaca API credentials required. Set in config or in accounts.yaml.")
        sys.exit(1)

    _set_order_manager_credentials(raw_cfg, alpaca_cfg["api_key"], alpaca_cfg["secret_key"])

    warmup = alpaca_cfg.get("warmup")
    if warmup:
        warmup.setdefault("api_key", alpaca_cfg["api_key"])
        warmup.setdefault("secret_key", alpaca_cfg["secret_key"])
        warmup.setdefault("symbols", alpaca_cfg.get("symbols_to_subscribe", []))
        for key in ("historical_data_url", "data_url", "data_override_url"):
            if alpaca_cfg.get(key):
                warmup.setdefault(key, alpaca_cfg[key])
        if str(alpaca_cfg.get("override_url", "")).startswith(("http://", "https://")):
            warmup.setdefault("override_url", alpaca_cfg["override_url"])

    raw_cfg.pop("data_provider", None)

    experiment = build_experiment_config(raw_cfg)
    built = ExperimentService.build(experiment)
    config_hash = ExperimentService.describe(experiment).config_hash
    source_run_url = _infer_source_run_url(
        args.config,
        explicit_source_run_url=getattr(args, "source_run_url", None),
    )
    state_store_meta = raw_cfg.setdefault("state_store", {}).setdefault("metadata", {})
    state_store_meta.update(
        {
            "launch_config_ref": args.config,
            "config_hash": config_hash,
        }
    )
    if source_run_url:
        state_store_meta["source_run_url"] = source_run_url
    source_session_id = getattr(args, "source_session_id", None)
    if source_session_id:
        state_store_meta["source_session_id"] = source_session_id

    logger.info(
        f"Starting live trading with profile: {args.config} "
        f"(hash={config_hash})"
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
    return {
        "session_id": raw_cfg.get("state_store", {}).get("session_id"),
        "state_store_database": raw_cfg.get("state_store", {}).get("database"),
        "config_path": args.config,
        "config_hash": config_hash,
        "source_run_url": source_run_url,
        "source_session_id": source_session_id,
        "symbols": alpaca_cfg.get("symbols_to_subscribe", []),
    }
