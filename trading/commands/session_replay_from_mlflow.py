from __future__ import annotations

import argparse
import copy
from typing import Any

from trading.commands.session_replay import cmd_session_replay
from trading.launchers.mlflow_hpo_launcher import load_source_run_context, persist_edited_config
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def _prepare_replay_config(source_context, args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(source_context.raw_config)
    cfg["mode"] = "session-replay"

    analysis_cfg = cfg.setdefault("analysis", {})
    experiment_name = getattr(args, "mlflow_experiment_name", None)
    if experiment_name:
        analysis_cfg["experiment_name"] = experiment_name
    else:
        analysis_cfg.setdefault("experiment_name", "Session Replay From MLflow")
    analysis_cfg.setdefault("run_name", f"mlflow-replay-{source_context.run_name}-{args.session_id[:16]}")
    analysis_cfg.setdefault(
        "description",
        f"Session replay using components recovered from MLflow run {source_context.run_id} "
        f"against MongoDB session {args.session_id}.",
    )

    mlflow_cfg = cfg.setdefault("mlflow", {})
    mlflow_cfg["enabled"] = True
    mlflow_cfg["tracking_uri"] = source_context.tracking_uri

    try:
        from utils.config_manager import ConfigManager

        root_state_store = ConfigManager().get("state_store") or {}
    except Exception:
        root_state_store = {}

    state_store = cfg.setdefault("state_store", {})
    state_store["enabled"] = True
    state_store["connection_uri"] = (
        getattr(args, "connection_uri", None)
        or root_state_store.get("connection_uri")
        or state_store.get("connection_uri")
    )
    state_store["database"] = (
        getattr(args, "database", None)
        or root_state_store.get("default_live_database")
        or root_state_store.get("database")
        or state_store.get("database")
    )
    state_store["session_id"] = args.session_id
    return cfg


def cmd_session_replay_from_mlflow(args: argparse.Namespace):
    source_context = load_source_run_context(
        args.run_url,
        tracking_uri=getattr(args, "tracking_uri", None),
    )
    replay_cfg = _prepare_replay_config(source_context, args)
    config_path = persist_edited_config(
        source_context,
        replay_cfg,
        output_dir_name="generated_session_replay_configs",
        filename_prefix="session_replay",
    )
    logger.info(f"Recovered MLflow config saved to {config_path}")

    replay_args = argparse.Namespace(**vars(args))
    replay_args.config = config_path
    replay_args.use_config_components = True
    replay_args.clean_mongo_backtest = True
    replay_args.source_run_url = source_context.source_url
    replay_args.source_run_id = source_context.run_id
    return cmd_session_replay(replay_args)
