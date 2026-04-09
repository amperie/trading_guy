"""
Production launcher for backtesting, live trading, and session replay.

Subcommands
-----------
backtest
    Run a full backtest from a CSV or Alpaca data source.

live
    Connect to Alpaca WebSocket for real-time trading.  Requires --session-id.
    Automatically warms up the algorithm from historical bars if the config
    includes an alpaca.warmup section.

walk-forward
    Rolling optimization (Ray Tune HPO) followed by out-of-sample backtesting
    over consecutive windows.

hpo
    Standalone Ray Tune hyperparameter search over a single date range.

session-replay
    Replay a stored live session using gap-free Alpaca historical bars.
    Fetches warmup bars + the full live session window, runs the same
    algo/portfolio from session metadata, and produces a three-way MLflow
    comparison: Alpaca replay (no prefix), MongoDB-tick replay (mongo_
    prefix), and the original live session (live_ prefix).

    With --start-date: also runs a fourth "extended" backtest from the
    given date to the session end using a fresh portfolio (no opening state
    restored).  Logged with extended_ prefix and a combined extended-vs-live
    chart.  The three-way comparison still runs normally.

Usage
-----
    python run.py backtest       --config configs/example_backtest.yaml --account paper
    python run.py backtest       --config configs/example_backtest.yaml --account paper --symbol TSLA --cash 50000
    python run.py live           --config configs/example_live.yaml --session-id my-session --account paper
    python run.py live           --config configs/example_live.yaml --session-id my-session --account paper \\
                                     --alpaca-override-url wss://stream.data.alpaca.markets/v2/test
    python run.py live           --config configs/example_live_self_optimizing.yaml --session-id opt-session --account paper
    python run.py walk-forward   --config configs/example_walk_forward.yaml --account paper
    python run.py hpo            --config configs/example_hpo.yaml --account paper
    python run.py hpo            --config configs/example_hpo.yaml --account paper --num-samples 50 --max-concurrent-trials 4
    python run.py session-replay --config configs/example_session_replay.yaml --session-id my-session --account paper
    python run.py session-replay --config configs/example_session_replay.yaml --session-id my-session --account paper \\
                                     --no-mlflow

Shared flags (all subcommands)
-------------------------------
    --config       Path to YAML config profile (required)
    --account      Account name from accounts.yaml (selects api_key/secret_key) (required)
    --symbol       Override portfolio symbol
    --cash         Override starting cash
    --algorithm    Override algorithm class (dotted path)
    --no-mlflow    Disable MLflow logging
    --run-name     Override MLflow run name
    --session-id   MongoDB state_store session ID
    --agg-period N Override aggregation period (also sets aggregation.enabled=true)

session-replay specific flags
------------------------------
    --timeframe    Bar size override for sessions missing metadata
                   (e.g. "Minute", "Hour", "Day"). Defaults to "Minute".
    --start-date   Run an additional extended Alpaca backtest starting from
                   this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) through the
                   end of the MongoDB session.  Fresh portfolio — no opening
                   state restored from MongoDB.  The normal three-way replay
                   still runs.
"""

import argparse
import copy
import os
import re
import sys
import tempfile
import yaml
import pandas as pd

from utils.utils import instantiate_from_string
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def _apply_session_log_file(cfg: dict, args: argparse.Namespace):
    """Point the file handler at a session-named log file if a run name is known."""
    run_name = getattr(args, "run_name", None) or cfg.get("analysis", {}).get("run_name")
    if run_name:
        Logger().set_log_file(run_name)


def _load_account_creds(account_name: str, path: str = "accounts.yaml") -> dict:
    """Load api_key/secret_key for the named account from accounts.yaml."""
    if not os.path.exists(path):
        logger.error(
            f"accounts.yaml not found at '{path}'. "
            "Copy accounts.yaml.example to accounts.yaml and fill in your credentials."
        )
        sys.exit(1)
    with open(path, "r") as f:
        accounts = yaml.safe_load(f) or {}
    if account_name not in accounts:
        logger.error(
            f"Account '{account_name}' not found in {path}. "
            f"Available: {list(accounts.keys())}"
        )
        sys.exit(1)
    entry = accounts[account_name]
    if not entry.get("api_key") or not entry.get("secret_key"):
        logger.error(
            f"Account '{account_name}' in {path} is missing api_key or secret_key."
        )
        sys.exit(1)
    return {"api_key": entry["api_key"], "secret_key": entry["secret_key"]}


def _import_class(dotted_path: str):
    """Import and return a class from a dotted module path without instantiating it."""
    import importlib
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. Override values win."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(config_path: str) -> dict:
    """Load profile YAML and merge with root config.yaml for shared infra."""
    from utils.config_manager import ConfigManager

    # Root config provides shared infra (logging, mlflow, state_store)
    root_cfg = ConfigManager().config

    with open(config_path, "r") as f:
        profile = yaml.safe_load(f) or {}

    return _deep_merge(root_cfg, profile)


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI flag overrides on top of the merged config."""
    if getattr(args, "symbol", None):
        if "portfolio" in cfg:
            cfg["portfolio"]["symbol"] = args.symbol
        if "alpaca" in cfg:
            cfg["alpaca"]["symbols_to_subscribe"] = [args.symbol]

    if getattr(args, "cash", None) is not None:
        if "portfolio" in cfg:
            cfg["portfolio"]["cash"] = args.cash

    if getattr(args, "algorithm", None):
        if "algorithm" in cfg:
            cfg["algorithm"]["algorithm"] = args.algorithm

    if getattr(args, "data", None):
        if "data_provider" in cfg:
            cfg["data_provider"]["path"] = args.data

    if getattr(args, "no_mlflow", False):
        cfg.setdefault("analysis", {})["log_to_mlflow"] = False

    if getattr(args, "run_name", None):
        cfg.setdefault("analysis", {})["run_name"] = args.run_name

    if getattr(args, "alpaca_override_url", None):
        cfg.setdefault("alpaca", {})["override_url"] = args.alpaca_override_url

    if getattr(args, "session_id", None):
        cfg.setdefault("state_store", {})["session_id"] = args.session_id

    if getattr(args, "agg_period", None) is not None:
        cfg.setdefault("aggregation", {})["aggregation_period_minutes"] = args.agg_period
        cfg.setdefault("aggregation", {}).setdefault("enabled", True)

    return cfg


def _build_components(cfg: dict):
    """Instantiate pipeline components from config dicts."""
    # Order Manager
    om_section = cfg["order_manager"]
    om_path = om_section["order_manager"]
    om_cfg = {k: v for k, v in om_section.items() if k != "order_manager"}
    om = instantiate_from_string(om_path, cfg=om_cfg) if om_cfg else instantiate_from_string(om_path)

    # Algorithm
    # Note: Algorithm.__init__ reads history_length from a keyword arg,
    # not from cfg, so we pass it explicitly.
    al_section = cfg["algorithm"]
    al_path = al_section["algorithm"]
    al_cfg = {k: v for k, v in al_section.items() if k != "algorithm"}
    history_length = al_cfg.pop("history_length", 0)
    al = instantiate_from_string(al_path, cfg=al_cfg, history_length=history_length)

    # Portfolio
    pf_section = cfg["portfolio"]
    pf_path = pf_section["portfolio"]
    pf_cfg = {k: v for k, v in pf_section.items() if k != "portfolio"}
    pf = instantiate_from_string(pf_path, cfg=pf_cfg, order_manager=om)

    # Data Provider (optional — live engines may not need one at top level)
    dp = None
    if "data_provider" in cfg:
        dp_section = cfg["data_provider"]
        dp_path = dp_section["provider"]
        dp_cfg = {k: v for k, v in dp_section.items() if k != "provider"}
        dp = instantiate_from_string(dp_path, cfg=dp_cfg)

    return dp, al, om, pf


def _fill_alpaca_creds(section: dict, creds: dict) -> None:
    """Override api_key / secret_key in a config section from accounts.yaml (always wins)."""
    section["api_key"] = creds.get("api_key", "")
    section["secret_key"] = creds.get("secret_key", "")


def _resolve_alpaca_credentials(cfg: dict, creds: dict) -> dict:
    """Fill in Alpaca API keys from creds if not set in config."""
    alpaca = cfg.setdefault("alpaca", {})
    _fill_alpaca_creds(alpaca, creds)
    return cfg


_SENSITIVE_KEYS = {"api_key", "secret_key", "password", "token"}


def _flatten_config(cfg: dict, prefix: str = "config") -> dict:
    """Recursively flatten a nested config dict to dot-notation key/value pairs."""
    result = {}
    for k, v in cfg.items():
        if k in _SENSITIVE_KEYS:
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_config(v, full_key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            result[full_key[:250]] = v
        # lists and complex objects are skipped
    return result


def _get_git_info() -> dict:
    """Return git commit hash and GitHub URL as a flat dict, or {} on failure."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return {}

    # Normalise remote to HTTPS URL (handles both https:// and git@github.com: forms)
    remote = re.sub(r"\.git$", "", remote)
    if remote.startswith("git@"):
        # git@github.com:user/repo  →  https://github.com/user/repo
        remote = re.sub(r"^git@([^:]+):", r"https://\1/", remote)

    commit_url = f"{remote}/commit/{commit}" if "github.com" in remote else ""

    result = {"git.commit": commit}
    if commit_url:
        result["git.commit_url"] = commit_url
    return result


def _run_analysis(cfg: dict, pf, om, config_path: str = None):
    """Run post-backtest analysis if enabled in config."""
    analysis_cfg = cfg.get("analysis", {})
    if not analysis_cfg.get("enabled", False):
        return

    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

    # Merge explicit analysis.parameters with flattened config values
    parameters = dict(analysis_cfg.get("parameters") or {})
    parameters.update(_flatten_config(cfg))

    tags = _get_git_info()
    benchmark_paths = analysis_cfg.get("benchmarks") or {}

    engine = PortfolioAnalyzer(pf, om)
    results = engine.run_full_analysis(
        log_to_mlflow=analysis_cfg.get("log_to_mlflow", True),
        experiment_name=analysis_cfg.get("experiment_name"),
        run_name=analysis_cfg.get("run_name"),
        description=analysis_cfg.get("description"),
        parameters=parameters,
        show_summary=True,
        tags=tags if tags else None,
        artifact_paths=[config_path] if config_path else None,
        benchmark_paths=benchmark_paths if benchmark_paths else None,
    )
    return results


def _validate_session_id(cfg: dict) -> None:
    """Error out if state_store is enabled but session_id is not set."""
    ss_cfg = cfg.get("state_store", {})
    if ss_cfg.get("enabled", False) and not ss_cfg.get("session_id"):
        logger.error(
            "state_store is enabled but session_id is not set. "
            "Set it in your config file or pass --session-id <id> to avoid "
            "creating anonymous MongoDB sessions."
        )
        sys.exit(1)


def cmd_backtest(args: argparse.Namespace):
    """Run a backtest from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _apply_session_log_file(cfg, args)
    _validate_session_id(cfg)

    creds = _load_account_creds(args.account)

    # Fill Alpaca credentials if using AlpacaDataProvider
    dp_section = cfg.get("data_provider", {})
    if "alpaca" in dp_section.get("provider", "").lower():
        _fill_alpaca_creds(dp_section, creds)
        if not dp_section.get("api_key") or not dp_section.get("secret_key"):
            logger.error(
                "AlpacaDataProvider requires credentials. "
                "Set api_key / secret_key in the config file or in accounts.yaml."
            )
            sys.exit(1)

    logger.info(f"Starting backtest with profile: {args.config}")

    dp, al, om, pf = _build_components(cfg)

    from trading.engines.backtest_engine import BacktestingEngine

    engine = BacktestingEngine(cfg={"state_store": cfg.get("state_store", {})}, dp=dp, al=al, om=om, pf=pf)

    agg_cfg = cfg.get("aggregation", {})
    if agg_cfg.get("enabled", False):
        from trading.engines.tick_aggregation_passthrough_engine import TickAggregationPassthroughEngine
        agg_engine = TickAggregationPassthroughEngine(cfg={**agg_cfg, "downstream_engine": engine})
        agg_engine.dp = dp
        logger.info(f"Aggregation enabled: {agg_cfg.get('aggregation_period_minutes', 5)}-min bars")
        agg_engine.run()
    else:
        engine.run()

    logger.info(
        f"Backtest complete — Value: ${pf.total_value:,.2f}, "
        f"Cash: ${pf.cash:,.2f}, Positions: {list(pf.positions.keys())}"
    )

    _run_analysis(cfg, pf, om, config_path=args.config)


def cmd_live(args: argparse.Namespace):
    """Run live trading from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _apply_session_log_file(cfg, args)

    if not getattr(args, "session_id", None):
        logger.error(
            "Live mode requires --session-id <id> on the command line. "
            "Each live run must have an explicit session ID for MongoDB persistence."
        )
        sys.exit(1)

    cfg.setdefault("state_store", {})["enabled"] = True

    _validate_session_id(cfg)
    creds = _load_account_creds(args.account)
    cfg = _resolve_alpaca_credentials(cfg, creds)

    alpaca_cfg = cfg.get("alpaca", {})
    if not alpaca_cfg.get("api_key") or not alpaca_cfg.get("secret_key"):
        logger.error("Alpaca API credentials required. Set in config or in accounts.yaml.")
        sys.exit(1)

    # Propagate alpaca credentials to order_manager config if needed
    om_section = cfg.get("order_manager", {})
    if not om_section.get("api_key"):
        om_section["api_key"] = alpaca_cfg["api_key"]
    if not om_section.get("secret_key"):
        om_section["secret_key"] = alpaca_cfg["secret_key"]
    cfg["order_manager"] = om_section

    # Propagate alpaca config to warmup data provider if needed
    warmup = alpaca_cfg.get("warmup")
    if warmup:
        if not warmup.get("api_key"):
            warmup["api_key"] = alpaca_cfg["api_key"]
        if not warmup.get("secret_key"):
            warmup["secret_key"] = alpaca_cfg["secret_key"]
        if not warmup.get("symbols"):
            warmup["symbols"] = alpaca_cfg.get("symbols_to_subscribe", [])

    # Live mode doesn't use a top-level data_provider (warmup provider
    # comes from the alpaca section instead). Remove any inherited from
    # root config.yaml so _build_components doesn't try to instantiate it.
    cfg.pop("data_provider", None)

    logger.info(f"Starting live trading with profile: {args.config}")

    dp, al, om, pf = _build_components(cfg)

    from trading.engines.alpaca_engine import AlpacaRealTimeEngine

    alpaca_cfg["state_store"] = cfg.get("state_store", {})
    alpaca_engine = AlpacaRealTimeEngine(cfg=alpaca_cfg, dp=dp, al=al, om=om, pf=pf)
    engine = alpaca_engine

    # Wrap with SelfOptimizingLiveEngine if optimization is enabled
    opt_cfg = cfg.get("optimization", {})
    if opt_cfg.get("enabled", False):
        from trading.engines.self_optimizing_live_engine import SelfOptimizingLiveEngine

        # Propagate alpaca credentials to the historical data provider
        hist_dp = opt_cfg.get("historical_data_provider", {})
        if hist_dp and not hist_dp.get("api_key"):
            hist_dp["api_key"] = alpaca_cfg["api_key"]
        if hist_dp and not hist_dp.get("secret_key"):
            hist_dp["secret_key"] = alpaca_cfg["secret_key"]

        logger.info(
            f"Self-optimization enabled: schedule={opt_cfg.get('schedule', 'daily')}, "
            f"window={opt_cfg.get('rolling_window_days', 90)}d"
        )
        engine = SelfOptimizingLiveEngine(alpaca_engine, opt_cfg)

    # Wire aggregation engine into the inner Alpaca engine if enabled
    agg_cfg = cfg.get("aggregation", {})
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


def cmd_walk_forward(args: argparse.Namespace):
    """Run a walk-forward backtest from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _apply_session_log_file(cfg, args)
    _validate_session_id(cfg)
    creds = _load_account_creds(args.account)

    # Fill Alpaca credentials if using AlpacaDataProvider
    dp_section = cfg.get("data_provider", {})
    if "alpaca" in dp_section.get("provider", "").lower():
        _fill_alpaca_creds(dp_section, creds)

    logger.info(f"Starting walk-forward backtest with profile: {args.config}")

    dp, al, om, pf = _build_components(cfg)

    # Build engine config from walk_forward section + analysis/MLflow settings
    engine_cfg = dict(cfg.get("walk_forward", {}))

    # Nest the walk_forward settings under the key the engine expects
    engine_cfg = {
        "walk_forward": cfg.get("walk_forward", {}),
        "experiment_name": cfg.get("analysis", {}).get("experiment_name", "Walk Forward Backtest"),
        "run_name": cfg.get("analysis", {}).get("run_name", "WalkForward"),
        "description": cfg.get("analysis", {}).get("description", ""),
        "state_store": cfg.get("state_store", {}),
    }

    from trading.engines.walk_forward_engine import WalkForwardEngine

    engine = WalkForwardEngine(cfg=engine_cfg, dp=dp, al=al, om=om, pf=pf)
    results = engine.run()

    agg = results.get("aggregate", {})
    logger.info("Walk-forward complete:")
    for key, val in agg.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.4f}")
        else:
            logger.info(f"  {key}: {val}")


def cmd_hpo(args: argparse.Namespace):
    """Run standalone Ray Tune HPO from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _apply_session_log_file(cfg, args)
    creds = _load_account_creds(args.account)

    # Fill Alpaca credentials if using AlpacaDataProvider
    dp_section = cfg.get("data_provider", {})
    if "alpaca" in dp_section.get("provider", "").lower():
        _fill_alpaca_creds(dp_section, creds)

    # CLI overrides for HPO-specific settings
    hpo_cfg = cfg.setdefault("hpo", {})
    if getattr(args, "num_samples", None) is not None:
        hpo_cfg["num_samples"] = args.num_samples
    if getattr(args, "max_concurrent_trials", None) is not None:
        hpo_cfg["max_concurrent_trials"] = args.max_concurrent_trials

    logger.info(f"Starting HPO with profile: {args.config}")

    # Extract classes (do NOT instantiate — tune_backtest_hyperparameters needs the class itself)
    al_section = cfg["algorithm"]
    pf_section = cfg["portfolio"]
    dp_section = cfg["data_provider"]
    om_section = cfg["order_manager"]

    al_class = _import_class(al_section["algorithm"])
    pf_class = _import_class(pf_section["portfolio"])
    dp_class = _import_class(dp_section["provider"])
    om_class = _import_class(om_section["order_manager"])

    # Build base configs (strip the class-path keys and positional constructor args)
    base_al_cfg = {k: v for k, v in al_section.items() if k not in ("algorithm", "history_length")}
    base_pf_cfg = {k: v for k, v in pf_section.items() if k not in ("portfolio", "cash", "keep_history")}
    base_dp_cfg = {k: v for k, v in dp_section.items() if k != "provider"}

    starting_cash = pf_section.get("cash", 10000.0)
    analysis_cfg = cfg.get("analysis", {})

    base_backtest_cfg = {
        "starting_cash":        starting_cash,
        "experiment_name":      analysis_cfg.get("experiment_name", "HPO"),
        "run_name":             analysis_cfg.get("run_name", "HPO_Run"),
        "description":          analysis_cfg.get("description", ""),
        "symbol":               base_pf_cfg.get("symbol") or base_pf_cfg.get("upro_symbol", ""),
        "config_artifact_path": args.config,
        "git_tags":             _get_git_info(),
        "benchmark_paths":      analysis_cfg.get("benchmarks") or {},
    }
    base_backtest_cfg.update(_flatten_config(cfg))

    # Parse search space (YAML dict → Ray Tune distributions)
    from utils.utils import parse_search_space
    search_space = parse_search_space(hpo_cfg.get("search_space", {}))

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
        search_space=search_space,
        algorithm_param_keys=hpo_cfg.get("algorithm_param_keys", []),
        portfolio_param_keys=hpo_cfg.get("portfolio_param_keys", []),
        num_samples=hpo_cfg.get("num_samples", 50),
        max_concurrent_trials=hpo_cfg.get("max_concurrent_trials", 8),
    )

    logger.info("HPO complete. Best config:")
    for k, v in best_config.items():
        logger.info(f"  {k}: {v}")


def cmd_session_replay(args: argparse.Namespace):
    """Replay a stored live session using Alpaca historical bars."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _apply_session_log_file(cfg, args)
    creds = _load_account_creds(args.account)
    cfg = _resolve_alpaca_credentials(cfg, creds)

    alpaca_cfg = cfg.get("alpaca", {})
    if not alpaca_cfg.get("api_key") or not alpaca_cfg.get("secret_key"):
        logger.error("Alpaca API credentials required. Set in config or in accounts.yaml.")
        sys.exit(1)

    if not getattr(args, "session_id", None):
        logger.error("session-replay requires --session-id <id>.")
        sys.exit(1)

    # --- Step 1: Load session metadata from MongoDB ---
    from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider
    from trading.core.classes import BracketOrder, OrderStatus, Position
    from utils.trading_state_store import TradingStateStore

    ss_cfg = cfg.get("state_store", {})
    meta_loader = SessionReplayDataProvider(cfg={
        "session_id":     args.session_id,
        "api_key":        alpaca_cfg["api_key"],
        "secret_key":     alpaca_cfg["secret_key"],
        "connection_uri": ss_cfg.get("connection_uri"),
        "database":       ss_cfg.get("database"),
        "timeframe":      getattr(args, "timeframe", None),
    })
    meta_loader.load_data()
    meta = meta_loader._session_metadata

    logger.info(
        f"Session replay: symbols={meta.get('symbols')} "
        f"timeframe={meta.get('timeframe')} warmup_bars={meta.get('warmup_bars')}"
    )

    def _build_opening_state(store: TradingStateStore, session_id: str, start_ts: pd.Timestamp) -> tuple[float | None, dict[str, Position], list]:
        """Restore the live session's opening cash, positions, and active orders."""
        start_dt = start_ts.to_pydatetime()
        first_snapshot = store._snapshots.find_one(
            {"session_id": session_id},
            sort=[("timestamp", 1)],
        )

        opening_cash = None
        opening_positions: dict[str, Position] = {}
        if first_snapshot is not None:
            opening_cash = first_snapshot.get("cash")
            for symbol, pos in (first_snapshot.get("positions") or {}).items():
                qty = int(pos.get("quantity", 0) or 0)
                if qty > 0:
                    opening_positions[symbol] = Position(symbol=symbol, quantity=qty)

        def _reset_order_for_replay(order):
            order.processed_by_portfolio = False
            if isinstance(order, BracketOrder):
                order.MANUAL_SALE = False
                order.SOLD_ORDER = None
                order.status = OrderStatus.PENDING_SALE if (
                    order.executed_datetime is not None and order.executed_datetime <= start_dt
                ) else OrderStatus.PENDING
                for child_name in list(order.get_child_order_names()):
                    child = order.get_child_order(child_name)
                    if child_name == "MANUAL_ORDER":
                        order.add_child_order("MANUAL_ORDER", None)
                        continue
                    if child is None:
                        continue
                    child.status = OrderStatus.PENDING
                    child.executed_datetime = None
                    child.cash = 0.0
                    child.processed_by_portfolio = False
            else:
                order.status = OrderStatus.PENDING
            return order

        opening_orders = []
        order_data = store.load_orders(session_id)
        for order in order_data["all_orders"].values():
            placed_ts = order.placed_datetime
            executed_ts = order.executed_datetime
            if placed_ts is None or placed_ts > start_dt:
                continue

            if isinstance(order, BracketOrder):
                if executed_ts is None or executed_ts > start_dt:
                    opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))
                    continue

                sold_order = order.SOLD_ORDER
                sold_ts = None
                if sold_order is not None:
                    sold_ts = sold_order.executed_datetime or sold_order.placed_datetime
                if sold_ts is None or sold_ts > start_dt:
                    opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))
            elif order.status == OrderStatus.PENDING:
                opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))

        return opening_cash, opening_positions, opening_orders

    # --- Step 2: Reconstruct algo + portfolio from session metadata ---
    al_class_path = meta.get("algorithm_class")
    al_cfg_raw    = dict(meta.get("algorithm_config") or {})
    pf_class_path = meta.get("portfolio_class")
    pf_cfg_raw    = dict(meta.get("portfolio_config") or {})
    if getattr(args, "cash", None) is not None:
        pf_cfg_raw["cash"] = args.cash

    if not al_class_path or not pf_class_path:
        logger.error(
            "Session metadata is missing algorithm_class / portfolio_class. "
            "This session was created before replay metadata was stored. "
            "Re-run the live session with a current version to generate full metadata."
        )
        sys.exit(1)

    warmup_bars = meta.get("warmup_bars", 0)
    history_length = al_cfg_raw.pop("history_length", 0)
    al = instantiate_from_string(al_class_path, cfg=al_cfg_raw, history_length=history_length)

    from trading.core.om.backtesting_om import BacktestingOrderManager
    om = BacktestingOrderManager(cfg={"market_hours_only": True})

    pf = instantiate_from_string(pf_class_path, cfg={**pf_cfg_raw, "keep_history": True}, order_manager=om)

    session_start = pd.to_datetime(meta["session_start"])
    session_end   = pd.to_datetime(meta["session_end"])
    store = TradingStateStore(
        connection_uri=ss_cfg.get("connection_uri") or "mongodb://localhost:27017",
        database=ss_cfg.get("database") or "trading",
    )
    opening_cash, opening_positions, opening_orders = _build_opening_state(
        store, args.session_id, session_start
    )
    opening_positions_template = copy.deepcopy(opening_positions)
    opening_orders_template = copy.deepcopy(opening_orders)
    if getattr(args, "cash", None) is None and opening_cash is not None:
        pf.cash = float(opening_cash)
    if opening_positions_template:
        pf.positions = copy.deepcopy(opening_positions_template)
    for order in copy.deepcopy(opening_orders_template):
        om.all_orders[order.order_id] = order
        om.pending_orders_by_id[order.order_id] = order
        if getattr(order, "platform_id", None):
            logger.debug(f"Restored opening order {order.order_id} platform_id={order.platform_id}")
    if hasattr(pf, "_symbol_entry_time"):
        pf._symbol_entry_time = {}
        for order in om.pending_orders_by_id.values():
            if (
                isinstance(order, BracketOrder)
                and order.status == OrderStatus.PENDING_SALE
                and order.executed_datetime is not None
            ):
                pf._symbol_entry_time[order.symbol] = order.executed_datetime
    logger.info(
        f"Restored opening replay state: cash={pf.cash:.2f}, "
        f"positions={{{', '.join(f'{k}:{v.quantity}' for k, v in pf.positions.items())}}}, "
        f"active_orders={len(om.pending_orders_by_id)}"
    )

    from trading.data_providers.alpaca_data_provider import AlpacaDataProvider

    # --- Step 3: Fetch warmup bars (date-range only; algo deque auto-truncates to history_length) ---
    if warmup_bars > 0:
        from utils.utils import compute_warmup_start_date
        warmup_start = compute_warmup_start_date(warmup_bars, meta["timeframe"], session_start.to_pydatetime())
        warmup_dp = AlpacaDataProvider(cfg={
            "api_key":    alpaca_cfg["api_key"],
            "secret_key": alpaca_cfg["secret_key"],
            "symbols":    meta["symbols"],
            "timeframe":  meta["timeframe"],
            "start_date": warmup_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date":   session_start.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        warmup_dp.load_data()

        # --- Step 4: Warm up algorithm manually (no portfolio, no engine) ---
        al.warm_up(list(warmup_dp.iterate()))
        logger.info(f"Algorithm warmed up (is_warmed_up={al.is_warmed_up})")

    # --- Step 5: Fetch live session bars ---
    live_dp = AlpacaDataProvider(cfg={
        "api_key":    alpaca_cfg["api_key"],
        "secret_key": alpaca_cfg["secret_key"],
        "symbols":    meta["symbols"],
        "timeframe":  meta["timeframe"],
        "start_date": session_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_date":   session_end.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    live_dp.load_data()

    # --- Step 6: Run BacktestingEngine on Alpaca bars ---
    from trading.engines.backtest_engine import BacktestingEngine

    engine = BacktestingEngine(cfg={}, dp=live_dp, al=al, om=om, pf=pf)
    engine.run()

    logger.info(
        f"Alpaca replay complete — Value: ${pf.total_value:,.2f}, "
        f"Cash: ${pf.cash:,.2f}, Positions: {list(pf.positions.keys())}"
    )

    # --- Step 6b: Extended Alpaca backtest (only when --start-date is given) ---
    pf_extended = None
    if getattr(args, "start_date", None):
        start_dt = pd.to_datetime(args.start_date).to_pydatetime()
        al_ext = instantiate_from_string(al_class_path, cfg=dict(al_cfg_raw), history_length=history_length)
        om_ext = BacktestingOrderManager(cfg={"market_hours_only": True})
        pf_ext = instantiate_from_string(pf_class_path, cfg={**pf_cfg_raw, "keep_history": True}, order_manager=om_ext)
        if getattr(args, "cash", None) is not None:
            pf_ext.cash = args.cash
        ext_dp = AlpacaDataProvider(cfg={
            "api_key":    alpaca_cfg["api_key"],
            "secret_key": alpaca_cfg["secret_key"],
            "symbols":    meta["symbols"],
            "timeframe":  meta["timeframe"],
            "start_date": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date":   session_end.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        ext_dp.load_data()
        engine_ext = BacktestingEngine(cfg={}, dp=ext_dp, al=al_ext, om=om_ext, pf=pf_ext)
        engine_ext.run()
        logger.info(
            f"Extended Alpaca replay complete — Value: ${pf_ext.total_value:,.2f}, "
            f"Cash: ${pf_ext.cash:,.2f}, Positions: {list(pf_ext.positions.keys())}"
        )
        pf_extended = pf_ext

    # --- Step 7: MongoDB-tick replay (exact bars the live session processed) ---
    from trading.data_providers.mongodb_data_provider import MongoDBDataProvider

    al_mongo = instantiate_from_string(al_class_path, cfg=dict(al_cfg_raw), history_length=history_length)
    om_mongo = BacktestingOrderManager(cfg={"market_hours_only": True})
    pf_mongo = instantiate_from_string(pf_class_path, cfg={**pf_cfg_raw, "keep_history": True}, order_manager=om_mongo)
    if getattr(args, "cash", None) is None and opening_cash is not None:
        pf_mongo.cash = float(opening_cash)
    if opening_positions_template:
        pf_mongo.positions = copy.deepcopy(opening_positions_template)
    for restored in copy.deepcopy(opening_orders_template):
        om_mongo.all_orders[restored.order_id] = restored
        om_mongo.pending_orders_by_id[restored.order_id] = restored
    if hasattr(pf_mongo, "_symbol_entry_time"):
        pf_mongo._symbol_entry_time = {}
        for order in om_mongo.pending_orders_by_id.values():
            if (
                isinstance(order, BracketOrder)
                and order.status == OrderStatus.PENDING_SALE
                and order.executed_datetime is not None
            ):
                pf_mongo._symbol_entry_time[order.symbol] = order.executed_datetime

    if warmup_bars > 0:
        al_mongo.warm_up(list(warmup_dp.iterate()))
        logger.info(f"MongoDB-replay algorithm warmed up (is_warmed_up={al_mongo.is_warmed_up})")

    mongo_dp = MongoDBDataProvider(cfg={
        "session_id":     args.session_id,
        "connection_uri": ss_cfg.get("connection_uri"),
        "database":       ss_cfg.get("database"),
    })

    engine_mongo = BacktestingEngine(cfg={}, dp=mongo_dp, al=al_mongo, om=om_mongo, pf=pf_mongo)
    engine_mongo.run()

    logger.info(
        f"MongoDB replay complete — Value: ${pf_mongo.total_value:,.2f}, "
        f"Cash: ${pf_mongo.cash:,.2f}, Positions: {list(pf_mongo.positions.keys())}"
    )

    analysis_cfg = cfg.get("analysis", {})
    log_mlflow = analysis_cfg.get("log_to_mlflow", True)
    if getattr(args, "no_mlflow", False):
        log_mlflow = False

    experiment_name = analysis_cfg.get("experiment_name", "Session Replay")
    run_name = (
        getattr(args, "run_name", None)
        or analysis_cfg.get("run_name")
        or f"session-replay-{args.session_id[:28]}"
    )

    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

    replay_analyzer = PortfolioAnalyzer(pf)
    mongo_analyzer  = PortfolioAnalyzer(pf_mongo)
    live_analyzer   = PortfolioAnalyzer.from_mongodb(
        args.session_id,
        connection_uri=ss_cfg.get("connection_uri"),
        database=ss_cfg.get("database"),
    )

    if log_mlflow:
        from utils.mlflow_client import MLflowClient

        client = MLflowClient.from_config(experiment_name=experiment_name)
        if not client.enabled:
            logger.info("MLflow disabled — skipping logging")
        else:
            tags = _get_git_info() or {}
            params = {
                "session_id": args.session_id,
                **_flatten_config({"meta": meta}),
            }
            session_start_str = meta.get("session_start")
            align_start = pd.to_datetime(session_start_str, utc=True) if session_start_str else None
            _tmp = tempfile.mkdtemp()

            with client.start_run(run_name=run_name, tags=tags if tags else None):
                client.log_params({k[:250]: v for k, v in params.items()
                                   if isinstance(v, (str, int, float, bool)) or v is None})

                # --- Three (or four) parallel analyses ---
                replay_analyzer._contribute_to_run(
                    client, metric_prefix="", artifact_prefix=""
                )
                mongo_analyzer._contribute_to_run(
                    client, metric_prefix="mongo_", artifact_prefix="mongo_"
                )
                live_analyzer._contribute_to_run(
                    client, metric_prefix="live_", artifact_prefix="live_"
                )
                if pf_extended is not None:
                    extended_analyzer = PortfolioAnalyzer(pf_extended)
                    client.log_params({"extended_start_date": str(args.start_date)})
                    extended_analyzer._contribute_to_run(
                        client, metric_prefix="extended_", artifact_prefix="extended_"
                    )

                # --- Combined charts: Alpaca replay vs Live ---
                combined_path = os.path.join(_tmp, "combined_equity_curve.png")
                try:
                    replay_analyzer.save_combined_equity_curve(
                        live_analyzer, combined_path,
                        self_label="Alpaca Replay", other_label="Live",
                        align_start=align_start,
                    )
                    client.log_artifact(combined_path)
                except Exception as e:
                    logger.warning(f"Failed to log combined_equity_curve.png: {e}")

                combined_lc_path = os.path.join(_tmp, "combined_lifecycle.html")
                try:
                    replay_analyzer.save_combined_lifecycle_chart_interactive(
                        live_analyzer, combined_lc_path,
                        self_label="Alpaca Replay", other_label="Live",
                        align_start=align_start,
                    )
                    client.log_artifact(combined_lc_path)
                except Exception as e:
                    logger.warning(f"Failed to log combined_lifecycle.html: {e}")

                # --- Combined charts: MongoDB replay vs Live ---
                combined_mongo_path = os.path.join(_tmp, "combined_equity_curve_mongo.png")
                try:
                    mongo_analyzer.save_combined_equity_curve(
                        live_analyzer, combined_mongo_path,
                        self_label="MongoDB Replay", other_label="Live",
                        align_start=align_start,
                    )
                    client.log_artifact(combined_mongo_path)
                except Exception as e:
                    logger.warning(f"Failed to log combined_equity_curve_mongo.png: {e}")

                combined_lc_mongo_path = os.path.join(_tmp, "combined_lifecycle_mongo.html")
                try:
                    mongo_analyzer.save_combined_lifecycle_chart_interactive(
                        live_analyzer, combined_lc_mongo_path,
                        self_label="MongoDB Replay", other_label="Live",
                        align_start=align_start,
                    )
                    client.log_artifact(combined_lc_mongo_path)
                except Exception as e:
                    logger.warning(f"Failed to log combined_lifecycle_mongo.html: {e}")

                # --- Combined charts: Extended Alpaca vs Live (only when --start-date given) ---
                if pf_extended is not None:
                    combined_ext_path = os.path.join(_tmp, "combined_equity_curve_extended.png")
                    try:
                        extended_analyzer.save_combined_equity_curve(
                            live_analyzer, combined_ext_path,
                            self_label="Extended Alpaca", other_label="Live",
                            align_start=align_start,
                        )
                        client.log_artifact(combined_ext_path)
                    except Exception as e:
                        logger.warning(f"Failed to log combined_equity_curve_extended.png: {e}")

                    combined_lc_ext_path = os.path.join(_tmp, "combined_lifecycle_extended.html")
                    try:
                        extended_analyzer.save_combined_lifecycle_chart_interactive(
                            live_analyzer, combined_lc_ext_path,
                            self_label="Extended Alpaca", other_label="Live",
                            align_start=align_start,
                        )
                        client.log_artifact(combined_lc_ext_path)
                    except Exception as e:
                        logger.warning(f"Failed to log combined_lifecycle_extended.html: {e}")

            logger.info("MLflow run complete")
    else:
        # Just print the summary
        replay_analyzer.run_full_analysis(log_to_mlflow=False, show_summary=True)


def main():
    parser = argparse.ArgumentParser(
        description="Trading framework launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py backtest        --config configs/example_backtest.yaml --account paper\n"
            "  python run.py backtest        --config configs/example_backtest.yaml --account paper --symbol TSLA --cash 50000\n"
            "  python run.py live            --config configs/example_live.yaml --session-id my-session --account paper\n"
            "  python run.py live            --config configs/example_live_self_optimizing.yaml --session-id opt-session --account paper\n"
            "  python run.py walk-forward    --config configs/example_walk_forward.yaml --account paper\n"
            "  python run.py hpo             --config configs/example_hpo.yaml --account paper\n"
            "  python run.py hpo             --config configs/example_hpo.yaml --account paper --num-samples 50 --max-concurrent-trials 4\n"
            "  python run.py session-replay  --config configs/example_live.yaml --session-id my-session --account paper\n"
            "  python run.py session-replay  --config configs/example_live.yaml --session-id my-session --account paper --start-date 2024-01-01\n"
            "  python run.py session-replay  --config configs/example_live.yaml --session-id my-session --account paper --no-mlflow\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # -- Shared arguments --
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True, help="Path to YAML config profile")
    shared.add_argument("--account", required=True, help="Account name from accounts.yaml (selects api_key/secret_key)")
    shared.add_argument("--symbol", help="Override portfolio symbol")
    shared.add_argument("--cash", type=float, help="Override starting cash")
    shared.add_argument("--algorithm", help="Override algorithm class (dotted path)")
    shared.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    shared.add_argument("--run-name", help="Override analysis run name")
    shared.add_argument("--session-id", dest="session_id", help="MongoDB state_store session ID (required when state_store.enabled is true)")
    shared.add_argument("--agg-period", dest="agg_period", type=int, help="Override aggregation.aggregation_period_minutes (also sets aggregation.enabled=true)")

    # -- backtest subcommand --
    bt = subparsers.add_parser("backtest", parents=[shared], help="Run a backtest")
    bt.add_argument("--data", help="Override data provider path")
    bt.set_defaults(func=cmd_backtest)

    # -- live subcommand --
    live = subparsers.add_parser("live", parents=[shared], help="Run live trading (wraps with self-optimization if optimization.enabled is true)")
    live.add_argument("--alpaca-override-url", dest="alpaca_override_url", help="Override Alpaca WebSocket URL (e.g. wss://stream.data.alpaca.markets/v2/test)")
    live.set_defaults(func=cmd_live)

    # -- walk-forward subcommand --
    wf = subparsers.add_parser("walk-forward", parents=[shared], help="Run walk-forward backtest with rolling HPO re-optimization")
    wf.add_argument("--data", help="Override data provider path")
    wf.set_defaults(func=cmd_walk_forward)

    # -- hpo subcommand --
    hpo_p = subparsers.add_parser("hpo", parents=[shared], help="Run standalone Ray Tune hyperparameter optimization over a single date range")
    hpo_p.add_argument("--data", help="Override data provider path")
    hpo_p.add_argument("--num-samples", dest="num_samples", type=int, help="Override hpo.num_samples (total Ray Tune trials)")
    hpo_p.add_argument("--max-concurrent-trials", dest="max_concurrent_trials", type=int, help="Override hpo.max_concurrent_trials (parallel Ray workers)")
    hpo_p.set_defaults(func=cmd_hpo)

    # -- session-replay subcommand --
    sr = subparsers.add_parser(
        "session-replay",
        parents=[shared],
        help="Replay a stored live session using Alpaca historical bars (warmup + live period)",
    )
    sr.add_argument(
        "--timeframe",
        help="Override bar timeframe for sessions missing metadata (e.g. Minute, Hour, Day)",
    )
    sr.add_argument(
        "--start-date",
        dest="start_date",
        help="Also run an extended Alpaca backtest from this date to session end (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). "
             "MongoDB-tick replay still runs as normal.",
    )
    sr.set_defaults(func=cmd_session_replay)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
