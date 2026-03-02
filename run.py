"""
Production launcher for backtesting and live trading.

Usage:
    python run.py backtest      --config configs/example_backtest.yaml
    python run.py backtest      --config configs/example_backtest.yaml --symbol TSLA --cash 50000
    python run.py live          --config configs/example_live.yaml
    python run.py live          --config configs/example_live.yaml --alpaca-override-url wss://stream.data.alpaca.markets/v2/test
    python run.py live          --config configs/example_live_self_optimizing.yaml
    python run.py walk-forward  --config configs/example_walk_forward.yaml
    python run.py hpo           --config configs/example_hpo.yaml
    python run.py hpo           --config configs/example_hpo.yaml --num-samples 50 --max-concurrent-trials 4
"""

import argparse
import copy
import os
import re
import sys
import yaml
from dotenv import load_dotenv

load_dotenv()

from utils.utils import instantiate_from_string
from utils.logger import Logger

logger = Logger().get_logger(__name__)


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


def _resolve_alpaca_credentials(cfg: dict) -> dict:
    """Fill in Alpaca API keys from env vars if not set in config."""
    alpaca = cfg.get("alpaca", {})
    if not alpaca.get("api_key"):
        alpaca["api_key"] = os.environ.get("ALPACA_API_KEY", "")
    if not alpaca.get("secret_key"):
        alpaca["secret_key"] = os.environ.get("ALPACA_SECRET_KEY", "")
    cfg["alpaca"] = alpaca
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

    from trading.analysis.analysis_engine import AnalysisEngine

    # Merge explicit analysis.parameters with flattened config values
    parameters = dict(analysis_cfg.get("parameters") or {})
    parameters.update(_flatten_config(cfg))

    tags = _get_git_info()
    benchmark_paths = analysis_cfg.get("benchmarks") or {}

    engine = AnalysisEngine(pf, om)
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
    _validate_session_id(cfg)

    logger.info(f"Starting backtest with profile: {args.config}")

    dp, al, om, pf = _build_components(cfg)

    from trading.engines.backtest_engine import BacktestingEngine

    engine = BacktestingEngine(cfg={"state_store": cfg.get("state_store", {})}, dp=dp, al=al, om=om, pf=pf)
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
    _validate_session_id(cfg)
    cfg = _resolve_alpaca_credentials(cfg)

    alpaca_cfg = cfg.get("alpaca", {})
    if not alpaca_cfg.get("api_key") or not alpaca_cfg.get("secret_key"):
        logger.error("Alpaca API credentials required. Set in config or via ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.")
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
    engine = AlpacaRealTimeEngine(cfg=alpaca_cfg, dp=dp, al=al, om=om, pf=pf)

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
        engine = SelfOptimizingLiveEngine(engine, opt_cfg)

    engine.run()


def cmd_walk_forward(args: argparse.Namespace):
    """Run a walk-forward backtest from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _validate_session_id(cfg)

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


def main():
    parser = argparse.ArgumentParser(
        description="Trading framework launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py backtest --config configs/example_backtest.yaml\n"
            "  python run.py backtest --config configs/example_backtest.yaml --symbol TSLA --cash 50000\n"
            "  python run.py live --config configs/example_live.yaml\n"
            "  python run.py live --config configs/example_live_self_optimizing.yaml\n"
            "  python run.py walk-forward --config configs/example_walk_forward.yaml\n"
            "  python run.py hpo --config configs/example_hpo.yaml\n"
            "  python run.py hpo --config configs/example_hpo.yaml --num-samples 50 --max-concurrent-trials 4\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Shared arguments --
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True, help="Path to YAML config profile")
    shared.add_argument("--symbol", help="Override portfolio symbol")
    shared.add_argument("--cash", type=float, help="Override starting cash")
    shared.add_argument("--algorithm", help="Override algorithm class (dotted path)")
    shared.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    shared.add_argument("--run-name", help="Override analysis run name")
    shared.add_argument("--session-id", dest="session_id", help="MongoDB state_store session ID (required when state_store.enabled is true)")

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
