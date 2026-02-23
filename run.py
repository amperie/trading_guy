"""
Production launcher for backtesting and live trading.

Usage:
    python run.py backtest      --config configs/example_backtest.yaml
    python run.py backtest      --config configs/example_backtest.yaml --symbol TSLA --cash 50000
    python run.py live          --config configs/example_live.yaml
    python run.py live          --config configs/example_live_self_optimizing.yaml
    python run.py walk-forward  --config configs/example_walk_forward.yaml
"""

import argparse
import copy
import os
import sys
import yaml
from dotenv import load_dotenv

load_dotenv()

from utils.utils import instantiate_from_string
from utils.logger import Logger
import logging
logging.getLogger("websockets.client").setLevel(logging.INFO)

logger = Logger().get_logger(__name__)


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


def _run_analysis(cfg: dict, pf, om):
    """Run post-backtest analysis if enabled in config."""
    analysis_cfg = cfg.get("analysis", {})
    if not analysis_cfg.get("enabled", False):
        return

    from trading.analysis.analysis_engine import AnalysisEngine

    engine = AnalysisEngine(pf, om)
    results = engine.run_full_analysis(
        log_to_mlflow=analysis_cfg.get("log_to_mlflow", True),
        experiment_name=analysis_cfg.get("experiment_name"),
        run_name=analysis_cfg.get("run_name"),
        description=analysis_cfg.get("description"),
        parameters=analysis_cfg.get("parameters"),
        show_summary=True,
    )
    return results


def cmd_backtest(args: argparse.Namespace):
    """Run a backtest from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    logger.info(f"Starting backtest with profile: {args.config}")

    dp, al, om, pf = _build_components(cfg)

    # Engine-level config (state_store, etc.)
    engine_cfg = {}
    if "state_store" in cfg:
        engine_cfg["state_store"] = cfg["state_store"]

    from trading.engines.backtest_engine import BacktestingEngine

    engine = BacktestingEngine(cfg=engine_cfg, dp=dp, al=al, om=om, pf=pf)
    engine.run()

    logger.info(
        f"Backtest complete — Value: ${pf.total_value:,.2f}, "
        f"Cash: ${pf.cash:,.2f}, Positions: {list(pf.positions.keys())}"
    )

    _run_analysis(cfg, pf, om)


def cmd_live(args: argparse.Namespace):
    """Run live trading from a config profile."""
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
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

    # -- backtest subcommand --
    bt = subparsers.add_parser("backtest", parents=[shared], help="Run a backtest")
    bt.add_argument("--data", help="Override data provider path")
    bt.set_defaults(func=cmd_backtest)

    # -- live subcommand --
    live = subparsers.add_parser("live", parents=[shared], help="Run live trading (wraps with self-optimization if optimization.enabled is true)")
    live.set_defaults(func=cmd_live)

    # -- walk-forward subcommand --
    wf = subparsers.add_parser("walk-forward", parents=[shared], help="Run walk-forward backtest with rolling HPO re-optimization")
    wf.add_argument("--data", help="Override data provider path")
    wf.set_defaults(func=cmd_walk_forward)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
