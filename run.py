"""
CLI entrypoint for backtesting, live trading, HPO, walk-forward, and session replay.

The operational logic lives in importable command modules so external systems can
reuse the same config normalization and runtime wiring without shelling out.
"""

from __future__ import annotations

import argparse

from trading.commands import (
    cmd_backtest,
    cmd_hpo,
    cmd_live,
    cmd_session_replay,
    cmd_walk_forward,
)
from trading.commands.hpo_from_mlflow import cmd_hpo_from_mlflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trading framework CLI. Commands share a typed config normalization layer "
            "so internal runs and external experiment runners use the same interface."
        ),
        epilog=(
            "Top-level command summary:\n"
            "  backtest:\n"
            "    --config --account [--data] [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--no-mlflow] [--run-name] [--session-id] [--agg-period]\n"
            "  live:\n"
            "    --config --account [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--alpaca-override-url] [--run-name] [--session-id] [--agg-period]\n"
            "  walk-forward:\n"
            "    --config --account [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--no-mlflow] [--run-name] [--session-id] [--agg-period]\n"
            "  hpo:\n"
            "    --config --account [--num-samples] [--max-concurrent-trials] [--symbol] [--cash] [--algorithm]\n"
            "    [--algorithm-url] [--portfolio] [--portfolio-url] [--run-name] [--agg-period]\n"
            "  hpo-from-mlflow:\n"
            "    --account --run-url [--tracking-uri] [--editor]\n"
            "  session-replay:\n"
            "    --config --account --session-id [--timeframe] [--start-date] [--run-name] [--agg-period]\n"
            "\n"
            "Use `python run.py <command> -h` for full command-specific help.\n"
            "\n"
            "Examples:\n"
            "  python run.py backtest --config configs/example_backtest.yaml --account paper\n"
            "  python run.py hpo --config configs/example_hpo.yaml --account paper --num-samples 50\n"
            "  python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --editor vim"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True, help="Path to YAML config profile")
    shared.add_argument("--account", required=True, help="Account name from accounts.yaml")
    shared.add_argument("--symbol", help="Override portfolio symbol")
    shared.add_argument("--cash", type=float, help="Override starting cash")
    shared.add_argument("--algorithm", help="Override algorithm implementation path")
    shared.add_argument("--algorithm-url", dest="algorithm_url", help="Load algorithm class code from an HTTP(S) URL")
    shared.add_argument("--portfolio", help="Override portfolio implementation path")
    shared.add_argument("--portfolio-url", dest="portfolio_url", help="Load portfolio class code from an HTTP(S) URL")
    shared.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    shared.add_argument("--run-name", dest="run_name", help="Override MLflow run name")
    shared.add_argument(
        "--session-id",
        dest="session_id",
        help="MongoDB state_store session ID (required when state_store.enabled is true)",
    )
    shared.add_argument(
        "--agg-period",
        dest="agg_period",
        type=int,
        help="Override aggregation.aggregation_period_minutes (also sets aggregation.enabled=true)",
    )

    shared_account = argparse.ArgumentParser(add_help=False)
    shared_account.add_argument("--account", required=True, help="Account name from accounts.yaml")

    backtest_p = subparsers.add_parser("backtest", parents=[shared], help="Run a backtest")
    backtest_p.add_argument("--data", help="Override data provider path")
    backtest_p.set_defaults(func=cmd_backtest)

    live_p = subparsers.add_parser("live", parents=[shared], help="Run live trading")
    live_p.add_argument("--alpaca-override-url", dest="alpaca_override_url", help="Override Alpaca websocket URL")
    live_p.set_defaults(func=cmd_live)

    wf_p = subparsers.add_parser("walk-forward", parents=[shared], help="Run walk-forward optimization")
    wf_p.set_defaults(func=cmd_walk_forward)

    hpo_p = subparsers.add_parser(
        "hpo",
        parents=[shared],
        help="Run standalone Ray Tune hyperparameter optimization over a single date range",
    )
    hpo_p.add_argument("--num-samples", dest="num_samples", type=int, help="Override hpo.num_samples")
    hpo_p.add_argument(
        "--max-concurrent-trials",
        dest="max_concurrent_trials",
        type=int,
        help="Override hpo.max_concurrent_trials",
    )
    hpo_p.set_defaults(func=cmd_hpo)

    hpo_mlflow_p = subparsers.add_parser(
        "hpo-from-mlflow",
        parents=[shared_account],
        help="Recreate an HPO search from an MLflow run, edit the generated YAML, then execute it",
        description=(
            "Load a prior MLflow run, reconstruct its runtime config, prefill HPO settings from MLflow artifacts "
            "when available, open the generated HPO YAML in a local editor, then run HPO from the saved file."
        ),
    )
    hpo_mlflow_p.add_argument("--run-url", required=True, help="MLflow run URL to recreate")
    hpo_mlflow_p.add_argument(
        "--tracking-uri",
        dest="tracking_uri",
        help="Optional MLflow tracking URI override if the URL does not point to the desired tracking server",
    )
    hpo_mlflow_p.add_argument(
        "--editor",
        help="Editor executable or command. Defaults to notepad.exe on Windows and vim on Linux when EDITOR/VISUAL is unset",
    )
    hpo_mlflow_p.set_defaults(func=cmd_hpo_from_mlflow)

    sr_p = subparsers.add_parser(
        "session-replay",
        parents=[shared],
        help="Replay a stored live session using Alpaca historical bars and MongoDB state",
    )
    sr_p.add_argument("--timeframe", help="Override replay timeframe when session metadata is missing it")
    sr_p.add_argument(
        "--start-date",
        dest="start_date",
        help="Run an additional extended Alpaca replay from this date to the session end",
    )
    sr_p.set_defaults(func=cmd_session_replay)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
