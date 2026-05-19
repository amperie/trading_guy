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
    cmd_hpo_split,
    cmd_live,
    cmd_mongo_backtest,
    cmd_promote,
    cmd_session_replay,
    cmd_walk_forward,
    cmd_walk_forward_hpo,
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
            "    Run a historical simulation over a configured date range and data source.\n"
            "    --config --account [--data] [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--no-mlflow] [--run-name] [--session-id] [--agg-period]\n"
            "    Example: python run.py backtest --config configs/example_backtest.yaml --account paper\n"
            "    Example: python run.py backtest --config trading/promoted/my_live_bundle/my_live_bundle.yaml --account paper --session-id live-20260513-a\n"
            "  mongo-backtest:\n"
            "    Run a simulation against bars already stored in MongoDB for a prior session.\n"
            "    --config --account --session-id [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--no-mlflow] [--run-name] [--agg-period]\n"
            "    Example: python run.py mongo-backtest --config trading/promoted/my_live_bundle/my_live_bundle.yaml --account paper --session-id live-20260513-a\n"
            "  live:\n"
            "    Start a live trading session using the configured broker, strategy, and portfolio wiring.\n"
            "    Supports plain live execution, background self-optimization, and live walk-forward optimization\n"
            "    when configured under the optimization block.\n"
            "    --config --account [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--alpaca-override-url] [--run-name] [--session-id] [--agg-period]\n"
            "    Example: python run.py live --config configs/example_live.yaml --account paper --run-name morning-open\n"
            "    Example: python run.py live --config configs/example_live_self_optimizing.yaml --account paper --session-id live-20260517-a\n"
            "    Example: python run.py live --config configs/example_live_walk_forward.yaml --account paper --session-id live-20260517-b\n"
            "  walk-forward:\n"
            "    Compute rolling optimize/validate decisions, then run one continuous historical simulation\n"
            "    that applies approved config changes at each trade-window boundary.\n"
            "    --config --account [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--no-mlflow] [--run-name] [--session-id] [--agg-period]\n"
            "    Example: python run.py walk-forward --config configs/example_walk_forward.yaml --account paper\n"
            "  walk-forward-hpo:\n"
            "    Search over walk-forward optimization, validation, and trading window sizes. Candidate runs are\n"
            "    logged to a temporary MLflow experiment, then the winning window schedule is rerun into the\n"
            "    permanent experiment. Optional cleanup can mark the temp experiment deleted, run MLflow GC,\n"
            "    and remove a dedicated staging S3 prefix.\n"
            "    --config --account [--symbol] [--cash] [--algorithm] [--algorithm-url] [--portfolio] [--portfolio-url]\n"
            "    [--no-mlflow] [--run-name] [--session-id] [--agg-period]\n"
            "    Example: python run.py walk-forward-hpo --config configs/example_walk_forward_hpo.yaml --account paper\n"
            "  hpo:\n"
            "    Launch a Ray Tune hyperparameter search for one config profile over a single date range.\n"
            "    --config --account [--num-samples] [--max-concurrent-trials] [--symbol] [--cash] [--algorithm]\n"
            "    [--algorithm-url] [--portfolio] [--portfolio-url] [--run-name] [--agg-period]\n"
            "    Example: python run.py hpo --config configs/example_hpo.yaml --account paper --num-samples 50\n"
            "  hpo-split:\n"
            "    Tune on the training span, then log best-config training and validation backtests.\n"
            "    Accepts --validation-period-days to override hpo.validation_period_days.\n"
            "    Example: python run.py hpo-split --config configs/example_hpo_split.yaml --account paper --validation-period-days 30\n"
            "  hpo-from-mlflow:\n"
            "    Rebuild an HPO config from a prior MLflow run, open it for edits, then execute the search.\n"
            "    --account --run-url [--tracking-uri] [--editor]\n"
            "    Example: python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --editor vim\n"
            "  session-replay:\n"
            "    Re-run a saved live session offline with historical Alpaca bars and MongoDB-backed state.\n"
            "    --config --account --session-id [--timeframe] [--start-date] [--run-name] [--agg-period]\n"
            "    Example: python run.py session-replay --config configs/example_live.yaml --account paper --session-id live-20260512\n"
            "  promote:\n"
            "    Promote a prior MLflow run into a portable live config plus checked-in component source files.\n"
            "    --run-url [--tracking-uri] [--name]\n"
            "    Example: python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id>\n"
            "\n"
            "Workflow Examples:\n"
            "  Backtest -> promote directly to live bundle:\n"
            "    python run.py backtest --config configs/example_backtest.yaml --account paper --run-name my_backtest\n"
            "    python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --name my_live_bundle\n"
            "    python run.py live --config trading/promoted/my_live_bundle/my_live_bundle.yaml --account paper --session-id live-20260513-a\n"
            "  Backtest -> recreate HPO from MLflow -> promote the derived winner:\n"
            "    python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id>\n"
            "    python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<derived_run_id>\n"
            "  Derived-run fallback:\n"
            "    If a recreated HPO run does not contain algorithm code artifacts, promote will inspect the MLflow\n"
            "    description for the original source run URL and retry component recovery from that run.\n"
            "\n"
            "Use `python run.py <command> -h` for full command-specific help.\n"
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

    mongo_backtest_p = subparsers.add_parser(
        "mongo-backtest",
        parents=[shared],
        help="Run a backtest against MongoDB bars for a stored session",
        description=(
            "Reuse the algorithm and portfolio wiring from the provided config, but force the runtime "
            "onto MongoDBDataProvider plus BacktestingOrderManager for the supplied session_id."
        ),
    )
    mongo_backtest_p.set_defaults(func=cmd_mongo_backtest, mongo_backtest=True)

    live_p = subparsers.add_parser(
        "live",
        parents=[shared],
        help="Run live trading",
        description=(
            "Start a live Alpaca-driven session. Depending on the optimization config, this can run as plain live, "
            "background self-optimizing live, or live walk-forward mode (optimization.mode=walk_forward_live)."
        ),
        epilog=(
            "Examples:\n"
            "  python run.py live --config configs/example_live.yaml --account paper --session-id live-20260517-plain\n"
            "  python run.py live --config configs/example_live_self_optimizing.yaml --account paper --session-id live-20260517-selfopt\n"
            "  python run.py live --config configs/example_live_walk_forward.yaml --account paper --session-id live-20260517-wf\n"
        ),
    )
    live_p.add_argument("--alpaca-override-url", dest="alpaca_override_url", help="Override Alpaca websocket URL")
    live_p.set_defaults(func=cmd_live)

    wf_p = subparsers.add_parser(
        "walk-forward",
        parents=[shared],
        help="Run walk-forward optimization",
        description=(
            "Run historical walk-forward optimization with three rolling windows: optimization, validation, and "
            "trading. The challenger is tuned on the optimization window, compared against the incumbent on the "
            "same validation window, and approved changes are then applied inside one continuous end-to-end "
            "simulation over the full data span."
        ),
        epilog=(
            "Example:\n"
            "  python run.py walk-forward --config configs/example_walk_forward.yaml --account paper --run-name wf_spy_v1\n"
        ),
    )
    wf_p.set_defaults(func=cmd_walk_forward)

    wf_hpo_p = subparsers.add_parser(
        "walk-forward-hpo",
        parents=[shared],
        help="Optimize walk-forward window sizes",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run an outer Optuna search over optimization_window_days, validation_window_days, and "
            "trading_window_days. Each candidate runs the normal walk-forward process, including the "
            "inner strategy HPO configured under walk_forward.search_space. Candidate runs are logged "
            "to a staging MLflow experiment; the winning windows are rerun into the final experiment. "
            "Configure staging/final experiment names, optional artifact locations, MLflow GC, and direct "
            "S3-prefix cleanup under walk_forward_window_hpo."
        ),
        epilog=(
            "Example:\n"
            "  python run.py walk-forward-hpo --config configs/example_walk_forward_hpo.yaml --account paper\n"
            "\n"
            "Config keys:\n"
            "  walk_forward_window_hpo.num_samples              Outer window trials\n"
            "  walk_forward_window_hpo.objective_metric         Metric to maximize, e.g. wf_annualized_return\n"
            "  walk_forward_window_hpo.min_periods              Reject schedules with too few walk-forward periods\n"
            "  walk_forward_window_hpo.search_space             Must include optimization_window_days,\n"
            "                                                    validation_window_days, trading_window_days\n"
            "  walk_forward_window_hpo.final_experiment_name    Permanent MLflow experiment for the rerun winner\n"
            "  walk_forward_window_hpo.staging_experiment_name  Optional temp experiment name; omitted means unique\n"
            "  walk_forward_window_hpo.staging_artifact_location Optional staging artifact root, such as an S3 prefix\n"
            "  walk_forward_window_hpo.cleanup_staging_experiment Mark temp experiment deleted after winner rerun\n"
            "  walk_forward_window_hpo.run_mlflow_gc            Run `mlflow gc` after deleting the temp experiment\n"
            "  walk_forward_window_hpo.cleanup_s3_prefix        Run `aws s3 rm <staging_artifact_location> --recursive`\n"
        ),
    )
    wf_hpo_p.set_defaults(func=cmd_walk_forward_hpo)

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

    hpo_split_p = subparsers.add_parser(
        "hpo-split",
        parents=[shared],
        help="Run split HPO with best-config training and validation analysis",
        description=(
            "Reserve the final validation_period_days as an out-of-sample holdout, run HPO on the "
            "earlier training span, then log the best config on both spans inside one MLflow run."
        ),
        epilog=(
            "Example:\n"
            "  python run.py hpo-split --config configs/example_hpo_split.yaml --account paper --validation-period-days 30\n"
        ),
    )
    hpo_split_p.add_argument("--num-samples", dest="num_samples", type=int, help="Override hpo.num_samples")
    hpo_split_p.add_argument(
        "--max-concurrent-trials",
        dest="max_concurrent_trials",
        type=int,
        help="Override hpo.max_concurrent_trials",
    )
    hpo_split_p.add_argument(
        "--validation-period-days",
        dest="validation_period_days",
        type=int,
        help="Override hpo.validation_period_days",
    )
    hpo_split_p.set_defaults(func=cmd_hpo_split)

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

    promote_p = subparsers.add_parser(
        "promote",
        help="Create a portable live config and local promoted code from an MLflow run",
        description=(
            "Load a prior MLflow run, recover its runtime config and component source artifacts, "
            "copy algorithm/portfolio code into trading/promoted/<bundle>/, and write the live config into that same bundle directory."
        ),
    )
    promote_p.add_argument("--run-url", required=True, help="MLflow run URL to promote")
    promote_p.add_argument(
        "--tracking-uri",
        dest="tracking_uri",
        help="Optional MLflow tracking URI override if the run URL does not point at the desired server",
    )
    promote_p.add_argument(
        "--name",
        help="Optional promotion bundle name. Defaults to a slug based on the source run name and run ID.",
    )
    promote_p.set_defaults(func=cmd_promote)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
