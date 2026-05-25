"""CLI help text constants."""

BACKTEST_DESCRIPTION = (
    "Run a historical backtest over the configured data provider, algorithm, portfolio, "
    "and order manager wiring."
)

BACKTEST_EPILOG = (
    "Examples:\n"
    "  python run.py backtest --config configs/example_backtest.yaml --account paper\n"
    "  python run.py backtest --config configs/example_backtest.yaml --account paper --data data/spy.csv --cash 50000\n"
)

HPO_DESCRIPTION = (
    "Run standalone Ray Tune hyperparameter optimization over a single historical date range."
)

HPO_EPILOG = (
    "Examples:\n"
    "  python run.py hpo --config configs/example_hpo.yaml --account paper\n"
    "  python run.py hpo --config configs/example_hpo.yaml --account paper --num-samples 50 --max-concurrent-trials 4\n"
)

SESSION_REPLAY_DESCRIPTION = (
    "Replay a stored paper/live session using historical Alpaca bars plus MongoDB-backed "
    "session state reconstruction."
)

SESSION_REPLAY_EPILOG = (
    "Examples:\n"
    "  python run.py session-replay --config configs/example_live.yaml --account paper --session-id live-20260512\n"
    "  python run.py session-replay --config configs/example_live.yaml --account paper --session-id live-20260512 --timeframe Minute --start-date 2026-05-01\n"
)

PROMOTE_EPILOG = (
    "Example:\n"
    "  python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --name my_live_bundle\n"
)

PIPELINE_EPILOG = (
    "Stages:\n"
    "  research  Run backtest -> split HPO -> walk-forward and evaluate research gates.\n"
    "  paper     Materialize a paper bundle from MLflow and start a paper session.\n"
    "  review    Replay a paper/live session and evaluate review gates.\n"
    "  live      Launch live trading from a promoted or approved bundle.\n"
    "\n"
    "Examples:\n"
    "  python run.py pipeline research --config configs/example_hpo_split.yaml --account paper\n"
    "  python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --account paper\n"
    "  python run.py pipeline review --config trading/promoted/my_live_bundle/my_live_bundle.yaml --account paper --session-id paper-20260523-120000\n"
    "  python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account live\n"
)

PIPELINE_RESEARCH_EPILOG = (
    "Example:\n"
    "  python run.py pipeline research --config configs/example_hpo_split.yaml --account paper --name candidate_bundle\n"
)

PIPELINE_PAPER_EPILOG = (
    "Example:\n"
    "  python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --account paper --session-id paper-20260523-120000\n"
)

PIPELINE_REVIEW_EPILOG = (
    "Example:\n"
    "  python run.py pipeline review --config trading/promoted/my_live_bundle/my_live_bundle.yaml --account paper --session-id paper-20260523-120000\n"
)

PIPELINE_LIVE_EPILOG = (
    "Example:\n"
    "  python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account live --session-id live-20260524-a\n"
)
