"""Long-form argparse help text for ``run.py`` commands."""

BACKTEST_DESCRIPTION = (
    "Run a historical simulation over a configured date range and data source. "
    "The command builds the configured data provider, algorithm, order manager, "
    "and portfolio, then runs the standard backtesting pipeline."
)

BACKTEST_EPILOG = """Examples:
  python run.py backtest --config configs/example_backtest.yaml --account paper
  python run.py backtest --config configs/example_backtest.yaml --account paper --data data/test_data.csv
  python run.py backtest --config trading/promoted/my_live_bundle/my_live_bundle.yaml --account paper --session-id live-20260513-a

Common overrides:
  --data PATH             Override simulator.data_provider.path
  --symbol SYMBOL         Override the configured portfolio symbol
  --cash AMOUNT           Override starting cash
  --algorithm PATH        Override algorithm dotted path
  --portfolio PATH        Override portfolio dotted path
  --agg-period N          Enable aggregation and set the bar period in minutes
"""

HPO_DESCRIPTION = (
    "Launch a Ray Tune hyperparameter optimization search for one config profile "
    "over a single date range. The winning candidate is logged with its metrics "
    "and reconstructable runtime config."
)

HPO_EPILOG = """Examples:
  python run.py hpo --config configs/example_hpo.yaml --account paper --num-samples 50
  python run.py hpo --config configs/example_hpo.yaml --account paper --max-concurrent-trials 4

Config:
  hpo.num_samples             Number of trials when --num-samples is omitted
  hpo.max_concurrent_trials   Parallel trial limit
  hpo.search_space            Parameter search space parsed by utils.parse_search_space()
"""

SESSION_REPLAY_DESCRIPTION = (
    "Replay a stored live session offline using historical bars and MongoDB-backed "
    "session state. This is intended for post-mortem analysis and paper/live "
    "review workflows."
)

SESSION_REPLAY_EPILOG = """Examples:
  python run.py session-replay --config configs/example_live.yaml --account paper --session-id live-20260512
  python run.py session-replay --config configs/example_live.yaml --account paper --session-id live-20260512 --timeframe Minute
  python run.py session-replay --config configs/example_live.yaml --account paper --session-id live-20260512 --start-date 2026-05-01

Notes:
  --timeframe is only needed when stored session metadata does not include it.
  --start-date runs an additional extended Alpaca replay from that date to the session end.
"""

PROMOTE_EPILOG = """Examples:
  python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id>
  python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --name my_live_bundle

Output:
  trading/promoted/<bundle>/<bundle>.yaml
  trading/promoted/<bundle>/algorithm.py
  trading/promoted/<bundle>/portfolio.py
"""

PIPELINE_EPILOG = """Stages:
  research   Run backtest, split HPO, walk-forward, and candidate registration
  paper      Promote a source run into a paper bundle and launch paper trading
  review     Replay a paper/live session and register an approved bundle
  live       Launch real-money trading from a local or MLflow bundle config

Examples:
  python run.py pipeline research --config configs/example_hpo_split.yaml --account paper
  python run.py pipeline research --config http://localhost:5000/#/experiments/1/runs/<run_id> --account paper --editor vim
  python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --account paper
  python run.py pipeline review --config trading/promoted/my_bundle/my_bundle.yaml --account paper --session-id paper-20260523
  python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_run_id> --account live
"""

PIPELINE_RESEARCH_EPILOG = """Examples:
  python run.py pipeline research --config configs/example_hpo_split.yaml --account paper --num-samples 50
  python run.py pipeline research --config http://localhost:5000/#/experiments/1/runs/<run_id> --account paper --editor vim

Flow:
  1. Run a baseline backtest.
  2. Run split HPO with validation holdout.
  3. Run walk-forward validation.
  4. Evaluate pipeline.gates.research and optionally register a candidate bundle.

MLflow config input:
  --config may be an MLflow run URL when the run contains a YAML config artifact
  or logged config.* params. The reconstructed config is opened in a local editor
  before any research stage runs, then saved under scratch/generated_pipeline_configs/.
  That edited YAML path is passed to backtest, split HPO, and walk-forward so the
  config artifact logged later reflects the run that actually executed.

Editor selection:
  Pass --editor, or set CODEX_EDITOR, VISUAL, or EDITOR. If none are set, vim is used.
  Direct MLflow artifact URIs such as runs:/<run_id>/config/runtime_config.yaml are
  not accepted here; pass the MLflow run URL instead.
"""

PIPELINE_PAPER_EPILOG = """Example:
  python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<candidate_run_id> --account paper --session-id paper-20260523

The command materializes a promoted bundle, logs it to the pipeline experiment,
and starts a paper-trading live session from that bundle.
"""

PIPELINE_REVIEW_EPILOG = """Example:
  python run.py pipeline review --config trading/promoted/my_bundle/my_bundle.yaml --account paper --session-id paper-20260523

The command replays the stored session, evaluates pipeline.gates.review, and
registers an approved live bundle when the review passes.
"""

PIPELINE_LIVE_EPILOG = """Examples:
  python run.py pipeline live --config trading/promoted/my_bundle/my_bundle.yaml --account live
  python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_run_id> --account live --session-id live-20260524

Use --alpaca-override-url for test websocket endpoints.
"""
