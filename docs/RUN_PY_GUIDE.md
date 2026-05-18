# `run.py` Guide

`run.py` is the main operational entrypoint for this repository. It exposes the supported trading workflows as subcommands and routes all of them through the same config normalization and component-loading layer.

This matters because:
- backtests, live runs, HPO, session replay, and MLflow-based reconstruction all use the same config shape
- command-line overrides are applied consistently
- the same algorithm and portfolio wiring can move from research to live promotion without inventing a second interface

## Quick Start

Inspect the top-level help:

```bash
python run.py -h
```

Inspect a specific command:

```bash
python run.py backtest -h
python run.py mongo-backtest -h
python run.py live -h
python run.py promote -h
```

## Command Summary

`run.py` currently supports these subcommands:

- `backtest`: run a historical simulation over a configured data source
- `mongo-backtest`: run a historical simulation over bars already stored in MongoDB for a prior session
- `live`: start a live Alpaca-driven trading session
- `walk-forward`: run rolling optimize/validate decisions plus one continuous out-of-sample simulation
- `hpo`: run a standalone hyperparameter search
- `hpo-from-mlflow`: reconstruct an HPO config from a prior MLflow run, edit it, then launch it
- `session-replay`: replay a stored live session offline
- `promote`: turn a prior MLflow run into a portable live bundle

## Shared CLI Model

Most commands use a shared set of flags:

- `--config`: YAML profile to load
- `--account`: account name from `accounts.yaml`
- `--symbol`: override the portfolio symbol
- `--cash`: override starting cash
- `--algorithm`: override the algorithm class path
- `--algorithm-url`: load algorithm code from an HTTP(S) URL
- `--portfolio`: override the portfolio class path
- `--portfolio-url`: load portfolio code from an HTTP(S) URL
- `--no-mlflow`: disable MLflow logging for commands that support it
- `--run-name`: override the analysis/MLflow run name
- `--session-id`: set the state-store session ID
- `--agg-period`: enable aggregation and override bar size in minutes

Important details:
- `live` requires `--session-id`
- `backtest` also supports `--data`
- `mongo-backtest` requires `--session-id` and forces MongoDB bars plus the backtesting order manager
- `hpo` also supports `--num-samples` and `--max-concurrent-trials`
- `hpo-from-mlflow` and `promote` operate from MLflow run URLs instead of local config paths

## Config Loading Rules

At runtime, command execution follows this pattern:

1. Load root config from `config.yaml`
2. Load the command-specific profile YAML
3. Deep-merge the profile over the root config
4. Apply CLI overrides on top
5. Normalize legacy component blocks into typed component configs
6. Build components from `implementation`, `source_path`, or `source_url`

This means a profile in `configs/` usually only needs to define the parts that are different from the root defaults.

## Backtest

Basic example:

```bash
python run.py backtest --config configs/example_backtest.yaml --account paper
```

Override the data path and cash:

```bash
python run.py backtest --config configs/example_backtest.yaml --account paper --data data/SPY_5min_MarketHours.csv --cash 50000
```

Run a remote algorithm implementation by URL:

```bash
python run.py backtest --config configs/example_backtest.yaml --account paper --algorithm-url https://example.com/my_algo.py
```

What `backtest` does:
- builds the configured data provider, algorithm, portfolio, and order manager
- optionally wraps the pipeline with tick aggregation
- runs the historical engine
- optionally runs analysis
- optionally logs metrics, charts, config artifacts, and component code artifacts to MLflow

### MLflow Artifacts Logged by Backtests

When analysis logging is enabled, the system now attempts to log:

- the original config file passed via `--config`
- a redacted `runtime_config.yaml` capturing the fully merged runtime config
- algorithm source code
- portfolio source code
- analysis outputs such as trades, charts, and reports

Those extra config artifacts are important for later `promote` and `hpo-from-mlflow` flows.

### Using a promoted bundle with `backtest`

`backtest` can still accept a promoted live bundle plus `--session-id` and auto-adapt it onto MongoDB bars.

Example:

```bash
python run.py backtest --config trading/promoted/winner_v1/winner_v1.yaml --account paper --session-id live-20260513-winner
```

Prefer `mongo-backtest` when you want that intent to be explicit.

## Mongo Backtest

Basic example:

```bash
python run.py mongo-backtest --config trading/promoted/winner_v1/winner_v1.yaml --account paper --session-id live-20260513-winner
```

What `mongo-backtest` does:
- keeps the config file intact on disk
- reuses the configured algorithm and portfolio wiring from that config
- forces the runtime to use `MongoDBDataProvider`
- forces the runtime to use `BacktestingOrderManager`
- runs analysis and MLflow logging the same way `backtest` does

Use this when:
- you want a normal backtest execution path
- the bar data already exists in MongoDB under a previous `session_id`
- you want to run a promoted bundle unchanged instead of authoring a separate backtest YAML

Important distinction:
- `mongo-backtest` reuses MongoDB bars only
- `session-replay` reconstructs more live-session context and also performs the Alpaca historical replay flow

## Live

Basic example:

```bash
python run.py live --config configs/example_live.yaml --account paper --session-id live-20260513-a
```

With an explicit run name:

```bash
python run.py live --config configs/example_live.yaml --account paper --session-id open-drive-1 --run-name morning_open
```

What `live` does:
- requires a `session_id` so MongoDB-backed state can be persisted and replayed
- injects Alpaca credentials from `accounts.yaml`
- rewires the order manager to use Alpaca-compatible credentials
- optionally enables tick aggregation
- optionally enables one of two optimization wrappers if the config asks for it:
- `optimization.enabled: true` with no `optimization.mode` or any non-`walk_forward_live` mode uses the background self-optimizing wrapper
- `optimization.enabled: true` with `optimization.mode: walk_forward_live` uses the live walk-forward wrapper

Operational note:
- every live session should get a unique `--session-id`
- if you reuse a session ID accidentally, you will mix state across runs

### Live Optimization Modes

`live` now supports three operational patterns:

- Plain live:

```bash
python run.py live --config configs/example_live.yaml --account paper --session-id live-20260517-plain
```

- Live with background self-optimization:

```bash
python run.py live --config configs/example_live_self_optimizing.yaml --account paper --session-id live-20260517-selfopt
```

- Live with walk-forward optimization:

```bash
python run.py live --config configs/example_live_walk_forward.yaml --account paper --session-id live-20260517-wf
```

The key config switch for live walk-forward mode is:

```yaml
optimization:
  enabled: true
  mode: walk_forward_live
```

In `walk_forward_live` mode the system:
- optimizes a challenger on the optimization window
- compares challenger vs incumbent on the same validation window
- records the decision in MongoDB `optimization_events`
- activates the winner according to `adoption_policy`

## Walk-Forward

Example:

```bash
python run.py walk-forward --config configs/example_walk_forward.yaml --account paper
```

Use this when you want repeated optimize-validate-trade windows to estimate out-of-sample robustness instead of a single historical pass.

The backtest walk-forward path now uses three windows:
- optimization window: tune the challenger
- validation window: compare incumbent and challenger on the same holdout slice
- trading window: boundary where approved changes become active in the main simulation

The important behavioral detail is that the historical engine no longer treats each trade window as a separate backtest result. It now:

1. computes each optimization/validation decision in sequence
2. stores each decision in MongoDB `optimization_events`
3. runs one continuous backtest across the full data span
4. applies approved parameter changes at the relevant trade-window boundaries
5. logs one end-to-end MLflow run from the main portfolio object

This gives you:
- one authoritative equity curve and transaction history for the full span
- end-to-end metrics from the actual main portfolio
- optimization tables and event markers instead of one nested MLflow run per period

Typical invocation:

```bash
python run.py walk-forward --config configs/example_walk_forward.yaml --account paper --run-name wf_spy_v1
```

Typical config block:

```yaml
walk_forward:
  optimization_window_days: 90
  validation_window_days: 20
  trading_window_days: 30
  improvement_threshold_pct: 5.0
  min_validation_trades: 10
  objective_metric: annualized_return
```

### Walk-Forward Logging

For historical walk-forward runs, MLflow now records:

- one full run for the entire simulation
- final end-to-end metrics from the continuous main portfolio
- the standard walk-forward report
- an equity curve chart with optimization/adoption markers
- optimization event artifacts such as `optimization_events.json`, `optimization_events.csv`, and `optimization_events.md`

MongoDB records the underlying optimization decisions in `optimization_events`. The event ids from those records are also used in the chart annotations so you can correlate the visualization with the stored audit trail.

## HPO

Example:

```bash
python run.py hpo --config configs/example_hpo.yaml --account paper --num-samples 50 --max-concurrent-trials 4
```

Use this when:
- you already know the local config you want to optimize
- you want to search parameter space directly
- you do not need to reconstruct the config from a previous MLflow run

## HPO From MLflow

Example:

```bash
python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id>
```

Optional editor override:

```bash
python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --editor "C:\\Program Files\\Notepad++\\notepad++.exe"
```

What happens:

1. The command parses the MLflow run URL
2. It loads the source run config from YAML artifacts when available
3. If no YAML artifact exists, it reconstructs config from logged `config.*` parameters
4. It attempts to restore algorithm and portfolio code references from run artifacts
5. It prefills HPO config data from MLflow artifacts such as `config/hpo_config.json`
6. It opens a generated YAML in an editor
7. After you save and close the editor, it executes the HPO run

This is useful when the best starting point is an already-executed research run instead of a hand-authored HPO profile.

## Promote

`promote` is the bridge from research to live deployment.

Example:

```bash
python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id>
```

Explicit bundle name:

```bash
python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --name spy_macd_live_v1
```

What `promote` does:

1. Load the source MLflow run
2. Recover the runtime config from YAML artifacts or logged params
3. Try to recover algorithm and portfolio code artifacts
4. Create a live-mode config
5. Copy algorithm and portfolio code into `trading/promoted/<bundle>/`
6. Write a live YAML to `trading/promoted/<bundle>/<bundle>.yaml`
7. Write `promotion_manifest.json` with source metadata and a launch example

Generated paths:

- `trading/promoted/<bundle>/<bundle>.yaml`
- `trading/promoted/<bundle>/promotion_manifest.json`
- `trading/promoted/<bundle>/*.py`

### Promotion Fallback for Derived Runs

Some runs, especially recreated HPO runs, may not directly contain the algorithm code artifact you need.

`promote` now handles this fallback:

1. Try the selected run first
2. If algorithm or portfolio source is still missing, inspect the MLflow description
3. If the description contains an original source run URL, load that run
4. Reuse matching algorithm and portfolio source references from the original run

This covers descriptions like:

```text
Recreated HPO from source MLflow run ddea820c81e843d1831ba531bd4c14ce (http://hp.lan:8899/#/experiments/596060974901698399/runs/ddea820c81e843d1831ba531bd4c14ce)
```

### Running a Promoted Bundle on Another Node

The intended flow is:

1. Clone the repo on the target node
2. Pull the promoted config and promoted component files
3. Configure `accounts.yaml` on that node
4. Run the promoted config with a fresh session ID

Example:

```bash
python run.py live --config trading/promoted/spy_macd_live_v1/spy_macd_live_v1.yaml --account paper --session-id live-20260513-b
```

Important constraints:
- the target node still needs the repo and Python environment
- promotion copies algorithm and portfolio code, not the entire repository state
- secrets are not embedded into the promoted config
- state-store session IDs should be unique per live launch

## Session Replay

Example:

```bash
python run.py session-replay --config configs/example_session_replay.yaml --account paper --session-id live-20260512
```

With an extended replay start date:

```bash
python run.py session-replay --config configs/example_session_replay.yaml --account paper --session-id live-20260512 --start-date 2026-05-01
```

Use this when:
- a live run already executed and persisted state
- you want to replay it offline with historical bars
- you want analysis or debugging without touching the live broker

## Common Workflows

### Research a strategy

```bash
python run.py backtest --config configs/example_backtest.yaml --account paper --run-name test_a
```

### Recreate HPO from a promising MLflow run

```bash
python run.py hpo-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id>
```

### Promote a winning run to a live bundle

```bash
python run.py promote --run-url http://localhost:5000/#/experiments/1/runs/<run_id> --name winner_v1
```

### Launch the promoted live config

```bash
python run.py live --config trading/promoted/winner_v1/winner_v1.yaml --account paper --session-id live-20260513-winner
```

### Launch the promoted config with live walk-forward optimization

```bash
python run.py live --config trading/promoted/winner_v1/winner_v1.yaml --account paper --session-id live-20260517-winner
```

Then enable live walk-forward by adding an optimization block like:

```yaml
optimization:
  enabled: true
  mode: walk_forward_live
  schedule: weekly
  optimization_window_days: 90
  validation_window_days: 20
  trading_window_days: 7
  improvement_threshold_pct: 5.0
  min_validation_trades: 10
```

### Run the promoted config as a Mongo-backed backtest

```bash
python run.py mongo-backtest --config trading/promoted/winner_v1/winner_v1.yaml --account paper --session-id live-20260513-winner
```

### Replay the live session later

```bash
python run.py session-replay --config trading/promoted/winner_v1/winner_v1.yaml --account paper --session-id live-20260513-winner
```

## Practical Notes

- Prefer config files in `configs/` for stable workflows and CLI overrides for one-off changes
- If you care about promotion later, keep MLflow logging enabled during backtests
- If component code is generated or loaded remotely, make sure the code artifact is actually available to MLflow
- Use `python run.py <command> -h` whenever you are unsure whether a flag is shared or command-specific

## Related Files

- [run.py](/E:/Programming/trading_guy/run.py)
- [README.md](/E:/Programming/trading_guy/README.md)
- [README_MLFLOW.md](/E:/Programming/trading_guy/docs/README_MLFLOW.md)
- [ALPACA_SETUP.md](/E:/Programming/trading_guy/docs/ALPACA_SETUP.md)
