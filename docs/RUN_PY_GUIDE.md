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
python run.py pipeline -h
python run.py walk-forward-hpo -h
python run.py promote -h
```

## Command Summary

`run.py` currently supports these subcommands:

- `backtest`: run a historical simulation over a configured data source
- `mongo-backtest`: run a historical simulation over bars already stored in MongoDB for a prior session
- `live`: start a live Alpaca-driven trading session
- `walk-forward`: run rolling optimize/validate decisions plus one continuous out-of-sample simulation
- `walk-forward-hpo`: search over walk-forward optimization, validation, and trading window sizes
- `hpo`: run a standalone hyperparameter search
- `hpo-split`: run HPO on a training span, then log train/validation backtests for the winner
- `hpo-from-mlflow`: reconstruct an HPO config from a prior MLflow run, edit it, then launch it
- `hpo-split-from-mlflow`: reconstruct an HPO config from a prior MLflow run, edit it, then launch split HPO with a validation holdout
- `session-replay`: replay a stored live session offline
- `promote`: turn a prior MLflow run into a portable live bundle
- `pipeline research`: run backtest, split HPO, and walk-forward, then evaluate research gates
- `pipeline paper`: materialize a paper bundle from MLflow, log it to the pipeline experiment, and start paper trading
- `pipeline review`: replay a paper/live session, evaluate review gates, and register an approved live bundle
- `pipeline live`: launch a local or MLflow-backed promoted bundle

## Shared CLI Model

Most commands use a shared set of flags:

- `--config`: local YAML profile to load, or an MLflow run URL with a reconstructable config
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
- `hpo-split` also supports `--num-samples`, `--max-concurrent-trials`, and `--validation-period-days`
- `walk-forward-hpo` uses `walk_forward_window_hpo` config keys rather than command-specific trial flags
- `hpo-from-mlflow`, `hpo-split-from-mlflow`, and `promote` operate from MLflow run URLs instead of local config paths
- `pipeline live` can also use an MLflow run URL as `--config`

## Config Loading Rules

At runtime, command execution follows this pattern:

1. Load root config from `config.yaml`
2. Load the command-specific profile YAML
3. Deep-merge the profile over the root config
4. Apply CLI overrides on top
5. Normalize legacy component blocks into typed component configs
6. Build components from `implementation`, `source_path`, or `source_url`

This means a profile in `configs/` usually only needs to define the parts that are different from the root defaults.

`--config` is no longer only a filesystem concept. When you pass an MLflow run URL, `run.py` reconstructs the runtime config from that run's artifacts and logged params, then merges that reconstructed config over `config.yaml` just like a normal local profile.

## Pipeline Workflow

The recommended release workflow is:

```text
research -> paper -> review -> live
```

Use the high-level `pipeline` command when you want the system to enforce that flow instead of manually stitching together low-level commands.

### `pipeline research`

Example:

```bash
python run.py pipeline research --config configs/example_hpo_split.yaml --account paper
```

What it does:

1. Runs `backtest`
2. Runs `hpo-split`
3. Runs `walk-forward`
4. Evaluates `pipeline.gates.research`
5. If the gates pass and `pipeline.auto_promote_research` is true, creates a candidate bundle
6. Logs that candidate bundle to the dedicated pipeline MLflow experiment
7. Prints the backtest, split-HPO, walk-forward, and candidate-bundle MLflow URLs plus the local bundle paths

### `pipeline paper`

Example:

```bash
python run.py pipeline paper --run-url http://localhost:5000/#/experiments/1/runs/<candidate_run_id> --account paper
```

What it does:

1. Loads the source MLflow run
2. Materializes a paper bundle under `trading/promoted/<bundle>/`
3. Logs the bundle into the pipeline MLflow experiment
4. Prints both the local bundle paths and the pipeline MLflow run URL
5. Starts paper trading with a generated or explicit session id

### `pipeline paper-from-session`

Relaunch a stored paper session using its MongoDB metadata:

```bash
python run.py pipeline paper-from-session --source-session-id <session_id>
```

The command reads `metadata.account_name`, `metadata.source_run_url`, and/or
`metadata.launch_config_ref` from the session document. `--account` is optional
and only needed to override or backfill an older session without
`metadata.account_name`.

## Paper Session Autostart

The helper [paper_session_autostart.py](/E:/Programming/trading_guy/scripts/paper_session_autostart.py)
restarts selected paper sessions in separate tmux panes. Session documents use
`autostart: true` as desired state; the field does not claim that a process is
currently healthy or running.

The helper uses the MongoDB URI and live database from `config.yaml`. Override
them with `--connection-uri` or `--database` when necessary.

```bash
# Mark sessions for restart
python scripts/paper_session_autostart.py enable paper-session-a
python scripts/paper_session_autostart.py enable paper-session-b

# Remove a session from the restart set
python scripts/paper_session_autostart.py disable paper-session-b

# List enabled sessions and their stored account names
python scripts/paper_session_autostart.py list

# Preview commands, then launch one tmux pane per enabled session
python scripts/paper_session_autostart.py start --dry-run
python scripts/paper_session_autostart.py start

# Inspect output
tmux attach -t paper-sessions
```

Use `Ctrl-b` followed by an arrow key to move between panes. Detach without
stopping the sessions using `Ctrl-b d`.

The default tmux session is `paper-sessions`. Override it with:

```bash
python scripts/paper_session_autostart.py start --tmux-session trading-paper
```

The launcher refuses to start if that tmux session already exists. This is a
deliberate duplicate-process guard. Stop the existing processes or kill the
tmux session before relaunching:

```bash
tmux kill-session -t paper-sessions
```

### Start on boot with systemd

Create `/etc/systemd/system/trading-paper-autostart.service`, replacing the user
and repository paths:

```ini
[Unit]
Description=Trading paper session autostart
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=trading
WorkingDirectory=/opt/trading_guy
ExecStart=/opt/trading_guy/.venv/bin/python scripts/paper_session_autostart.py start
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trading-paper-autostart.service
systemctl status trading-paper-autostart.service
```

Requirements:

- `tmux` must be installed and available to the service user.
- The service user must be able to read the repository and account credentials.
- MongoDB, MLflow, and the broker/network endpoints must be reachable.
- Enabled session documents need `metadata.account_name` and a launch source.

### `pipeline review`

Example:

```bash
python run.py pipeline review --config trading/promoted/<paper_bundle>/<paper_bundle>.yaml --account paper --session-id <paper_session_id>
```

What it does:

1. Runs `session-replay`
2. Computes drift metrics such as Alpaca replay vs live and Mongo replay vs live
3. Evaluates `pipeline.gates.review`
4. If the gates pass, creates an approved live bundle
5. Logs the approved bundle into the pipeline MLflow experiment
6. Prints the replay MLflow URL, the approved local bundle path, and the approved MLflow bundle URL

### `pipeline live`

Examples:

```bash
python run.py pipeline live --config trading/promoted/<approved_bundle>/<approved_bundle>.yaml --account live --session-id <live_session_id>
python run.py pipeline live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account live --session-id <live_session_id>
```

This is the last stage. It launches from either:

- a local promoted bundle after a git pull
- an MLflow run URL that contains a reconstructable promoted bundle config

That dual launch path is intentional. It lets another node run the same approved artifact from local source control or directly from MLflow.

### Pipeline Config

The root `config.yaml` can define a dedicated pipeline experiment and promotion gates:

```yaml
pipeline:
  experiment_name: "Trading Pipeline Bundle Registry"
  artifact_location: null
  auto_promote_research: true
  gates:
    research:
      min_val_annualized_return: 0
      max_val_max_drawdown_pct: 25
      min_val_total_trades: 5
      min_wf_annualized_return: 0
      max_wf_max_drawdown_pct: 30
    review:
      max_alpaca_live_equity_drift_pct: 5
      max_mongo_live_equity_drift_pct: 5
```

That dedicated experiment is where candidate, paper, and approved bundle-registration runs are stored. Those runs contain manifests, promoted bundle files, and launch instructions, not strategy metrics.

## Debug REPL

The root [config.yaml](/E:/Programming/trading_guy/config.yaml) now controls whether `Ctrl-C` opens the debug REPL:

```yaml
debug_on_sigint: false
```

Set it to `true` when you want `Ctrl-C` to pause an engine-driven run and open the REPL instead of interrupting the process.

Available REPL commands include:

- `progress`: show engine progress, elapsed time, and estimated remaining work
- `loglevel`: print current root, console, and file log levels
- `loglevel DEBUG`: set console and file handlers to `DEBUG`
- `loglevel console INFO`: change only console verbosity
- `loglevel file DEBUG`: change only file logging verbosity
- `al`, `pf`, `om`: inspect algorithm, portfolio, and order manager state
- `dump ...`: write state snapshots or CSV histories to disk
- `c`: resume execution
- `q`: stop the engine

For interactive terminals, walk-forward modes also emit a live one-line status bar. When stdout is redirected or not attached to a TTY, the system falls back to normal line-based logging.

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

### Backtest a promoted session with fresh Alpaca bars

Use a normal backtest YAML when you want strategy parameters from a prior live
session but do not want to replay its stored MongoDB bars or opening state.

The `topology_promoted_from_space_search` profile copies the session's exact
topology parameters into a regular Alpaca backtest and enables a 2% trailing
stop with the original 5% profit target:

```bash
python run.py backtest --config configs/topology_promoted_from_space_search_trailing_stop_backtest.yaml --account secondary_paper3
```

The profile requests split-adjusted, market-hours-only UPRO minute bars from
June 3 through June 10, 2026. Its configured `end_date` is June 11 so the full
June 10 session is included. MongoDB is not accessed when this command runs.

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

## Walk-Forward Window HPO

Example:

```bash
python run.py walk-forward-hpo --config configs/example_walk_forward_hpo.yaml --account paper
```

Use this when you want to optimize the walk-forward schedule itself. The normal
`walk-forward` command treats these values as fixed:

```yaml
walk_forward:
  optimization_window_days: 180
  validation_window_days: 30
  trading_window_days: 30
```

`walk-forward-hpo` adds an outer Optuna search over those three values:

```yaml
walk_forward_window_hpo:
  num_samples: 8
  objective_metric: wf_annualized_return
  min_periods: 3
  search_space:
    optimization_window_days: { type: choice, values: [90, 120, 180, 252] }
    validation_window_days:   { type: choice, values: [20, 30, 45, 60] }
    trading_window_days:      { type: choice, values: [10, 20, 30] }
```

For each sampled window schedule, the command:

1. creates a candidate walk-forward config with the sampled window sizes
2. runs the normal historical walk-forward engine
3. lets each walk-forward period run the existing inner Ray Tune HPO over `walk_forward.search_space`
4. scores the candidate using `walk_forward_window_hpo.objective_metric`
5. records the candidate MLflow run id and window settings
6. reruns the winning window schedule into the permanent MLflow experiment

The objective metric can come from the aggregate walk-forward summary, such as:

- `wf_annualized_return`
- `wf_total_return_pct`
- `wf_sharpe_ratio`
- `wf_final_equity`

It can also refer to a metric attribute from the final performance metrics
object, such as `annualized_return`, if that is more convenient.

### Staging and Final MLflow Experiments

Candidate runs are intentionally logged to a staging experiment so the
permanent experiment only receives the rerun winner.

Key config:

```yaml
walk_forward_window_hpo:
  # Omit this for a unique temp experiment name per run.
  # staging_experiment_name: "wf_window_hpo_tmp_20260519"

  final_experiment_name: "walk_forward_runs"

  # Optional. Use a dedicated staging prefix if you enable direct S3 cleanup.
  # staging_artifact_location: "s3://your-mlflow-bucket/tmp/wf-window-hpo/unique-run-id"
  # final_artifact_location: "s3://your-mlflow-bucket/permanent/walk-forward"
```

If `staging_experiment_name` is omitted, the command creates a unique name like
`wf_window_hpo_tmp_<timestamp>_<group_id>`. This avoids collisions between
concurrent or repeated searches.

### Cleanup Behavior

Cleanup is controlled by these keys:

```yaml
walk_forward_window_hpo:
  cleanup_staging_experiment: true
  run_mlflow_gc: true
  cleanup_s3_prefix: false
```

Behavior:

- `cleanup_staging_experiment` marks the staging MLflow experiment deleted after the winner is rerun.
- `run_mlflow_gc` runs `mlflow gc --experiment-ids <id>` after the staging experiment is marked deleted.
- `cleanup_s3_prefix` runs `aws s3 rm <staging_artifact_location> --recursive`.

Important S3 note: MLflow experiment deletion by itself should be treated as a
metadata/lifecycle operation. If candidate artifacts are stored in S3 and you
need hard cleanup, use `run_mlflow_gc` when your MLflow setup can resolve
artifact URIs, or configure `cleanup_s3_prefix` with a dedicated staging prefix.
Never point `staging_artifact_location` at a shared or permanent artifact root
when direct S3 cleanup is enabled.

### Cost Model

This command is intentionally expensive. Total work is roughly:

```text
outer_window_trials * walk_forward_periods_per_candidate * inner_hpo_trials
```

For example, `8` outer trials, `10` walk-forward periods, and `30` inner
strategy HPO trials can mean around `2,400` strategy backtests before validation
and final continuous reruns. Start with small discrete `choice` lists and scale
up after the staging/final MLflow flow looks right.

## HPO

Example:

```bash
python run.py hpo --config configs/example_hpo.yaml --account paper --num-samples 50 --max-concurrent-trials 4
```

Use this when:
- you already know the local config you want to optimize
- you want to search parameter space directly
- you do not need to reconstruct the config from a previous MLflow run

## Split HPO

Example:

```bash
python run.py hpo-split --config configs/example_hpo_split.yaml --account paper --validation-period-days 30
```

What it does:

1. Uses all but the last `validation_period_days` of the configured date range as the HPO training window
2. Reserves the final `validation_period_days` as an out-of-sample validation window
3. Runs Ray Tune only on the training window
4. Selects the winning config using `hpo.objective_metric`
5. Re-runs the chosen config once on training and once on validation
6. Logs one MLflow run with:
   - training metrics prefixed `trn_`
   - validation metrics prefixed `val_`
   - validation artifacts prefixed `val_`

Supported `hpo.objective_metric` values:
- `val_annualized_return`: default, choose the winner by validation annualized return
- `trn_annualized_return`: choose the winner by training annualized return

Useful flags:

```bash
python run.py hpo-split --config configs/example_hpo_split.yaml --account paper --num-samples 25 --max-concurrent-trials 4
python run.py hpo-split --config configs/example_hpo_split.yaml --account paper --validation-period-days 45
```

Use this when you want one HPO pass plus a clean out-of-sample holdout without the rolling complexity of full walk-forward.

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

## Split HPO From MLflow

Example:

```bash
python run.py hpo-split-from-mlflow --account paper --run-url http://localhost:5000/#/experiments/1/runs/<run_id>
```

This follows the same reconstruction and edit flow as `hpo-from-mlflow`, then runs the `hpo-split` workflow. The edited YAML must include `hpo.validation_period_days`; the final slice of that many days is reserved for validation.

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

Direct MLflow launch is also supported now:

```bash
python run.py live --config http://localhost:5000/#/experiments/1/runs/<approved_bundle_run_id> --account paper --session-id live-20260513-c
```

That works because `--config` can reconstruct a runtime config from MLflow artifacts and logged params when the bundle run contains them.

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
