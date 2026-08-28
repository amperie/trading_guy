# Algo Crucible

The goal of the algo crucible is not to find impressive backtests. The goal is
to reject fragile strategies and promote only strategies whose behavior survives
out-of-sample validation, regime analysis, parameter perturbation, and live-like
execution checks.

Backtests are useful as a falsification tool. They are weak evidence when used
to select the highest-return parameter set.

## Core Principle

Optimize for survivability, not headline performance.

A strategy can be useful in two ways:

- **Generalist:** acceptable behavior across broad market conditions.
- **Specialist:** stable behavior in specific regimes, even if total-period
  performance is poor.

Do not discard an algorithm early just because aggregate performance is bad.
If it repeatedly performs well in an identifiable regime, keep it as a candidate
for a future regime-routed multi-algo strategy.

The crucible is a high-level orchestrator, not an execution engine. It should
coordinate existing repo components such as `WalkForwardEngine`,
`BacktestingEngine`, analysis tools, regime scoring, and Ray-backed HPO runners.

All expensive stages should run through Ray-backed runners:

- HPO tuning.
- Walk-forward validation.
- Out-of-sample backtests.
- Parameter plateau analysis.
- Start-date and execution-assumption perturbation tests.
- Confirmation backtests.

Serial execution should only be used for cheap setup, result aggregation, gate
evaluation, and artifact writing.

## Config Split

The crucible should use two YAML files:

- **Platform config:** reusable crucible policy and execution settings.
- **Workload config:** algorithm, portfolio, data, and search-space definition
  for one strategy family.

This prevents every strategy config from carrying a large copy of platform
thresholds, plateau rules, Ray settings, and promotion policy.

Example CLI shape:

```bash
uv run python run.py crucible \
  --platform-config configs/crucible/platform_default.yaml \
  --workload-config configs/crucible/workloads/spy_macd_switch.yaml
```

The orchestrator should load both files, validate them separately, then combine
them into one resolved run config. Workload config can override explicitly
allowed platform fields, but broad platform policy should not be silently
mutable from workload YAML.

## Platform Config

Platform config controls how the crucible runs and how candidates are judged.
It should be reusable across many algorithms.

Example:

```yaml
crucible:
  name: default_crucible
  run_name: spy_macd_switch_v1
  output_dir: scratch/crucible_runs

ray:
  address: null
  max_concurrent_trials: 8
  num_cpus_per_job: 1
  memory_per_job_gb: 2
  log_worker_output: false

logging:
  level: INFO
  structured_events: true
  log_stage_transitions: true
  log_job_lifecycle: true
  log_gate_decisions: true
  log_every_n_completed_jobs: 25
  suppress_per_bar_logs: true
  persist_logs_to_mlflow: true

budgets:
  max_total_jobs: 5000
  max_stage_jobs: 1000
  max_runtime_hours: 12
  max_detail_runs: 200
  stop_stage_if_failure_rate_above: 0.80

walk_forward:
  optimization_window_days: 756
  validation_window_days: 252
  embargo_days: 20
  min_windows: 5

hpo:
  num_samples: 300
  objective_metric: "annualized_return"
  top_seed_count: 12
  diverse_seed_count: 6
  min_peak_distance: 0.18

gates:
  generalist:
    min_trades: 50
    min_median_oos_return: 0.0
    min_profitable_windows_pct: 0.55
    max_drawdown: 0.25
    max_train_to_validation_degradation: 0.75
  specialist:
    min_regime_trades: 30
    min_regime_windows: 3
    min_regime_median_oos_return: 0.0
    min_regime_profitable_windows_pct: 0.60
    max_regime_drawdown: 0.18
  defensive:
    max_drawdown_reduction_required: 0.10
    max_return_drag: 0.05

plateau:
  local_search_radius_pct: 0.30
  narrowed_search_radius_pct: 0.12
  min_neighbor_trials: 40
  min_neighbor_pass_rate: 0.60
  min_worst_quartile_oos_return: -0.03
  max_peak_to_median_degradation: 0.50
  max_trade_count_cv: 0.75
  max_drawdown_p95: 0.20

perturbations:
  start_date_offsets_days: [-60, -30, 0, 30, 60]
  training_window_multipliers: [0.75, 1.0, 1.25]
  validation_window_multipliers: [0.75, 1.0, 1.25]
  embargo_day_values: [5, 20, 40]
  slippage_multipliers: [1.0, 1.5, 2.0]
  cost_multipliers: [1.0, 2.0]
  entry_delay_bars: [0, 1]
  exit_delay_bars: [0, 1]

confirmation:
  enabled: true
  untouched_period_required: true
  start_date: null
  end_date: null
  min_return_pct: 0.0
  max_drawdown_pct: 25.0
  min_trades: 0

promotion:
  write_json: true
  write_yaml: true
  write_markdown: true
  log_to_mlflow: true
  parent_run_summary_metrics: true
  link_detail_runs: true
  create_promoted_folder: false
  output_dir: trading/promoted

mlflow:
  parent_experiment_name: "Algo Crucible"
  detail_experiment_strategy: per_crucible_run
  detail_experiment_name_template: "Algo Crucible Details/{workload_name}/{run_id}"
  full_detail_run_policy: survivors
  max_parent_artifact_table_rows: 5000

resume:
  enabled: true
  system_of_record: mlflow
  local_cache_dir: scratch/crucible_runs
  rerun_failed_jobs: true
  reuse_completed_jobs: true

state_store:
  backend: mlflow
  allow_backends:
    - mlflow
    - mongodb
    - composite
```

## Workload Config

Workload config defines what is being tested.

Example:

```yaml
workload:
  name: spy_macd_switch
  run_name: spy_macd_switch_v1
  description: "SPY MACD regime switch between UPRO and SPXU."

data_provider:
  provider: "trading.data_providers.alpaca_data_provider.AlpacaDataProvider"
  symbols: ["SPY", "UPRO", "SPXU"]
  timeframe: "Minute"
  start_date: "2016-01-01"
  end_date: "2026-01-01"

order_manager:
  order_manager: "trading.core.om.backtesting_om.BacktestingOM"

algorithm:
  algorithm: "trading.algorithms.spy_trend_macd_algorithm.SpyTrendMACDAlgorithm"
  params:
    spy_symbol: "SPY"
    upro_symbol: "UPRO"
    spxu_symbol: "SPXU"
    history_length: 5000
    market_regime:
      enabled: true
      trend_lookback_days: 50
      trend_threshold: 0.03
      baseline_ma_window_days: 200
      volatility_lookback_days: 50
      volatility_percentile_window_days: 252
      drawdown_lookback_days: 252
      annualization_days: 252
      market_hours_per_day: 6.5

portfolio:
  portfolio: "trading.core.pf.dual_symbol_switch_portfolio.DualSymbolSwitchPortfolio"
  params:
    cash: 100000
    keep_history: true
    symbol: "UPRO"

search_space:
  algorithm_param_keys:
    - macd_fast_period
    - macd_slow_period
    - macd_signal_period
    - strength_scale
  portfolio_param_keys:
    - stop_pct
    - profit_pct
  space:
    macd_fast_period: {type: randint, low: 5, high: 300}
    macd_slow_period: {type: randint, low: 20, high: 700}
    macd_signal_period: {type: randint, low: 3, high: 200}
    strength_scale: {type: uniform, low: 5.0, high: 30.0}
    stop_pct: {type: uniform, low: 1.0, high: 20.0}
    profit_pct: {type: uniform, low: 1.0, high: 25.0}

fixed_assumptions:
  starting_cash: 100000
  transaction_cost: 0.0
  slippage_bps: 5
  benchmark_symbol: "SPY"

hpo:
  objective:
    metric: composite_v1
    weights:
      annualized_return: 1.0
      max_drawdown_pct: -1.5
      volatility: -0.25
      sortino_ratio: 5.0
      calmar_ratio: 2.0
    gates:
      min_trades: 20
      max_drawdown_pct: 25
    low_trade_penalty: 1000
    drawdown_gate_penalty: 100

objectives:
  hpo_objective_metric: "composite_v1"
  generalist_gate_metric: "median_oos_return"
  specialist_gate_metric: "regime_median_oos_return"
  plateau_metric: "median_oos_return"
  perturbation_survival_metric: "pass_rate"
  confirmation_metric: "median_oos_return"
```

## Inputs

Each crucible run should define:

- Algorithm class.
- Portfolio class.
- Data provider config.
- Tunable algorithm parameters.
- Tunable portfolio parameters.
- Regime detector config.
- Fixed assumptions:
  - symbols
  - timeframe
  - warmup bars
  - embargo days
  - transaction costs
  - slippage model
  - starting cash
  - benchmark, if applicable

The algorithm and portfolio configs should be treated as frozen artifacts once a
candidate reaches final confirmation or paper trading.

## Immutable Named Runs

Each crucible execution should have a human-readable `run_name` in config.

The actual run identity should combine that name with the hash of the fully
resolved config:

```text
crucible_run_id = <run_name>_<resolved_config_hash_16>
```

Example:

```text
spy_macd_switch_v1_8f31c2a91b7d4e22
```

When a crucible run starts:

1. Load platform config.
2. Load workload config.
3. Validate both.
4. Resolve them into one config.
5. Compute `resolved_config_hash`.
6. Create `crucible_run_id`.
7. Persist `resolved_config.yaml`.
8. Treat that resolved config as immutable.

After kickoff, the stored resolved config is the config for that run. Local YAML
files may change later, but they do not mutate the existing run.

### Dedup Rules

On startup, the state store should search by `run_name` and
`resolved_config_hash`.

Rules:

- Same `run_name` + same `resolved_config_hash` + incomplete run: resume it.
- Same `run_name` + same `resolved_config_hash` + complete run: refuse by
  default; require `--rerun`.
- Same `run_name` + different `resolved_config_hash`: fail fast and require a
  new run name, such as `spy_macd_switch_v2`.
- New `run_name`: create a new run.

This prevents accidental pollution where a familiar run name points to multiple
different policies, thresholds, search spaces, or data ranges.

Pseudocode:

```python
existing_exact = state_store.find_run({
    "run_name": run_name,
    "resolved_config_hash": resolved_config_hash,
})

if existing_exact and existing_exact.status != "complete":
    resume(existing_exact)
elif existing_exact:
    raise AlreadyComplete("Use --rerun to create a new attempt.")
elif state_store.find_run({"run_name": run_name}):
    raise ConfigChangedForRunName("Use a new run_name.")
else:
    create_new_run()
```

Recommended MLflow tags:

```text
crucible.run_name
crucible.run_id
crucible.resolved_config_hash
crucible.status
```

## Identity and Fingerprints

The crucible needs deterministic identity for both runs and candidates.

### Data Fingerprint

Every run should record a data fingerprint:

```yaml
data_fingerprint:
  provider: "trading.data_providers.alpaca_data_provider.AlpacaDataProvider"
  symbols: ["SPY", "UPRO", "SPXU"]
  timeframe: "Minute"
  start_date: "2016-01-01"
  end_date: "2026-01-01"
  row_count: 982345
  min_timestamp: "2016-01-04T09:30:00"
  max_timestamp: "2025-12-31T16:00:00"
  source_hash: "sha256:..."
  vendor: "alpaca"
```

For local files, include file hashes. For API-backed data, include provider,
symbols, timeframe, date range, row counts, timestamp bounds, and any available
vendor/source metadata. Repeated or resumed runs should warn if the data
fingerprint changes.

### Candidate ID

Each candidate should have a deterministic ID:

```text
candidate_id = candidate_<hash16>
```

The hash should include:

- Algorithm class.
- Portfolio class.
- Resolved algorithm params.
- Resolved portfolio params.
- Workload config hash.
- Regime config hash.
- Code version metadata when available.

Candidate IDs are used by Ray jobs, MLflow links, plateau neighborhoods,
perturbation results, promotion packets, and the promotion registry.

### Regime Label Versioning

Regime scoring is only comparable when the regime definition is stable.

Track:

```text
regime_label_version
regime_config_hash
regime_detector_code_hash
```

If the regime detector config or code changes, old regime scorecards should be
treated as a different labeling system.

## Objectives and Metrics

Do not use one metric everywhere.

Separate:

- HPO objective metric.
- Generalist gate metric.
- Specialist gate metric.
- Plateau metric.
- Perturbation survival metric.
- Confirmation metric.

If every stage optimizes annualized return, the crucible will still overfit.
Each stage should optimize or gate on the metric appropriate to that stage.
The shared Ray HPO launcher still defaults to `annualized_return` for backward
compatibility. Crucible workloads should opt into `composite_v1`, which scores:

```text
annualized_return
- drawdown_weight * abs(max_drawdown_pct)
- volatility_weight * volatility
+ sortino_weight * sortino_ratio
+ calmar_weight * calmar_ratio
- configured gate penalties
```

## Failure Taxonomy

Rejected candidates should record structured failure reasons:

```yaml
failure_reasons:
  - insufficient_trades
  - unstable_parameters
  - no_regime_repeatability
  - drawdown_too_high
  - perturbation_failed
  - confirmation_failed
```

Use stable reason codes so rejected candidates can be analyzed later across many
crucible runs.

Suggested reason codes:

- `insufficient_trades`
- `insufficient_windows`
- `negative_median_oos_return`
- `low_profitable_windows_pct`
- `drawdown_too_high`
- `train_to_validation_degradation_too_high`
- `no_regime_repeatability`
- `unstable_parameters`
- `no_plateau`
- `plateau_pass_rate_too_low`
- `scenario_perturbation_failed`
- `cost_slippage_fragile`
- `execution_delay_fragile`
- `confirmation_failed`
- `paper_replay_mismatch`
- `operational_failure`

## Budgets and Resource Controls

The crucible can explode combinatorially. Platform config should define hard
budgets:

```yaml
budgets:
  max_total_jobs: 5000
  max_stage_jobs: 1000
  max_runtime_hours: 12
  max_detail_runs: 200
  stop_stage_if_failure_rate_above: 0.80
```

Ray resource controls should also be explicit:

```yaml
ray:
  num_cpus_per_job: 1
  memory_per_job_gb: 2
  max_concurrent_trials: 8
```

Budget breaches should fail cleanly and leave resumable state behind.

## Leakage Guards

Leakage checks should be validators, not conventions.

Required guards:

- Validation starts after train end plus embargo.
- Embargo must be compatible with max lookback and warmup requirements.
- Regime labels for bar `T` cannot use data after `T`.
- Warmup data cannot include validation bars in a way that influences training
  selection.
- Confirmation data cannot be used before the candidate is frozen.
- Paper replay cannot tune or rewrite the frozen strategy.

If a leakage guard fails, the crucible should fail the stage before launching
Ray jobs.

## Promotion Registry

Promotion packets should feed the existing promoted-strategy pattern:

```text
trading/promoted/
  <candidate_id>/
    promotion_manifest.json
    frozen_config.yaml
    evidence/
      promotion_packet.json
      promotion_packet.yaml
      promotion_packet.md
```

Generated algorithm or portfolio source files should be copied into the promoted
folder only when the candidate depends on generated code. Native repo classes
should be referenced by dotted class path and code version metadata.

## Human Approval

The crucible can recommend promotion, but it should not silently start paper
trading.

Before paper trading:

- Require a human approval flag.
- Freeze the candidate config.
- Write the promotion packet.
- Record approval metadata in MLflow and the promotion manifest.

## Operational Kill Switches

Paper and live runs need operational brakes even after a candidate passes the
crucible.

Suggested kill switch rules:

- Max daily loss.
- Max total drawdown.
- Unexpected regime behavior.
- Live-vs-replay signal mismatch.
- Broker/order sync errors.
- Data feed gaps.
- Order rejection rate.
- Position reconciliation failure.

Kill switch breaches should be logged and included in paper-trading quality
gate evidence.

## Logging Requirements

The crucible should have enough logging to debug long distributed runs without
turning logs into per-bar noise.

Required logging:

- Run creation and resume decisions.
- Config hashes and resolved run ID.
- Data fingerprint creation and mismatch warnings.
- Stage start, completion, failure, skip, and resume.
- Ray job submission counts.
- Ray job completion counts.
- Ray job failures with job ID, stage, candidate ID, and reason.
- Gate decisions with candidate ID, candidate type, pass/fail, and reason codes.
- Plateau seed selection and rejection reasons.
- Perturbation failure summaries.
- Confirmation and promotion decisions.
- MLflow artifact upload failures.
- State-store read/write failures.

Avoid:

- Per-bar logs during backtests.
- Full config dumps in every worker log.
- Full metric payloads for every failed candidate.
- Repeated unchanged progress messages.

Recommended structured event shape:

```json
{
  "event": "gate_decision",
  "crucible_run_id": "spy_macd_switch_v1_8f31c2a91b7d4e22",
  "stage": "03_regime_gate",
  "candidate_id": "candidate_abc123",
  "candidate_type": "specialist",
  "passed": true,
  "reason_codes": ["regime_repeatability_passed"],
  "timestamp": "2026-08-27T12:34:56Z"
}
```

Driver logs should focus on orchestration:

- Current stage.
- Jobs submitted.
- Jobs completed / failed / reused.
- ETA when available.
- Gate summaries.
- Next action.

Worker logs should focus on one job:

- Job start.
- Input IDs and hashes.
- Backtest or validation completion.
- Compact metric summary.
- Artifact/result write.
- Failure traceback if the job fails.

Progress logging should be throttled:

```yaml
logging:
  log_every_n_completed_jobs: 25
```

The MLflow parent run should include log artifacts or links when
`persist_logs_to_mlflow` is true:

```text
logs/driver.log
logs/stage_02_hpo_walk_forward.log
logs/failed_jobs/<job_id>.log
```

For large runs, only persist detailed worker logs for failed jobs and retained
detail runs. Compact job results should carry enough summary information for
normal inspection.

## Stage 1: Historical Regime Labeling

Before judging strategy performance, label all historical bars or validation
windows by market regime using only data available up to that point.

Regime configuration should be expressed in market hours or trading days, not
raw bar counts. The detector is responsible for inferring the incoming bar
granularity from timestamps and converting durations into bar windows.

Example:

```yaml
market_regime:
  enabled: true
  trend_lookback_days: 50
  trend_threshold: 0.03
  baseline_ma_window_days: 200
  volatility_lookback_days: 50
  volatility_percentile_window_days: 252
  drawdown_lookback_days: 252
  annualization_days: 252
  market_hours_per_day: 6.5
```

At minimum, track:

- Trend regime:
  - `UPTREND`
  - `DOWNTREND`
  - `RANGE`
  - `UNKNOWN`
- Volatility regime:
  - `LOW_VOL`
  - `NORMAL_VOL`
  - `HIGH_VOL`
  - `UNKNOWN`
- Drawdown state.
- Distance from long-term baseline.

Regime labels must be online-safe. The label for bar `T` cannot use any data
after bar `T`.

## Stage 2: Quick Sanity Check

Run parallel hyperparameter tuning across multiple walk-forward windows.

For each trial:

1. Train or optimize on past data.
2. Apply an embargo gap.
3. Validate on future unseen data.
4. Repeat across many windows.
5. Score both total validation performance and regime-specific validation
   performance.

This stage should be cheap, fast, and harsh. Most candidates should fail here.

Implementation note: this stage should use the existing Ray Tune path, currently
centered around `trading.launchers.run_backtest_ray.tune_backtest_hyperparameters`
and the split-HPO validation helpers in `trading.commands.hpo`. The crucible
should wrap those runners and collect all trial summaries, not just the best
config, because regime scoring and specialist detection need the full candidate
distribution.

## Stage 3: Regime-Aware Quality Gate

Do not reject solely on total performance.

Advance a candidate if either condition is true:

- It performs acceptably overall.
- It performs consistently well in one or more identifiable regimes.

Example:

- Algo A loses overall but works in `RANGE_LOW_VOL`.
- Algo B works in `UPTREND_NORMAL_VOL`.
- Algo C is useful in `DOWNTREND_HIGH_VOL` because it protects capital.

Those are all potentially useful.

## Regime Scorecard

For every algorithm and parameter candidate, calculate metrics by regime:

- Return.
- Sharpe or Sortino.
- Max drawdown.
- Trade count.
- Win rate.
- Average trade PnL.
- Average win / average loss.
- Exposure.
- Turnover.
- Percent of profitable windows.
- Worst 10th percentile window result.
- Train-to-validation degradation.
- Consistency across windows within the same regime.

The key question is:

> Does this algorithm repeatedly behave well in the same regime across different
> time windows?

Best-regime return alone is not enough. A useful specialist needs repeatability.

## Specialist Tagging

Candidates that pass regime-specific gates should be tagged explicitly:

```yaml
candidate_type: specialist
specialist_regimes:
  - RANGE_LOW_VOL
failure_regimes:
  - DOWNTREND_HIGH_VOL
notes: "Mean-reversion candidate. Only eligible when range and low-vol filters agree."
```

Possible candidate types:

- `generalist`: acceptable across broad market conditions.
- `specialist`: useful only in specific regimes.
- `defensive`: useful because it reduces drawdown or avoids hostile conditions.
- `reject`: insufficient evidence or too fragile.

## Stage 4: Parameter Stability

For candidates that pass the early gate, perturb parameters around the best
regions.

The goal is not to find the best point. The goal is to find a plateau.

Good signs:

- Nearby parameters behave similarly.
- Performance surface is smooth enough.
- Drawdown remains sane.
- Trade count remains sufficient.
- Regime-specific edge persists across nearby settings.

Bad signs:

- One tiny magic parameter island.
- Performance collapses with small changes.
- Trade count changes wildly.
- The target-regime edge disappears outside one exact setting.

Parameter stability should be evaluated both overall and within each promising
regime.

Implementation note: plateau analysis should submit neighborhood backtests as
Ray tasks. The orchestrator should generate the local parameter grid or sample,
then run those candidate configs in parallel using the same backtest runner used
by HPO validation.

## Stage 5: Regime-Specific Plateau Gate

Advance only if the candidate has a real plateau in its relevant evaluation
scope:

- Generalists need broad stability.
- Specialists need stability inside their target regimes.
- Defensive candidates need stable drawdown protection or loss avoidance.

Reject candidates that only work at a brittle parameter point.

## Stage 6: Scenario Perturbation

Stress the setup itself, not just the strategy parameters.

Perturb:

- Start dates.
- End dates.
- Training window sizes.
- Validation window sizes.
- Embargo lengths.
- Warmup lengths.
- Transaction costs.
- Slippage assumptions.
- Entry timing.
- Exit timing.
- Skipped trades.
- Worse fills during high-volatility bars.

Score both:

- Total behavior.
- Target-regime behavior.

Implementation note: each scenario is independent and should be a Ray task. A
scenario task should receive a frozen candidate config plus one perturbation
patch, run the appropriate backtest or walk-forward validation, then return a
small scorecard. The driver process should avoid loading full portfolio histories
from every worker unless detailed artifacts are explicitly requested.

## Stage 7: Out-of-Band Quality Gate

Promote only if performance survives perturbations outside the exact tuning
setup.

Minimum promotion evidence should include:

- Stable performance across date perturbations.
- Survival under higher costs and slippage.
- Survival under one-bar delayed entries and exits.
- No hidden catastrophic regime dependency.
- Sufficient trades to make the evidence meaningful.
- Failure modes are understood and documented.

## Stage 8: Confirmation

Use a final untouched historical range only after the candidate is frozen.

Once this data is inspected, it is burned. Do not tune after seeing confirmation
results.

Confirmation should verify:

- Overall behavior, if the candidate is a generalist.
- Target-regime behavior, if the candidate is a specialist.
- Defensive behavior, if the candidate is defensive.
- Signal and order behavior matches expectations.

Implementation note: confirmation is still a backtest job and should go through
the Ray backtest runner for consistency, even if only one candidate remains.

## Stage 9: Promote to Paper Trading

Before paper trading, freeze:

- Algorithm code.
- Portfolio code.
- Parameters.
- Regime eligibility rules.
- Data source.
- Timeframe.
- Cost and slippage assumptions.
- Expected performance envelope.

Paper trading is not another optimization loop. It is an implementation and
forward-behavior test.

## Stage 10: Paper Trading Quality Gate

Paper trading passes only if live behavior is consistent with the frozen model.

Check:

- Live regime labels are produced online and match replayed labels.
- The algo trades only in approved regimes, if it is a specialist.
- Live signals match signals from replaying the same bars through a backtest.
- Intended orders match.
- Fill differences are explainable by the slippage model.
- Performance is within the expected regime-specific distribution.
- No operational issues appear in data ingestion, order submission, state
  persistence, warmup, or broker sync.

If paper bars replayed as a backtest would have produced materially different
signals or intended orders, the candidate fails until the mismatch is explained.

## Recommended Outputs

Each crucible run should produce a promotion packet:

- Candidate config.
- Candidate type.
- Specialist regimes, if any.
- Failure regimes.
- Walk-forward scorecard.
- Regime scorecard.
- Parameter plateau summary.
- Scenario perturbation summary.
- Confirmation result.
- Paper trading expectations.
- Promotion decision.

Example decision:

```yaml
decision: promote_to_paper
candidate_type: specialist
specialist_regimes:
  - RANGE_LOW_VOL
quality_gates_passed:
  - walk_forward_oos
  - regime_scorecard
  - parameter_plateau
  - scenario_perturbation
  - confirmation
primary_risks:
  - "Loses money in persistent high-volatility downtrends."
  - "Requires regime router to disable trading outside approved regimes."
```

## Implementation Shape

The crucible workflow lives outside `trading.engines`:

```text
algo_crucible/
    __init__.py
    orchestrator.py
    config.py
    models.py
    gates.py
    jobs.py
    scoring.py
    plateau.py
    perturbations.py
    confirmation.py
    state_store.py
```

Suggested responsibility split:

- `orchestrator.py`: coordinates stages and gates.
- `config.py`: validates crucible config.
- `models.py`: stores candidate, window, regime, plateau, perturbation, and
  promotion result objects.
- `gates.py`: applies generalist, specialist, defensive, plateau, perturbation,
  and paper-trading gates.
- `jobs.py`: contains local/Ray wrappers for independent work units.
- `scoring.py`: builds total and per-regime scorecards.
- `plateau.py`: generates parameter neighborhoods and summarizes plateaus.
- `perturbations.py`: generates scenario patches and summarizes robustness.
- `confirmation.py`: runs untouched confirmation and writes
  JSON/YAML/Markdown promotion packets.
- `state_store.py`: defines the `CrucibleStateStore` interface plus concrete
  implementations such as `MlflowCrucibleStateStore`, future
  `MongoCrucibleStateStore`, and optional `CompositeCrucibleStateStore`.

The orchestrator should look roughly like this:

```python
class AlgoCrucible:
    def __init__(self, cfg):
        self.cfg = AlgoCrucibleConfig.model_validate(cfg)

    def run(self):
        regime_labels = self.label_regimes()

        hpo_trials = self.run_ray_hpo_and_walk_forward(regime_labels)
        candidates = self.score_candidates_by_regime(hpo_trials, regime_labels)
        candidates = self.gates.regime_aware_filter(candidates)

        plateau_jobs = self.build_plateau_jobs(candidates)
        plateau_results = self.ray_runner.run_backtests(plateau_jobs)
        candidates = self.gates.plateau_filter(candidates, plateau_results)

        perturbation_jobs = self.build_perturbation_jobs(candidates)
        perturbation_results = self.ray_runner.run_backtests(perturbation_jobs)
        candidates = self.gates.perturbation_filter(candidates, perturbation_results)

        confirmation_jobs = self.build_confirmation_jobs(candidates)
        confirmation_results = self.ray_runner.run_backtests(confirmation_jobs)

        return self.promotion_builder.build(candidates, confirmation_results)
```

The Ray layer should expose coarse independent jobs, not tiny per-bar work:

```python
@ray.remote
def run_crucible_backtest_job(job: dict) -> dict:
    # Instantiate data provider, algorithm, portfolio, and order manager.
    # Run BacktestingEngine.
    # Score overall and per-regime results.
    # Return compact metrics plus artifact references.
    ...
```

The driver process should handle:

- Config parsing.
- Search-space expansion.
- Job submission.
- `ray.wait` / `ray.get` coordination.
- Gate decisions.
- Promotion packet generation.

The worker process should handle:

- One expensive backtest or validation run.
- Local analysis for that run.
- Compact scorecard return.
- Optional MLflow artifact logging.

This keeps Ray overhead low and avoids sending large tick histories back to the
driver unnecessarily.

## Resumability

Assume crucible runs will fail mid-run. Long HPO, plateau, perturbation, and
confirmation batches must be resumable.

Use a `CrucibleStateStore` abstraction for checkpointing and resume. MLflow
should be the first implementation because it is already useful for inspection,
lineage, artifacts, and review, but the orchestrator should not hard-code MLflow
APIs throughout the workflow.

MLflow can be the visible system of record for crucible state so runs can resume
from different hosts. This requires a shared MLflow tracking server with:

- A database backend store, preferably PostgreSQL or MySQL for concurrent
  multi-host writes.
- A remote artifact store, such as S3, Azure Blob Storage, GCS, MinIO, or NFS.
- Stable artifact access from every host that may resume the run.

A local run directory should still exist, but it is a cache and staging area,
not the canonical source of truth.

Long term, MongoDB may be better for operational job state, locking, and
high-frequency status updates. A composite backend may eventually use MongoDB
for operational resume state and MLflow for UI, metrics, lineage, and promotion
artifacts.

The orchestrator should persist the same logical structure to MLflow artifacts
and mirror it locally:

```text
scratch/crucible_runs/
  20260827T120000_spy_macd_switch/
    run_manifest.yaml
    resolved_config.yaml
    stages/
      01_regime_labeling/
        stage_state.json
        result_summary.json
      02_hpo_walk_forward/
        stage_state.json
        jobs.jsonl
        results/
          <job_id>.json
      03_regime_gate/
        stage_state.json
        candidates.json
      04_plateau/
        stage_state.json
        jobs.jsonl
        results/
          <job_id>.json
      05_perturbations/
        stage_state.json
        jobs.jsonl
        results/
          <job_id>.json
      06_confirmation/
        stage_state.json
        jobs.jsonl
        results/
          <job_id>.json
    promotion_packet.json
    promotion_packet.yaml
    promotion_packet.md
```

### Run Identity

Each crucible run should have a stable `run_id`.

Default:

- New run: `<run_name>_<resolved_config_hash_16>`.
- Resume run: exact `run_id`, or matching `run_name` and config hash.

Example:

```bash
uv run python run.py crucible \
  --platform-config configs/crucible/platform_default.yaml \
  --workload-config configs/crucible/workloads/spy_macd_switch.yaml \
  --resume-dir scratch/crucible_runs/20260827T120000_spy_macd_switch
```

The resolved platform + workload config should be written once at run creation.
On resume, the orchestrator should load `resolved_config.yaml` from MLflow by
default. This prevents accidental continuation with a changed YAML.

If the user wants to resume with changed config, require an explicit override
such as `--rerun` with a new run ID. Do not mutate an existing run's resolved
config.

### State Store Interface

The orchestrator should talk to a small interface rather than directly to
MLflow, MongoDB, or local files:

```python
class CrucibleStateStore:
    def create_run(self, manifest: dict) -> str: ...
    def load_run(self, run_id: str) -> dict: ...
    def find_run(self, query: dict) -> dict | None: ...

    def write_stage_state(self, run_id: str, stage: str, state: dict) -> None: ...
    def read_stage_state(self, run_id: str, stage: str) -> dict | None: ...

    def write_job_manifest(self, run_id: str, stage: str, jobs: list[dict]) -> None: ...
    def read_job_manifest(self, run_id: str, stage: str) -> list[dict]: ...

    def write_job_result(self, run_id: str, stage: str, job_id: str, result: dict) -> None: ...
    def read_job_result(self, run_id: str, stage: str, job_id: str) -> dict | None: ...
    def list_job_results(self, run_id: str, stage: str) -> dict[str, dict]: ...

    def write_promotion_packet(self, run_id: str, packet: dict) -> None: ...
```

First implementation:

```text
MlflowCrucibleStateStore
```

Likely later implementations:

```text
MongoCrucibleStateStore
CompositeCrucibleStateStore
```

Backend roles:

- `MlflowCrucibleStateStore`: best for UI inspection, experiment lineage,
  metrics, artifacts, and human review.
- `MongoCrucibleStateStore`: better for operational job records, querying,
  locking, partial updates, and high-concurrency resume.
- `CompositeCrucibleStateStore`: MongoDB as operational truth, MLflow as the
  review and artifact layer.

Start with MLflow, but keep the abstraction from the first implementation so the
crucible is not trapped inside MLflow-specific behavior.

### Stage State

Every stage should have a small state file:

```json
{
  "stage": "04_plateau",
  "status": "running",
  "started_at": "2026-08-27T12:10:00Z",
  "completed_at": null,
  "input_hash": "sha256:...",
  "job_count": 240,
  "completed_jobs": 181,
  "failed_jobs": 3
}
```

Allowed stage statuses:

- `pending`
- `running`
- `complete`
- `failed`
- `skipped`

On resume:

1. Load run manifest.
2. Load each stage state in order.
3. Skip completed stages whose input hash still matches.
4. For running or failed stages, inspect job results and submit only missing or
   failed jobs according to resume policy.
5. Recompute downstream stages if an upstream stage was rerun and changed.

### Deterministic Job IDs

Each independent Ray job must have a deterministic `job_id` based on the job
contents.

Example:

```python
def make_job_id(stage: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{stage}_{digest}"
```

The job payload should include every input that affects the result:

- Candidate config.
- Data date range.
- Warmup range.
- Walk-forward window.
- Regime config version.
- Cost/slippage assumptions.
- Perturbation patch.
- Code version metadata, if available.

It should not include unstable fields such as wall-clock timestamps.

### Idempotent Job Results

Each Ray worker should write its result to local temp storage first, then upload
the completed result artifact to MLflow:

```text
stages/<stage>/results/<job_id>.json
```

The worker should be idempotent:

1. Check MLflow for `stages/<stage>/results/<job_id>.json`.
2. If it exists and validates, return the existing result.
3. Otherwise write a local `<job_id>.json.tmp`.
4. Atomically rename the temp file to `<job_id>.json`.
5. Upload the completed artifact to MLflow.

This prevents corrupted partial results when a process dies during write.

Returned job results should be compact:

```json
{
  "job_id": "plateau_abc123",
  "status": "complete",
  "candidate_id": "candidate_007",
  "metrics": {},
  "regime_metrics": {},
  "artifact_refs": {},
  "error": null
}
```

Do not send full tick history or full portfolio history back to the driver by
default. Store large artifacts separately and return references.

### Job Manifests

Before submitting Ray jobs for a stage, write `jobs.jsonl` locally and upload it
to MLflow:

```jsonl
{"job_id":"plateau_001","status":"pending","payload":{...}}
{"job_id":"plateau_002","status":"pending","payload":{...}}
```

On resume, load `jobs.jsonl` and existing result artifacts from MLflow.

Submit jobs when:

- no result file exists
- result file is invalid
- result file has `"status": "failed"` and `rerun_failed_jobs` is true

Skip jobs when:

- result file exists
- result file validates
- result status is `complete`
- `reuse_completed_jobs` is true

### Driver State

The driver should flush progress after every completed Ray job or small batch of
jobs. A long run should never depend on in-memory state only.

Recommended orchestration loop:

```python
pending = submit_missing_jobs(stage)
while pending:
    ready, pending = ray.wait(pending, num_returns=1, timeout=30)
    for ref in ready:
        result = ray.get(ref)
        state_store.record_job_result(stage, result)
    state_store.write_stage_state(stage)
```

If the driver crashes, already completed worker result artifacts remain usable
from any host.

### MLflow System of Record

MLflow should store:

- Parent crucible run.
- Parent-level summary metrics for every stage.
- Stage state artifacts.
- Job manifests.
- Compact job result artifacts.
- Promotion packets.
- Tags for `crucible_run_id`, workload name, platform config hash, workload
  config hash, resolved config hash, git commit, and run status.

Recommended MLflow structure:

```text
Experiment: Algo Crucible
  Parent run: crucible/<workload>/<run_id>
    metrics:
      hpo.completed_trials
      hpo.candidates_after_regime_gate
      plateau.completed_jobs
      plateau.pass_rate
      perturbation.completed_jobs
      perturbation.pass_rate
      confirmation.promoted_candidates
    artifacts:
      run_manifest.yaml
      resolved_config.yaml
      stages/...
      promotion_packet.*
      summaries/...
    nested runs:
      stage/02_hpo_walk_forward
      stage/04_plateau
      job/<job_id>
```

The parent run is the canonical index. Nested runs are useful for UI filtering
and metrics, but artifacts under the parent run should be sufficient to resume.

Do not rely on MLflow params for mutable state. Params are best treated as
immutable run descriptors. Use tags for small mutable status fields and artifacts
for durable structured state.

Ray workers should not independently mutate broad shared state. They may write
deterministic job results through the state store, but stage transitions, gate
decisions, invalidation, and promotion decisions should stay in the driver.

### Interpretable Parent Run

The parent MLflow run will become large, so it must work as an index and summary,
not as a dumping ground.

The parent run should answer these questions from the MLflow UI without opening
hundreds of artifacts:

- What workload ran?
- Which stage is current or failed?
- How many candidates entered and exited each stage?
- How many jobs completed, failed, or were reused from cache?
- Which candidates survived?
- Which regimes produced specialists?
- Which detailed HPO/backtest runs should be inspected?
- What is the final promotion decision?

Recommended parent-run tags:

```text
crucible_run_id
workload_name
status
current_stage
candidate_count.initial
candidate_count.after_regime_gate
candidate_count.after_plateau
candidate_count.after_perturbation
candidate_count.promoted
specialist_regimes
promoted_candidate_ids
detail_experiment_name
platform_config_hash
workload_config_hash
resolved_config_hash
git_commit
```

Recommended parent-run metrics:

```text
hpo.trials_total
hpo.trials_complete
hpo.trials_failed
hpo.best_metric
hpo.median_metric
hpo.candidates_after_gate

regime_gate.generalists
regime_gate.specialists
regime_gate.defensive
regime_gate.rejected

plateau.jobs_total
plateau.jobs_complete
plateau.jobs_failed
plateau.best_pass_rate
plateau.median_pass_rate
plateau.best_score
plateau.best_peak_to_median_degradation
plateau.accepted_plateaus
plateau.rejected_peaks
plateau.candidates_after_gate

perturbation.jobs_total
perturbation.jobs_complete
perturbation.jobs_failed
perturbation.best_pass_rate
perturbation.median_pass_rate
perturbation.candidates_after_gate

confirmation.jobs_total
confirmation.jobs_complete
confirmation.jobs_failed
confirmation.promoted_candidates
```

Recommended parent artifacts:

```text
summaries/stage_summary.json
summaries/candidate_summary.csv
summaries/regime_summary.csv
summaries/plateau_summary.csv
summaries/plateau_seed_summary.csv
summaries/plateau_neighbor_summary.csv
summaries/plateau_artifact_index.json
summaries/perturbation_summary.csv
summaries/detail_run_links.json
promotion_packet.json
promotion_packet.yaml
promotion_packet.md
```

Keep these summaries compact. The parent run should contain enough data to
navigate the crucible, not every raw backtest artifact.

### Detail Runs and Scale

There is a difference between:

- **Checkpointing every job result:** required for resume.
- **Creating a full inspectable MLflow run for every job:** optional and often
  too noisy.

Every Ray job should produce a compact deterministic result artifact so the
crucible can resume. Not every Ray job needs a full MLflow run with charts,
reports, and large artifacts.

For large crucible runs, use a global parent experiment plus a per-crucible-run
detail experiment.

Recommended split:

```text
Experiment: Algo Crucible
  Parent summary runs for all crucible executions.

Experiment: Algo Crucible Details/<workload_name>/<run_id>
  Detailed runs for one crucible execution.
```

Why this split:

- The global `Algo Crucible` experiment stays readable as an index of crucible
  attempts.
- Each massive execution gets its own detail experiment, so thousands of detail
  runs do not pollute every other workload.
- Detail experiments can be archived, deleted, or retained independently.
- The parent summary run still links to the detail experiment and specific runs.

Each detail run should be linked back to the parent with tags:

```text
crucible_run_id
parent_run_id
stage
job_id
candidate_id
workload_name
```

Recommended detail-run policy:

```yaml
mlflow:
  full_detail_run_policy: survivors
```

Policy options:

- `none`: no full detail runs; only compact checkpoint artifacts.
- `failures_only`: full detail runs only for failed jobs that need debugging.
- `survivors`: full detail runs for candidates that pass a stage gate, plus
  failures that need debugging.
- `top_n`: full detail runs for the top N jobs per stage.
- `all`: full detail runs for every job. This is usually too much.

Default recommendation: `survivors`.

Store compact checkpoint results for all jobs:

```text
stages/<stage>/results/<job_id>.json
```

Store full MLflow detail runs only for:

- Candidates that pass the regime-aware gate.
- Plateau centers and accepted neighborhood evidence.
- Perturbation scenarios for candidates that reach final review.
- Confirmation runs.
- Failed jobs that need debugging.
- Optional top-N rejected candidates for post-mortem inspection.

The parent should store a compact link table:

```json
{
  "candidate_007": {
    "detail_experiment_name": "Algo Crucible Details/spy_macd_switch/20260827T120000",
    "hpo_run_id": "...",
    "plateau_job_run_ids": ["..."],
    "perturbation_job_run_ids": ["..."],
    "confirmation_run_id": "..."
  }
}
```

This makes the parent run readable while still preserving drill-down paths for
specific evidence.

Do not store every detailed run under the parent if that makes the parent run
impossible to inspect. The parent should be the table of contents. Detail runs
should be the appendix.

Do not default to full detail runs for every failed or rejected candidate. In a
large crucible, most candidates are supposed to fail. Store their compact metrics
and gate reasons, then only preserve full artifacts when they are useful for
debugging, audit, or model improvement.

### Plateau Observability

Plateau finding needs its own summary artifacts. Otherwise the process becomes
impossible to audit from the MLflow UI.

The parent run should store compact plateau metrics:

```text
plateau.seed_count
plateau.neighborhood_jobs_total
plateau.neighborhood_jobs_complete
plateau.accepted_plateaus
plateau.rejected_peaks
plateau.best_score
plateau.best_pass_rate
plateau.median_pass_rate
plateau.best_peak_to_median_degradation
plateau.best_worst_quartile_return
```

The parent run should also store plateau summary tables:

```text
summaries/plateau_seed_summary.csv
summaries/plateau_neighbor_summary.csv
summaries/plateau_summary.csv
summaries/plateau_artifact_index.json
```

`plateau_seed_summary.csv` should have one row per selected peak:

```text
seed_id
candidate_id
candidate_type
specialist_regimes
center_config_json
center_score
center_oos_return
center_max_drawdown
center_trade_count
normalized_param_distance_to_nearest_seed
selected_reason
```

`plateau_neighbor_summary.csv` should have one row per local neighborhood run:

```text
seed_id
job_id
candidate_id
candidate_type
specialist_regime
param_distance_from_seed
config_patch_json
passed_gate
oos_return
max_drawdown
trade_count
profitable_windows_pct
worst_quartile_oos_return
detail_run_id
```

`plateau_summary.csv` should have one row per plateau candidate:

```text
seed_id
candidate_id
candidate_type
specialist_regimes
accepted
plateau_score
neighbor_trials
neighbor_pass_rate
median_oos_return
worst_quartile_oos_return
max_drawdown_p95
trade_count_cv
peak_to_median_degradation
regime_consistency_score
failure_reason
```

Recommended visual artifacts:

- `plateau_heatmap_<seed_id>_<param_x>_<param_y>.html`
- `plateau_heatmap_<seed_id>_<param_x>_<param_y>.png`
- `plateau_parallel_coordinates_<seed_id>.html`
- `plateau_distance_decay_<seed_id>.png`
- `plateau_regime_breakdown_<seed_id>.png`

Use heatmaps when there are one or two dominant numeric parameters. Use parallel
coordinates when there are many tunable parameters. Use distance-decay plots for
the core plateau question: does performance degrade slowly as parameters move
away from the seed, or collapse immediately?

For each seed, the distance-decay plot should show:

- X-axis: normalized parameter distance from seed.
- Y-axis: selected performance metric.
- Color: pass/fail gate result or specialist regime.
- Horizontal gate line for the relevant metric threshold.
- Marker or shaded region where neighborhood performance falls under the gate.
- Optional line: rolling median performance by distance bucket.

When possible, every plateau visual should include the relevant gate boundary:

- Return plots: minimum return gate.
- Drawdown plots: maximum drawdown gate.
- Pass-rate plots: minimum pass-rate gate.
- Trade-count plots: minimum trade-count gate.
- Degradation plots: maximum peak-to-median degradation gate.

For distance-decay plots, explicitly mark the first distance bucket where the
rolling median or pass rate falls below the gate. This makes the effective
plateau radius visible instead of burying it in the summary table.

For regime specialists, plateau visuals should be generated for the target
regime metric, not only aggregate performance. A candidate can have an ugly
aggregate heatmap and still have a useful `RANGE_LOW_VOL` plateau.

The `plateau_artifact_index.json` file should map seeds to artifacts:

```json
{
  "seed_003": {
    "summary_row": 3,
    "heatmaps": [
      "plots/plateau_heatmap_seed_003_stop_pct_profit_pct.html"
    ],
    "parallel_coordinates": "plots/plateau_parallel_coordinates_seed_003.html",
    "distance_decay": "plots/plateau_distance_decay_seed_003.png",
    "detail_run_ids": ["..."]
  }
}
```

The promotion packet should include a short plateau narrative for every promoted
candidate:

```yaml
plateau_evidence:
  seed_id: seed_003
  accepted: true
  neighbor_pass_rate: 0.68
  median_oos_return: 0.041
  worst_quartile_oos_return: -0.012
  peak_to_median_degradation: 0.31
  interpretation: "Performance degrades gradually around the seed and remains acceptable in RANGE_LOW_VOL."
```

This gives the MLflow parent run enough context to explain why a peak was
accepted as a plateau or rejected as a spike.

## Build Milestones and Quality Gates

Build the crucible in vertical slices. Each milestone must have unit coverage
plus at least one small end-to-end acceptance test using synthetic data or a test
state store. Do not advance to the next milestone until the current milestone's
gates pass.

### Milestone 1: Single-Candidate Scoring Spine

Required implementation:

- Load platform and workload YAML.
- Resolve configs into immutable `resolved_config.yaml`.
- Compute deterministic `crucible_run_id`.
- Compute deterministic `candidate_id`.
- Run one fixed candidate through `BacktestingEngine`.
- Produce online-safe regime labels.
- Produce overall scorecard.
- Produce regime scorecard.
- Create MLflow parent run through `CrucibleStateStore`.
- Write compact summary artifacts.

Quality gates:

- Same config produces same `crucible_run_id`.
- Same candidate produces same `candidate_id`.
- One backtest result exists.
- Overall scorecard exists.
- Regime scorecard exists.
- Parent run has summary metrics and artifacts.
- Same `run_name` with different config hash fails.

End-to-end acceptance:

```text
Run tiny synthetic workload with 50-200 bars.
Assert parent run exists in test state store or MLflow.
Assert candidate_summary.csv has one candidate.
Assert regime_summary.csv has rows.
Assert rerunning same config resumes or refuses duplicate according to policy.
```

### Milestone 2: Resumable Ray Job Runner

Required implementation:

- Add Ray-backed job submission.
- Add deterministic job IDs.
- Add job manifests.
- Add compact job result artifacts.
- Add resume logic for missing, complete, and failed jobs.
- Add local cache/staging.

Quality gates:

- Completed job results are reused.
- Failed or missing jobs are rerun when policy allows.
- Duplicate resume does not duplicate completed results.
- State is loaded from `CrucibleStateStore`, not process memory.
- Result writes are atomic.

End-to-end acceptance:

```text
Submit 5 synthetic backtest jobs.
Force stop after 2 complete.
Resume.
Assert exactly 5 completed results.
Assert first 2 job IDs/results were reused.
Assert no duplicate result artifacts were created.
```

### Milestone 3: Walk-Forward OOS Batch

Required implementation:

- Generate train / embargo / validation windows.
- Run validation jobs through Ray.
- Produce per-window scorecards.
- Produce aggregate OOS scorecard.
- Produce validation-only regime scorecard.
- Add leakage validators.

Quality gates:

- Validation starts after train end plus embargo.
- Warmup does not leak validation data into training selection.
- Regime scoring uses validation timestamps only.
- Every window has a scorecard.
- Aggregate OOS scorecard exists.

End-to-end acceptance:

```text
Use synthetic dated data with obvious train/validation boundaries.
Run 3 walk-forward windows.
Assert every validation_start > train_end + embargo.
Assert 3 window results exist.
Assert regime metrics only use validation timestamps.
```

### Milestone 4: Broad HPO Integration

Required implementation:

- Reuse or wrap existing Ray HPO runner.
- Capture full trial summaries, not just best config.
- Convert trials into candidate results.
- Record failed trials with reason codes.
- Log HPO summary metrics to parent run.

Quality gates:

- Trial configs map correctly into algorithm and portfolio params.
- Full trial summary table exists.
- Candidate objects are created from completed trials.
- Failed trials are visible with structured reasons.
- HPO stage can resume without duplicating completed trial results.

End-to-end acceptance:

```text
Run HPO with 6-10 trials on tiny synthetic workload.
Assert trial_summary.csv has all completed trials.
Assert at least one CandidateResult is created.
Assert each candidate has config, metrics, and candidate_id.
```

### Milestone 5: Regime-Aware Gates

Required implementation:

- Add generalist gate.
- Add specialist gate.
- Add defensive gate.
- Add reject classification.
- Add structured failure reasons.
- Write candidate type and specialist tags to summary artifacts.

Quality gates:

- Generalist can pass.
- Specialist can pass even when aggregate performance fails.
- Rejects include stable failure reason codes.
- Candidate type is deterministic.
- Specialist regimes are recorded.

End-to-end acceptance:

```text
Feed fabricated candidate scorecards:
A passes overall.
B fails overall but passes RANGE_LOW_VOL.
C fails all.
Assert types: generalist, specialist, reject.
Assert failure reason codes are present.
```

Integration acceptance:

```text
Use synthetic market data with one clear regime where the strategy works.
Assert the candidate is kept as specialist despite bad aggregate result.
```

### Milestone 6: Plateau Finder

Required implementation:

- Select diverse peaks from HPO results.
- Normalize parameter space.
- Generate local parameter neighborhoods.
- Run neighborhood jobs through Ray.
- Score plateau pass rate, median, worst quartile, degradation, drawdown, trade
  count stability, and regime consistency.
- Produce plateau summary tables.
- Produce plateau plots with gate lines.
- Apply plateau gate.

Quality gates:

- Diverse peak selection avoids duplicate nearby peaks.
- Neighborhood configs stay within search-space bounds.
- Spike-like surfaces are rejected.
- Broad stable regions are accepted.
- Specialist plateau can pass on target-regime metric even if aggregate plateau
  fails.
- Plot artifacts include gate boundary metadata or visible gate lines.

End-to-end acceptance:

```text
Use fake trial surface with one narrow spike and one broad plateau.
Run plateau finder.
Assert spike rejected.
Assert plateau accepted.
Assert plateau_summary.csv exists.
Assert distance-decay plot artifact exists.
Assert gate threshold is represented in plot metadata/config.
```

### Milestone 7: Perturbation Analysis

Required implementation:

- Generate perturbation scenarios from platform config.
- Submit perturbation jobs through Ray.
- Score each scenario overall and by target regime.
- Apply perturbation gate.
- Record scenario-specific failure reasons.

Quality gates:

- Each perturbation has deterministic job ID.
- Perturbations are bounded by budget caps.
- Results identify which scenario failed.
- Candidate fails if too many required scenarios fail.
- Specialist candidates are scored against their target regimes.

End-to-end acceptance:

```text
Create candidate that passes base but fails 2x cost.
Run perturbation stage.
Assert candidate rejected with cost_slippage_fragile.
Assert perturbation_summary.csv identifies failing scenarios.
```

### Milestone 8: Confirmation and Promotion Packet

Required implementation:

- Run confirmation on untouched historical range.
- Freeze candidate config before confirmation.
- Block tuning after confirmation.
- Generate promotion packet JSON/YAML/Markdown.
- Create promoted-candidate folder only after approval or explicit flag.
- Link MLflow parent run to detail runs and promotion packet.

Quality gates:

- Confirmation uses configured untouched dates.
- Candidate config is immutable before confirmation.
- Promotion packet includes required sections.
- No paper trading launch happens without approval.
- Parent run has final summary metrics and links.

End-to-end acceptance:

```text
Run candidate through mocked prior stages into confirmation.
Assert confirmation job uses configured untouched dates.
Assert promotion_packet.json/yaml/md exist.
Assert no paper trading launch happens without approval.
```

### Milestone 9: Paper Trading Replay Gate

Required implementation:

- Replay paper/live bars through frozen backtest path.
- Compare live and replay regime labels.
- Compare live and replay signals.
- Compare intended orders.
- Compare fills against tolerance.
- Record paper quality gate result.

Quality gates:

- Same bars produce same regime labels.
- Same bars produce same signals.
- Same bars produce same intended orders.
- Fill differences are inside configured tolerance or explained.
- Mismatches produce structured failure reasons.

End-to-end acceptance:

```text
Take recorded paper bars.
Replay through frozen candidate.
Assert live signal log equals replay signal log.
Inject one changed bar/order.
Assert paper_replay_mismatch failure.
```

### Cross-Cutting Gates

These gates apply from the beginning:

- Config immutability: same `run_name` with different config hash fails.
- Data fingerprint: changed data is detected.
- Budget cap: excessive job count fails before launch.
- MLflow interpretability: parent run has summary metrics, not only artifacts.
- Resume: completed work is reused.
- Leakage: invalid train/validation boundaries fail before jobs launch.
- Detail-run policy: full detail runs follow configured retention policy.

Every milestone should include a resume/re-run check. Synthetic data should be
preferred for end-to-end acceptance because it can force known regimes, known
winners, known failures, and known plateau shapes.

### Local Cache

Each host may keep a local mirror under `local_cache_dir`.

Use the cache to:

- Avoid repeatedly downloading large manifests.
- Stage temp files before MLflow upload.
- Recover from transient tracking-server outages.

On resume, the orchestrator should reconcile local cache against MLflow and
prefer MLflow when there is disagreement.

### Config and Code Drift

At run creation, record:

- Platform config hash.
- Workload config hash.
- Resolved config hash.
- Git commit, if available.
- Dirty git status, if available.

On resume, warn if:

- Git commit changed.
- Tracked files changed.
- Resolved config changed.

Do not silently mix results from different code/config states. Either block the
resume or require an explicit override.

### Resumability Rules

- Completed stages are skipped.
- Completed jobs are reused.
- Failed jobs can be rerun.
- Downstream stages must be invalidated if upstream candidate outputs change.
- Result writes must be atomic.
- Job IDs must be deterministic.
- MLflow is the system of record.
- Local files are cache/staging, not canonical state.

## Brutal Standard

Most algorithms should fail.

An algorithm that makes money overall but has no stable regime story is
suspicious. An algorithm that only works in one regime but does so repeatedly
and predictably is useful.

The crucible should preserve those specialists instead of flattening everything
into aggregate performance and throwing away the components needed for a
regime-routed strategy.
