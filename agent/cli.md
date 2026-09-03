# run.py CLI

Shared flags for `backtest`, `live`, `walk-forward`, `hpo`, `session-replay`:

| Flag | Description |
|------|-------------|
| `--config` | YAML profile (required) |
| `--symbol` | Override portfolio symbol |
| `--cash` | Override starting cash |
| `--algorithm` | Override algorithm class (dotted path) |
| `--no-mlflow` | Disable MLflow |
| `--run-name` | Override analysis run name |
| `--session-id` | MongoDB `state_store` session ID |
| `--agg-period N` | Sets `aggregation.aggregation_period_minutes` and `aggregation.enabled=true` |

`backtest` / `walk-forward`: `--data` overrides the data provider path.
`live`: `--alpaca-override-url`.
`hpo`: `--num-samples`, `--max-concurrent-trials`.
`session-replay`: `--timeframe` when session metadata is missing (`Minute`, `Hour`, `Day`).
