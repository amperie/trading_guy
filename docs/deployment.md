# Deployment

## Current Deployment Model

`trading_guy` is primarily a local or server-side Python runtime. It does not define a single production deployment target. Deployment depends on the workflow:

- Backtests and HPO can run locally or on a compute host.
- Remote optimization can use Ray.
- MLflow stores experiment tracking and promoted bundle artifacts.
- MongoDB stores live trading session state and replay data.
- Alpaca provides brokerage connectivity for paper/live accounts.
- Quant Crucible platform execution runs this repo in a Docker container through `trading.platform.runner`.

## Environment And Secrets

Local account credentials are represented by `accounts.yaml` and environment files. Treat broker keys, MongoDB URLs, MLflow credentials, and object-store credentials as secrets.

For platform execution, `qc-platform-api` passes run identity and config through environment variables and command-line arguments. The trading repo must be mounted into the executor container, and the image must contain dependencies needed to run `uv run python -m trading.platform.runner`.

## Promotion Artifacts

Promoted bundles live under:

```text
trading/promoted/<bundle_name>/
```

Bundles can also be reconstructed from MLflow run URLs when the run contains the required config and artifacts. The pipeline prints both local paths and MLflow links so paper/live launch paths are explicit.

## Operational Deployment Notes

- Validate account selection before any live command.
- Keep paper and live accounts separate in `accounts.yaml`.
- Use MLflow experiment names deliberately; pipeline registry runs are bundle metadata, not strategy performance runs.
- For restartable paper sessions, use `scripts/paper_session_autostart.py` and ensure MongoDB/networking are available before launch.
- Use `tmux` or a process supervisor for long-running paper/live sessions.
- A Docker image used by `qc-platform-api` needs TA-Lib, Ray/MLflow dependencies, and access to any required config/data paths.

