# Operations

## Testing

Run the full suite:

```bash
uv run pytest
```

Focused tests live under:

- `tests/unit`
- `tests/integration`
- `tests/e2e`

`docs/TEST_STRUCTURE.md` explains the intended organization.

## Debugging

Set `debug_on_sigint: true` in `config.yaml` to open the interactive debug REPL on `Ctrl-C` during engine-driven runs.

Useful REPL commands include `progress`, `loglevel`, `c`, and `q`. See `docs/DEBUG_INSTRUCTIONS.md` for details.

## Observability

- MLflow records metrics, parameters, charts, bundle artifacts, and promotion metadata.
- MongoDB stores live session and replay state.
- Analysis outputs include metrics, equity curves, trade logs, and other artifacts.
- Platform runner mode emits structured JSON progress for API workers.

## Failure Areas To Check First

- Config import paths for algorithm, portfolio, data provider, and order manager.
- Account names and broker credentials in `accounts.yaml`.
- Data file path, symbol coverage, and timestamp format.
- MongoDB availability for live/session replay workflows.
- MLflow tracking URI and artifact store permissions.
- Ray availability for HPO and remote optimization.
- TA-Lib installation when indicators fail to import.
- Docker mount paths when run from `qc-platform-api`.

## Current Limitations

- The repo is broad and command-rich; the root README is the best quick map, but detailed docs are split across topic files.
- Production live trading needs external process supervision and explicit operational discipline; the repo does not provide a complete deployment platform by itself.
- Platform execution assumes the API-side Docker image and mounted checkout are kept compatible with this repo.

