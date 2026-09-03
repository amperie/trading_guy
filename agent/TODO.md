# Agent and project TODO

Standing work list. Not an instruction file — no 30-line cap. Check this when the task overlaps an item; mark items done in the same change when you actually finish them. Add new follow-ups here instead of appending to topic instruction files.

## Instruction set

- [x] Split `agent/AGENTS.md` into topic files of 30 lines or less.
- [x] Add `agent/authoring.md` with the 30-line rule for future instruction files.
- [x] Add `agent/INDEX.md` cataloguing instruction files and code locations.
- [ ] Keep `agent/INDEX.md` in lockstep when adding, renaming, or deleting topic files.
- [ ] Keep `agent/layout.md` in lockstep when adding modules, configs, or test files.
- [ ] After any instruction edit, run `wc -l agent/*.md` and split any instruction file over 30 lines (`INDEX.md`, `layout.md`, `TODO.md` may exceed 30).
- [ ] Point human docs at `agent/INDEX.md` or the relevant topic file instead of treating `AGENTS.md` as a full architecture dump (`docs/TEST_STRUCTURE.md`, `docs/INTERACTIVE_CHART_README.md`, `tests/README.md`).

## Tests

- [ ] Finish fixtures for `tests/unit/test_analysis_engine.py` (called out as incomplete in the old instruction dump).
- [ ] Reconcile pass counts between `agent/tests.md`, `docs/TEST_STRUCTURE.md`, and `tests/README.md` (those docs still list older suites and 6/10 analysis, 22/23 portfolio).
- [ ] Confirm whether `test_portfolio.py` still has a failing edge case; if fixed, update docs; if not, document the case here.
- [ ] Add `tests/unit/` files that exist in the repo but were missing from the old AGENTS test list to `agent/tests.md` only if they are considered required regression suites (do not grow `tests.md` past 30 lines — link out if needed).
- [ ] Coverage command in `agent/tests.md` uses `--cov=core --cov=utils`; confirm those package paths match the current layout (`trading/core` vs `core`).
- [ ] Prefer `.venv/bin/pytest` on macOS/Linux; keep Windows `.venv/Scripts/pytest` as a note only.

## Base classes and engines

- [ ] Document `long_short_oscillator_portfolio.py` in `agent/portfolio.md` or a split file if agents start editing it.
- [ ] Document `core/om/alpaca_om.py` (live routing) in `agent/orders.md` or a split file when that backend is in scope.
- [ ] Document `engines/split_period_backtest_engine.py` if it becomes a supported entry path from `run.py`.
- [ ] Document `engines/base_engine.py` (`BaseEngine` / `AsyncEngine`) contracts if new engines are added.
- [ ] Confirm `TickAggregationPassthroughEngine` live vs backtest wiring in `run.py` still matches `agent/engines.md`.

## Config and CLI

- [ ] Example YAML in the old AGENTS dump was truncated; real shapes live in `configs/`. If a profile adds required keys, add a single line to `agent/implementation.md` or split a `config-keys.md` file.
- [ ] Verify `--agg-period`, `--session-id`, and `--timeframe` in `run.py` still match `agent/cli.md`.
- [ ] Confirm MLflow tracking URI in `agent/utilities.md` (`http://z440.lan:5000`) still matches `config.yaml` / `utils/mlflow_client.py`.

## Analysis and persistence

- [ ] Prefer `PortfolioAnalyzer` for Mongo session post-mortems; keep `AnalysisEngine` notes only for in-memory backtest analysis.
- [ ] Document `utils/trading_state_store.py` session metadata fields if replay or `from_mongodb` behavior changes.

## Hygiene

- [ ] Do not re-merge topic files into `AGENTS.md`.
- [ ] Do not put TODOs, pass-count tables, or full directory trees into instruction files.
