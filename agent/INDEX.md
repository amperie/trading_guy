# Agent instruction index

Catalogues `agent/` files and where the matching code lives. Read this first, then only the topic files you need. Instruction files are capped at 30 lines; this index, `layout.md`, and `TODO.md` are not.

## How to load

| Task | Read |
|------|------|
| Any change | `AGENTS.md`, this index |
| Add/edit `agent/` files | `authoring.md` |
| New class, wiring, config keys | `architecture.md`, `implementation.md`, `layout.md` |
| Algorithm / warmup / reconfigure | `algorithm.md` |
| Multi-timeframe bars | `multi-timeframe.md` |
| Loading or iterating market data | `data-provider.md` |
| Signals → orders, cash, positions | `portfolio.md` |
| Order types, OM backends | `orders.md` |
| Backtest, live, bar aggregation | `engines.md` |
| Metrics, charts, MLflow, Mongo replay | `analysis.md` |
| Walk-forward or live HPO | `optimization.md` |
| Helpers, logger, MLflow client | `utilities.md` |
| `run.py` flags | `cli.md` |
| Pytest | `tests.md` |
| Known follow-ups | `TODO.md` |

## Instruction files

| File | Role |
|------|------|
| `agent/AGENTS.md` | Boot: load order and global constraints |
| `agent/authoring.md` | How to write future instruction files |
| `agent/architecture.md` | Pipeline, config singleton, swappable components |
| `agent/algorithm.md` | `Algorithm` contract, warmup gate, `reconfigure` |
| `agent/multi-timeframe.md` | `MultiTimeframeAlgorithm` |
| `agent/data-provider.md` | `DataProvider` / `TestDataProvider` |
| `agent/portfolio.md` | `Portfolio` contract and stock implementations |
| `agent/orders.md` | `OrderManager`, order enums, brackets |
| `agent/engines.md` | Backtest, Alpaca live, tick aggregation |
| `agent/analysis.md` | `AnalysisEngine`, `PortfolioAnalyzer` |
| `agent/optimization.md` | Walk-forward and self-optimizing live HPO |
| `agent/utilities.md` | `utils/` helpers |
| `agent/cli.md` | `run.py` subcommands and flags |
| `agent/implementation.md` | Imports, timestamps, DataFrames, config/aggregation shape |
| `agent/tests.md` | How to run tests |

## Catalogues (not 30-line instruction files)

| File | Role |
|------|------|
| `agent/INDEX.md` | This file |
| `agent/layout.md` | Repository directory map |
| `agent/TODO.md` | Standing work list |

## Code map (short)

| Area | Path |
|------|------|
| Enums / dataclasses | `trading/core/classes.py` |
| Algorithm bases | `trading/core/algorithm.py`, `trading/core/multi_timeframe_algorithm.py` |
| Portfolio base + impls | `trading/core/portfolio.py`, `trading/core/pf/` |
| OrderManager impls | `trading/core/om/` |
| Algorithms | `trading/algorithms/` |
| Data providers | `trading/data_providers/` |
| Engines | `trading/engines/` |
| Analysis | `trading/analysis/` |
| YAML profiles | `configs/` |
| Utils | `utils/` |
| Tests | `tests/unit/`, `tests/integration/` |
| CLI | `run.py` |

Full tree: `agent/layout.md`. Cursor always-apply: `.cursor/rules/agent-instructions.mdc`. Claude: `.claude/CLAUDE.md`.
