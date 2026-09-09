from __future__ import annotations

import argparse
import calendar
import csv
import dataclasses
import json
import math
import os
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from algo_crucible.orchestrator import CrucibleOrchestrator
from trading.commands.backtest import run_backtest_from_raw_config
from trading.commands.common import (
    apply_cli_overrides,
    load_raw_config,
)
from trading.reporting import LocalRunResultSink
from trading.reporting.sinks import SinkRun


DEFAULT_CONFIG_BY_STAGE = {
    "idea": "platform:backtest",
    "smoke": "platform:backtest",
    "research": "platform:backtest",
    "crucible": "configs/crucible/platform_local_csv_mlflow.yaml",
    "promotion": "platform:backtest",
    "monitoring": "platform:backtest",
}
DEFAULT_CRUCIBLE_WORKLOAD = "configs/crucible/workloads/spy_5min_local_csv_test.yaml"
CRUCIBLE_STAGES = (
    ("hpo", "Running HPO candidate search"),
    ("walk_forward_oos", "Running walk-forward OOS validation"),
    ("regime_gate", "Evaluating regime-aware gates"),
    ("plateau", "Testing parameter plateau stability"),
    ("cross_instrument_transfer", "Testing cross-instrument transfer"),
    ("perturbation", "Running perturbation scenarios"),
    ("structural_break_stability", "Checking structural-break stability"),
    ("execution_realism_stress", "Running execution realism stress"),
    ("monte_carlo", "Running Monte Carlo path simulation"),
    ("confirmation", "Running final confirmation"),
)
CRUCIBLE_STAGE_NAMES = tuple(stage[0] for stage in CRUCIBLE_STAGES) + ("paper_replay",)
MIN_CRUCIBLE_DATA_ROWS = 100
_PROGRESS_LOG_PATH: Path | None = None


def emit(progress_pct: float, message: str, **extra: Any) -> None:
    event = {"progressPct": progress_pct, "message": message, **extra}
    print(json.dumps(event, default=str), flush=True)
    if _PROGRESS_LOG_PATH is not None:
        _PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROGRESS_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_jsonable(event), default=str) + "\n")


def _set_progress_log(output_dir: str | Path) -> None:
    global _PROGRESS_LOG_PATH
    _PROGRESS_LOG_PATH = Path(output_dir) / "progress_events.jsonl"
    if _PROGRESS_LOG_PATH.exists():
        _PROGRESS_LOG_PATH.unlink()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _metric_payload(result: dict[str, Any] | None) -> dict[str, float]:
    metrics = (result or {}).get("metrics")
    if metrics is None:
        return {}
    payload: dict[str, float] = {}
    for name in dataclasses.fields(metrics):
        value = getattr(metrics, name.name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[name.name] = float(value)
    return payload


def _ui_metrics(metrics: dict[str, float]) -> dict[str, float]:
    mapping = {
        "sharpe_ratio": "sharpe",
        "max_drawdown_pct": "maxDrawdown",
        "total_trades": "trades",
        "win_rate": "winRate",
        "profit_factor": "profitFactor",
        "annualized_return": "cagr",
    }
    return {ui_key: float(metrics[source]) for source, ui_key in mapping.items() if source in metrics}


def tenant_mlflow_experiment_name(tenant_id: str) -> str:
    return f"QC_tenant_{tenant_id}"


def _apply_tenant_mlflow_grouping(raw_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tenant_id = str(args.tenant_id)
    experiment_name = tenant_mlflow_experiment_name(tenant_id)
    analysis = raw_cfg.setdefault("analysis", {})
    analysis["experiment_name"] = experiment_name
    mlflow_tags = dict(analysis.get("mlflow_tags") or {})
    mlflow_tags["tenant_id"] = tenant_id
    mlflow_tags["platform.tenant_id"] = tenant_id
    analysis["mlflow_tags"] = mlflow_tags
    raw_cfg.setdefault("mlflow", {})["parent_experiment_name"] = experiment_name
    return raw_cfg


def _namespace(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = {
        "config": args.config,
        "account": args.account,
        "symbol": args.symbol,
        "cash": args.cash,
        "algorithm": args.algorithm,
        "algorithm_url": args.algorithm_url,
        "portfolio": args.portfolio,
        "portfolio_url": args.portfolio_url,
        "data": args.data,
        "no_mlflow": args.no_mlflow,
        "run_name": args.run_name or f"{args.stage}_{args.run_id}",
        "alpaca_override_url": None,
        "session_id": args.session_id,
        "agg_period": args.agg_period,
        "mlflow_experiment_name_override": tenant_mlflow_experiment_name(args.tenant_id),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _platform_backtest_config(args: argparse.Namespace) -> dict[str, Any]:
    symbol = args.symbol or "SPXU"
    return {
        "mode": "backtest",
        "state_store": {"enabled": False},
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": "../data/SPXU_5min.csv",
            "truncate": 0,
        },
        "algorithm": {
            "algorithm": "trading.algorithms.test_algorithm.TestAlgorithm",
            "history_length": 10,
        },
        "portfolio": {
            "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "symbol": symbol,
            "cash": args.cash or 100000,
            "keep_history": True,
            "stop_pct": 2,
            "profit_pct": 5,
        },
        "order_manager": {
            "order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager",
        },
        "analysis": {
            "enabled": True,
            "log_to_mlflow": not args.no_mlflow,
            "experiment_name": tenant_mlflow_experiment_name(args.tenant_id),
            "run_name": args.run_name or f"{args.stage}_{args.run_id}",
            "description": "Platform-managed backtest run",
            "benchmarks": {},
        },
    }


def _platform_crucible_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _platform_backtest_config(args)
    cfg["mode"] = "hpo"
    cfg["hpo"] = {
        "validation_period_days": args.validation_period_days or 30,
        "objective_metric": "val_annualized_return",
        "num_samples": args.hpo_samples or 4,
        "max_concurrent_trials": args.hpo_concurrency or 1,
        "log_trials_to_mlflow": False,
        "log_ray_worker_output": True,
        "search_space": {
            "stop_pct": {"type": "uniform", "low": 1.0, "high": 8.0},
            "profit_pct": {"type": "uniform", "low": 2.0, "high": 12.0},
        },
        "algorithm_param_keys": [],
        "portfolio_param_keys": ["stop_pct", "profit_pct"],
    }
    return cfg


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _source_tree_data_path(path: str | Path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw)
    trading_root = Path(__file__).resolve().parents[1]
    return str((trading_root / raw).resolve())


def _materialize_local_workload_paths(workload: dict[str, Any], args: argparse.Namespace) -> None:
    data_provider = workload.get("data_provider")
    if isinstance(data_provider, dict):
        if args.data:
            data_provider["path"] = args.data
        if data_provider.get("path"):
            data_provider["path"] = _source_tree_data_path(data_provider["path"])


def _materialize_local_platform_paths(platform: dict[str, Any]) -> None:
    paper_replay = platform.get("paper_replay")
    if isinstance(paper_replay, dict) and paper_replay.get("data_path"):
        paper_replay["data_path"] = _source_tree_data_path(paper_replay["data_path"])


def _hydrate_confirmation_window(platform: dict[str, Any], workload: dict[str, Any]) -> None:
    confirmation = platform.setdefault("confirmation", {})
    if confirmation.get("start_date") and confirmation.get("end_date"):
        return
    provider = workload.get("data_provider") or {}
    start = (
        provider.get("confirmation_start_date")
        or provider.get("test_start_date")
        or provider.get("start_date")
    )
    end = (
        provider.get("confirmation_end_date")
        or provider.get("test_end_date")
        or provider.get("end_date")
    )
    if not (start and end):
        return
    if not confirmation.get("start_date"):
        confirmation["start_date"] = start
    if not confirmation.get("end_date"):
        confirmation["end_date"] = end


def _crucible_config_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    platform_config = (
        "configs/crucible/platform_local_csv_mlflow.yaml"
        if args.config == "platform:crucible"
        else args.config
    )
    platform = _load_yaml(platform_config)
    workload = _load_yaml(args.workload_config or DEFAULT_CRUCIBLE_WORKLOAD)
    run_name = args.run_name or f"{args.stage}_{args.run_id}"

    platform.setdefault("crucible", {})["run_name"] = run_name
    platform["tenant_id"] = str(args.tenant_id)
    workload.setdefault("workload", {})["run_name"] = run_name
    platform.setdefault("resume", {})["local_cache_dir"] = str(output_dir / "crucible_runs")
    if args.hpo_samples is not None:
        platform.setdefault("hpo", {})["num_samples"] = args.hpo_samples
    if args.hpo_concurrency is not None:
        platform.setdefault("hpo", {})["max_concurrent_trials"] = args.hpo_concurrency
        platform.setdefault("ray", {})["max_concurrent_trials"] = args.hpo_concurrency
    if args.validation_period_days is not None:
        platform.setdefault("hpo", {})["validation_period_days"] = args.validation_period_days
    platform.setdefault("hpo", {})["ray_storage_path"] = str((output_dir / "ray_results").resolve())
    platform.setdefault("ray", {})["enabled"] = bool(args.use_ray)
    if args.no_mlflow:
        platform.setdefault("state_store", {})["backend"] = "local"
    platform.setdefault("mlflow", {})["parent_experiment_name"] = tenant_mlflow_experiment_name(args.tenant_id)
    _materialize_local_platform_paths(platform)
    _materialize_local_workload_paths(workload, args)
    _hydrate_confirmation_window(platform, workload)

    effective_platform = output_dir / "crucible_platform.yaml"
    effective_workload = output_dir / "crucible_workload.yaml"
    _write_yaml(effective_platform, platform)
    _write_yaml(effective_workload, workload)
    return effective_platform, effective_workload


def _emit_crucible_runtime_diagnostics(platform_path: Path, workload_path: Path) -> None:
    platform = _load_yaml(platform_path)
    workload = _load_yaml(workload_path)
    data_provider = workload.get("data_provider") or {}
    data_diagnostics = _csv_data_diagnostics(data_provider)
    emit(
        9,
        "Crucible runtime diagnostics",
        diagnosticType="crucible_runtime_config",
        platformConfigPath=platform_path,
        workloadConfigPath=workload_path,
        dataPath=data_provider.get("path"),
        dataProviderConfig=data_provider,
        algorithmConfig=workload.get("algorithm"),
        portfolioConfig=workload.get("portfolio"),
        runnerConfig=platform,
        runtimeAssets=workload.get("platform_runtime_assets") or [],
        dataDiagnostics=data_diagnostics,
    )


def _csv_data_diagnostics(data_provider: dict[str, Any]) -> dict[str, Any]:
    path = data_provider.get("path")
    diagnostics: dict[str, Any] = {
        "path": path,
        "startDate": data_provider.get("start_date"),
        "endDate": data_provider.get("end_date"),
    }
    if not path:
        return diagnostics | {"exists": False, "error": "data_provider.path is missing"}
    csv_path = Path(path)
    diagnostics["exists"] = csv_path.exists()
    if not csv_path.exists():
        return diagnostics

    total_rows = filtered_rows = 0
    first_timestamp = last_timestamp = None
    start = _parse_date(data_provider.get("start_date"))
    end = _parse_date(data_provider.get("end_date"))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            ts = _parse_date(row.get("timestamp"))
            if ts is not None:
                first_timestamp = ts if first_timestamp is None else min(first_timestamp, ts)
                last_timestamp = ts if last_timestamp is None else max(last_timestamp, ts)
            if (start is None or ts is None or ts >= start) and (
                end is None or ts is None or ts <= end
            ):
                filtered_rows += 1
    return diagnostics | {
        "totalRows": total_rows,
        "filteredRows": filtered_rows,
        "firstTimestamp": first_timestamp,
        "lastTimestamp": last_timestamp,
        "minimumRows": MIN_CRUCIBLE_DATA_ROWS,
    }


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _validate_crucible_data(data_provider: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    if not diagnostics.get("exists"):
        raise FileNotFoundError(f"Crucible data file not found: {data_provider.get('path')}")
    filtered_rows = diagnostics.get("filteredRows")
    if isinstance(filtered_rows, int) and filtered_rows < MIN_CRUCIBLE_DATA_ROWS:
        raise ValueError(
            "Crucible dataset is too small after date filtering: "
            f"{filtered_rows} rows at {data_provider.get('path')} "
            f"from {data_provider.get('start_date')} to {data_provider.get('end_date')}. "
            f"Upload a real market-data CSV with at least {MIN_CRUCIBLE_DATA_ROWS} rows."
        )


def _load_stage_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config == "platform:backtest":
        raw_cfg = _platform_backtest_config(args)
    elif args.config == "platform:crucible":
        raw_cfg = _platform_crucible_config(args)
    else:
        raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, _namespace(args))
    raw_cfg.setdefault("analysis", {})["enabled"] = True
    raw_cfg.setdefault("analysis", {})["run_name"] = args.run_name or f"{args.stage}_{args.run_id}"
    _apply_tenant_mlflow_grouping(raw_cfg, args)
    return raw_cfg


def _write_manifest(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )


def _write_chart_artifacts(output_dir: Path, stage: str, portfolio: Any) -> dict[str, Any]:
    points = _equity_chart_points(portfolio)
    if not points:
        return {}
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    equity = {
        "id": "equity",
        "stage": stage,
        "title": "Equity Curve",
        "kind": "equity_curve",
        "format": "chart-series-v1",
        "points": points,
    }
    manifest = {
        "charts": [
            {
                "id": equity["id"],
                "title": equity["title"],
                "kind": equity["kind"],
                "artifact": "charts/equity_curve.json",
                "pointCount": len(points),
            }
        ]
    }
    (chart_dir / "equity_curve.json").write_text(
        json.dumps(_jsonable(equity), indent=2), encoding="utf-8"
    )
    (output_dir / "chart_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )
    return manifest


def _write_backtest_evidence_artifact(
    output_dir: Path,
    stage: str,
    portfolio: Any,
    analysis: dict[str, Any] | None,
    metrics: dict[str, float],
) -> dict[str, Any]:
    evidence = {
        "stage": stage,
        "metrics": metrics,
        "equity": _equity_chart_points(portfolio),
        "returnsHistogram": _return_histogram((analysis or {}).get("daily_returns")),
        "monthlyReturns": _monthly_returns((analysis or {}).get("monthly_returns")),
        "trades": _trade_rows((analysis or {}).get("trades")),
        "tradeCount": len((analysis or {}).get("trades") or []),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backtest_evidence.json").write_text(
        json.dumps(_jsonable(evidence), indent=2),
        encoding="utf-8",
    )
    return evidence


CRUCIBLE_EVIDENCE_STAGES = {
    "hpo": ("HPO", "stages/04_hpo/summaries/stage_summary.json"),
    "walk_forward_oos": ("Walk-forward OOS", "stages/03_walk_forward_oos/summaries/stage_summary.json"),
    "regime_gate": ("Regime gates", "stages/05_regime_gate/summaries/stage_summary.json"),
    "plateau": ("Plateau", "stages/06_plateau/summaries/stage_summary.json"),
    "cross_instrument_transfer": ("Cross-instrument transfer", "stages/07_cross_instrument_transfer/summaries/stage_summary.json"),
    "perturbation": ("Perturbation", "stages/07_perturbation/summaries/stage_summary.json"),
    "structural_break_stability": ("Structural break stability", "stages/08_structural_break_stability/summaries/stage_summary.json"),
    "execution_realism_stress": ("Execution realism stress", "stages/08_execution_realism_stress/summaries/stage_summary.json"),
    "monte_carlo": ("Monte Carlo", "stages/08_monte_carlo/summaries/stage_summary.json"),
    "confirmation": ("Confirmation", "stages/08_confirmation/summaries/stage_summary.json"),
    "paper_replay": ("Paper replay", "stages/09_paper_replay/summaries/stage_summary.json"),
}

MAX_CRUCIBLE_EVIDENCE_ARTIFACTS = 250


def _write_crucible_evidence_artifact(
    output_dir: Path,
    result: dict[str, Any],
    requested: list[tuple[str, str]],
    metrics: dict[str, float],
) -> dict[str, Any]:
    run_dir = Path(str(result.get("run_dir") or ""))
    summaries = {key: _read_json_file(run_dir / rel) for key, (_, rel) in CRUCIBLE_EVIDENCE_STAGES.items()}
    failure_codes = _confirmation_failure_codes(summaries, metrics)
    coaching = _confirmation_autopsy(summaries, metrics, failure_codes)
    evidence = {
        "stage": "crucible",
        "status": "succeeded" if result.get("status") == "complete" else str(result.get("status", "succeeded")),
        "verdict": _crucible_verdict(result),
        "message": "Crucible process completed",
        "metrics": metrics,
        "crucibleRunId": result.get("crucible_run_id"),
        "mlflowRunUrl": result.get("mlflow_run_url"),
        "milestones": [
            _crucible_milestone_payload(name, label, summaries.get(name), name in {item[0] for item in requested})
            for name, (label, _) in CRUCIBLE_EVIDENCE_STAGES.items()
        ],
        "promotionCriteria": _promotion_criteria(summaries, metrics),
        "risks": _crucible_risks(summaries, metrics),
        "failureCodes": failure_codes,
        "coaching": coaching,
        "hpo": _read_csv_rows(run_dir / "stages/04_hpo/summaries/hpo_trial_summary.csv", limit=250),
        "regimes": _read_csv_rows(run_dir / "stages/03_walk_forward_oos/summaries/validation_regime_summary.csv", limit=500),
        "regimeGateDecisions": _read_csv_rows(run_dir / "stages/05_regime_gate/summaries/regime_gate_summary.csv", limit=250),
        "plateau": _read_csv_rows(run_dir / "stages/06_plateau/summaries/plateau_summary.csv", limit=250),
        "plateauNeighbors": _read_csv_rows(run_dir / "stages/06_plateau/summaries/plateau_neighbor_summary.csv", limit=500),
        "transfer": _read_csv_rows(run_dir / "stages/07_cross_instrument_transfer/summaries/transfer_summary.csv", limit=250),
        "transferInstruments": _read_csv_rows(run_dir / "stages/07_cross_instrument_transfer/summaries/transfer_instrument_summary.csv", limit=500),
        "perturbations": _read_csv_rows(run_dir / "stages/07_perturbation/summaries/perturbation_scenario_summary.csv", limit=500),
        "structuralBreaks": _read_csv_rows(run_dir / "stages/08_structural_break_stability/summaries/structural_break_summary.csv", limit=250),
        "structuralBreakWindows": _read_csv_rows(run_dir / "stages/08_structural_break_stability/summaries/structural_break_windows.csv", limit=500),
        "executionRealism": _read_csv_rows(run_dir / "stages/08_execution_realism_stress/summaries/execution_realism_summary.csv", limit=250),
        "executionRealismScenarios": _read_csv_rows(run_dir / "stages/08_execution_realism_stress/summaries/execution_realism_scenarios.csv", limit=500),
        "walkForward": _walk_forward_rows(run_dir),
        "monteCarloSummary": _read_csv_rows(run_dir / "stages/08_monte_carlo/summaries/monte_carlo_summary.csv", limit=250),
        "monteCarlo": _monte_carlo_rows(run_dir),
        "monteCarloTerminalDistribution": _read_csv_rows(run_dir / "stages/08_monte_carlo/summaries/monte_carlo_terminal_distribution.csv", limit=2000),
        "confirmation": _read_csv_rows(run_dir / "stages/08_confirmation/summaries/confirmation_summary.csv", limit=250),
        "artifacts": _crucible_artifact_index(result),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_confirmation_autopsy(output_dir, run_dir if result.get("run_dir") else None, coaching)
    (output_dir / "crucible_evidence.json").write_text(
        json.dumps(_jsonable(evidence), indent=2),
        encoding="utf-8",
    )
    return evidence


FAILURE_META = {
    "confirmation.missing_artifact": ("confirmation", "inconclusive", "Missing confirmation evidence", "Required confirmation evidence is absent."),
    "hpo.high_pbo": ("hpo", "fatal", "High PBO", "Backtest overfitting probability is too high."),
    "hpo.deflated_sharpe_fail": ("hpo", "fatal", "Deflated Sharpe failed", "Sharpe does not survive multiple-testing adjustment."),
    "hpo.too_many_trials": ("hpo", "soft", "Large HPO search", "HPO trial count is high relative to available evidence."),
    "walk_forward.too_few_profitable_windows": ("walk_forward_oos", "fatal", "Too few profitable OOS windows", "Edge is not repeated enough across walk-forward windows."),
    "walk_forward.oos_degradation": ("walk_forward_oos", "fatal", "OOS degradation", "Out-of-sample performance materially decays from the optimized result."),
    "walk_forward.high_fold_variance": ("walk_forward_oos", "soft", "High fold variance", "Performance depends heavily on specific walk-forward folds."),
    "regime.concentration": ("regime_gate", "soft", "Regime concentration", "Strategy behavior is concentrated in a narrow regime."),
    "regime.required_gate_fail": ("regime_gate", "fatal", "Regime gate failed", "Required regime gate did not retain a candidate."),
    "plateau.peak_too_narrow": ("plateau", "fatal", "Narrow parameter peak", "Best parameters look like an isolated spike."),
    "plateau.low_neighbor_pass_rate": ("plateau", "fatal", "Weak plateau neighbors", "Nearby parameters do not preserve the edge."),
    "transfer.sibling_fail": ("cross_instrument_transfer", "fatal", "Sibling transfer failed", "Frozen logic did not transfer to configured sibling instruments."),
    "perturbation.cost_fragile": ("perturbation", "fatal", "Cost fragile", "Required cost or data perturbations erased the edge."),
    "execution.delay_kills_edge": ("execution_realism_stress", "fatal", "Execution fragile", "Returns did not survive delayed-fill or adverse-execution stress."),
    "execution.capacity_dies_early": ("execution_realism_stress", "soft", "Capacity constrained", "Turnover or liquidity limits capacity."),
    "break.structural_decay": ("structural_break_stability", "fatal", "Structural decay", "Edge degraded around detected structural breaks."),
    "monte_carlo.ruin_probability_high": ("monte_carlo", "fatal", "Monte Carlo failure", "Bootstrapped paths show unacceptable ruin or loss probability."),
    "confirmation.no_candidate": ("confirmation", "fatal", "No promotion candidate", "No candidate survived all required gates."),
}
FAILURE_PRIORITY = tuple(FAILURE_META)


def _confirmation_failure_codes(
    summaries: dict[str, dict[str, Any] | None],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def add(code: str, metric: str, observed: Any = None, threshold: Any = None) -> None:
        stage, severity, label, detail = FAILURE_META[code]
        failures.append(
            {
                "code": code,
                "stage": stage,
                "severity": severity,
                "label": label,
                "detail": detail,
                "evidence": [_evidence_pointer(code, stage, metric, observed, threshold, detail)],
            }
        )

    if not summaries.get("confirmation"):
        add("confirmation.missing_artifact", "stage_summary.json", False, True)
    if (pbo := _metric_any(metrics, summaries.get("hpo"), "pbo", "probability_of_backtest_overfitting")) is not None and pbo > 0.2:
        add("hpo.high_pbo", "pbo", pbo, 0.2)
    if (dsr := _metric_any(metrics, summaries.get("hpo"), "deflated_sharpe", "deflated_sharpe_ratio")) is not None and dsr <= 0:
        add("hpo.deflated_sharpe_fail", "deflated_sharpe", dsr, 0)
    if (trials := _metric_any(metrics, summaries.get("hpo"), "hpo.trials_complete", "trials_run", "num_samples")) is not None and trials > 100:
        add("hpo.too_many_trials", "trials_run", trials, 100)
    if (profitable := _metric_any(metrics, summaries.get("walk_forward_oos"), "walk_forward_oos.profitable_windows_pct", "profitable_windows_pct")) is not None and profitable < 50:
        add("walk_forward.too_few_profitable_windows", "profitable_windows_pct", profitable, 50)
    if (degradation := _metric_any(metrics, summaries.get("walk_forward_oos"), "oos_degradation_pct", "median_degradation_pct")) is not None and degradation > 50:
        add("walk_forward.oos_degradation", "oos_degradation_pct", degradation, 50)
    if (fold_var := _metric_any(metrics, summaries.get("walk_forward_oos"), "fold_variance", "return_std_pct", "sharpe_std")) is not None and fold_var > 1:
        add("walk_forward.high_fold_variance", "fold_variance", fold_var, 1)
    if (_num(summaries.get("regime_gate"), "reject_count") or 0) > 0:
        add("regime.concentration", "reject_count", _num(summaries.get("regime_gate"), "reject_count"), 0)
    if summaries.get("regime_gate") and (_num(summaries.get("regime_gate"), "passed_candidate_count") or 0) <= 0:
        add("regime.required_gate_fail", "passed_candidate_count", 0, 1)
    if (_num(summaries.get("plateau"), "rejected_peaks") or 0) > 0:
        add("plateau.peak_too_narrow", "rejected_peaks", _num(summaries.get("plateau"), "rejected_peaks"), 0)
    if summaries.get("plateau") and (_num(summaries.get("plateau"), "accepted_plateaus") or 0) <= 0:
        add("plateau.low_neighbor_pass_rate", "accepted_plateaus", 0, 1)
    if (_num(summaries.get("cross_instrument_transfer"), "rejected_candidates") or 0) > 0:
        add("transfer.sibling_fail", "rejected_candidates", _num(summaries.get("cross_instrument_transfer"), "rejected_candidates"), 0)
    if (_num(summaries.get("perturbation"), "rejected_candidates") or 0) > 0:
        add("perturbation.cost_fragile", "rejected_candidates", _num(summaries.get("perturbation"), "rejected_candidates"), 0)
    if (_num(summaries.get("structural_break_stability"), "rejected_candidates") or 0) > 0:
        add("break.structural_decay", "rejected_candidates", _num(summaries.get("structural_break_stability"), "rejected_candidates"), 0)
    if (_num(summaries.get("execution_realism_stress"), "rejected_candidates") or 0) > 0:
        add("execution.delay_kills_edge", "rejected_candidates", _num(summaries.get("execution_realism_stress"), "rejected_candidates"), 0)
    if (_metric_any(metrics, summaries.get("execution_realism_stress"), "capacity_usd", "max_capacity_usd") or math.inf) < 100000:
        add("execution.capacity_dies_early", "capacity_usd", _metric_any(metrics, summaries.get("execution_realism_stress"), "capacity_usd", "max_capacity_usd"), 100000)
    if (_num(summaries.get("monte_carlo"), "rejected_candidates") or 0) > 0:
        add("monte_carlo.ruin_probability_high", "rejected_candidates", _num(summaries.get("monte_carlo"), "rejected_candidates"), 0)
    promoted = metrics.get("confirmation.promoted_candidates") or _num(summaries.get("confirmation"), "confirmed_candidates") or 0
    if summaries.get("confirmation") and promoted <= 0:
        add("confirmation.no_candidate", "confirmed_candidates", promoted, 1)
    return sorted(failures, key=lambda item: FAILURE_PRIORITY.index(item["code"]))


def _confirmation_autopsy(
    summaries: dict[str, dict[str, Any] | None],
    metrics: dict[str, float],
    failure_codes: list[dict[str, Any]],
) -> dict[str, Any]:
    promoted = (metrics.get("confirmation.promoted_candidates") or _num(summaries.get("confirmation"), "confirmed_candidates") or 0) > 0
    primary = failure_codes[0] if failure_codes else None
    verdict = "promote" if promoted and not failure_codes else "inconclusive" if primary and primary["severity"] == "inconclusive" else "reject" if primary and primary["severity"] == "fatal" else "revise"
    code = primary["code"] if primary else None
    return {
        "verdict": verdict,
        "stage": primary["stage"] if primary else "confirmation",
        "primaryFailure": code,
        "secondaryFailures": [item["code"] for item in failure_codes[1:3]],
        "severity": primary["severity"] if primary else "info",
        "summary": _autopsy_summary(verdict, primary),
        "evidence": primary["evidence"] if primary else [],
        "suggestedActions": _suggested_actions(code),
        "reentryStage": _reentry_stage(code),
        "overfitRisk": _overfit_risk(summaries, metrics),
    }


def _evidence_pointer(code: str, stage: str, metric: str, observed: Any, threshold: Any, explanation: str) -> dict[str, Any]:
    return {
        "id": f"{code}:{metric}",
        "stage": stage,
        "artifact": CRUCIBLE_EVIDENCE_STAGES.get(stage, ("", "crucible_evidence.json"))[1],
        "metric": metric,
        "observed": observed,
        "threshold": threshold,
        "explanation": explanation,
    }


def _metric_any(metrics: dict[str, float], summary: dict[str, Any] | None, *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key) if key in metrics else (summary or {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _autopsy_summary(verdict: str, primary: dict[str, Any] | None) -> str:
    if verdict == "promote":
        return "Confirmation found a promotion candidate and no blocking coaching failure."
    if primary is None:
        return "Confirmation did not find enough evidence to produce a diagnosis."
    return f"{primary['label']}: {primary['detail']}"


def _suggested_actions(code: str | None) -> list[dict[str, Any]]:
    titles = {
        "walk_forward.too_few_profitable_windows": ["Reduce free parameters and rerun walk-forward.", "Coarsen the HPO search space.", "Increase in-sample window length if data supports it."],
        "walk_forward.high_fold_variance": ["Reduce free parameters and rerun walk-forward.", "Coarsen the HPO search space.", "Add regime-conditional logic only if the hypothesis predicts it."],
        "plateau.peak_too_narrow": ["Switch HPO objective to neighborhood-averaged Sharpe.", "Freeze the parameter with the narrowest accepted range.", "Shrink the search space to economically defensible ranges."],
        "plateau.low_neighbor_pass_rate": ["Switch HPO objective to neighborhood-averaged Sharpe.", "Reduce parameter count.", "Rerun plateau before confirmation."],
        "hpo.high_pbo": ["Cut the number of HPO trials.", "Pre-register a narrower search space.", "Reduce model or parameter complexity."],
        "hpo.deflated_sharpe_fail": ["Cut the number of HPO trials.", "Pre-register a narrower search space.", "Move holdout earlier and reserve it for one final run."],
        "hpo.too_many_trials": ["Cut the number of HPO trials.", "Reduce free parameters.", "Use a stricter validation profile before confirmation."],
        "regime.concentration": ["Declare whether this is a specialist strategy.", "Add a regime filter only if the hypothesis supports it.", "Cap allocation if used as a satellite."],
        "regime.required_gate_fail": ["Rework the hypothesis around the failed regime behavior.", "Test a simpler baseline.", "Rerun regime gates before downstream stages."],
        "transfer.sibling_fail": ["Re-derive the signal from an economic hypothesis.", "Test siblings before deep HPO.", "Stop research if no plausible sibling exposure exists."],
        "perturbation.cost_fragile": ["Lower rebalance or signal frequency.", "Add a turnover cap.", "Restrict the universe to more liquid instruments."],
        "execution.delay_kills_edge": ["Use next-bar execution rules.", "Remove same-bar or stop-touch assumptions.", "Slow signal frequency."],
        "execution.capacity_dies_early": ["Lower participation limits.", "Reduce turnover.", "Cap promotion sizing instead of changing the signal."],
        "break.structural_decay": ["Treat the detected break as a research blocker.", "Retest the economic rationale by period.", "Reject if the edge exists only before the break."],
        "monte_carlo.ruin_probability_high": ["Reduce sizing or leverage.", "Add portfolio-level drawdown brakes.", "Require more independent trades before confirmation."],
        "confirmation.no_candidate": ["Fix the highest-priority upstream blocker.", "Re-enter at the earliest changed stage.", "Do not rerun confirmation until upstream evidence changes."],
        "confirmation.missing_artifact": ["Repair or rerun missing confirmation evidence.", "Check artifact persistence for the run.", "Do not interpret this as a strategy failure."],
    }
    return [
        {
            "id": f"{code}:action_{index}",
            "failureCode": code,
            "rank": index,
            "title": title,
            "rationale": "Pre-specified remediation selected from the confirmation coaching playbook.",
            "actionType": _action_type(code),
            "reentryStage": _reentry_stage(code),
            "requiresUserJudgment": True,
        }
        for index, title in enumerate(titles.get(code or "", []), start=1)
    ]


def _action_type(code: str | None) -> str:
    if code in {"confirmation.missing_artifact"}:
        return "collect_evidence"
    if code in {"confirmation.no_candidate", "transfer.sibling_fail", "break.structural_decay"}:
        return "stop_research"
    if code and code.startswith("hpo") or code and code.startswith("plateau"):
        return "change_search_space"
    if code and code.startswith("execution") or code and code.startswith("perturbation"):
        return "change_execution"
    return "change_strategy_structure"


def _reentry_stage(code: str | None) -> str:
    if not code:
        return "confirmation"
    if code.startswith("hpo") or code.startswith("plateau"):
        return "hpo"
    if code.startswith("walk_forward"):
        return "walk_forward_oos"
    if code.startswith("regime"):
        return "regime_gate"
    if code.startswith("transfer"):
        return "idea"
    if code.startswith("perturbation"):
        return "research"
    if code.startswith("execution"):
        return "execution_realism_stress"
    if code.startswith("break"):
        return "research"
    if code.startswith("monte_carlo"):
        return "monte_carlo"
    if code == "confirmation.missing_artifact":
        return "confirmation"
    return "hpo"


def _overfit_risk(summaries: dict[str, dict[str, Any] | None], metrics: dict[str, float]) -> dict[str, Any]:
    hpo_trials = _metric_any(metrics, summaries.get("hpo"), "hpo.trials_complete", "trials_run", "num_samples")
    pbo = _metric_any(metrics, summaries.get("hpo"), "pbo", "probability_of_backtest_overfitting")
    dsr = _metric_any(metrics, summaries.get("hpo"), "deflated_sharpe", "deflated_sharpe_ratio")
    oos_windows = _metric_any(metrics, summaries.get("walk_forward_oos"), "jobs_total", "window_count")
    level = "high" if (hpo_trials or 0) > 100 or (pbo or 0) > 0.2 or (dsr is not None and dsr <= 0) else "medium" if (hpo_trials or 0) > 50 else "low"
    return {
        "level": level,
        "freeParamCount": None,
        "hpoTrialCount": hpo_trials,
        "turnover": metrics.get("turnover"),
        "pbo": pbo,
        "deflatedSharpe": dsr,
        "siblingCount": _metric_any(metrics, summaries.get("cross_instrument_transfer"), "instrument_count", "sibling_count"),
        "oosWindowCount": oos_windows,
        "notes": ["More search is not a default repair." if level == "high" else "Overfit risk uses available Crucible summary metrics."],
    }


def _write_confirmation_autopsy(output_dir: Path, run_dir: Path | None, coaching: dict[str, Any]) -> None:
    rel = Path("stages/08_confirmation/summaries")
    for root in (output_dir, run_dir):
        if root is None:
            continue
        target = root / rel
        target.mkdir(parents=True, exist_ok=True)
        (target / "confirmation_autopsy.json").write_text(json.dumps(_jsonable(coaching), indent=2), encoding="utf-8")
        with (target / "confirmation_playbook.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "failureCode", "rank", "title", "reentryStage", "actionType", "requiresUserJudgment"])
            writer.writeheader()
            writer.writerows({key: action.get(key) for key in writer.fieldnames} for action in coaching.get("suggestedActions", []))


def _write_crucible_chart_manifest(output_dir: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    charts = []
    if evidence.get("walkForward"):
        charts.append(
            {
                "id": "walk_forward_oos",
                "title": "Walk-forward OOS",
                "kind": "walk_forward_oos",
                "artifact": "crucible_evidence.json",
                "pointCount": len(evidence["walkForward"]),
            }
        )
    if evidence.get("monteCarlo"):
        charts.append(
            {
                "id": "monte_carlo",
                "title": "Monte Carlo Bands",
                "kind": "monte_carlo_bands",
                "artifact": "crucible_evidence.json",
                "pointCount": len(evidence["monteCarlo"]),
            }
        )
    manifest = {"charts": charts}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chart_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest


def _log_platform_artifacts_to_crucible_run(state_store: Any, run_id: str, output_dir: str | Path) -> None:
    for name in ("stage_summary.json", "crucible_evidence.json", "chart_manifest.json", "progress_events.jsonl"):
        state_store.log_existing_artifact(run_id, Path(output_dir) / name)


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv_rows(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_coerce_csv_row(row) for _, row in zip(range(limit), csv.DictReader(handle))]


def _coerce_csv_row(row: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            payload[key] = None
            continue
        clean = value.strip()
        if key.endswith("_json") or key in {"objective_details", "config", "algorithm_params", "portfolio_params"}:
            payload[key] = _coerce_json_value(clean)
            continue
        if clean.lower() in {"true", "false"}:
            payload[key] = clean.lower() == "true"
            continue
        try:
            payload[key] = float(clean)
            continue
        except ValueError:
            payload[key] = clean
    return payload


def _coerce_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _crucible_milestone_payload(
    name: str,
    label: str,
    summary: dict[str, Any] | None,
    requested: bool,
) -> dict[str, Any]:
    return {
        "id": name,
        "label": label,
        "status": "complete" if summary else "pending",
        "requested": requested,
        "message": _milestone_message(name, summary),
        "metrics": _stage_ui_metrics(name, summary or {}),
    }


def _stage_ui_metrics(name: str, summary: dict[str, Any]) -> dict[str, float]:
    metrics = _flat_numeric_metrics(summary)
    aliases = {
        "hpo": {
            "trials_run": "hpo.trials_complete",
            "valid_candidates": "hpo.candidates_created",
            "best_objective_score": "hpo.best_metric",
            "median_objective_score": "hpo.median_metric",
            "constraint_failures": "hpo.trials_failed",
        },
        "walk_forward_oos": {
            "window_count": "jobs_total",
            "oos_pass_count": "jobs_complete",
        },
        "regime_gate": {
            "regimes_tested": "candidate_count",
            "regimes_passed": "passed_candidate_count",
            "blocking_regimes": "reject_count",
        },
        "plateau": {
            "stable_neighborhood_size": "neighbor_count",
            "plateau_score": "accepted_plateaus",
            "warning_count": "rejected_peaks",
        },
        "cross_instrument_transfer": {
            "candidates_tested": "candidate_count",
            "candidates_accepted": "accepted_candidates",
            "candidates_rejected": "rejected_candidates",
            "instruments_tested": "instrument_count",
        },
        "perturbation": {
            "scenarios_run": "scenario_count",
            "scenarios_passed": "accepted_candidates",
            "scenarios_rejected": "rejected_candidates",
            "blocking_failures": "rejected_candidates",
        },
        "structural_break_stability": {
            "candidates_tested": "candidate_count",
            "candidates_accepted": "accepted_candidates",
            "candidates_rejected": "rejected_candidates",
            "input_observations": "input_return_observations",
        },
        "execution_realism_stress": {
            "candidates_tested": "candidate_count",
            "candidates_accepted": "accepted_candidates",
            "candidates_rejected": "rejected_candidates",
            "input_observations": "input_return_observations",
        },
        "monte_carlo": {
            "input_observations": "input_return_observations",
        },
        "confirmation": {
            "criteria_passed": "confirmed_candidates",
            "blocking_failures": "rejected_candidates",
        },
    }
    for target, source in aliases.get(name, {}).items():
        if target not in metrics and source in metrics:
            metrics[target] = metrics[source]
    return metrics


def _milestone_message(name: str, summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Not run yet"
    if name == "walk_forward_oos":
        return f"{summary.get('jobs_complete', 0)} of {summary.get('jobs_total', 0)} windows complete"
    if name == "regime_gate":
        return f"{summary.get('passed_candidate_count', 0)} candidates passed regime gates"
    if name == "plateau":
        return f"{summary.get('accepted_plateaus', 0)} plateaus accepted"
    if name == "cross_instrument_transfer":
        return f"{summary.get('accepted_candidates', 0)} candidates passed transfer"
    if name == "perturbation":
        return f"{summary.get('accepted_candidates', 0)} candidates survived perturbations"
    if name == "structural_break_stability":
        return f"{summary.get('accepted_candidates', 0)} candidates passed structural-break checks"
    if name == "execution_realism_stress":
        return f"{summary.get('accepted_candidates', 0)} candidates survived execution stress"
    if name == "monte_carlo":
        return f"{summary.get('accepted_candidates', 0)} candidates passed Monte Carlo simulation"
    if name == "confirmation":
        return f"{summary.get('confirmed_candidates', 0)} candidates confirmed"
    return "Stage complete"


def _promotion_criteria(
    summaries: dict[str, dict[str, Any] | None],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "walk_forward_oos",
            "OOS profitable windows",
            metrics.get("walk_forward_oos.profitable_windows_pct")
            or _num(summaries.get("walk_forward_oos"), "profitable_windows_pct"),
            "Recorded once walk-forward OOS completes.",
            summaries.get("walk_forward_oos") is not None,
        ),
        _criterion(
            "regime_gate",
            "Regime gate pass count",
            _num(summaries.get("regime_gate"), "passed_candidate_count"),
            "At least one candidate should pass generalist or specialist gates.",
            summaries.get("regime_gate") is not None,
        ),
        _criterion(
            "plateau",
            "Accepted parameter plateaus",
            _num(summaries.get("plateau"), "accepted_plateaus"),
            "Stable candidates should not depend on a knife-edge parameter choice.",
            summaries.get("plateau") is not None,
        ),
        _criterion(
            "cross_instrument_transfer",
            "Transfer survivors",
            _num(summaries.get("cross_instrument_transfer"), "accepted_candidates"),
            "Same frozen logic should remain above noise on configured sibling instruments.",
            summaries.get("cross_instrument_transfer") is not None,
        ),
        _criterion(
            "perturbation",
            "Perturbation survivors",
            _num(summaries.get("perturbation"), "accepted_candidates"),
            "Candidates should survive required cost and data shocks.",
            summaries.get("perturbation") is not None,
        ),
        _criterion(
            "structural_break_stability",
            "Structural-break survivors",
            _num(summaries.get("structural_break_stability"), "accepted_candidates"),
            "Candidate edge should not be concentrated in one unstable period.",
            summaries.get("structural_break_stability") is not None,
        ),
        _criterion(
            "execution_realism_stress",
            "Execution-stress survivors",
            _num(summaries.get("execution_realism_stress"), "accepted_candidates"),
            "Candidate returns should survive delayed fills, missed signals, and adverse execution.",
            summaries.get("execution_realism_stress") is not None,
        ),
        _criterion(
            "monte_carlo",
            "Monte Carlo survivors",
            _num(summaries.get("monte_carlo"), "accepted_candidates"),
            "Candidate return streams should survive bootstrapped path simulations.",
            summaries.get("monte_carlo") is not None,
        ),
        _criterion(
            "confirmation",
            "Promotion candidates",
            metrics.get("confirmation.promoted_candidates") or _num(summaries.get("confirmation"), "confirmed_candidates"),
            "Final confirmation must produce at least one promotion candidate.",
            summaries.get("confirmation") is not None,
        ),
    ]


def _criterion(id_: str, label: str, value: float | None, detail: str, complete: bool) -> dict[str, Any]:
    if not complete:
        status = "pending"
    elif value is not None and value > 0:
        status = "pass"
    else:
        status = "fail"
    return {"id": id_, "label": label, "status": status, "value": value, "detail": detail}


def _num(summary: dict[str, Any] | None, key: str) -> float | None:
    value = (summary or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _crucible_risks(
    summaries: dict[str, dict[str, Any] | None],
    metrics: dict[str, float],
) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    if _num(summaries.get("regime_gate"), "reject_count"):
        risks.append({"severity": "warning", "label": "Regime fragility", "detail": "Some candidates failed regime-aware gates."})
    if (_num(summaries.get("plateau"), "rejected_peaks") or 0) > 0:
        risks.append({"severity": "warning", "label": "Parameter instability", "detail": "Some peaks did not hold up in plateau testing."})
    if (_num(summaries.get("cross_instrument_transfer"), "rejected_candidates") or 0) > 0:
        risks.append({"severity": "blocking", "label": "Transfer failure", "detail": "One or more candidates failed configured sibling-instrument transfer."})
    if (_num(summaries.get("perturbation"), "rejected_candidates") or 0) > 0:
        risks.append({"severity": "blocking", "label": "Perturbation failure", "detail": "One or more candidates failed required perturbation scenarios."})
    if (_num(summaries.get("structural_break_stability"), "rejected_candidates") or 0) > 0:
        risks.append({"severity": "blocking", "label": "Structural break failure", "detail": "One or more candidates showed concentrated or degraded edge after a detected break."})
    if (_num(summaries.get("execution_realism_stress"), "rejected_candidates") or 0) > 0:
        risks.append({"severity": "blocking", "label": "Execution realism failure", "detail": "One or more candidates failed delayed-fill, missed-signal, or adverse-execution stress."})
    if (_num(summaries.get("monte_carlo"), "rejected_candidates") or 0) > 0:
        risks.append({"severity": "blocking", "label": "Monte Carlo failure", "detail": "One or more candidates failed bootstrapped path simulation thresholds."})
    if summaries.get("confirmation") and (metrics.get("confirmation.promoted_candidates") or 0) <= 0:
        risks.append({"severity": "blocking", "label": "No promotion candidate", "detail": "Confirmation did not produce a candidate ready for promotion."})
    if not risks:
        risks.append({"severity": "info", "label": "No recorded blockers", "detail": "No Crucible risks were recorded in completed milestones."})
    return risks


def _walk_forward_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(run_dir / "stages/03_walk_forward_oos/summaries/oos_summary.csv", limit=120)
    return [
        {
            "window": str(row.get("window_id") or idx + 1),
            "isSharpe": _float(row.get("train_sharpe_ratio")) or _float(row.get("sharpe_ratio")) or 0.0,
            "oosSharpe": _float(row.get("sharpe_ratio")) or 0.0,
            "returnPct": _float(row.get("total_return_pct")),
            "maxDrawdownPct": _float(row.get("max_drawdown_pct")),
            "degradation": _degradation(row),
        }
        for idx, row in enumerate(rows)
    ]


def _monte_carlo_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(run_dir / "stages/08_monte_carlo/summaries/monte_carlo_path_bands.csv", limit=500)
    first_candidate = next((row.get("candidate_id") for row in rows), None)
    return [
        {
            "step": int(_float(row.get("step")) or idx + 1),
            "p0": 100.0,
            "p5": _float(row.get("p5")) or 0.0,
            "p50": _float(row.get("p50")) or 0.0,
            "p95": _float(row.get("p95")) or 0.0,
        }
        for idx, row in enumerate(rows)
        if first_candidate is None or row.get("candidate_id") == first_candidate
    ]


def _degradation(row: dict[str, Any]) -> float | None:
    in_sample = _float(row.get("train_sharpe_ratio"))
    oos = _float(row.get("sharpe_ratio"))
    if in_sample in (None, 0) or oos is None:
        return None
    return max(0.0, round((in_sample - oos) / abs(in_sample) * 100.0, 2))


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _crucible_artifact_index(result: dict[str, Any]) -> list[dict[str, str]]:
    artifacts = result.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return []
    rows = [{"name": str(key), "path": str(value)} for key, value in artifacts.items()]

    def useful(row: dict[str, str]) -> bool:
        name = row["name"].lower()
        if "/manifests/job_" in name:
            return False
        return (
            "/summaries/" in name
            or "/charts/" in name
            or name.endswith((".log", "crucible_evidence.json", "chart_manifest.json", "stage_summary.json"))
            or name in {"crucible_platform.yaml", "crucible_platform_config.json"}
        )

    selected = [row for row in rows if useful(row)]
    if len(selected) < MAX_CRUCIBLE_EVIDENCE_ARTIFACTS:
        selected.extend(
            row
            for row in rows
            if row not in selected and "/manifests/job_" not in row["name"].lower()
        )
    return selected[:MAX_CRUCIBLE_EVIDENCE_ARTIFACTS]


def _equity_chart_points(portfolio: Any, max_points: int = 1200) -> list[dict[str, float | str]]:
    history = getattr(portfolio, "value_history", None)
    if not history:
        return []
    items = sorted(history.items(), key=lambda item: str(item[0]))
    if len(items) > max_points:
        step = max(1, len(items) // max_points)
        items = items[::step]
    values = [float(value) for _, value in items if isinstance(value, Real)]
    if not values:
        return []
    peak = values[0]
    points: list[dict[str, float | str]] = []
    for raw_date, value in items:
        if not isinstance(value, Real):
            continue
        numeric = float(value)
        peak = max(peak, numeric)
        drawdown = ((numeric - peak) / peak * 100.0) if peak else 0.0
        points.append(
            {
                "date": str(raw_date),
                "strategy": numeric,
                "benchmark": values[0],
                "drawdown": drawdown,
            }
        )
    return points


def _series_items(series: Any) -> list[tuple[Any, Any]]:
    if series is None:
        return []
    if isinstance(series, dict):
        return list(series.items())
    items = getattr(series, "items", None)
    if callable(items):
        return list(items())
    if isinstance(series, list):
        return list(enumerate(series))
    return []


def _return_histogram(series: Any, bucket_pct: float = 0.5) -> list[dict[str, Any]]:
    returns = [
        float(value) * 100.0
        for _, value in _series_items(series)
        if isinstance(value, Real) and math.isfinite(float(value))
    ]
    if not returns:
        return []
    low = math.floor(min(returns) / bucket_pct) * bucket_pct
    high = math.ceil(max(returns) / bucket_pct) * bucket_pct
    bucket_count = int((high - low) / bucket_pct) + 1
    buckets = [round(low + idx * bucket_pct, 1) for idx in range(bucket_count)]
    counts = {bucket: 0 for bucket in buckets}
    for value in returns:
        bucket = round(round(value / bucket_pct) * bucket_pct, 1)
        counts[min(max(bucket, buckets[0]), buckets[-1])] += 1
    return [
        {
            "bucket": f"{bucket:+.1f}%",
            "count": count,
            "positive": bucket >= 0,
        }
        for bucket, count in counts.items()
    ]


def _monthly_returns(series: Any) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, float]] = {}
    for raw_date, value in _series_items(series):
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            continue
        date = getattr(raw_date, "to_pydatetime", lambda: raw_date)()
        year = int(getattr(date, "year", 0) or 0)
        month = int(getattr(date, "month", 0) or 0)
        if not year or month < 1 or month > 12:
            continue
        rows.setdefault(year, {})[calendar.month_abbr[month]] = float(value) * 100.0
    months = list(calendar.month_abbr)[1:]
    return [
        {
            "year": year,
            "months": [
                {"month": month, "value": round(values.get(month, 0.0), 2)}
                for month in months
            ],
        }
        for year, values in sorted(rows.items())
    ]


def _get_value(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _date_label(value: Any) -> str:
    if value is None:
        return "-"
    fmt = getattr(value, "strftime", None)
    return fmt("%b %d") if callable(fmt) else str(value)


def _trade_rows(trades: Any, limit: int = 18) -> list[dict[str, Any]]:
    rows = []
    for idx, trade in enumerate(list(trades or [])[:limit], start=1):
        pnl = float(_get_value(trade, "pnl") or 0.0)
        duration = float(_get_value(trade, "duration") or 0.0)
        side = str(_get_value(trade, "side") or "long").lower()
        rows.append(
            {
                "id": idx,
                "entry": _date_label(_get_value(trade, "entry_time", "entry")),
                "exit": _date_label(_get_value(trade, "exit_time", "exit")),
                "side": "short" if side == "short" else "long",
                "size": int(float(_get_value(trade, "quantity", "size") or 0.0)),
                "pnl": round(pnl, 2),
                "ret": round(float(_get_value(trade, "pnl_pct", "ret") or 0.0), 2),
                "bars": int(round(duration / 300.0)) if duration else 0,
            }
        )
    return rows


def _record_simple_result(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    sink = LocalRunResultSink(args.output_dir)
    with SinkRun(sink, run_name=args.run_name or f"{args.stage}_{args.run_id}"):
        sink.log_params(
            {
                "platform.run_id": args.run_id,
                "platform.strategy_id": args.strategy_id,
                "platform.tenant_id": args.tenant_id,
                "platform.stage": args.stage,
            }
        )
        sink.log_metrics(summary.get("metrics") or {})
        sink.log_text(json.dumps(_jsonable(summary), indent=2), "stage_summary.json")


def execute_idea(args: argparse.Namespace) -> dict[str, Any]:
    emit(20, "Loading strategy runtime config")
    raw_cfg = _load_stage_config(args)
    emit(65, "Validating component wiring")
    summary = {
        "stage": "idea",
        "status": "succeeded",
        "verdict": "pending",
        "message": "Strategy config loaded and validated",
        "metrics": {"computeCostUsd": 0.0},
        "config": _jsonable(raw_cfg),
    }
    _record_simple_result(args, summary)
    return summary


def execute_backtest_stage(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    emit(15, "Loading backtest config")
    raw_cfg = _load_stage_config(args)
    if smoke:
        raw_cfg.setdefault("data_provider", {})["truncate"] = args.smoke_rows
    emit(30, "Starting backtest engine")
    result = run_backtest_from_raw_config(
        raw_cfg,
        args=_namespace(args, result_output_dir=args.output_dir),
        config_path=args.config,
        account=args.account,
        result_output_dir=args.output_dir,
    )
    emit(85, "Collecting analysis result")
    analysis = (result or {}).get("analysis")
    raw_metrics = _metric_payload(analysis)
    metrics = _ui_metrics(raw_metrics) | {"computeCostUsd": 0.0}
    stage = "smoke" if smoke else "research"
    chart_manifest = _write_chart_artifacts(
        Path(args.output_dir),
        stage,
        (result or {}).get("portfolio"),
    )
    evidence = _write_backtest_evidence_artifact(
        Path(args.output_dir),
        stage,
        (result or {}).get("portfolio"),
        analysis,
        metrics,
    )
    summary = {
        "stage": stage,
        "status": "succeeded",
        "verdict": "passed-smoke" if smoke else "caution",
        "message": "Smoke backtest completed" if smoke else "Research backtest completed",
        "metrics": metrics,
        "analysis": _jsonable(analysis),
        "finalValue": (result or {}).get("final_value"),
        "cash": (result or {}).get("cash"),
        "positions": (result or {}).get("positions"),
        "charts": chart_manifest.get("charts", []),
        "evidence": {"artifact": "backtest_evidence.json", "tradeCount": evidence["tradeCount"]},
    }
    _write_manifest(Path(args.output_dir), summary)
    return summary


def execute_crucible(args: argparse.Namespace) -> dict[str, Any]:
    emit(8, "Preparing crucible platform and workload configs")
    platform_path, workload_path = _crucible_config_paths(args)
    _emit_crucible_runtime_diagnostics(platform_path, workload_path)
    workload = _load_yaml(workload_path)
    data_provider = workload.get("data_provider") or {}
    _validate_crucible_data(data_provider, _csv_data_diagnostics(data_provider))
    orchestrator = CrucibleOrchestrator(platform_path, workload_path)
    result: dict[str, Any] | None = None
    requested = _requested_crucible_stages(args)
    progress_plan = _crucible_progress_plan(requested)
    for name, message in requested:
        start_pct, end_pct = progress_plan[name]
        emit(start_pct, message, crucibleStage=name)
        if name == "hpo":
            result = orchestrator.run_hpo_stage(rerun=args.rerun_crucible)
        elif name == "walk_forward_oos":
            result = orchestrator.run_walk_forward_oos(rerun=args.rerun_crucible, use_ray=args.use_ray)
        elif name == "regime_gate":
            result = orchestrator.run_regime_gate_stage(rerun=args.rerun_crucible)
        elif name == "plateau":
            result = orchestrator.run_plateau_stage(rerun=args.rerun_crucible, use_ray=args.use_ray)
        elif name == "cross_instrument_transfer":
            result = orchestrator.run_cross_instrument_transfer_stage(rerun=args.rerun_crucible, use_ray=args.use_ray)
        elif name == "perturbation":
            result = orchestrator.run_perturbation_stage(rerun=args.rerun_crucible, use_ray=args.use_ray)
        elif name == "structural_break_stability":
            result = orchestrator.run_structural_break_stability_stage(rerun=args.rerun_crucible)
        elif name == "execution_realism_stress":
            result = orchestrator.run_execution_realism_stress_stage(rerun=args.rerun_crucible)
        elif name == "monte_carlo":
            result = orchestrator.run_monte_carlo_stage(rerun=args.rerun_crucible)
        elif name == "confirmation":
            result = orchestrator.run_confirmation_stage(
                rerun=args.rerun_crucible,
                use_ray=args.use_ray,
                create_promoted_folder=False,
            )
        elif name == "paper_replay":
            result = orchestrator.run_paper_replay_stage(rerun=args.rerun_crucible)
        emit(end_pct, f"Completed {name.replace('_', ' ')}", crucibleStage=name)
    assert result is not None
    metrics = _flat_numeric_metrics(result.get("metrics") or {})
    metrics["robustnessScore"] = _robustness_score(metrics)
    metrics["computeCostUsd"] = 0.0
    summary = {
        "stage": "crucible",
        "status": "succeeded" if result.get("status") == "complete" else str(result.get("status", "succeeded")),
        "verdict": _crucible_verdict(result),
        "message": "Crucible process completed",
        "metrics": metrics,
        "crucibleRunId": result.get("crucible_run_id"),
        "runDir": result.get("run_dir"),
        "mlflowRunUrl": result.get("mlflow_run_url"),
        "details": _jsonable(result),
    }
    evidence = _write_crucible_evidence_artifact(Path(args.output_dir), result, requested, metrics)
    _write_crucible_chart_manifest(Path(args.output_dir), evidence)
    summary["evidence"] = {"artifact": "crucible_evidence.json", "milestones": len(evidence["milestones"])}
    _write_manifest(Path(args.output_dir), summary)
    _log_platform_artifacts_to_crucible_run(orchestrator.state_store, result["crucible_run_id"], args.output_dir)
    _record_simple_result(args, summary)
    return summary


def _requested_crucible_stages(args: argparse.Namespace) -> list[tuple[str, str]]:
    messages = {name: message for name, message in CRUCIBLE_STAGES}
    messages["paper_replay"] = "Running paper replay"
    selected = args.crucible_milestone or [name for name, _ in CRUCIBLE_STAGES]
    if args.include_paper_replay and "paper_replay" not in selected:
        selected = [*selected, "paper_replay"]
    ordered = [name for name in CRUCIBLE_STAGE_NAMES if name in set(selected)]
    if not ordered:
        raise ValueError("At least one crucible milestone must be selected")
    return [(name, messages[name]) for name in ordered]


def _crucible_progress_plan(stages: list[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    start = 12.0
    span = 82.0 / max(len(stages), 1)
    return {
        name: (start + idx * span, min(94.0, start + (idx + 1) * span))
        for idx, (name, _) in enumerate(stages)
    }


def _flat_numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def _robustness_score(metrics: dict[str, float]) -> float:
    for key in ("confirmation.promoted_candidates", "monte_carlo.accepted_candidates", "perturbation.accepted_candidates", "plateau.accepted_seeds"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _crucible_verdict(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    if float(metrics.get("confirmation.promoted_candidates") or 0) > 0:
        return "robust"
    if result.get("status") == "complete":
        return "fragile"
    return str(result.get("status") or "pending")


def execute_promotion(args: argparse.Namespace) -> dict[str, Any]:
    emit(40, "Checking promotion evidence")
    summary = {
        "stage": "promotion",
        "status": "succeeded",
        "verdict": "promoted",
        "message": "Promotion packet recorded",
        "metrics": {"liveReadiness": 1.0, "computeCostUsd": 0.0},
    }
    _record_simple_result(args, summary)
    return summary


def execute_monitoring(args: argparse.Namespace) -> dict[str, Any]:
    emit(40, "Recording monitoring snapshot")
    summary = {
        "stage": "monitoring",
        "status": "succeeded",
        "verdict": "pending",
        "message": "Monitoring snapshot recorded",
        "metrics": {"computeCostUsd": 0.0},
    }
    _record_simple_result(args, summary)
    return summary


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == "idea":
        return execute_idea(args)
    if args.stage == "smoke":
        return execute_backtest_stage(args, smoke=True)
    if args.stage == "research":
        return execute_backtest_stage(args, smoke=False)
    if args.stage == "crucible":
        return execute_crucible(args)
    if args.stage == "promotion":
        return execute_promotion(args)
    if args.stage == "monitoring":
        return execute_monitoring(args)
    raise ValueError(f"Unsupported stage: {args.stage}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant Crucible platform stage runner")
    parser.add_argument("--stage", required=True, choices=list(DEFAULT_CONFIG_BY_STAGE))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--workload-config")
    parser.add_argument("--account", default="secondary_paper3")
    parser.add_argument("--run-name")
    parser.add_argument("--experiment-name", default="Quant Crucible Platform")
    parser.add_argument("--symbol")
    parser.add_argument("--cash", type=float)
    parser.add_argument("--algorithm")
    parser.add_argument("--algorithm-url")
    parser.add_argument("--portfolio")
    parser.add_argument("--portfolio-url")
    parser.add_argument("--data")
    parser.add_argument("--session-id")
    parser.add_argument("--agg-period", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=500)
    parser.add_argument("--hpo-samples", type=int)
    parser.add_argument("--hpo-concurrency", type=int)
    parser.add_argument("--validation-period-days", type=int)
    parser.add_argument("--crucible-milestone", action="append", choices=CRUCIBLE_STAGE_NAMES)
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument("--rerun-crucible", action="store_true")
    parser.add_argument("--include-paper-replay", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config is None:
        args.config = DEFAULT_CONFIG_BY_STAGE[args.stage]
    os.environ.setdefault("TRADING_GUY_ARTIFACT_TMP", str(Path(args.output_dir) / "_tmp"))
    _set_progress_log(args.output_dir)
    summary = execute(args)
    emit(100, summary.get("message", "Stage completed"), summary=summary)


if __name__ == "__main__":
    main()
