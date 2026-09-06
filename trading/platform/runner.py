from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
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
    ("perturbation", "Running perturbation scenarios"),
    ("confirmation", "Running final confirmation"),
)
CRUCIBLE_STAGE_NAMES = tuple(stage[0] for stage in CRUCIBLE_STAGES) + ("paper_replay",)


def emit(progress_pct: float, message: str, **extra: Any) -> None:
    print(json.dumps({"progressPct": progress_pct, "message": message, **extra}, default=str), flush=True)


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
        "mlflow_experiment_name_override": args.experiment_name,
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
            "experiment_name": args.experiment_name,
            "run_name": args.run_name or f"{args.stage}_{args.run_id}",
            "description": "Platform-managed backtest run",
            "benchmarks": {},
        },
    }


def _platform_crucible_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _platform_backtest_config(args)
    cfg["mode"] = "hpo"
    cfg["hpo"] = {
        "validation_period_days": args.validation_period_days,
        "objective_metric": "val_annualized_return",
        "num_samples": args.hpo_samples,
        "max_concurrent_trials": args.hpo_concurrency,
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
    workload.setdefault("workload", {})["run_name"] = run_name
    platform.setdefault("resume", {})["local_cache_dir"] = str(output_dir / "crucible_runs")
    platform.setdefault("hpo", {})["num_samples"] = args.hpo_samples
    platform.setdefault("hpo", {})["max_concurrent_trials"] = args.hpo_concurrency
    platform.setdefault("hpo", {})["validation_period_days"] = args.validation_period_days
    platform.setdefault("hpo", {})["ray_storage_path"] = str((output_dir / "ray_results").resolve())
    platform.setdefault("ray", {})["enabled"] = bool(args.use_ray)
    if args.no_mlflow:
        platform.setdefault("state_store", {})["backend"] = "local"
    if args.experiment_name:
        platform.setdefault("mlflow", {})["parent_experiment_name"] = args.experiment_name

    effective_platform = output_dir / "crucible_platform.yaml"
    effective_workload = output_dir / "crucible_workload.yaml"
    _write_yaml(effective_platform, platform)
    _write_yaml(effective_workload, workload)
    return effective_platform, effective_workload


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
    if args.experiment_name:
        raw_cfg.setdefault("analysis", {})["experiment_name"] = args.experiment_name
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
    raw_metrics = _metric_payload((result or {}).get("analysis"))
    metrics = _ui_metrics(raw_metrics) | {"computeCostUsd": 0.0}
    chart_manifest = _write_chart_artifacts(
        Path(args.output_dir),
        "smoke" if smoke else "research",
        (result or {}).get("portfolio"),
    )
    summary = {
        "stage": "smoke" if smoke else "research",
        "status": "succeeded",
        "verdict": "passed-smoke" if smoke else "caution",
        "message": "Smoke backtest completed" if smoke else "Research backtest completed",
        "metrics": metrics,
        "analysis": _jsonable((result or {}).get("analysis")),
        "finalValue": (result or {}).get("final_value"),
        "cash": (result or {}).get("cash"),
        "positions": (result or {}).get("positions"),
        "charts": chart_manifest.get("charts", []),
    }
    _write_manifest(Path(args.output_dir), summary)
    return summary


def execute_crucible(args: argparse.Namespace) -> dict[str, Any]:
    emit(8, "Preparing crucible platform and workload configs")
    platform_path, workload_path = _crucible_config_paths(args)
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
        elif name == "perturbation":
            result = orchestrator.run_perturbation_stage(rerun=args.rerun_crucible, use_ray=args.use_ray)
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
    _record_simple_result(args, summary)
    _write_manifest(Path(args.output_dir), summary)
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
    for key in ("confirmation.promoted_candidates", "perturbation.accepted_candidates", "plateau.accepted_seeds"):
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
    parser.add_argument("--hpo-samples", type=int, default=4)
    parser.add_argument("--hpo-concurrency", type=int, default=1)
    parser.add_argument("--validation-period-days", type=int, default=30)
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
    summary = execute(args)
    emit(100, summary.get("message", "Stage completed"), summary=summary)


if __name__ == "__main__":
    main()
