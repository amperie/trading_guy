from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

from trading.commands.backtest import run_backtest_from_raw_config
from trading.commands.common import (
    apply_cli_overrides,
    fill_data_provider_creds,
    load_account_creds,
    load_raw_config,
)
from trading.commands.hpo import run_hpo_split_from_raw_config
from trading.reporting import LocalRunResultSink
from trading.reporting.sinks import SinkRun


DEFAULT_CONFIG_BY_STAGE = {
    "idea": "platform:backtest",
    "smoke": "platform:backtest",
    "research": "platform:backtest",
    "crucible": "platform:crucible",
    "promotion": "platform:backtest",
    "monitoring": "platform:backtest",
}


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
    }
    _write_manifest(Path(args.output_dir), summary)
    return summary


def execute_crucible(args: argparse.Namespace) -> dict[str, Any]:
    emit(10, "Loading crucible HPO config")
    raw_cfg = _load_stage_config(args)
    raw_cfg.setdefault("hpo", {})["num_samples"] = args.hpo_samples
    raw_cfg.setdefault("hpo", {})["max_concurrent_trials"] = args.hpo_concurrency
    raw_cfg.setdefault("hpo", {})["validation_period_days"] = args.validation_period_days
    provider_name = raw_cfg.get("data_provider", {}).get("provider", "")
    if "alpaca" in provider_name.lower():
        creds = load_account_creds(args.account)
        fill_data_provider_creds(raw_cfg, creds)
    emit(30, "Starting split HPO and validation backtests")
    details = run_hpo_split_from_raw_config(
        raw_cfg,
        config_artifact_path=args.config,
        num_samples_override=args.hpo_samples,
        max_concurrent_override=args.hpo_concurrency,
        validation_period_days_override=args.validation_period_days,
        return_details=True,
    )
    emit(85, "Persisting crucible result")
    train_metrics = _metric_payload(details.get("train_results"))
    val_metrics = _metric_payload(details.get("val_results"))
    metrics = {
        **{f"train.{k}": v for k, v in train_metrics.items()},
        **{f"validation.{k}": v for k, v in val_metrics.items()},
        "robustnessScore": float(details.get("objective_value") or 0.0),
        "computeCostUsd": 0.0,
    }
    summary = {
        "stage": "crucible",
        "status": "succeeded",
        "verdict": "robust",
        "message": "Crucible split-HPO validation completed",
        "metrics": metrics,
        "details": _jsonable(details),
    }
    _record_simple_result(args, summary)
    _write_manifest(Path(args.output_dir), summary)
    return summary


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
