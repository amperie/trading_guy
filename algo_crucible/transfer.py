from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from algo_crucible.builders import build_candidate_from_params
from algo_crucible.models import Candidate
from utils.utils import merge_nested_config


def load_transfer_candidates(run_dir: str | Path, resolved_cfg) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    plateau_path = _existing_path(run_dir, "stages/06_plateau/summaries/plateau_summary.csv", "summaries/plateau_summary.csv")
    hpo_path = _existing_path(run_dir, "stages/04_hpo/summaries/hpo_trial_summary.csv", "summaries/hpo_trial_summary.csv")
    if not plateau_path.exists() or not hpo_path.exists():
        raise FileNotFoundError("run_plateau_stage and run_hpo_stage must finish before transfer can run")
    accepted = {str(row["candidate_id"]): row for row in _read_csv_rows(plateau_path) if str(row.get("accepted")).lower() in {"true", "1"}}
    candidates = []
    for row in _read_csv_rows(hpo_path):
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in accepted:
            continue
        candidate = build_candidate_from_params(resolved_cfg, _decode(row.get("algorithm_params")) or {}, _decode(row.get("portfolio_params")) or {})
        candidates.append({"seed_id": accepted[candidate_id].get("seed_id"), "candidate": candidate})
    return candidates


def build_transfer_instruments(platform: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = platform.get("cross_instrument_transfer", {})
    instruments = cfg.get("instruments") or []
    max_instruments = int(cfg.get("max_instruments", len(instruments)))
    return [
        {
            "instrument_id": f"instrument_{idx + 1:03d}_{_slug(item.get('name') or item.get('symbol') or str(idx + 1))}",
            "name": item.get("name") or item.get("symbol") or f"instrument_{idx + 1:03d}",
            "symbol": item.get("symbol", ""),
            "required": bool(item.get("required", True)),
            "data_provider_patch": copy.deepcopy(item.get("data_provider_patch") or {}),
            "algorithm_param_patch": copy.deepcopy(item.get("algorithm_param_patch") or {}),
            "portfolio_param_patch": copy.deepcopy(item.get("portfolio_param_patch") or {}),
        }
        for idx, item in enumerate(instruments[:max_instruments])
    ]


def apply_transfer_instrument(resolved_cfg, candidate: Candidate, instrument: dict[str, Any]) -> tuple[dict[str, Any], Candidate]:
    workload = copy.deepcopy(resolved_cfg.workload)
    algorithm_params = copy.deepcopy(candidate.algorithm_params)
    portfolio_params = copy.deepcopy(candidate.portfolio_params)
    if instrument.get("symbol"):
        for params in (algorithm_params, portfolio_params):
            for key in ("symbol", "trade_symbol"):
                if key in params:
                    params[key] = instrument["symbol"]
            for key in ("symbols", "tradable_symbols", "evaluation_symbols"):
                if key in params and isinstance(params[key], list):
                    params[key] = [instrument["symbol"]]
    merge_nested_config(workload["data_provider"], copy.deepcopy(instrument.get("data_provider_patch") or {}))
    merge_nested_config(algorithm_params, copy.deepcopy(instrument.get("algorithm_param_patch") or {}))
    merge_nested_config(portfolio_params, copy.deepcopy(instrument.get("portfolio_param_patch") or {}))
    return workload, build_candidate_from_params(resolved_cfg, algorithm_params, portfolio_params)


def summarize_transfer(candidates: list[dict[str, Any]], instruments: list[dict[str, Any]], job_results: list[dict[str, Any]], platform: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not instruments:
        return {"instrument_rows": [], "summary_rows": []}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in job_results:
        if result.get("status") == "complete":
            payload = result["result"]
            grouped.setdefault((payload["source_candidate_id"], payload["instrument_id"]), []).append(payload["overall_scorecard"])

    instrument_rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate"].candidate_id
        for instrument in instruments:
            scored = _score(grouped.get((candidate_id, instrument["instrument_id"]), []), platform)
            instrument_rows.append({
                "candidate_id": candidate_id,
                "seed_id": candidate.get("seed_id"),
                "instrument_id": instrument["instrument_id"],
                "instrument_name": instrument["name"],
                "symbol": instrument["symbol"],
                "required": instrument["required"],
                **scored,
            })

    summary_rows = []
    cfg = platform.get("cross_instrument_transfer", {})
    max_required_failures = int(cfg.get("max_required_failures", 0))
    min_pass_rate = _pct_threshold(cfg.get("min_instrument_pass_rate", 0.50))
    for candidate in candidates:
        rows = [row for row in instrument_rows if row["candidate_id"] == candidate["candidate"].candidate_id]
        required_failures = [row for row in rows if row["required"] and not row["passed_gate"]]
        pass_rate = _pct([bool(row["passed_gate"]) for row in rows]) or 0.0
        accepted = bool(rows) and len(required_failures) <= max_required_failures and pass_rate >= min_pass_rate
        summary_rows.append({
            "candidate_id": candidate["candidate"].candidate_id,
            "seed_id": candidate.get("seed_id"),
            "accepted": accepted,
            "instrument_count": len(rows),
            "instrument_pass_rate": pass_rate,
            "required_failure_count": len(required_failures),
            "failure_reason": "" if accepted else ",".join(sorted({row["failure_reason"] for row in required_failures if row["failure_reason"]})) or "no_transfer_instruments_passed",
        })
    return {"instrument_rows": instrument_rows, "summary_rows": summary_rows}


def transfer_return_stream_rows(job_results: list[dict[str, Any]], accepted_candidate_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for result in job_results:
        if result.get("status") != "complete":
            continue
        payload = result.get("result") or {}
        candidate_id = str(payload.get("source_candidate_id") or payload.get("candidate_id") or "")
        if candidate_id not in accepted_candidate_ids:
            continue
        for item in payload.get("return_stream") or []:
            rows.append({
                "candidate_id": candidate_id,
                "instrument_id": payload.get("instrument_id", ""),
                "window_id": payload.get("window_id", ""),
                "step": item.get("step"),
                "timestamp": item.get("timestamp"),
                "return_pct": item.get("return_pct"),
                "equity": item.get("equity"),
            })
    return rows


def transfer_metrics(summary_rows: list[dict[str, Any]], jobs_total: int, jobs_complete: int, jobs_failed: int) -> dict[str, float]:
    pass_rates = [value for value in (_num(row.get("instrument_pass_rate")) for row in summary_rows) if value is not None]
    return {
        "transfer.jobs_total": float(jobs_total),
        "transfer.jobs_complete": float(jobs_complete),
        "transfer.jobs_failed": float(jobs_failed),
        "transfer.accepted_candidates": float(sum(1 for row in summary_rows if row.get("accepted") is True)),
        "transfer.rejected_candidates": float(sum(1 for row in summary_rows if row.get("accepted") is not True)),
        "transfer.best_pass_rate": max(pass_rates, default=0.0),
    }


def _score(rows: list[dict[str, Any]], platform: dict[str, Any]) -> dict[str, Any]:
    cfg = platform.get("cross_instrument_transfer", {})
    returns = [value for value in (_num(row.get("total_return_pct")) for row in rows) if value is not None]
    sharpes = [value for value in (_num(row.get("sharpe_ratio")) for row in rows) if value is not None]
    drawdowns = [value for value in (_num(row.get("max_drawdown_pct")) for row in rows) if value is not None]
    trades = [value for value in (_num(row.get("total_trades")) for row in rows) if value is not None]
    min_windows = int(cfg.get("min_windows", 1))
    min_return = _pct_threshold(cfg.get("min_median_oos_return", 0.0))
    min_profitable = _pct_threshold(cfg.get("min_profitable_windows_pct", 0.50))
    max_drawdown = abs(_pct_threshold(cfg.get("max_drawdown", 100.0)))
    min_trades = float(cfg.get("min_trades", 0))
    median_return = _median(returns)
    profitable_pct = _pct([value > 0 for value in returns])
    worst_drawdown = min(drawdowns, default=None)
    passed = len(rows) >= min_windows and median_return is not None and median_return >= min_return and (profitable_pct or 0.0) >= min_profitable and (worst_drawdown is None or abs(worst_drawdown) <= max_drawdown) and sum(trades) >= min_trades
    return {
        "passed_gate": passed,
        "window_count": len(rows),
        "median_oos_return": median_return,
        "median_sharpe": _median(sharpes),
        "worst_drawdown": worst_drawdown,
        "profitable_windows_pct": profitable_pct,
        "total_trades": sum(trades),
        "failure_reason": "" if passed else "transfer_gate_failed",
    }


def _decode(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def _existing_path(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = run_dir / relative_path
        if path.exists():
            return path
    return run_dir / relative_paths[0]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except (EmptyDataError, FileNotFoundError):
        return []


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else float((values[mid - 1] + values[mid]) / 2.0)


def _pct(flags: list[bool]) -> float | None:
    return 100.0 * sum(1 for flag in flags if flag) / len(flags) if flags else None


def _pct_threshold(value: Any) -> float:
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")[:48] or "instrument"
