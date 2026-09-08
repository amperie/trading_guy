from __future__ import annotations

import ast
import copy
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from algo_crucible.builders import build_candidate_from_params
from utils.utils import merge_nested_config


def load_perturbation_candidates(run_dir: str | Path, resolved_cfg) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    plateau_path = _existing_path(run_dir, "stages/06_plateau/summaries/plateau_summary.csv", "summaries/plateau_summary.csv")
    hpo_path = _existing_path(run_dir, "stages/04_hpo/summaries/hpo_trial_summary.csv", "summaries/hpo_trial_summary.csv")
    if not plateau_path.exists():
        raise FileNotFoundError("run_plateau_stage must produce plateau_summary.csv before perturbation analysis can run")
    if not hpo_path.exists():
        raise FileNotFoundError("run_hpo_stage must produce hpo_trial_summary.csv before perturbation analysis can run")
    accepted = {
        str(row["candidate_id"]): row
        for row in _read_csv_rows(plateau_path)
        if str(row.get("accepted")).lower() in {"true", "1"}
    }
    candidates = []
    for row in _read_csv_rows(hpo_path):
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in accepted:
            continue
        algorithm_params = _decode(row.get("algorithm_params")) or {}
        portfolio_params = _decode(row.get("portfolio_params")) or {}
        candidate = build_candidate_from_params(resolved_cfg, algorithm_params, portfolio_params)
        candidates.append({
            "seed_id": accepted[candidate_id].get("seed_id"),
            "candidate": candidate,
            "candidate_type": accepted[candidate_id].get("candidate_type", "generalist"),
            "specialist_regimes": accepted[candidate_id].get("specialist_regimes", ""),
        })
    return candidates


def build_perturbation_scenarios(platform: dict[str, Any], *, data_start: Any = None, data_end: Any = None) -> list[dict[str, Any]]:
    cfg = platform.get("perturbations", {})
    raw = cfg.get("scenarios") or [
        {"name": "baseline", "required": True, "patch": {}},
        {
            "name": "cost_slippage_2x",
            "required": True,
            "portfolio_param_multipliers": {"tx_cost": 2.0, "transaction_cost": 2.0, "slippage_bps": 2.0},
        },
    ]
    max_scenarios = int(cfg.get("max_scenarios", 20))
    scenarios = []
    for idx, scenario in enumerate(raw[:max_scenarios]):
        scenarios.append({
            "scenario_id": f"scenario_{idx + 1:03d}_{_slug(scenario.get('name', str(idx + 1)))}",
            "name": scenario.get("name", f"scenario_{idx + 1:03d}"),
            "required": bool(scenario.get("required", True)),
            "patch": copy.deepcopy(scenario.get("patch", {})),
            "algorithm_param_multipliers": copy.deepcopy(scenario.get("algorithm_param_multipliers", {})),
            "portfolio_param_multipliers": copy.deepcopy(scenario.get("portfolio_param_multipliers", {})),
            "data_provider_patch": copy.deepcopy(scenario.get("data_provider_patch", {})),
            "date_time_offset_seconds": int(scenario.get("date_time_offset_seconds", 0) or 0),
            "scale": scenario.get("scale", "configured"),
        })
    scenarios.extend(_date_time_scenarios(
        cfg,
        start_idx=len(scenarios),
        remaining=max_scenarios - len(scenarios),
        data_start=data_start,
        data_end=data_end,
    ))
    return scenarios


def apply_scenario(resolved_cfg, candidate, scenario: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    workload = copy.deepcopy(resolved_cfg.workload)
    algorithm_params = copy.deepcopy(candidate.algorithm_params)
    portfolio_params = copy.deepcopy(candidate.portfolio_params)
    _apply_multipliers(algorithm_params, scenario.get("algorithm_param_multipliers", {}))
    _apply_multipliers(portfolio_params, scenario.get("portfolio_param_multipliers", {}))
    patch = copy.deepcopy(scenario.get("patch") or {})
    merge_nested_config(workload, patch)
    merge_nested_config(algorithm_params, copy.deepcopy(patch.get("algorithm", {}).get("params", {})))
    merge_nested_config(portfolio_params, copy.deepcopy(patch.get("portfolio", {}).get("params", {})))
    merge_nested_config(workload["data_provider"], copy.deepcopy(scenario.get("data_provider_patch") or {}))
    perturbed = build_candidate_from_params(resolved_cfg, algorithm_params, portfolio_params)
    return workload, perturbed


def summarize_perturbations(
    *,
    candidates: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    job_results: list[dict[str, Any]],
    platform: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for result in job_results:
        if result.get("status") != "complete":
            continue
        payload = result["result"]
        key = (payload["source_candidate_id"], payload["scenario_id"])
        grouped.setdefault(key, {"overall": [], "regimes": []})
        grouped[key]["overall"].append(payload["overall_scorecard"])
        grouped[key]["regimes"].extend(payload["regime_scorecard"])

    scenario_rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate"].candidate_id
        for scenario in scenarios:
            scored = _score_rows(grouped.get((candidate_id, scenario["scenario_id"]), {"overall": [], "regimes": []}), candidate, platform)
            reason = "" if scored["passed_gate"] else _failure_reason(scenario["name"])
            scenario_rows.append({
                "candidate_id": candidate_id,
                "seed_id": candidate.get("seed_id"),
                "candidate_type": candidate.get("candidate_type"),
                "specialist_regimes": candidate.get("specialist_regimes", ""),
                "scenario_id": scenario["scenario_id"],
                "scenario_name": scenario["name"],
                "required": scenario["required"],
                "scale": scenario.get("scale", ""),
                "date_time_offset_seconds": scenario.get("date_time_offset_seconds", 0),
                **scored,
                "failure_reason": reason,
            })

    summary_rows = []
    for candidate in candidates:
        rows = [row for row in scenario_rows if row["candidate_id"] == candidate["candidate"].candidate_id]
        required_failures = [row for row in rows if row["required"] and not row["passed_gate"]]
        pass_rate = _pct([bool(row["passed_gate"]) for row in rows]) or 0.0
        cfg = platform.get("perturbations", {})
        max_required_failures = int(cfg.get("max_required_failures", 0))
        min_pass_rate = _pct_threshold(cfg.get("min_scenario_pass_rate", 0.80))
        accepted = len(required_failures) <= max_required_failures and pass_rate >= min_pass_rate
        summary_rows.append({
            "candidate_id": candidate["candidate"].candidate_id,
            "seed_id": candidate.get("seed_id"),
            "candidate_type": candidate.get("candidate_type"),
            "specialist_regimes": candidate.get("specialist_regimes", ""),
            "accepted": accepted,
            "scenario_count": len(rows),
            "scenario_pass_rate": pass_rate,
            "required_failure_count": len(required_failures),
            "failure_reason": ",".join(sorted({row["failure_reason"] for row in required_failures if row["failure_reason"]})),
        })
    return {"scenario_rows": scenario_rows, "summary_rows": summary_rows}


def perturb_windows(
    windows: list[dict[str, Any]],
    scenario: dict[str, Any],
    *,
    data_start: Any,
    data_end: Any,
) -> list[dict[str, Any]]:
    offset = int(scenario.get("date_time_offset_seconds") or 0)
    if not offset:
        return windows
    return [_shift_window(window, offset, data_start=data_start, data_end=data_end) for window in windows]


def perturbation_metrics(summary_rows: list[dict[str, Any]], jobs_total: int, jobs_complete: int, jobs_failed: int) -> dict[str, float]:
    pass_rates = [_num(row.get("scenario_pass_rate")) for row in summary_rows]
    pass_rates = [value for value in pass_rates if value is not None]
    return {
        "perturbation.jobs_total": float(jobs_total),
        "perturbation.jobs_complete": float(jobs_complete),
        "perturbation.jobs_failed": float(jobs_failed),
        "perturbation.accepted_candidates": float(sum(1 for row in summary_rows if row.get("accepted") is True)),
        "perturbation.rejected_candidates": float(sum(1 for row in summary_rows if row.get("accepted") is not True)),
        "perturbation.best_pass_rate": max(pass_rates, default=0.0),
        "perturbation.median_pass_rate": _median(pass_rates) or 0.0,
    }


def _score_rows(group: dict[str, list[dict[str, Any]]], candidate: dict[str, Any], platform: dict[str, Any]) -> dict[str, Any]:
    rows = group["overall"]
    thresholds = _thresholds(platform, candidate.get("candidate_type", "generalist"))
    if candidate.get("candidate_type") == "specialist" and candidate.get("specialist_regimes"):
        targets = set(str(candidate["specialist_regimes"]).split(","))
        rows = [row for row in group["regimes"] if row.get("regime") in targets]
        thresholds = _thresholds(platform, "specialist")
    returns = [_num(row.get("total_return_pct")) for row in rows]
    returns = [value for value in returns if value is not None]
    drawdowns = [_num(row.get("max_drawdown_pct")) for row in rows]
    drawdowns = [value for value in drawdowns if value is not None]
    trades = [_num(row.get("total_trades")) for row in rows]
    trades = [value for value in trades if value is not None]
    median_return = _median(returns)
    profitable = _pct([value > 0 for value in returns])
    worst_drawdown = min(drawdowns, default=None)
    passed = (
        len(rows) >= thresholds["min_windows"]
        and median_return is not None
        and median_return >= thresholds["min_return"]
        and (profitable or 0.0) >= thresholds["min_profitable"]
        and (worst_drawdown is None or abs(worst_drawdown) <= thresholds["max_drawdown"])
        and sum(trades) >= thresholds["min_trades"]
    )
    return {
        "passed_gate": passed,
        "median_oos_return": median_return,
        "worst_drawdown": worst_drawdown,
        "profitable_windows_pct": profitable,
        "total_trades": sum(trades),
    }


def _apply_multipliers(params: dict[str, Any], multipliers: dict[str, float]) -> None:
    for key, multiplier in multipliers.items():
        if key in params and isinstance(params[key], (int, float)):
            params[key] *= float(multiplier)


def _failure_reason(name: str) -> str:
    lowered = name.lower()
    if "cost" in lowered or "slippage" in lowered:
        return "cost_slippage_fragile"
    if "date" in lowered:
        return "date_fragile"
    return "scenario_perturbation_failed"


def _date_time_scenarios(
    cfg: dict[str, Any],
    *,
    start_idx: int,
    remaining: int,
    data_start: Any,
    data_end: Any,
) -> list[dict[str, Any]]:
    if remaining <= 0 or not bool(cfg.get("date_time_enabled", True)) or data_start is None or data_end is None:
        return []
    count = int(cfg.get("date_time_scenarios", remaining))
    count = max(0, min(count, remaining))
    dataset_seconds = max(0, int((pd.Timestamp(data_end).to_pydatetime() - pd.Timestamp(data_start).to_pydatetime()).total_seconds()))
    max_offset_seconds = max(1, int(dataset_seconds * max(0.0, float(cfg.get("max_date_time_range_pct", 0.10)))))
    mix = cfg.get("date_time_scale_mix", {}) or {}
    scales = _scale_counts(count, {
        "small": float(mix.get("small", 0.50)),
        "weeks": float(mix.get("weeks", 0.30)),
        "months": float(mix.get("months", 0.20)),
    })
    scenarios = []
    for scale, scale_count in scales.items():
        for offset in _scale_offsets(scale, scale_count, max_offset_seconds):
            if not offset:
                continue
            idx = start_idx + len(scenarios) + 1
            direction = "plus" if offset > 0 else "minus"
            scenarios.append({
                "scenario_id": f"scenario_{idx:03d}_date_time_{scale}_{direction}_{abs(offset)}s",
                "name": f"date_time_{scale}_{direction}_{abs(offset)}s",
                "required": bool(cfg.get("date_time_required", True)),
                "patch": {},
                "algorithm_param_multipliers": {},
                "portfolio_param_multipliers": {},
                "data_provider_patch": {},
                "date_time_offset_seconds": offset,
                "scale": scale,
            })
    return scenarios[:remaining]


def _scale_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    positive = {key: max(0.0, value) for key, value in weights.items()}
    weight_sum = sum(positive.values()) or 1.0
    counts = {key: int(total * value / weight_sum) for key, value in positive.items()}
    while sum(counts.values()) < total:
        key = max(positive, key=lambda item: (positive[item] / weight_sum) - (counts[item] / max(total, 1)))
        counts[key] += 1
    return counts


def _scale_offsets(scale: str, count: int, max_offset_seconds: int) -> list[int]:
    max_seconds = max(1, int(max_offset_seconds))
    caps = {
        "small": min(max_seconds, 3 * 24 * 3600),
        "weeks": min(max_seconds, 8 * 7 * 24 * 3600),
        "months": min(max_seconds, 6 * 30 * 24 * 3600),
    }
    cap = caps.get(scale, max_seconds)
    if count <= 0 or cap <= 0:
        return []
    magnitudes = [max(3600, int(cap * (idx + 1) / max(count, 1))) for idx in range((count + 1) // 2)]
    offsets = []
    for magnitude in magnitudes:
        offsets.extend([-magnitude, magnitude])
    return offsets[:count]


def _shift_window(window: dict[str, Any], offset_seconds: int, *, data_start: Any, data_end: Any) -> dict[str, Any]:
    shifted = copy.deepcopy(window)
    keys = ["train_start", "train_end", "embargo_start", "embargo_end", "validation_start", "validation_end"]
    parsed = {key: pd.Timestamp(shifted[key]).to_pydatetime() for key in keys}
    low = pd.Timestamp(data_start).to_pydatetime() - parsed["train_start"]
    high = pd.Timestamp(data_end).to_pydatetime() - parsed["validation_end"]
    offset = max(int(low.total_seconds()), min(offset_seconds, int(high.total_seconds())))
    delta = timedelta(seconds=offset)
    for key in keys:
        shifted[key] = (parsed[key] + delta).isoformat()
    shifted["window_id"] = f"{window['window_id']}_dt_{offset:+d}s"
    shifted["date_time_offset_seconds"] = offset
    return shifted


def _thresholds(platform: dict[str, Any], candidate_type: str) -> dict[str, float]:
    cfg = platform.get("gates", {}).get("specialist" if candidate_type == "specialist" else "generalist", {})
    return {
        "min_windows": int(cfg.get("min_regime_windows" if candidate_type == "specialist" else "min_windows", 1)),
        "min_trades": float(cfg.get("min_trades", 0)),
        "min_return": _pct_threshold(cfg.get("min_regime_median_oos_return" if candidate_type == "specialist" else "min_median_oos_return", 0.0)),
        "min_profitable": _pct_threshold(cfg.get("min_regime_profitable_windows_pct" if candidate_type == "specialist" else "min_profitable_windows_pct", 0.0)),
        "max_drawdown": abs(_pct_threshold(cfg.get("max_regime_drawdown" if candidate_type == "specialist" else "max_drawdown", 100.0))),
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


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except EmptyDataError:
        return []


def _slug(value: Any) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
    return text or "scenario"


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else float((values[mid - 1] + values[mid]) / 2.0)


def _pct(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return 100.0 * sum(1 for flag in flags if flag) / len(flags)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct_threshold(value: Any) -> float:
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _existing_path(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = run_dir / relative_path
        if path.exists():
            return path
    return run_dir / relative_paths[0]
