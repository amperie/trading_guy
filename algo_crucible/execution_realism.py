from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


def load_execution_realism_inputs(run_dir: str | Path) -> list[dict[str, Any]]:
    return _read_csv_rows(Path(run_dir) / "stages/07_perturbation/summaries/perturbation_return_stream.csv")


def build_execution_realism_scenarios(platform: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = platform.get("execution_realism_stress", {})
    raw = cfg.get("scenarios") or [
        {"name": "baseline", "required": True},
        {"name": "one_bar_delay", "required": True, "delay_bars": 1},
        {"name": "miss_10pct_positive_returns", "required": True, "miss_signal_pct": 10.0},
        {"name": "adverse_stops_5bps", "required": True, "adverse_stop_bps": 5.0},
        {"name": "stress_spread_2x", "required": False, "spread_bps": 2.0},
    ]
    return [
        {
            "scenario_id": f"exec_{idx + 1:03d}_{_slug(row.get('name', str(idx + 1)))}",
            "stress_type": row.get("stress_type") or _stress_type(row),
            "severity": row.get("severity") or _severity(row),
            **row,
        }
        for idx, row in enumerate(raw[: int(cfg.get("max_scenarios", 20))])
    ]


def analyze_execution_realism(rows: list[dict[str, Any]], platform: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cfg = platform.get("execution_realism_stress", {})
    min_pass_rate = _ratio_threshold(cfg.get("min_scenario_pass_rate", 0.80))
    max_required_failures = int(cfg.get("max_required_failures", 0))
    min_return = float(cfg.get("min_total_return_pct", -100.0))
    max_degradation = float(cfg.get("max_total_return_degradation_pct", 50.0))
    scenarios = build_execution_realism_scenarios(platform)
    scenario_rows = []
    summary_rows = []
    grouped = _group_returns(rows)
    for candidate_id, returns in grouped.items():
        baseline_stats = _stats(returns)
        passed = 0
        required_failures = 0
        failures = []
        for scenario in scenarios:
            stressed = _stress_returns(returns, scenario)
            stats = _stats(stressed)
            total = stats["total_return_pct"]
            degradation = max(0.0, baseline_stats["total_return_pct"] - total)
            accepted = total >= min_return and degradation <= max_degradation
            if accepted:
                passed += 1
            elif scenario.get("required", True):
                required_failures += 1
                failures.append(_failure_reason(scenario))
            scenario_rows.append({
                "candidate_id": candidate_id,
                "scenario_id": scenario["scenario_id"],
                "scenario_name": scenario.get("name"),
                "stress_type": scenario.get("stress_type"),
                "severity": scenario.get("severity"),
                "analysis_mode": "return_stream_proxy",
                "required": bool(scenario.get("required", True)),
                "accepted": accepted,
                "baseline_total_return_pct": baseline_stats["total_return_pct"],
                "baseline_sharpe_like": baseline_stats["sharpe_like"],
                "baseline_hit_rate_pct": baseline_stats["hit_rate_pct"],
                "baseline_max_drawdown_pct": baseline_stats["max_drawdown_pct"],
                "stressed_total_return_pct": total,
                "stressed_sharpe_like": stats["sharpe_like"],
                "stressed_hit_rate_pct": stats["hit_rate_pct"],
                "stressed_max_drawdown_pct": stats["max_drawdown_pct"],
                "total_return_degradation_pct": degradation,
                "sharpe_degradation": max(0.0, baseline_stats["sharpe_like"] - stats["sharpe_like"]),
                "hit_rate_degradation_pct": max(0.0, baseline_stats["hit_rate_pct"] - stats["hit_rate_pct"]),
                "failure_reason": "" if accepted else _failure_reason(scenario),
            })
        pass_rate = passed / len(scenarios) if scenarios else 0.0
        accepted = required_failures <= max_required_failures and pass_rate >= min_pass_rate
        summary_rows.append({
            "candidate_id": candidate_id,
            "scenario_count": len(scenarios),
            "scenario_pass_rate": pass_rate,
            "required_failure_count": required_failures,
            "accepted": accepted,
            "failure_reason": "" if accepted else (failures[0] if failures else "execution_realism_pass_rate_below_gate"),
        })
    return {"scenario_rows": scenario_rows, "summary_rows": summary_rows}


def execution_realism_metrics(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    accepted = [row for row in summary_rows if row.get("accepted") is True]
    rejected = [row for row in summary_rows if row.get("accepted") is not True]
    pass_rates = [float(row.get("scenario_pass_rate") or 0.0) for row in summary_rows]
    return {
        "execution_realism.candidate_count": float(len(summary_rows)),
        "execution_realism.accepted_candidates": float(len(accepted)),
        "execution_realism.rejected_candidates": float(len(rejected)),
        "execution_realism.worst_scenario_pass_rate": min(pass_rates, default=0.0),
    }


def _stress_returns(returns: list[float], scenario: dict[str, Any]) -> list[float]:
    stressed = list(returns)
    delay = int(scenario.get("delay_bars", 0) or 0)
    if delay > 0 and stressed:
        stressed = [0.0] * min(delay, len(stressed)) + stressed[: max(0, len(stressed) - delay)]
    miss_pct = _ratio_threshold(scenario.get("miss_signal_pct", 0.0) or 0.0)
    if miss_pct > 0:
        step = max(1, round(1 / miss_pct))
        seen = 0
        for idx, value in enumerate(stressed):
            if value <= 0:
                continue
            seen += 1
            if seen % step == 0:
                stressed[idx] = 0.0
    adverse = float(scenario.get("adverse_stop_bps", 0.0) or 0.0) / 100.0
    spread = float(scenario.get("spread_bps", 0.0) or 0.0) / 100.0
    spread += max(0.0, float(scenario.get("spread_multiplier", 1.0) or 1.0) - 1.0) * float(scenario.get("base_spread_bps", 1.0)) / 100.0
    fill_ratio = _ratio_threshold(scenario.get("partial_fill_pct", 100.0) or 100.0)
    return [(value * fill_ratio) - spread - (adverse if value < 0 else 0.0) for value in stressed]


def _stats(returns: list[float]) -> dict[str, float]:
    total = sum(returns)
    mean = total / len(returns) if returns else 0.0
    vol = _std(returns)
    equity = 100.0
    path = []
    for value in returns:
        equity *= 1.0 + value / 100.0
        path.append(equity)
    return {
        "total_return_pct": total,
        "sharpe_like": mean / vol if vol > 0 else 0.0,
        "hit_rate_pct": 100.0 * sum(1 for value in returns if value > 0) / len(returns) if returns else 0.0,
        "max_drawdown_pct": _max_drawdown_pct(path),
    }


def _max_drawdown_pct(path: list[float]) -> float:
    peak = path[0] if path else 100.0
    worst = 0.0
    for value in path:
        peak = max(peak, value)
        worst = min(worst, (value / peak - 1.0) * 100.0) if peak > 0 else worst
    return worst


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _group_returns(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        value = _num(row.get("return_pct"))
        if candidate_id and value is not None:
            grouped.setdefault(candidate_id, []).append(value)
    return grouped


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except EmptyDataError:
        return []


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_threshold(value: Any) -> float:
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _failure_reason(scenario: dict[str, Any]) -> str:
    name = str(scenario.get("name", "stress")).lower()
    if "delay" in name:
        return "fill_delay_fragile"
    if "miss" in name:
        return "missed_signal_fragile"
    if "stop" in name:
        return "adverse_stop_fragile"
    if "spread" in name:
        return "spread_stress_fragile"
    return "execution_realism_fragile"


def _stress_type(scenario: dict[str, Any]) -> str:
    if scenario.get("delay_bars"):
        return "latency_delay"
    if scenario.get("miss_signal_pct"):
        return "missed_signals"
    if scenario.get("adverse_stop_bps"):
        return "adverse_stops"
    if scenario.get("spread_bps") or scenario.get("spread_multiplier"):
        return "spread_regime"
    if scenario.get("partial_fill_pct"):
        return "partial_fills"
    return "baseline"


def _severity(scenario: dict[str, Any]) -> str:
    if scenario.get("delay_bars"):
        return f"{scenario.get('delay_bars')}bar"
    if scenario.get("miss_signal_pct"):
        return f"{scenario.get('miss_signal_pct')}pct"
    if scenario.get("adverse_stop_bps"):
        return f"{scenario.get('adverse_stop_bps')}bps"
    if scenario.get("spread_bps"):
        return f"{scenario.get('spread_bps')}bps"
    if scenario.get("spread_multiplier"):
        return f"{scenario.get('spread_multiplier')}x"
    if scenario.get("partial_fill_pct"):
        return f"{scenario.get('partial_fill_pct')}pct"
    return "baseline"


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "scenario"
