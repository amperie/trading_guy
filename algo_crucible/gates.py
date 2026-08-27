from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateDecision:
    candidate_id: str
    candidate_type: str
    passed: bool
    specialist_regimes: list[str]
    reason_codes: list[str]
    metrics: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "passed": self.passed,
            "specialist_regimes": ",".join(self.specialist_regimes),
            "reason_codes": ",".join(self.reason_codes),
            **self.metrics,
        }


def evaluate_regime_aware_gates(
    overall_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    platform: dict[str, Any],
) -> list[GateDecision]:
    gates = platform.get("gates", {})
    generalist_cfg = gates.get("generalist", {})
    specialist_cfg = gates.get("specialist", {})
    by_candidate = _group_by(overall_rows, "candidate_id")
    regimes_by_candidate = _group_by(regime_rows, "candidate_id")
    decisions = []
    for candidate_id in sorted(by_candidate):
        aggregate = _aggregate_overall(by_candidate[candidate_id])
        if _passes_generalist(aggregate, generalist_cfg):
            decisions.append(GateDecision(
                candidate_id=candidate_id,
                candidate_type="generalist",
                passed=True,
                specialist_regimes=[],
                reason_codes=["generalist_gate_passed"],
                metrics=aggregate,
            ))
            continue

        specialist_regimes = _passing_specialist_regimes(
            regimes_by_candidate.get(candidate_id, []),
            specialist_cfg,
        )
        if specialist_regimes:
            decisions.append(GateDecision(
                candidate_id=candidate_id,
                candidate_type="specialist",
                passed=True,
                specialist_regimes=specialist_regimes,
                reason_codes=["specialist_gate_passed"],
                metrics=aggregate,
            ))
            continue

        decisions.append(GateDecision(
            candidate_id=candidate_id,
            candidate_type="reject",
            passed=False,
            specialist_regimes=[],
            reason_codes=_failure_reasons(aggregate, generalist_cfg),
            metrics=aggregate,
        ))
    return decisions


def gate_summary_metrics(decisions: list[GateDecision]) -> dict[str, float]:
    return {
        "regime_gate.generalists": float(sum(1 for d in decisions if d.candidate_type == "generalist")),
        "regime_gate.specialists": float(sum(1 for d in decisions if d.candidate_type == "specialist")),
        "regime_gate.defensive": float(sum(1 for d in decisions if d.candidate_type == "defensive")),
        "regime_gate.rejected": float(sum(1 for d in decisions if d.candidate_type == "reject")),
        "regime_gate.passed": float(sum(1 for d in decisions if d.passed)),
    }


def _aggregate_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [_num(row.get("total_return_pct")) for row in rows]
    drawdowns = [_num(row.get("max_drawdown_pct")) for row in rows]
    trades = [_num(row.get("total_trades")) for row in rows]
    return {
        "windows": len(rows),
        "median_oos_return_pct": _median(returns),
        "profitable_windows_pct": _pct([ret > 0 for ret in returns if ret is not None]),
        "worst_drawdown_pct": min([dd for dd in drawdowns if dd is not None], default=None),
        "total_trades": int(sum(trade for trade in trades if trade is not None)),
    }


def _passing_specialist_regimes(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    passing = []
    min_windows = int(cfg.get("min_regime_windows", 1))
    min_bars = int(cfg.get("min_regime_bars", 1))
    min_return = _pct_threshold(cfg.get("min_regime_median_oos_return", 0.0))
    min_profitable = _pct_threshold(cfg.get("min_regime_profitable_windows_pct", 0.0))
    max_drawdown = _drawdown_threshold(cfg.get("max_regime_drawdown", 100.0))
    for regime, group in _group_by(rows, "regime").items():
        if regime.startswith("UNKNOWN"):
            continue
        returns = [_num(row.get("total_return_pct")) for row in group]
        drawdowns = [_num(row.get("max_drawdown_pct")) for row in group]
        bars = sum(int(_num(row.get("bars")) or 0) for row in group)
        if len(group) < min_windows or bars < min_bars:
            continue
        median_return = _median(returns)
        profitable = _pct([ret > 0 for ret in returns if ret is not None])
        worst_drawdown = min([dd for dd in drawdowns if dd is not None], default=0.0)
        if (
            median_return is not None
            and profitable is not None
            and median_return >= min_return
            and profitable >= min_profitable
            and abs(worst_drawdown) <= max_drawdown
        ):
            passing.append(regime)
    return sorted(passing)


def _passes_generalist(metrics: dict[str, Any], cfg: dict[str, Any]) -> bool:
    min_trades = int(cfg.get("min_trades", 0))
    min_return = _pct_threshold(cfg.get("min_median_oos_return", 0.0))
    min_profitable = _pct_threshold(cfg.get("min_profitable_windows_pct", 0.0))
    max_drawdown = _drawdown_threshold(cfg.get("max_drawdown", 100.0))
    if metrics["total_trades"] < min_trades:
        return False
    if metrics["median_oos_return_pct"] is None or metrics["median_oos_return_pct"] < min_return:
        return False
    if metrics["profitable_windows_pct"] is None or metrics["profitable_windows_pct"] < min_profitable:
        return False
    if metrics["worst_drawdown_pct"] is not None and abs(metrics["worst_drawdown_pct"]) > max_drawdown:
        return False
    return True


def _failure_reasons(metrics: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    reasons = []
    if metrics["total_trades"] < int(cfg.get("min_trades", 0)):
        reasons.append("insufficient_trades")
    if metrics["median_oos_return_pct"] is None or metrics["median_oos_return_pct"] < _pct_threshold(cfg.get("min_median_oos_return", 0.0)):
        reasons.append("negative_median_oos_return")
    if metrics["profitable_windows_pct"] is None or metrics["profitable_windows_pct"] < _pct_threshold(cfg.get("min_profitable_windows_pct", 0.0)):
        reasons.append("low_profitable_windows_pct")
    max_drawdown = _drawdown_threshold(cfg.get("max_drawdown", 100.0))
    if metrics["worst_drawdown_pct"] is not None and abs(metrics["worst_drawdown_pct"]) > max_drawdown:
        reasons.append("drawdown_too_high")
    if not reasons:
        reasons.append("no_regime_repeatability")
    return reasons


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _median(values: list[float | None]) -> float | None:
    nums = sorted(value for value in values if value is not None)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return float(nums[mid])
    return float((nums[mid - 1] + nums[mid]) / 2.0)


def _pct(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return 100.0 * sum(1 for flag in flags if flag) / len(flags)


def _num(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _pct_threshold(value) -> float:
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _drawdown_threshold(value) -> float:
    value = abs(float(value))
    return value * 100.0 if value <= 1.0 else value
