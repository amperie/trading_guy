from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(slots=True)
class WalkForwardPeriod:
    optimization_start: datetime
    optimization_end: datetime
    validation_start: datetime
    validation_end: datetime
    trading_start: datetime
    trading_end: datetime


@dataclass(slots=True)
class WalkForwardDecision:
    incumbent_metric: float
    challenger_metric: float
    incumbent_trades: int
    challenger_trades: int
    objective_metric: str
    improvement_pct: float
    threshold_pct: float
    min_validation_trades: int
    adopted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "incumbent_metric": self.incumbent_metric,
            "challenger_metric": self.challenger_metric,
            "incumbent_trades": self.incumbent_trades,
            "challenger_trades": self.challenger_trades,
            "objective_metric": self.objective_metric,
            "improvement_pct": self.improvement_pct,
            "threshold_pct": self.threshold_pct,
            "min_validation_trades": self.min_validation_trades,
            "adopted": self.adopted,
            "reason": self.reason,
        }


def compute_walk_forward_periods(
    data_start: datetime,
    data_end: datetime,
    optimization_window_days: int,
    validation_window_days: int,
    trading_window_days: int,
) -> list[WalkForwardPeriod]:
    periods: list[WalkForwardPeriod] = []
    opt_start = data_start

    while True:
        opt_end = opt_start + timedelta(days=optimization_window_days)
        validation_start = opt_end
        validation_end = validation_start + timedelta(days=validation_window_days)
        trading_start = validation_end
        trading_end = trading_start + timedelta(days=trading_window_days)

        if trading_start >= data_end:
            break
        if trading_end > data_end:
            trading_end = data_end

        periods.append(
            WalkForwardPeriod(
                optimization_start=opt_start,
                optimization_end=opt_end,
                validation_start=validation_start,
                validation_end=validation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            )
        )
        opt_start = opt_start + timedelta(days=trading_window_days)

    return periods


def metric_value(metrics: Any, objective_metric: str) -> float:
    value = getattr(metrics, objective_metric, None)
    if value is None:
        return 0.0
    return float(value)


def metric_improvement_pct(incumbent: float, challenger: float) -> float:
    if abs(incumbent) > 0:
        return ((challenger - incumbent) / abs(incumbent)) * 100.0
    return 100.0 if challenger > incumbent else 0.0


def decide_walk_forward_adoption(
    incumbent_metrics: Any,
    challenger_metrics: Any,
    *,
    objective_metric: str,
    improvement_threshold_pct: float,
    min_validation_trades: int = 0,
) -> WalkForwardDecision:
    incumbent_value = metric_value(incumbent_metrics, objective_metric)
    challenger_value = metric_value(challenger_metrics, objective_metric)
    incumbent_trades = int(getattr(incumbent_metrics, "total_trades", 0) or 0)
    challenger_trades = int(getattr(challenger_metrics, "total_trades", 0) or 0)
    improvement_pct = metric_improvement_pct(incumbent_value, challenger_value)

    if challenger_trades < min_validation_trades:
        return WalkForwardDecision(
            incumbent_metric=incumbent_value,
            challenger_metric=challenger_value,
            incumbent_trades=incumbent_trades,
            challenger_trades=challenger_trades,
            objective_metric=objective_metric,
            improvement_pct=improvement_pct,
            threshold_pct=improvement_threshold_pct,
            min_validation_trades=min_validation_trades,
            adopted=False,
            reason="challenger_below_min_validation_trades",
        )

    if improvement_pct < improvement_threshold_pct:
        return WalkForwardDecision(
            incumbent_metric=incumbent_value,
            challenger_metric=challenger_value,
            incumbent_trades=incumbent_trades,
            challenger_trades=challenger_trades,
            objective_metric=objective_metric,
            improvement_pct=improvement_pct,
            threshold_pct=improvement_threshold_pct,
            min_validation_trades=min_validation_trades,
            adopted=False,
            reason="improvement_below_threshold",
        )

    return WalkForwardDecision(
        incumbent_metric=incumbent_value,
        challenger_metric=challenger_value,
        incumbent_trades=incumbent_trades,
        challenger_trades=challenger_trades,
        objective_metric=objective_metric,
        improvement_pct=improvement_pct,
        threshold_pct=improvement_threshold_pct,
        min_validation_trades=min_validation_trades,
        adopted=True,
        reason="challenger_adopted",
    )
