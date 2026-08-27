from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from io import StringIO
from typing import Any

import pandas as pd

from trading.analysis.market_regime import classify_ticks


METRIC_KEYS = [
    "total_return_pct",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "win_rate",
    "profit_factor",
    "total_trades",
    "final_equity",
    "initial_equity",
    "trading_days",
]


def overall_scorecard(metrics: Any) -> dict[str, Any]:
    raw = asdict(metrics) if is_dataclass(metrics) else dict(vars(metrics))
    return {key: _scalar(raw.get(key)) for key in METRIC_KEYS}


def regime_scorecard(portfolio, ticks: list[list[Any]], regime_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    labels = classify_ticks(ticks, regime_cfg or {})
    values = pd.Series(portfolio.value_history)
    values.index = pd.to_datetime(list(portfolio.value_history.keys()))
    returns = values.pct_change().fillna(0.0)

    rows = []
    timestamps = list(values.index)
    label_by_ts: dict[pd.Timestamp, str] = {}
    for tick_labels in labels:
        for snapshot in tick_labels.values():
            label_by_ts[pd.Timestamp(snapshot.timestamp)] = snapshot.composite_regime

    frame = pd.DataFrame({
        "timestamp": timestamps,
        "return": [float(returns.loc[ts]) for ts in timestamps],
        "equity": [float(values.loc[ts]) for ts in timestamps],
        "regime": [label_by_ts.get(pd.Timestamp(ts), "UNKNOWN_UNKNOWN") for ts in timestamps],
    })
    for regime, group in frame.groupby("regime", sort=True):
        equity = group["equity"]
        running_max = equity.expanding().max()
        drawdown = ((equity / running_max) - 1.0).min() if len(equity) else 0.0
        rows.append({
            "regime": regime,
            "bars": int(len(group)),
            "total_return_pct": float(((1.0 + group["return"]).prod() - 1.0) * 100.0),
            "avg_bar_return_pct": float(group["return"].mean() * 100.0),
            "max_drawdown_pct": float(drawdown * 100.0),
        })
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
