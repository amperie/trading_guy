from __future__ import annotations

import csv
import html
import math
import statistics
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
    "volatility",
]

DISTRIBUTION_METRIC_KEYS = [
    "total_return_pct",
    "annualized_return",
    "max_drawdown_pct",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
    "profit_factor",
    "total_trades",
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


def distribution_stats(rows: list[dict[str, Any]], metric_keys: list[str] | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in metric_keys or DISTRIBUTION_METRIC_KEYS:
        values = [_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        values.sort()
        stats[f"{key}_count"] = len(values)
        stats[f"{key}_median"] = _median(values)
        stats[f"{key}_std_dev"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        stats[f"{key}_min"] = float(values[0])
        stats[f"{key}_max"] = float(values[-1])
    return stats


def prefixed_numeric_metrics(prefix: str, stats: dict[str, Any]) -> dict[str, float]:
    return {f"{prefix}.{key}": float(value) for key, value in stats.items() if isinstance(value, (int, float))}


def distribution_svg(rows: list[dict[str, Any]], title: str, metric_keys: list[str] | None = None) -> str:
    panels = []
    for key in metric_keys or DISTRIBUTION_METRIC_KEYS:
        values = [_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            panels.append(_histogram_panel(key, values))
    width = 900
    panel_height = 150
    height = 70 + panel_height * max(len(panels), 1)
    body = "\n".join(panel.replace("{y}", str(60 + idx * panel_height)) for idx, panel in enumerate(panels))
    if not body:
        body = '<text x="30" y="70" font-size="14" fill="#444">No numeric metric values available.</text>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#fff"/>'
        f'<text x="30" y="35" font-size="22" font-family="Arial" font-weight="700" fill="#111">{html.escape(title)}</text>'
        f"{body}</svg>"
    )


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(values: list[float]) -> float:
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _histogram_panel(key: str, values: list[float]) -> str:
    values = sorted(values)
    bins = _bins(values, 12)
    max_count = max(count for _, _, count in bins) or 1
    x0, y0, chart_w, chart_h = 30, "{y}", 700, 72
    bar_w = chart_w / len(bins)
    bars = []
    for idx, (_, _, count) in enumerate(bins):
        h = 1 if count == 0 else max(2, chart_h * count / max_count)
        x = x0 + idx * bar_w
        y = f"{{y_plus_{idx}}}"
        bars.append(f'<rect x="{x:.1f}" y="{y}" width="{max(bar_w - 3, 1):.1f}" height="{h:.1f}" fill="#3b82f6"/>')
    median = _median(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    label = (
        f"{key}: n={len(values)} median={median:.4g} std={std_dev:.4g} "
        f"min={values[0]:.4g} max={values[-1]:.4g}"
    )
    panel = (
        f'<g transform="translate(0,{y0})">'
        f'<text x="30" y="0" font-size="14" font-family="Arial" font-weight="700" fill="#111">{html.escape(label)}</text>'
        f'<line x1="30" y1="95" x2="730" y2="95" stroke="#888" stroke-width="1"/>'
        f'<text x="30" y="118" font-size="11" font-family="Arial" fill="#555">{values[0]:.4g}</text>'
        f'<text x="690" y="118" font-size="11" font-family="Arial" fill="#555">{values[-1]:.4g}</text>'
        f'{"".join(bars)}'
        f'</g>'
    )
    for idx, (_, _, count) in enumerate(bins):
        h = 1 if count == 0 else max(2, chart_h * count / max_count)
        panel = panel.replace(f"{{y_plus_{idx}}}", f"{95 - h:.1f}")
    return panel


def _bins(values: list[float], max_bins: int) -> list[tuple[float, float, int]]:
    if len(values) <= 1 or values[0] == values[-1]:
        return [(values[0], values[-1], len(values))]
    bin_count = min(max_bins, max(3, int(math.sqrt(len(values)))))
    low, high = values[0], values[-1]
    step = (high - low) / bin_count
    bins = [[low + idx * step, low + (idx + 1) * step, 0] for idx in range(bin_count)]
    bins[-1][1] = high
    for value in values:
        idx = min(int((value - low) / step), bin_count - 1)
        bins[idx][2] += 1
    return [(float(start), float(end), int(count)) for start, end, count in bins]
