from __future__ import annotations

import copy
import math
from typing import Any

import pandas as pd

from algo_crucible.builders import build_components
from algo_crucible.models import Candidate
from algo_crucible.scoring import overall_scorecard, regime_scorecard
from trading.analysis.analysis_engine import AnalysisEngine
from trading.engines.backtest_engine import BacktestingEngine


def run_validation_backtest(payload: dict[str, Any]) -> dict[str, Any]:
    workload = copy.deepcopy(payload["workload"])
    candidate = Candidate.from_dict(payload["candidate"])
    window = payload["window"]
    data_cfg = workload["data_provider"]
    data_cfg["start_date"] = window["validation_start"]
    data_cfg["end_date"] = window["validation_end"]

    dp, al, om, pf = build_components(workload, candidate)
    ticks = list(dp.iterate())
    engine = BacktestingEngine({"status_line_enabled": False, "state_store": {"enabled": False}}, dp, al, om, pf)
    engine.run()

    analysis = AnalysisEngine(pf, om)
    metrics = overall_scorecard(analysis.calculate_metrics())
    regimes = regime_scorecard(pf, ticks, candidate.algorithm_params.get("market_regime", {}), analysis.extract_trades())
    return {
        "candidate_id": candidate.candidate_id,
        "seed_id": payload.get("seed_id"),
        "neighbor_id": payload.get("neighbor_id"),
        "scenario_id": payload.get("scenario_id"),
        "instrument_id": payload.get("instrument_id"),
        "source_candidate_id": payload.get("source_candidate_id", candidate.candidate_id),
        "window_id": window["window_id"],
        "window": window,
        "overall_scorecard": metrics,
        "regime_scorecard": regimes,
        "return_stream": _return_stream(pf),
        "signal_forward_return_stream": _signal_forward_return_stream(pf),
    }


def _return_stream(portfolio, max_points: int = 1000) -> list[dict[str, Any]]:
    if not getattr(portfolio, "value_history", None):
        return []
    values = pd.Series(portfolio.value_history, dtype=float).sort_index()
    values.index = pd.to_datetime(list(values.index))
    if len(values) < 2:
        return []
    daily = values.resample("D").last().dropna()
    source = daily if len(daily) >= 2 else values
    returns = source.pct_change().dropna()
    if len(returns) > max_points:
        step = max(1, math.ceil(len(returns) / max_points))
        returns = returns.iloc[::step]
    return [
        {
            "step": idx + 1,
            "timestamp": ts.isoformat(),
            "return_pct": float(ret * 100.0),
            "equity": float(source.loc[ts]),
        }
        for idx, (ts, ret) in enumerate(returns.items())
        if math.isfinite(float(ret))
    ]


def _signal_forward_return_stream(portfolio, horizon: int = 1, max_points: int = 2000) -> list[dict[str, Any]]:
    signals_history = getattr(portfolio, "signals_history", None) or {}
    tick_history = getattr(portfolio, "tick_history", None) or {}
    if not signals_history or not tick_history:
        return []

    prices: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for timestamp, tick in sorted(tick_history.items()):
        for item in tick:
            if item.close is not None:
                prices.setdefault(item.symbol, []).append((pd.Timestamp(timestamp), float(item.close)))

    rows = []
    for timestamp, signals in sorted(signals_history.items()):
        ts = pd.Timestamp(timestamp)
        for signal in signals:
            series = prices.get(signal.symbol, [])
            match = next((idx for idx, (price_ts, _) in enumerate(series) if price_ts == ts), None)
            if match is None or match + horizon >= len(series):
                continue
            current = series[match][1]
            future_ts, future = series[match + horizon]
            if current <= 0:
                continue
            direction = 1.0 if getattr(signal.type, "name", "") == "BUY" else -1.0
            strength = float(getattr(signal, "strength", 100.0) or 0.0) / 100.0
            rows.append({
                "step": len(rows) + 1,
                "timestamp": ts.isoformat(),
                "symbol": signal.symbol,
                "signal": direction * strength,
                "signal_type": getattr(signal.type, "name", ""),
                "signal_strength": getattr(signal, "strength", None),
                "forward_timestamp": future_ts.isoformat(),
                "forward_return_pct": direction * ((future / current) - 1.0) * 100.0,
                "raw_forward_return_pct": ((future / current) - 1.0) * 100.0,
                "horizon_bars": horizon,
            })
    if len(rows) > max_points:
        step = max(1, math.ceil(len(rows) / max_points))
        rows = rows[::step]
    return rows
