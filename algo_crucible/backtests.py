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

    metrics = overall_scorecard(AnalysisEngine(pf, om).calculate_metrics())
    regimes = regime_scorecard(pf, ticks, candidate.algorithm_params.get("market_regime", {}))
    return {
        "candidate_id": candidate.candidate_id,
        "seed_id": payload.get("seed_id"),
        "neighbor_id": payload.get("neighbor_id"),
        "scenario_id": payload.get("scenario_id"),
        "source_candidate_id": payload.get("source_candidate_id", candidate.candidate_id),
        "window_id": window["window_id"],
        "window": window,
        "overall_scorecard": metrics,
        "regime_scorecard": regimes,
        "return_stream": _return_stream(pf),
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
