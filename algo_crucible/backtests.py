from __future__ import annotations

import copy
from typing import Any

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
    engine = BacktestingEngine({"status_line_enabled": False}, dp, al, om, pf)
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
    }
