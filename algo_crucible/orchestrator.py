from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from algo_crucible.builders import build_candidate, build_components
from algo_crucible.config import resolve_configs
from algo_crucible.scoring import overall_scorecard, regime_scorecard, rows_to_csv
from algo_crucible.state_store import create_state_store
from trading.analysis.analysis_engine import AnalysisEngine
from trading.engines.backtest_engine import BacktestingEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class CrucibleOrchestrator:
    def __init__(self, platform_config: str | Path, workload_config: str | Path, state_store=None):
        self.resolved_cfg = resolve_configs(platform_config, workload_config)
        self.state_store = state_store or create_state_store(self.resolved_cfg.platform)

    def run_milestone1(self, rerun: bool = False) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        if run.get("status") == "complete":
            return run

        logger.info(f"Starting crucible run {cfg.crucible_run_id}")
        candidate = build_candidate(cfg)
        dp, al, om, pf = build_components(cfg.workload, candidate)
        ticks = list(dp.iterate())
        engine = BacktestingEngine({"status_line_enabled": False}, dp, al, om, pf)
        engine.run()

        analysis = AnalysisEngine(pf, om)
        metrics = analysis.calculate_metrics()
        overall = overall_scorecard(metrics)
        regime_cfg = candidate.algorithm_params.get("market_regime", {})
        regimes = regime_scorecard(pf, ticks, regime_cfg)

        candidate_row = {
            "candidate_id": candidate.candidate_id,
            "algorithm_class": candidate.algorithm_class,
            "portfolio_class": candidate.portfolio_class,
            **overall,
        }
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "resolved_config_hash": cfg.resolved_config_hash,
            "candidate_count": 1,
            "backtest_count": 1,
            "candidate_id": candidate.candidate_id,
            "overall_scorecard": overall,
            "regime_count": len(regimes),
        }
        artifacts = {
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, "summaries/stage_summary.json", summary),
            "candidate_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/candidate_summary.csv", rows_to_csv([candidate_row])),
            "regime_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/regime_summary.csv", rows_to_csv(regimes)),
            "candidate": self.state_store.write_artifact_json(cfg.crucible_run_id, f"candidates/{candidate.candidate_id}.json", {
                "candidate_id": candidate.candidate_id,
                "algorithm_class": candidate.algorithm_class,
                "portfolio_class": candidate.portfolio_class,
                "algorithm_params": candidate.algorithm_params,
                "portfolio_params": candidate.portfolio_params,
            }),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "complete",
            "summary": summary,
            "metrics": {f"milestone1.{key}": value for key, value in overall.items() if isinstance(value, (int, float))},
            "artifacts": artifacts,
        })
        logger.info(f"Completed crucible run {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest
