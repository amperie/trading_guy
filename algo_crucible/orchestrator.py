from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from algo_crucible.backtests import run_validation_backtest
from algo_crucible.builders import build_candidate, build_components
from algo_crucible.config import resolve_configs
from algo_crucible.gates import evaluate_regime_aware_gates, gate_summary_metrics
from algo_crucible.hpo import run_hpo_search
from algo_crucible.jobs import CrucibleJob, RayJobRunner
from algo_crucible.scoring import overall_scorecard, regime_scorecard, rows_to_csv
from algo_crucible.state_store import create_state_store
from algo_crucible.windows import data_range_from_frame, generate_walk_forward_windows, windows_to_rows
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

    def run_walk_forward_oos(self, rerun: bool = False, use_ray: bool | None = None) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        candidate = build_candidate(cfg)
        dp, _, _, _ = build_components(cfg.workload, candidate)
        dp.load_data()
        data_start, data_end = data_range_from_frame(dp.data)
        wf_cfg = cfg.platform.get("walk_forward", {})
        windows = generate_walk_forward_windows(
            data_start=data_start,
            data_end=data_end,
            optimization_window_days=int(wf_cfg.get("optimization_window_days", 30)),
            validation_window_days=int(wf_cfg.get("validation_window_days", 10)),
            embargo_days=int(wf_cfg.get("embargo_days", 0)),
            step_days=wf_cfg.get("step_days"),
            min_windows=int(wf_cfg.get("min_windows", 1)),
        )
        window_rows = windows_to_rows(windows)
        jobs = [
            CrucibleJob("03_walk_forward_oos", "validation_backtest", {
                "crucible_run_id": cfg.crucible_run_id,
                "candidate": candidate.to_dict(),
                "window": row,
                "workload": cfg.workload,
            })
            for row in window_rows
        ]
        ray_cfg = cfg.platform.get("ray", {})
        runner = RayJobRunner(
            use_ray=bool(ray_cfg.get("enabled", True) if use_ray is None else use_ray),
            max_concurrent_jobs=ray_cfg.get("max_concurrent_trials"),
        )
        batch = runner.run_jobs(
            run_id=cfg.crucible_run_id,
            jobs=jobs,
            worker=run_validation_backtest,
            state_store=self.state_store,
            rerun_failed_jobs=bool(cfg.platform.get("resume", {}).get("rerun_failed_jobs", True)),
        )
        completed = [row["result"] for row in batch.results if row.get("status") == "complete"]
        oos_rows = [_window_metric_row(result) for result in completed]
        regime_rows = [
            {"window_id": result["window_id"], "candidate_id": result["candidate_id"], **regime}
            for result in completed
            for regime in result["regime_scorecard"]
        ]
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "candidate_id": candidate.candidate_id,
            "window_count": len(windows),
            "jobs_total": batch.jobs_total,
            "jobs_complete": batch.jobs_complete,
            "jobs_failed": batch.jobs_failed,
            "jobs_reused": batch.jobs_reused,
            "median_oos_return_pct": _median([row["total_return_pct"] for row in oos_rows]),
            "profitable_windows_pct": _pct([row["total_return_pct"] > 0 for row in oos_rows]),
        }
        artifacts = {
            "window_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/window_summary.csv", rows_to_csv(window_rows)),
            "oos_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/oos_summary.csv", rows_to_csv(oos_rows)),
            "validation_regime_summary": self.state_store.write_artifact_text(
                cfg.crucible_run_id,
                "summaries/validation_regime_summary.csv",
                rows_to_csv(regime_rows),
            ),
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, "summaries/stage_03_summary.json", summary),
        }
        metrics = {
            "walk_forward_oos.jobs_total": batch.jobs_total,
            "walk_forward_oos.jobs_complete": batch.jobs_complete,
            "walk_forward_oos.jobs_failed": batch.jobs_failed,
            "walk_forward_oos.jobs_reused": batch.jobs_reused,
        }
        if summary["median_oos_return_pct"] is not None:
            metrics["walk_forward_oos.median_return_pct"] = summary["median_oos_return_pct"]
        if summary["profitable_windows_pct"] is not None:
            metrics["walk_forward_oos.profitable_windows_pct"] = summary["profitable_windows_pct"]
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "complete",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        logger.info(f"Completed walk-forward OOS stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_hpo_stage(self, rerun: bool = False) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        existing = self.state_store.read_artifact_json(cfg.crucible_run_id, "summaries/hpo_stage_summary.json")
        if existing and run.get("status") == "complete" and not rerun:
            return run

        hpo = run_hpo_search(cfg)
        trial_rows = [_json_row(row) for row in hpo["trial_rows"]]
        candidate_rows = [
            {
                "candidate_id": candidate.candidate_id,
                "algorithm_class": candidate.algorithm_class,
                "portfolio_class": candidate.portfolio_class,
                "algorithm_params": json.dumps(candidate.algorithm_params, sort_keys=True),
                "portfolio_params": json.dumps(candidate.portfolio_params, sort_keys=True),
            }
            for candidate in hpo["candidates"]
        ]
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "best_config": hpo["best_config"],
            **hpo["metrics"],
        }
        artifacts = {
            "hpo_stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, "summaries/hpo_stage_summary.json", summary),
            "hpo_trial_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/hpo_trial_summary.csv", rows_to_csv(trial_rows)),
            "hpo_failed_trials": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/hpo_failed_trials.csv", rows_to_csv(hpo["failed_trials"])),
            "hpo_candidate_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, "summaries/hpo_candidate_summary.csv", rows_to_csv(candidate_rows)),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "complete",
            "summary": summary,
            "metrics": {key: value for key, value in hpo["metrics"].items() if isinstance(value, (int, float)) and value is not None},
            "artifacts": artifacts,
        })
        logger.info(f"Completed HPO stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_regime_gate_stage(self, rerun: bool = False) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        run_dir = Path(run["run_dir"])
        oos_path = run_dir / "summaries" / "oos_summary.csv"
        regime_path = run_dir / "summaries" / "validation_regime_summary.csv"
        if not oos_path.exists() or not regime_path.exists():
            raise FileNotFoundError("run_walk_forward_oos must produce OOS and regime summaries before regime gates can run")

        overall_rows = pd.read_csv(oos_path).to_dict(orient="records")
        regime_rows = pd.read_csv(regime_path).to_dict(orient="records")
        decisions = evaluate_regime_aware_gates(overall_rows, regime_rows, cfg.platform)
        decision_rows = [decision.to_row() for decision in decisions]
        metrics = gate_summary_metrics(decisions)
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "candidate_count": len(decisions),
            "passed_candidate_count": int(metrics["regime_gate.passed"]),
            "generalist_count": int(metrics["regime_gate.generalists"]),
            "specialist_count": int(metrics["regime_gate.specialists"]),
            "reject_count": int(metrics["regime_gate.rejected"]),
        }
        artifacts = {
            "regime_gate_summary": self.state_store.write_artifact_text(
                cfg.crucible_run_id,
                "summaries/regime_gate_summary.csv",
                rows_to_csv(decision_rows),
            ),
            "stage_summary": self.state_store.write_artifact_json(
                cfg.crucible_run_id,
                "summaries/stage_05_summary.json",
                summary,
            ),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "complete",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        logger.info(f"Completed regime gate stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest


def _window_metric_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": result["candidate_id"],
        "window_id": result["window_id"],
        **result["window"],
        **result["overall_scorecard"],
    }


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _pct(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return 100.0 * sum(1 for flag in flags if flag) / len(flags)
