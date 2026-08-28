from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from algo_crucible.backtests import run_validation_backtest
from algo_crucible.builders import build_candidate, build_components
from algo_crucible.confirmation import (
    build_promotion_packet,
    confirmation_metrics,
    confirmation_window,
    confirmation_workload,
    freeze_candidate,
    load_confirmation_candidates,
    packet_markdown,
    summarize_confirmation,
    write_promoted_packet,
)
from algo_crucible.config import resolve_configs
from algo_crucible.gates import evaluate_regime_aware_gates, gate_summary_metrics
from algo_crucible.hpo import run_hpo_search
from algo_crucible.jobs import CrucibleJob, RayJobRunner
from algo_crucible.paper_replay import (
    compare_traces,
    load_frozen_candidate,
    load_observed_trace,
    mismatch_rows,
    replay_trace,
    trace_to_csv,
)
from algo_crucible.plateau import (
    build_plateau_neighbors,
    distance_decay_svg,
    load_plateau_seeds,
    plateau_metrics,
    seed_summary_rows,
    summarize_plateaus,
)
from algo_crucible.perturbations import (
    apply_scenario,
    build_perturbation_scenarios,
    load_perturbation_candidates,
    perturbation_metrics,
    summarize_perturbations,
)
from algo_crucible.scoring import distribution_stats, distribution_svg, overall_scorecard, prefixed_numeric_metrics, regime_scorecard, rows_to_csv
from algo_crucible.state_store import create_state_store
from algo_crucible.windows import data_range_from_frame, generate_walk_forward_windows, windows_to_rows
from trading.analysis.analysis_engine import AnalysisEngine
from trading.engines.backtest_engine import BacktestingEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)

STAGE_01 = "stages/01_single_candidate"
STAGE_03 = "stages/03_walk_forward_oos"
STAGE_04 = "stages/04_hpo"
STAGE_05 = "stages/05_regime_gate"
STAGE_06 = "stages/06_plateau"
STAGE_07 = "stages/07_perturbation"
STAGE_08 = "stages/08_confirmation"
STAGE_09 = "stages/09_paper_replay"


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
        engine = BacktestingEngine({"status_line_enabled": False, "state_store": {"enabled": False}}, dp, al, om, pf)
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
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_01}/summaries/stage_summary.json", summary),
            "candidate_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_01}/summaries/candidate_summary.csv", rows_to_csv([candidate_row])),
            "regime_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_01}/summaries/regime_summary.csv", rows_to_csv(regimes)),
            "candidate": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_01}/candidates/{candidate.candidate_id}.json", {
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
        if self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_03}/summaries/stage_summary.json") and not rerun:
            return run
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
        distributions = distribution_stats(oos_rows)
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
            "distribution_stats": distributions,
        }
        artifacts = {
            "window_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_03}/summaries/window_summary.csv", rows_to_csv(window_rows)),
            "oos_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_03}/summaries/oos_summary.csv", rows_to_csv(oos_rows)),
            "validation_regime_summary": self.state_store.write_artifact_text(
                cfg.crucible_run_id,
                f"{STAGE_03}/summaries/validation_regime_summary.csv",
                rows_to_csv(regime_rows),
            ),
            "oos_distribution_chart": self.state_store.write_artifact_text(
                cfg.crucible_run_id,
                f"{STAGE_03}/charts/walk_forward_oos_distributions.svg",
                distribution_svg(oos_rows, "Walk-forward OOS metric distributions"),
            ),
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_03}/summaries/stage_summary.json", summary),
        }
        metrics = {
            "walk_forward_oos.jobs_total": batch.jobs_total,
            "walk_forward_oos.jobs_complete": batch.jobs_complete,
            "walk_forward_oos.jobs_failed": batch.jobs_failed,
            "walk_forward_oos.jobs_reused": batch.jobs_reused,
            **prefixed_numeric_metrics("walk_forward_oos", distributions),
        }
        if summary["median_oos_return_pct"] is not None:
            metrics["walk_forward_oos.median_return_pct"] = summary["median_oos_return_pct"]
        if summary["profitable_windows_pct"] is not None:
            metrics["walk_forward_oos.profitable_windows_pct"] = summary["profitable_windows_pct"]
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "running",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        logger.info(f"Completed walk-forward OOS stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_hpo_stage(self, rerun: bool = False) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        existing = self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_04}/summaries/stage_summary.json")
        if existing and not rerun:
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
            "hpo_stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_04}/summaries/stage_summary.json", summary),
            "hpo_trial_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_04}/summaries/hpo_trial_summary.csv", rows_to_csv(trial_rows)),
            "hpo_failed_trials": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_04}/summaries/hpo_failed_trials.csv", rows_to_csv(hpo["failed_trials"])),
            "hpo_candidate_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_04}/summaries/hpo_candidate_summary.csv", rows_to_csv(candidate_rows)),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "running",
            "summary": summary,
            "metrics": {key: value for key, value in hpo["metrics"].items() if isinstance(value, (int, float)) and value is not None},
            "artifacts": artifacts,
        })
        logger.info(f"Completed HPO stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_regime_gate_stage(self, rerun: bool = False) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        if self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_05}/summaries/stage_summary.json") and not rerun:
            return run
        run_dir = Path(run["run_dir"])
        oos_path = _existing_path(run_dir, f"{STAGE_03}/summaries/oos_summary.csv", "summaries/oos_summary.csv")
        regime_path = _existing_path(run_dir, f"{STAGE_03}/summaries/validation_regime_summary.csv", "summaries/validation_regime_summary.csv")
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
                f"{STAGE_05}/summaries/regime_gate_summary.csv",
                rows_to_csv(decision_rows),
            ),
            "stage_summary": self.state_store.write_artifact_json(
                cfg.crucible_run_id,
                f"{STAGE_05}/summaries/stage_summary.json",
                summary,
            ),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "running",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        logger.info(f"Completed regime gate stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_plateau_stage(self, rerun: bool = False, use_ray: bool | None = None) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        if self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_06}/summaries/stage_summary.json") and not rerun:
            return run
        run_dir = Path(run["run_dir"])
        logger.info(f"Starting plateau stage for {cfg.crucible_run_id}")
        space = cfg.workload.get("search_space", {}).get("space") or cfg.platform.get("hpo", {}).get("space", {})
        seeds = load_plateau_seeds(run_dir, cfg, cfg.platform)
        neighbors = build_plateau_neighbors(seeds, cfg, cfg.platform)
        logger.info(
            f"Prepared plateau stage run_id={cfg.crucible_run_id} "
            f"seeds={len(seeds)} neighbors={len(neighbors)}"
        )

        dp, _, _, _ = build_components(cfg.workload, build_candidate(cfg))
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
        logger.info(
            f"Plateau validation windows run_id={cfg.crucible_run_id} "
            f"windows={len(window_rows)} jobs={len(neighbors) * len(window_rows)}"
        )
        jobs = [
            CrucibleJob("06_plateau", "plateau_validation_backtest", {
                "crucible_run_id": cfg.crucible_run_id,
                "seed_id": neighbor["seed_id"],
                "neighbor_id": neighbor["neighbor_id"],
                "candidate": neighbor["candidate"].to_dict(),
                "window": window,
                "workload": cfg.workload,
            })
            for neighbor in neighbors
            for window in window_rows
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
        scored = summarize_plateaus(seeds=seeds, neighbors=neighbors, job_results=batch.results, platform=cfg.platform)
        seed_rows = seed_summary_rows(seeds, space)
        neighbor_rows = scored["neighbor_rows"]
        plateau_rows = scored["summary_rows"]
        metrics = plateau_metrics(plateau_rows, batch.jobs_total, batch.jobs_complete, batch.jobs_failed)
        gate_value = _pct_threshold(cfg.platform.get("gates", {}).get("generalist", {}).get("min_median_oos_return", 0.0))
        artifact_index = {}
        chart_artifacts = {}
        for seed in seeds:
            rows = [row for row in neighbor_rows if row["seed_id"] == seed["seed_id"]]
            path = f"{STAGE_06}/plots/plateau_distance_decay_{seed['seed_id']}.svg"
            chart_artifacts[seed["seed_id"]] = self.state_store.write_artifact_text(
                cfg.crucible_run_id,
                path,
                distance_decay_svg(seed["seed_id"], rows, gate_value),
            )
            artifact_index[seed["seed_id"]] = {
                "distance_decay": path,
                "gate_metric": "median_oos_return",
                "gate_value": gate_value,
                "detail_run_ids": [],
            }
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "seed_count": len(seeds),
            "neighbor_count": len(neighbors),
            "accepted_plateaus": int(metrics["plateau.accepted_plateaus"]),
            "rejected_peaks": int(metrics["plateau.rejected_peaks"]),
        }
        artifacts = {
            "plateau_seed_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_06}/summaries/plateau_seed_summary.csv", rows_to_csv(seed_rows)),
            "plateau_neighbor_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_06}/summaries/plateau_neighbor_summary.csv", rows_to_csv(neighbor_rows)),
            "plateau_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_06}/summaries/plateau_summary.csv", rows_to_csv(plateau_rows)),
            "plateau_artifact_index": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_06}/summaries/plateau_artifact_index.json", artifact_index),
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_06}/summaries/stage_summary.json", summary),
            **chart_artifacts,
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "running",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        for row in plateau_rows:
            logger.info(
                f"Plateau decision run_id={cfg.crucible_run_id} seed_id={row.get('seed_id')} "
                f"candidate_id={row.get('candidate_id')} accepted={row.get('accepted')} "
                f"pass_rate={row.get('neighbor_pass_rate')} reason={row.get('failure_reason')}"
            )
        logger.info(f"Completed plateau stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_perturbation_stage(self, rerun: bool = False, use_ray: bool | None = None) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        if self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_07}/summaries/stage_summary.json") and not rerun:
            return run
        run_dir = Path(run["run_dir"])
        logger.info(f"Starting perturbation stage for {cfg.crucible_run_id}")
        candidates = load_perturbation_candidates(run_dir, cfg)
        scenarios = build_perturbation_scenarios(cfg.platform)
        logger.info(
            f"Prepared perturbation stage run_id={cfg.crucible_run_id} "
            f"candidates={len(candidates)} scenarios={len(scenarios)}"
        )

        dp, _, _, _ = build_components(cfg.workload, build_candidate(cfg))
        dp.load_data()
        data_start, data_end = data_range_from_frame(dp.data)
        wf_cfg = cfg.platform.get("walk_forward", {})
        windows = windows_to_rows(generate_walk_forward_windows(
            data_start=data_start,
            data_end=data_end,
            optimization_window_days=int(wf_cfg.get("optimization_window_days", 30)),
            validation_window_days=int(wf_cfg.get("validation_window_days", 10)),
            embargo_days=int(wf_cfg.get("embargo_days", 0)),
            step_days=wf_cfg.get("step_days"),
            min_windows=int(wf_cfg.get("min_windows", 1)),
        ))
        jobs = []
        scenario_rows = []
        for candidate in candidates:
            for scenario in scenarios:
                workload, perturbed = apply_scenario(cfg, candidate["candidate"], scenario)
                scenario_rows.append({
                    "candidate_id": candidate["candidate"].candidate_id,
                    "scenario_id": scenario["scenario_id"],
                    "scenario_name": scenario["name"],
                    "required": scenario["required"],
                    "perturbed_candidate_id": perturbed.candidate_id,
                })
                for window in windows:
                    jobs.append(CrucibleJob("07_perturbation", "scenario_validation_backtest", {
                        "crucible_run_id": cfg.crucible_run_id,
                        "scenario_id": scenario["scenario_id"],
                        "source_candidate_id": candidate["candidate"].candidate_id,
                        "candidate": perturbed.to_dict(),
                        "window": window,
                        "workload": workload,
                    }))
        logger.info(
            f"Perturbation validation windows run_id={cfg.crucible_run_id} "
            f"windows={len(windows)} jobs={len(jobs)}"
        )

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
        scored = summarize_perturbations(
            candidates=candidates,
            scenarios=scenarios,
            job_results=batch.results,
            platform=cfg.platform,
        )
        metrics = perturbation_metrics(scored["summary_rows"], batch.jobs_total, batch.jobs_complete, batch.jobs_failed)
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "candidate_count": len(candidates),
            "scenario_count": len(scenarios),
            "accepted_candidates": int(metrics["perturbation.accepted_candidates"]),
            "rejected_candidates": int(metrics["perturbation.rejected_candidates"]),
        }
        artifacts = {
            "perturbation_scenarios": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_07}/summaries/perturbation_scenarios.csv", rows_to_csv(scenario_rows)),
            "perturbation_scenario_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_07}/summaries/perturbation_scenario_summary.csv", rows_to_csv(scored["scenario_rows"])),
            "perturbation_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_07}/summaries/perturbation_summary.csv", rows_to_csv(scored["summary_rows"])),
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_07}/summaries/stage_summary.json", summary),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "running",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        for row in scored["summary_rows"]:
            logger.info(
                f"Perturbation decision run_id={cfg.crucible_run_id} "
                f"candidate_id={row.get('candidate_id')} accepted={row.get('accepted')} "
                f"pass_rate={row.get('scenario_pass_rate')} reason={row.get('failure_reason')}"
            )
        logger.info(f"Completed perturbation stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_confirmation_stage(
        self,
        rerun: bool = False,
        use_ray: bool | None = None,
        create_promoted_folder: bool | None = None,
    ) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=rerun)
        if (
            self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_08}/summaries/stage_summary.json")
            and self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_08}/promotion/promotion_packet.json")
            and not rerun
        ):
            return run
        run_dir = Path(run["run_dir"])
        logger.info(f"Starting confirmation stage for {cfg.crucible_run_id}")
        candidates = load_confirmation_candidates(run_dir, cfg)
        window = confirmation_window(cfg.platform)
        workload = confirmation_workload(cfg, window)
        logger.info(
            f"Prepared confirmation stage run_id={cfg.crucible_run_id} "
            f"candidates={len(candidates)} start={window['validation_start']} end={window['validation_end']}"
        )

        frozen = [freeze_candidate(item["candidate"], cfg) for item in candidates]
        jobs = [
            CrucibleJob("08_confirmation", "confirmation_backtest", {
                "crucible_run_id": cfg.crucible_run_id,
                "candidate": item["candidate"].to_dict(),
                "window": window,
                "workload": workload,
            })
            for item in candidates
        ]
        ray_cfg = cfg.platform.get("ray", {})
        batch = RayJobRunner(
            use_ray=bool(ray_cfg.get("enabled", True) if use_ray is None else use_ray),
            max_concurrent_jobs=ray_cfg.get("max_concurrent_trials"),
        ).run_jobs(
            run_id=cfg.crucible_run_id,
            jobs=jobs,
            worker=run_validation_backtest,
            state_store=self.state_store,
            rerun_failed_jobs=bool(cfg.platform.get("resume", {}).get("rerun_failed_jobs", True)),
        )

        confirmation_rows = summarize_confirmation(candidates, batch.results, cfg.platform)
        metrics = confirmation_metrics(confirmation_rows, batch.jobs_total, batch.jobs_complete, batch.jobs_failed)
        artifacts = {
            "confirmation_summary": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_08}/summaries/confirmation_summary.csv", rows_to_csv(confirmation_rows)),
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_08}/summaries/stage_summary.json", {
                "crucible_run_id": cfg.crucible_run_id,
                "run_name": cfg.run_name,
                "candidate_count": len(candidates),
                "confirmed_candidates": int(metrics["confirmation.promoted_candidates"]),
                "rejected_candidates": int(metrics["confirmation.rejected_candidates"]),
            }),
        }
        for item in frozen:
            artifacts[f"frozen_{item['candidate']['candidate_id']}"] = self.state_store.write_artifact_json(
                cfg.crucible_run_id,
                f"{STAGE_08}/frozen_candidates/{item['candidate']['candidate_id']}.json",
                item,
            )

        packet = build_promotion_packet(
            resolved_cfg=cfg,
            confirmation_rows=confirmation_rows,
            frozen_candidates=frozen,
            artifact_paths=artifacts,
            mlflow_run_url=run.get("mlflow_run_url"),
        )
        artifacts.update({
            "promotion_packet_json": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_08}/promotion/promotion_packet.json", packet),
            "promotion_packet_yaml": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_08}/promotion/promotion_packet.yaml", yaml.safe_dump(packet, sort_keys=True)),
            "promotion_packet_md": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_08}/promotion/promotion_packet.md", packet_markdown(packet)),
        })
        should_create = bool(cfg.platform.get("promotion", {}).get("create_promoted_folder", False))
        if create_promoted_folder is not None:
            should_create = bool(create_promoted_folder)
        if should_create:
            artifacts.update(write_promoted_packet(packet, cfg.platform))

        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "candidate_count": len(candidates),
            "confirmed_candidates": int(metrics["confirmation.promoted_candidates"]),
            "rejected_candidates": int(metrics["confirmation.rejected_candidates"]),
            "promotion_packet": f"{STAGE_08}/promotion/promotion_packet.json",
            "paper_trading_started": False,
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "complete",
            "summary": summary,
            "metrics": metrics,
            "artifacts": artifacts,
        })
        for row in confirmation_rows:
            logger.info(
                f"Confirmation decision run_id={cfg.crucible_run_id} "
                f"candidate_id={row.get('candidate_id')} confirmed={row.get('confirmed')} "
                f"return={row.get('total_return_pct')} reason={row.get('failure_reason')}"
            )
        logger.info(f"Completed confirmation stage for {cfg.crucible_run_id}: {json.dumps(summary, sort_keys=True)}")
        return manifest

    def run_paper_replay_stage(self, rerun: bool = False) -> dict[str, Any]:
        cfg = self.resolved_cfg
        run = self.state_store.start_or_resume(cfg, rerun=True)
        if self.state_store.read_artifact_json(cfg.crucible_run_id, f"{STAGE_09}/summaries/stage_summary.json") and not rerun:
            return run
        paper_cfg = cfg.platform.get("paper_replay", {})
        observed_trace_path = paper_cfg.get("observed_trace_path")
        if not observed_trace_path:
            raise ValueError("paper_replay.observed_trace_path is required")

        run_dir = Path(run["run_dir"])
        logger.info(f"Starting paper replay stage for {cfg.crucible_run_id}")
        candidate = load_frozen_candidate(run_dir, paper_cfg.get("candidate_id"))
        replay_rows = replay_trace(cfg, candidate, paper_cfg.get("data_path"))
        observed_rows = load_observed_trace(observed_trace_path)
        compared = compare_traces(replay_rows, observed_rows, cfg.platform)
        summary = {
            "crucible_run_id": cfg.crucible_run_id,
            "run_name": cfg.run_name,
            "candidate_id": candidate.candidate_id,
            "passed": compared["passed"],
            "failure_reason": compared["failure_reason"],
            "mismatches_total": int(compared["metrics"]["paper_replay.mismatches_total"]),
            "paper_trace_path": str(observed_trace_path),
        }
        artifacts = {
            "paper_replay_trace": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_09}/summaries/paper_replay_trace.csv", trace_to_csv(replay_rows)),
            "paper_replay_mismatches": self.state_store.write_artifact_text(cfg.crucible_run_id, f"{STAGE_09}/summaries/paper_replay_mismatches.csv", rows_to_csv(mismatch_rows(compared["mismatches"]))),
            "stage_summary": self.state_store.write_artifact_json(cfg.crucible_run_id, f"{STAGE_09}/summaries/stage_summary.json", summary),
        }
        manifest = self.state_store.update_run(cfg.crucible_run_id, {
            "status": "paper_replay_passed" if compared["passed"] else "paper_replay_failed",
            "summary": summary,
            "metrics": compared["metrics"],
            "artifacts": artifacts,
        })
        logger.info(
            f"Completed paper replay stage for {cfg.crucible_run_id}: "
            f"passed={compared['passed']} mismatches={summary['mismatches_total']}"
        )
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


def _pct_threshold(value) -> float:
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _existing_path(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = run_dir / relative_path
        if path.exists():
            return path
    return run_dir / relative_paths[0]
