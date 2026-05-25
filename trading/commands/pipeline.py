from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace
from typing import Any

from trading.commands.backtest import cmd_backtest
from trading.commands.common import apply_cli_overrides, apply_session_log_file, load_raw_config
from trading.commands.hpo import run_hpo_split_from_raw_config
from trading.commands.live import cmd_live
from trading.commands.session_replay import cmd_session_replay
from trading.commands.walk_forward import cmd_walk_forward
from trading.pipeline import (
    build_session_id,
    evaluate_research_gates,
    evaluate_review_gates,
    log_registered_bundle,
    materialize_bundle,
)


def _is_mlflow_run_url(value: str | None) -> bool:
    return bool(value and value.startswith(("http://", "https://")) and "/runs/" in value)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_pairs(pairs: list[tuple[str, Any]]) -> None:
    for key, value in pairs:
        if value is None or value == "":
            continue
        print(f"{key}: {value}")


def _print_gate_report(report) -> None:
    print(f"Gate status: {'PASS' if report.passed else 'FAIL'}")
    if not report.checks:
        print("Gates: none configured")
        return
    for check in report.checks:
        actual = "n/a" if check.actual is None else f"{check.actual:.4f}"
        threshold = "n/a" if check.threshold is None else f"{check.threshold:.4f}"
        status = "PASS" if check.passed else "FAIL"
        print(f"{status} {check.name}: actual={actual} {check.comparator} threshold={threshold}")


def _as_live_args(args: argparse.Namespace, config_path: str, session_id: str) -> argparse.Namespace:
    return SimpleNamespace(
        config=config_path,
        account=args.account,
        symbol=getattr(args, "symbol", None),
        cash=getattr(args, "cash", None),
        algorithm=getattr(args, "algorithm", None),
        algorithm_url=getattr(args, "algorithm_url", None),
        portfolio=getattr(args, "portfolio", None),
        portfolio_url=getattr(args, "portfolio_url", None),
        no_mlflow=getattr(args, "no_mlflow", False),
        run_name=getattr(args, "run_name", None),
        session_id=session_id,
        agg_period=getattr(args, "agg_period", None),
        alpaca_override_url=getattr(args, "alpaca_override_url", None),
    )


def _materialize_editable_research_config(args: argparse.Namespace) -> str:
    if not _is_mlflow_run_url(args.config):
        return args.config

    from trading.launchers.mlflow_hpo_launcher import (
        edit_config_dict,
        load_source_run_context,
        persist_edited_config,
        sanitize_source_config,
    )

    source_context = load_source_run_context(
        args.config,
        tracking_uri=getattr(args, "tracking_uri", None),
    )
    edited_cfg = edit_config_dict(
        source_context.raw_config,
        editor=getattr(args, "editor", None),
        filename="pipeline_research_config.yaml",
        label="pipeline research config",
    )
    edited_cfg = sanitize_source_config(edited_cfg) if "execution_config" in edited_cfg else edited_cfg
    return persist_edited_config(
        source_context,
        edited_cfg,
        output_dir_name="generated_pipeline_configs",
        filename_prefix="pipeline_research",
    )


def cmd_pipeline_research(args: argparse.Namespace):
    effective_config = _materialize_editable_research_config(args)
    stage_args = copy.copy(args)
    stage_args.config = effective_config

    raw_cfg = load_raw_config(effective_config)
    raw_cfg = apply_cli_overrides(raw_cfg, stage_args)
    apply_session_log_file(raw_cfg, stage_args)

    backtest_result = cmd_backtest(stage_args)
    hpo_result = run_hpo_split_from_raw_config(
        raw_cfg,
        config_artifact_path=effective_config,
        num_samples_override=getattr(stage_args, "num_samples", None),
        max_concurrent_override=getattr(stage_args, "max_concurrent_trials", None),
        validation_period_days_override=getattr(stage_args, "validation_period_days", None),
        return_details=True,
    )
    walk_forward_result = cmd_walk_forward(stage_args)
    gate_report = evaluate_research_gates(raw_cfg, backtest_result, hpo_result, walk_forward_result)

    bundle = None
    bundle_record = None
    pipeline_cfg = raw_cfg.get("pipeline", {}) or {}
    if gate_report.passed and pipeline_cfg.get("auto_promote_research", True) and hpo_result.get("run_url"):
        bundle = materialize_bundle(hpo_result["run_url"], name=getattr(args, "name", None), paper=True)
        bundle_record = log_registered_bundle(
            raw_cfg,
            bundle,
            stage="candidate",
            status="candidate",
            source_run_url=hpo_result["run_url"],
            metadata={
                "backtest_mlflow_run_url": ((backtest_result or {}).get("analysis") or {}).get("mlflow_run_url"),
                "hpo_mlflow_run_url": hpo_result.get("run_url"),
                "walk_forward_mlflow_run_url": (walk_forward_result or {}).get("mlflow_run_url"),
                "research_config_path": effective_config,
            },
        )

    _print_header("PIPELINE RESEARCH")
    _print_pairs(
        [
            ("Config", effective_config),
            ("Source Config", args.config if effective_config != args.config else None),
            ("Backtest MLflow", ((backtest_result or {}).get("analysis") or {}).get("mlflow_run_url")),
            ("HPO Split MLflow", hpo_result.get("run_url")),
            ("Walk-Forward MLflow", (walk_forward_result or {}).get("mlflow_run_url")),
            ("Candidate Bundle", bundle.config_path if bundle else None),
            ("Candidate Manifest", bundle.manifest_path if bundle else None),
            ("Candidate Bundle MLflow", (bundle_record or {}).get("run_url")),
        ]
    )
    _print_gate_report(gate_report)
    return {
        "backtest": backtest_result,
        "hpo_split": hpo_result,
        "walk_forward": walk_forward_result,
        "gates": gate_report.to_dict(),
        "bundle": bundle_record,
    }


def cmd_pipeline_paper(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.run_url)
    bundle = materialize_bundle(args.run_url, name=getattr(args, "name", None), paper=True)
    session_id = getattr(args, "session_id", None) or build_session_id("paper")
    bundle_record = log_registered_bundle(
        raw_cfg,
        bundle,
        stage="paper",
        status="paper_ready",
        source_run_url=args.run_url,
        metadata={"session_id": session_id},
    )
    live_result = cmd_live(_as_live_args(args, bundle.config_path, session_id))
    _print_header("PIPELINE PAPER")
    _print_pairs(
        [
            ("Source MLflow", args.run_url),
            ("Local Bundle", bundle.config_path),
            ("Bundle Manifest", bundle.manifest_path),
            ("Pipeline Bundle MLflow", bundle_record.get("run_url")),
            ("Paper Session", live_result.get("session_id")),
        ]
    )
    return {"bundle": bundle_record, "live": live_result}


def cmd_pipeline_review(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    review_result = cmd_session_replay(args)
    gate_report = evaluate_review_gates(raw_cfg, review_result)

    approval_record = None
    approved_bundle = None
    if gate_report.passed:
        approved_bundle = materialize_bundle(args.config, name=getattr(args, "name", None), paper=False)
        approval_record = log_registered_bundle(
            raw_cfg,
            approved_bundle,
            stage="approved",
            status="approved",
            source_run_url=args.config if args.config.startswith(("http://", "https://")) else None,
            metadata=review_result,
        )

    _print_header("PIPELINE REVIEW")
    _print_pairs(
        [
            ("Config", args.config),
            ("Replay MLflow", review_result.get("mlflow_run_url")),
            ("Approved Bundle", approved_bundle.config_path if approved_bundle else None),
            ("Approved Bundle MLflow", (approval_record or {}).get("run_url")),
        ]
    )
    _print_gate_report(gate_report)
    return {"review": review_result, "gates": gate_report.to_dict(), "approved_bundle": approval_record}


def cmd_pipeline_live(args: argparse.Namespace):
    session_id = getattr(args, "session_id", None) or build_session_id("live")
    live_result = cmd_live(_as_live_args(args, args.config, session_id))
    _print_header("PIPELINE LIVE")
    _print_pairs(
        [
            ("Config", args.config),
            ("Session", live_result.get("session_id")),
            ("Config Hash", live_result.get("config_hash")),
        ]
    )
    return live_result
