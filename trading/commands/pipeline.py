from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace
from typing import Any

from trading.commands.backtest import cmd_backtest
from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    fill_data_provider_creds,
    load_account_creds,
    load_raw_config,
)
from trading.commands.hpo import _normalize_split_objective_metric, _resolve_hpo_split_dates, run_hpo_split_from_raw_config
from trading.commands.live import cmd_live
from trading.commands.session_replay import cmd_session_replay
from trading.commands.walk_forward import cmd_walk_forward
from trading.config.component_loader import import_component_class
from trading.pipeline import (
    build_session_id,
    evaluate_research_gates,
    evaluate_review_gates,
    log_registered_bundle,
    materialize_bundle,
)

PIPELINE_EXPERIMENT_DEFAULTS = {
    "backtest": "Pipeline HPO",
    "hpo_split": "Pipeline HPO Split",
    "walk_forward": "Pipeline Walk-Forward",
    "review": "Pipeline - Review",
    "bundle_registry": "Pipeline - Bundle Registry",
}


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


def _resolve_session_store(connection_uri: str | None = None, database: str | None = None):
    from utils.config_manager import ConfigManager
    from utils.trading_state_store import TradingStateStore

    cfg = ConfigManager().get("state_store") or {}
    return TradingStateStore(
        connection_uri=connection_uri or cfg.get("connection_uri", "mongodb://localhost:27017"),
        database=database or cfg.get("database", "trading"),
    )


def _resolve_session_launch_source(metadata: dict[str, Any]) -> str | None:
    source_run_url = metadata.get("source_run_url")
    if isinstance(source_run_url, str) and source_run_url:
        return source_run_url
    launch_config_ref = metadata.get("launch_config_ref")
    if isinstance(launch_config_ref, str) and launch_config_ref:
        return launch_config_ref
    return None


def _as_live_args(args: argparse.Namespace, config_path: str, session_id: str) -> argparse.Namespace:
    source_run_url = None
    config_ref = str(config_path)
    if config_ref.startswith(("http://", "https://")) and "/runs/" in config_ref:
        source_run_url = config_ref
    elif str(getattr(args, "config", "")).startswith(("http://", "https://")) and "/runs/" in str(getattr(args, "config", "")):
        source_run_url = str(getattr(args, "config"))
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
        source_run_url=source_run_url,
        source_session_id=getattr(args, "source_session_id", None),
        agg_period=getattr(args, "agg_period", None),
        alpaca_override_url=getattr(args, "alpaca_override_url", None),
    )


def _pipeline_experiment_name(raw_cfg: dict[str, Any], key: str) -> str:
    pipeline_cfg = raw_cfg.get("pipeline", {}) or {}
    experiments = pipeline_cfg.get("experiments", {}) or {}
    if experiments.get(key):
        return str(experiments[key])
    try:
        from utils.config_manager import ConfigManager

        global_pipeline_cfg = ConfigManager().get("pipeline") or {}
        global_experiments = global_pipeline_cfg.get("experiments", {}) or {}
        if global_experiments.get(key):
            return str(global_experiments[key])
    except Exception:
        pass
    return PIPELINE_EXPERIMENT_DEFAULTS[key]


def _with_analysis_experiment(raw_cfg: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(raw_cfg)
    cfg.setdefault("analysis", {})["experiment_name"] = experiment_name
    return cfg


def _materialize_editable_research_config(args: argparse.Namespace) -> str:
    if not _is_mlflow_run_url(args.config):
        return args.config

    from trading.launchers.mlflow_hpo_launcher import (
        edit_config_dict,
        load_source_run_context,
        normalize_hpo_search_space,
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
    edited_cfg = normalize_hpo_search_space(edited_cfg)
    return persist_edited_config(
        source_context,
        edited_cfg,
        output_dir_name="generated_pipeline_configs",
        filename_prefix="pipeline_research",
    )


def _preflight_pipeline_research(
    raw_cfg: dict[str, Any],
    *,
    validation_period_days_override: int | None = None,
) -> None:
    experiment = build_experiment_config(raw_cfg)
    if experiment.data_provider is None:
        raise ValueError("Pipeline research requires a data_provider section.")

    validation_period_days = int(
        validation_period_days_override
        if validation_period_days_override is not None
        else (raw_cfg.get("hpo") or {}).get("validation_period_days", 0)
    )
    _resolve_hpo_split_dates(dict(experiment.data_provider.params), validation_period_days)
    _normalize_split_objective_metric((raw_cfg.get("hpo") or {}).get("objective_metric"))

    dp_class = import_component_class(experiment.data_provider)
    provider_name = f"{dp_class.__module__}.{dp_class.__name__}".lower()
    if "alpaca" not in provider_name:
        return

    probe_cfg = dict(experiment.data_provider.params)
    probe_cfg.pop("start_date", None)
    probe_cfg.pop("end_date", None)
    probe_cfg["limit"] = 1
    try:
        dp_class(probe_cfg).get_data_length()
    except Exception as exc:
        raise RuntimeError(
            "Pipeline research preflight failed while probing Alpaca historical data access. "
            "Check accounts.yaml credentials, provider config, and network access before rerunning."
        ) from exc


def cmd_pipeline_research(args: argparse.Namespace):
    effective_config = _materialize_editable_research_config(args)
    stage_args = copy.copy(args)
    stage_args.config = effective_config

    raw_cfg = load_raw_config(effective_config)
    raw_cfg = apply_cli_overrides(raw_cfg, stage_args)
    apply_session_log_file(raw_cfg, stage_args)
    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        fill_data_provider_creds(raw_cfg, load_account_creds(stage_args.account))
    _preflight_pipeline_research(
        raw_cfg,
        validation_period_days_override=getattr(stage_args, "validation_period_days", None),
    )

    stage_args.mlflow_experiment_name_override = _pipeline_experiment_name(raw_cfg, "backtest")
    backtest_result = cmd_backtest(stage_args)
    hpo_cfg = _with_analysis_experiment(raw_cfg, _pipeline_experiment_name(raw_cfg, "hpo_split"))
    hpo_result = run_hpo_split_from_raw_config(
        hpo_cfg,
        config_artifact_path=effective_config,
        num_samples_override=getattr(stage_args, "num_samples", None),
        max_concurrent_override=getattr(stage_args, "max_concurrent_trials", None),
        validation_period_days_override=getattr(stage_args, "validation_period_days", None),
        return_details=True,
    )
    stage_args.mlflow_experiment_name_override = _pipeline_experiment_name(raw_cfg, "walk_forward")
    walk_forward_result = cmd_walk_forward(stage_args)
    gate_report = evaluate_research_gates(raw_cfg, backtest_result, hpo_result, walk_forward_result)

    bundle = None
    bundle_record = None
    pipeline_cfg = raw_cfg.get("pipeline", {}) or {}
    if gate_report.passed and pipeline_cfg.get("auto_promote_research", True) and hpo_result.get("run_url"):
        bundle = materialize_bundle(hpo_result["run_url"], name=getattr(args, "name", None), paper=True)
        bundle_record = log_registered_bundle(
            {**copy.deepcopy(raw_cfg), "pipeline": {**((raw_cfg.get("pipeline") or {})), "experiment_name": _pipeline_experiment_name(raw_cfg, "bundle_registry")}},
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
        {**copy.deepcopy(raw_cfg), "pipeline": {**((raw_cfg.get("pipeline") or {})), "experiment_name": _pipeline_experiment_name(raw_cfg, "bundle_registry")}},
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


def cmd_pipeline_paper_from_session(args: argparse.Namespace):
    store = _resolve_session_store(
        connection_uri=getattr(args, "connection_uri", None),
        database=getattr(args, "database", None),
    )
    session_doc = store.get_session(args.source_session_id)
    if session_doc is None:
        raise ValueError(f"Session '{args.source_session_id}' not found in MongoDB")

    metadata = session_doc.get("metadata") or {}
    launch_source = _resolve_session_launch_source(metadata)
    if not launch_source:
        raise ValueError(
            f"Session '{args.source_session_id}' is missing metadata.source_run_url and "
            "metadata.launch_config_ref; cannot reconstruct a runnable paper bundle."
        )

    bundle = materialize_bundle(
        launch_source,
        name=getattr(args, "name", None),
        paper=True,
        tracking_uri=getattr(args, "tracking_uri", None),
    )
    session_id = getattr(args, "session_id", None) or build_session_id("paper")
    raw_cfg = load_raw_config(bundle.config_path)
    bundle_record = log_registered_bundle(
        {**copy.deepcopy(raw_cfg), "pipeline": {**((raw_cfg.get("pipeline") or {})), "experiment_name": _pipeline_experiment_name(raw_cfg, "bundle_registry")}},
        bundle,
        stage="paper",
        status="paper_ready",
        source_run_url=metadata.get("source_run_url"),
        metadata={"session_id": session_id, "source_session_id": args.source_session_id},
    )
    live_args = _as_live_args(args, bundle.config_path, session_id)
    live_args.source_run_url = metadata.get("source_run_url") or live_args.source_run_url
    live_args.source_session_id = args.source_session_id
    live_result = cmd_live(live_args)

    _print_header("PIPELINE PAPER FROM SESSION")
    _print_pairs(
        [
            ("Source Session", args.source_session_id),
            ("Source MLflow", metadata.get("source_run_url")),
            ("Launch Source", launch_source),
            ("Local Bundle", bundle.config_path),
            ("Bundle Manifest", bundle.manifest_path),
            ("Pipeline Bundle MLflow", bundle_record.get("run_url")),
            ("Paper Session", live_result.get("session_id")),
            ("Config Hash", live_result.get("config_hash")),
        ]
    )
    return {
        "source_session_id": args.source_session_id,
        "launch_source": launch_source,
        "bundle": bundle_record,
        "bundle_config": bundle.config_path,
        "live": live_result,
    }


def cmd_pipeline_review(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    args.mlflow_experiment_name_override = _pipeline_experiment_name(raw_cfg, "review")
    review_result = cmd_session_replay(args)
    gate_report = evaluate_review_gates(raw_cfg, review_result)

    approval_record = None
    approved_bundle = None
    if gate_report.passed:
        approved_bundle = materialize_bundle(args.config, name=getattr(args, "name", None), paper=False)
        approval_record = log_registered_bundle(
            {**copy.deepcopy(raw_cfg), "pipeline": {**((raw_cfg.get("pipeline") or {})), "experiment_name": _pipeline_experiment_name(raw_cfg, "bundle_registry")}},
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
