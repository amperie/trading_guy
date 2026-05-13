from __future__ import annotations

import subprocess

from trading.commands.common import flatten_config
from trading.reporting import ExperimentReporter


def get_git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return {}

    return {
        "git.commit": commit,
        "git.remote_url": remote_url,
    }


def run_analysis(cfg: dict, pf, om, config_path: str | None = None):
    analysis_cfg = cfg.get("analysis", {})
    if not analysis_cfg.get("enabled", False):
        return None

    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

    parameters = dict(analysis_cfg.get("parameters") or {})
    parameters.update(flatten_config(cfg))

    analyzer = PortfolioAnalyzer(pf, om)
    report, summary = ExperimentReporter.build_single_report(
        analyzer=analyzer,
        experiment_name=analysis_cfg.get("experiment_name"),
        run_name=analysis_cfg.get("run_name"),
        description=analysis_cfg.get("description"),
        tags=get_git_info() or None,
        benchmark_paths=analysis_cfg.get("benchmarks") or None,
        parameters=parameters,
        config_artifact_paths=[config_path] if config_path else None,
    )
    ExperimentReporter.show_summary(summary)
    if analysis_cfg.get("log_to_mlflow", True):
        ExperimentReporter.log_to_mlflow(report)
    return ExperimentReporter.summary_to_legacy_dict(summary)
