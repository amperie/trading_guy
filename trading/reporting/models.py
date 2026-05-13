from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class AnalysisSummary:
    trades: Any
    metrics: Any
    lifecycle_chains: Any
    tick_returns: Any
    daily_returns: Any
    monthly_returns: Any
    bracket_analysis: Any
    report: str
    benchmarks: Any


@dataclass(slots=True)
class AnalyzerReportTarget:
    name: str
    analyzer: Any
    summary: AnalysisSummary
    metric_prefix: str = ""
    artifact_prefix: str = ""
    log_charts: bool = True
    log_trades: bool = True
    log_signals: bool = True
    log_report: bool = True


@dataclass(slots=True)
class CombinedArtifactSpec:
    filename: str
    builder: Callable[[str], None]


@dataclass(slots=True)
class ExperimentReport:
    experiment_name: str | None
    run_name: str | None
    tracking_uri: str | None = None
    description: str | None = None
    tags: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    analyzers: list[AnalyzerReportTarget] = field(default_factory=list)
    config_artifact_paths: list[str] = field(default_factory=list)
    combined_artifacts: list[CombinedArtifactSpec] = field(default_factory=list)
