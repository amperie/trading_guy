import json
from dataclasses import dataclass
from pathlib import Path

from trading.reporting import (
    AnalysisSummary,
    AnalyzerReportTarget,
    ExperimentReport,
    ExperimentReporter,
    LocalRunResultSink,
)


@dataclass
class Metrics:
    total_return: float = 12.5
    sharpe_ratio: float = 1.2
    total_trades: int = 2
    bracket_trades: int = 0


class Analyzer:
    def save_trades_csv(self, path):
        Path(path).write_text("id,pnl\n1,10\n", encoding="utf-8")


def test_local_result_sink_writes_canonical_manifest(tmp_path):
    summary = AnalysisSummary(
        trades=[],
        metrics=Metrics(),
        lifecycle_chains=[],
        tick_returns=[],
        daily_returns=[],
        monthly_returns=[],
        bracket_analysis=None,
        report="analysis report",
        benchmarks={},
    )
    report = ExperimentReport(
        experiment_name="experiment",
        run_name="run-1",
        tags={"tenant_id": "tenant-1"},
        parameters={"strategy": "demo"},
        analyzers=[
            AnalyzerReportTarget(
                name="primary",
                analyzer=Analyzer(),
                summary=summary,
                log_charts=False,
                log_signals=False,
                log_report=True,
                log_trades=True,
            )
        ],
    )

    result = ExperimentReporter.log(report, [LocalRunResultSink(tmp_path)])

    manifest = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["total_return"] == 12.5
    assert manifest["parameters"] == {"strategy": "demo"}
    assert manifest["tags"]["tenant_id"] == "tenant-1"
    assert {artifact["path"] for artifact in manifest["artifacts"]} == {
        "artifacts/performance_report.txt",
        "artifacts/trades.csv",
    }
    assert result.metrics["sharpe_ratio"] == 1.2


class CapturingSink(LocalRunResultSink):
    name = "capture"


def test_composite_sink_mirrors_to_all_sinks(tmp_path):
    summary = AnalysisSummary(
        trades=[],
        metrics=Metrics(),
        lifecycle_chains=[],
        tick_returns=[],
        daily_returns=[],
        monthly_returns=[],
        bracket_analysis=None,
        report="analysis report",
        benchmarks={},
    )
    report = ExperimentReport(
        experiment_name=None,
        run_name="run-1",
        analyzers=[
            AnalyzerReportTarget(
                name="primary",
                analyzer=Analyzer(),
                summary=summary,
                log_charts=False,
                log_signals=False,
                log_report=True,
                log_trades=False,
            )
        ],
    )
    left = LocalRunResultSink(tmp_path / "left")
    right = CapturingSink(tmp_path / "right")

    result = ExperimentReporter.log(report, [left, right])

    assert (tmp_path / "left" / "run_result.json").is_file()
    assert (tmp_path / "right" / "run_result.json").is_file()
    assert set(result.sink_runs) == {"local_result", "capture"}
