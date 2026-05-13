from __future__ import annotations

import dataclasses
import math
import os
import uuid
from typing import Any

from trading.reporting.models import (
    AnalysisSummary,
    AnalyzerReportTarget,
    CombinedArtifactSpec,
    ExperimentReport,
)
from utils.mlflow_client import MLflowClient


def summarize_analyzer(analyzer, benchmark_paths: dict[str, str] | None = None) -> AnalysisSummary:
    if benchmark_paths:
        analyzer._benchmark_paths = benchmark_paths

    metrics = analyzer.get_metrics()
    return AnalysisSummary(
        trades=analyzer.get_trades(),
        metrics=metrics,
        lifecycle_chains=analyzer.get_lifecycle_chains(),
        tick_returns=analyzer.get_tick_returns(),
        daily_returns=analyzer.get_daily_returns(),
        monthly_returns=analyzer.get_monthly_returns(),
        bracket_analysis=analyzer.analyze_bracket_effectiveness() if metrics.bracket_trades > 0 else None,
        report=analyzer.generate_report(),
        benchmarks=analyzer.calculate_external_benchmarks(),
    )


def _print_summary(summary: AnalysisSummary) -> None:
    metrics = summary.metrics
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total Return:       ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
    print(f"Annualized Return:  {metrics.annualized_return:.2f}%")
    print(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {metrics.max_drawdown_pct:.2f}%")
    print(f"\nTotal Trades:       {metrics.total_trades}")
    print(f"Win Rate:           {metrics.win_rate:.1f}%")
    print(f"Profit Factor:      {metrics.profit_factor:.2f}")
    if metrics.bracket_trades > 0:
        print(f"\nBracket Orders:     {metrics.bracket_trades}")
        print(f"Stop Loss Rate:     {metrics.bracket_stop_rate:.1f}%")
        print(f"Profit Taker Rate:  {metrics.bracket_profit_rate:.1f}%")
    print("=" * 80 + "\n")


def _workspace_temp_dir() -> str:
    base_dir = os.path.join(os.getcwd(), "scratch", "tmp_reporting")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    return path


class ExperimentReporter:
    @staticmethod
    def build_single_report(
        *,
        analyzer,
        experiment_name: str | None,
        run_name: str | None,
        description: str | None,
        tags: dict[str, Any] | None,
        parameters: dict[str, Any] | None,
        benchmark_paths: dict[str, str] | None = None,
        config_artifact_paths: list[str] | None = None,
    ) -> tuple[ExperimentReport, AnalysisSummary]:
        summary = summarize_analyzer(analyzer, benchmark_paths=benchmark_paths)
        report = ExperimentReport(
            experiment_name=experiment_name,
            run_name=run_name,
            description=description,
            tags=tags,
            parameters=parameters or {},
            analyzers=[AnalyzerReportTarget(name="primary", analyzer=analyzer, summary=summary)],
            config_artifact_paths=list(config_artifact_paths or []),
        )
        return report, summary

    @staticmethod
    def summary_to_legacy_dict(summary: AnalysisSummary) -> dict[str, Any]:
        return {
            "trades": summary.trades,
            "metrics": summary.metrics,
            "lifecycle_chains": summary.lifecycle_chains,
            "tick_returns": summary.tick_returns,
            "daily_returns": summary.daily_returns,
            "monthly_returns": summary.monthly_returns,
            "bracket_analysis": summary.bracket_analysis,
            "report": summary.report,
            "benchmarks": summary.benchmarks,
        }

    @staticmethod
    def show_summary(summary: AnalysisSummary) -> None:
        _print_summary(summary)

    @staticmethod
    def log_to_mlflow(report: ExperimentReport) -> None:
        if report.tracking_uri:
            client = MLflowClient(experiment_name=report.experiment_name, tracking_uri=report.tracking_uri)
        else:
            client = MLflowClient.from_config(experiment_name=report.experiment_name)
        if not client.enabled:
            return

        with client.start_run(run_name=report.run_name, description=report.description, tags=report.tags):
            if report.parameters:
                client.log_params(report.parameters)

            for target in report.analyzers:
                ExperimentReporter._log_target(client, target)

            for path in report.config_artifact_paths:
                if os.path.isfile(path):
                    client.log_artifact(path, artifact_path="config")

            if report.combined_artifacts:
                temp_dir = _workspace_temp_dir()
                for spec in report.combined_artifacts:
                    fpath = os.path.join(temp_dir, spec.filename)
                    try:
                        spec.builder(fpath)
                        client.log_artifact(fpath)
                    except Exception as exc:
                        target_name = spec.filename
                        from utils.logger import Logger
                        Logger().get_logger(__name__).warning(f"Failed to log {target_name}: {exc}")

    @staticmethod
    def _log_target(client: MLflowClient, target: AnalyzerReportTarget) -> None:
        metrics_dict = {}
        for field in dataclasses.fields(target.summary.metrics):
            value = getattr(target.summary.metrics, field.name)
            if isinstance(value, (int, float)):
                numeric = float(value)
                if not (math.isnan(numeric) or math.isinf(numeric)):
                    metrics_dict[f"{target.metric_prefix}{field.name}"] = numeric
        client.log_metrics(metrics_dict)

        analyzer = target.analyzer
        session_metadata = getattr(analyzer, "_session_metadata", {}) or {}
        if session_metadata:
            from trading.commands.common import flatten_config

            prefixed = {
                f"{target.metric_prefix}session": session_metadata
            } if target.metric_prefix else {"session": session_metadata}
            client.log_params(flatten_config(prefixed))

        if target.log_report:
            client.log_text(target.summary.report, f"{target.artifact_prefix}performance_report.txt")
            if hasattr(analyzer, "generate_all_orders_report"):
                try:
                    all_orders = analyzer.generate_all_orders_report()
                    if all_orders:
                        client.log_text(all_orders, f"{target.artifact_prefix}all_orders_report.txt")
                except Exception:
                    pass

        if target.log_charts:
            tmp = _workspace_temp_dir()
            ExperimentReporter._log_chart_artifacts(client, analyzer, tmp, target.artifact_prefix)

        if target.log_signals:
            tmp = _workspace_temp_dir()
            fpath = os.path.join(tmp, f"{target.artifact_prefix}signals_orders.csv")
            try:
                analyzer.save_signals_orders_csv(fpath)
                client.log_artifact(fpath)
            except Exception:
                pass

        if target.log_trades:
            tmp = _workspace_temp_dir()
            fpath = os.path.join(tmp, f"{target.artifact_prefix}trades.csv")
            try:
                analyzer.save_trades_csv(fpath)
                client.log_artifact(fpath)
            except Exception:
                pass

    @staticmethod
    def _log_chart_artifacts(client: MLflowClient, analyzer, tmp: str, artifact_prefix: str) -> None:
        save_artifacts = {
            "equity_curve.png": "save_equity_curve",
            "technical_analysis.png": "save_technical_chart",
            "drawdown.png": "save_drawdown_chart",
            "orders_chart.png": "save_orders_chart",
            "lifecycle.png": "save_lifecycle_chart",
            "lifecycle_interactive.html": "save_lifecycle_chart_interactive",
        }
        for fname, method_name in save_artifacts.items():
            method = getattr(analyzer, method_name, None)
            if callable(method):
                fpath = os.path.join(tmp, f"{artifact_prefix}{fname}")
                try:
                    method(fpath)
                    client.log_artifact(fpath)
                except Exception:
                    pass

        plot_artifacts = (
            ("portfolio_with_trades.png", "plot_portfolio_with_trades"),
            ("trade_pnl.png", "plot_trade_pnl"),
            ("returns_distribution.png", "plot_returns_distribution"),
            ("stock_performance.png", "plot_stock_performance"),
            ("dashboard.png", "plot_comprehensive_dashboard"),
            ("orders_timeline.png", "plot_orders_timeline"),
            ("interactive_portfolio.html", "plot_interactive_portfolio"),
        )
        for fname, method_name in plot_artifacts:
            method = getattr(analyzer, method_name, None)
            if callable(method):
                fpath = os.path.join(tmp, f"{artifact_prefix}{fname}")
                try:
                    method(show=False, save_path=fpath)
                    if os.path.isfile(fpath):
                        client.log_artifact(fpath)
                except Exception:
                    pass
