from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading.reporting import (
    AnalysisSummary,
    AnalyzerReportTarget,
    CombinedArtifactSpec,
    ExperimentReport,
    ExperimentReporter,
)


@dataclass
class FakeMetrics:
    total_return: float = 100.0
    total_return_pct: float = 10.0
    annualized_return: float = 12.0
    sharpe_ratio: float = 1.5
    max_drawdown_pct: float = 3.0
    total_trades: int = 4
    win_rate: float = 50.0
    profit_factor: float = 1.2
    bracket_trades: int = 0
    bracket_stop_rate: float = 0.0
    bracket_profit_rate: float = 0.0


class FakeAnalyzer:
    def __init__(self):
        self._session_metadata = {"session_id": "abc123"}

    def save_equity_curve(self, path): Path(path).write_text("eq", encoding="utf-8")
    def save_technical_chart(self, path): Path(path).write_text("tech", encoding="utf-8")
    def save_drawdown_chart(self, path): Path(path).write_text("dd", encoding="utf-8")
    def save_orders_chart(self, path): Path(path).write_text("orders", encoding="utf-8")
    def save_lifecycle_chart(self, path): Path(path).write_text("lc", encoding="utf-8")
    def save_lifecycle_chart_interactive(self, path): Path(path).write_text("<html></html>", encoding="utf-8")
    def save_signals_orders_csv(self, path): Path(path).write_text("signals", encoding="utf-8")
    def save_trades_csv(self, path): Path(path).write_text("trades", encoding="utf-8")
    def plot_portfolio_with_trades(self, show=False, save_path=None): Path(save_path).write_text("pwt", encoding="utf-8")
    def plot_trade_pnl(self, show=False, save_path=None): Path(save_path).write_text("tpnl", encoding="utf-8")
    def plot_returns_distribution(self, show=False, save_path=None): Path(save_path).write_text("rd", encoding="utf-8")
    def plot_stock_performance(self, show=False, save_path=None): Path(save_path).write_text("sp", encoding="utf-8")
    def plot_comprehensive_dashboard(self, show=False, save_path=None): Path(save_path).write_text("db", encoding="utf-8")
    def plot_orders_timeline(self, show=False, save_path=None): Path(save_path).write_text("ot", encoding="utf-8")
    def plot_interactive_portfolio(self, show=False, save_path=None): Path(save_path).write_text("<html>portfolio</html>", encoding="utf-8")
    def generate_all_orders_report(self): return "all orders"


class FakeClient:
    enabled = True

    def __init__(self):
        self.run_id = "run-123"
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.texts = []

    def start_run(self, run_name=None, description=None, tags=None):
        self.run_name = run_name
        self.description = description
        self.tags = tags
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log_params(self, params):
        self.params.append(params)

    def log_metrics(self, metrics):
        self.metrics.append(metrics)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))

    def log_text(self, text, path):
        self.texts.append((path, text))

    def get_run_url(self):
        return "http://mlflow/runs/run-123"


def _summary():
    return AnalysisSummary(
        trades=[],
        metrics=FakeMetrics(),
        lifecycle_chains=[],
        tick_returns=[],
        daily_returns=[],
        monthly_returns=[],
        bracket_analysis=None,
        report="hello report",
        benchmarks={},
    )


def _summary_with_trade_count(total_trades: int):
    summary = _summary()
    summary.metrics.total_trades = total_trades
    return summary


def test_reporter_logs_single_target(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr("trading.reporting.service.MLflowClient.from_config", lambda experiment_name=None: client)
    tmp_root = tmp_path / "reporter_tmp"
    tmp_root.mkdir()
    monkeypatch.setattr("trading.reporting.service._workspace_temp_dir", lambda: str(tmp_root))

    analyzer = FakeAnalyzer()
    report = ExperimentReport(
        experiment_name="exp",
        run_name="run-1",
        parameters={"config.mode": "backtest"},
        analyzers=[AnalyzerReportTarget(name="primary", analyzer=analyzer, summary=_summary())],
    )

    ExperimentReporter.log_to_mlflow(report)

    assert client.run_name == "run-1"
    assert any("total_return" in entry for entry in client.metrics)
    assert any(path == "performance_report.txt" for path, _ in client.texts)
    assert any(path == "all_orders_report.txt" for path, _ in client.texts)
    assert any(name == "trades.csv" for name, _ in client.artifacts)
    assert any(name == "interactive_portfolio.html" for name, _ in client.artifacts)
    assert any(name == "dashboard.png" for name, _ in client.artifacts)
    assert any("config.session.session_id" in entry for entry in client.params if isinstance(entry, dict))


def test_reporter_logs_prefixed_and_combined_artifacts(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("trading.reporting.service.MLflowClient.from_config", lambda experiment_name=None: client)

    analyzer = FakeAnalyzer()
    report = ExperimentReport(
        experiment_name="exp",
        run_name="run-2",
        analyzers=[
            AnalyzerReportTarget(
                name="mongo",
                analyzer=analyzer,
                summary=_summary(),
                metric_prefix="mongo_",
                artifact_prefix="mongo_",
            )
        ],
        combined_artifacts=[
            CombinedArtifactSpec(
                filename="combined_equity_curve.png",
                builder=lambda path: Path(path).write_text("combined", encoding="utf-8"),
            )
        ],
    )

    ExperimentReporter.log_to_mlflow(report)

    assert any("mongo_total_return" in entry for entry in client.metrics)
    assert any(name == "mongo_trades.csv" for name, _ in client.artifacts)
    assert any(name == "combined_equity_curve.png" for name, _ in client.artifacts)


def test_reporter_logs_config_artifacts(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr("trading.reporting.service.MLflowClient.from_config", lambda experiment_name=None: client)

    cfg_path = tmp_path / "reporter_test_config.yaml"
    cfg_path.write_text("mode: backtest\n", encoding="utf-8")
    try:
        report = ExperimentReport(
            experiment_name="exp",
            run_name="run-3",
            analyzers=[],
            config_artifact_paths=[str(cfg_path)],
        )
        ExperimentReporter.log_to_mlflow(report)
        assert ("reporter_test_config.yaml", "config") in client.artifacts
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


def test_reporter_skips_heavy_artifacts_for_large_trade_counts(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr("trading.reporting.service.MLflowClient.from_config", lambda experiment_name=None: client)
    tmp_root = tmp_path / "reporter_tmp_heavy"
    tmp_root.mkdir()
    monkeypatch.setattr("trading.reporting.service._workspace_temp_dir", lambda: str(tmp_root))

    analyzer = FakeAnalyzer()
    report = ExperimentReport(
        experiment_name="exp",
        run_name="run-heavy",
        analyzers=[
            AnalyzerReportTarget(
                name="primary",
                analyzer=analyzer,
                summary=_summary_with_trade_count(501),
            )
        ],
    )

    ExperimentReporter.log_to_mlflow(report)

    assert any(path == "performance_report.txt" for path, _ in client.texts)
    assert any(name == "trades.csv" for name, _ in client.artifacts)
    assert not any(path == "all_orders_report.txt" for path, _ in client.texts)
    assert not any(name == "signals_orders.csv" for name, _ in client.artifacts)
    assert not any(name == "orders_chart.png" for name, _ in client.artifacts)
    assert not any(name == "technical_analysis.png" for name, _ in client.artifacts)
    assert not any(name == "interactive_portfolio.html" for name, _ in client.artifacts)
