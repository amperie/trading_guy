from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from trading.analysis.analysis_engine import AnalysisEngine


class _FakeAnalyzer:
    def __init__(self, portfolio, order_manager):
        self.portfolio = portfolio
        self.order_manager = order_manager

    def save_equity_curve(self, path): Path(path).write_text("eq", encoding="utf-8")
    def save_drawdown_chart(self, path): Path(path).write_text("dd", encoding="utf-8")
    def save_technical_chart(self, path): Path(path).write_text("tech", encoding="utf-8")
    def save_orders_chart(self, path): Path(path).write_text("orders", encoding="utf-8")
    def save_lifecycle_chart(self, path): Path(path).write_text("life", encoding="utf-8")
    def save_lifecycle_chart_interactive(self, path): Path(path).write_text("<html></html>", encoding="utf-8")
    def save_performance_report(self, path): Path(path).write_text("report", encoding="utf-8")
    def save_signals_orders_csv(self, path): Path(path).write_text("signals", encoding="utf-8")
    def save_trades_csv(self, path): Path(path).write_text("trades", encoding="utf-8")


class _FakeClient:
    enabled = True

    def __init__(self):
        self.artifacts = []
        self.texts = []
        self.metrics = []
        self.params = []

    def start_run(self, run_name=None, description=None, tags=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log_params(self, params):
        self.params.append(params)

    def log_metrics(self, metrics):
        self.metrics.append(metrics)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))

    def log_text(self, text, filename):
        self.texts.append(filename)

    def log_json(self, data, filename):
        self.artifacts.append((filename, None))

    def log_markdown(self, markdown, filename):
        self.artifacts.append((filename, None))

    def get_run_url(self):
        return ""


def test_analysis_engine_mlflow_logs_legacy_portfolio_analyzer_artifacts(monkeypatch):
    engine = AnalysisEngine(SimpleNamespace(), SimpleNamespace())
    engine._metrics = SimpleNamespace(
        total_return=100.0,
        total_return_pct=10.0,
        annualized_return=12.0,
        sharpe_ratio=1.5,
        sortino_ratio=1.4,
        max_drawdown=-50.0,
        max_drawdown_pct=-3.5,
        max_drawdown_duration=2.0,
        calmar_ratio=1.1,
        ulcer_index=0.4,
        volatility=9.0,
        total_trades=4,
        winning_trades=2,
        losing_trades=2,
        win_rate=50.0,
        avg_win=10.0,
        avg_loss=-8.0,
        largest_win=20.0,
        largest_loss=-12.0,
        profit_factor=1.2,
        avg_trade_pnl=1.0,
        avg_trade_duration=2.0,
        avg_bars_in_trade=3.0,
        bracket_trades=0,
        bracket_stop_triggers=0,
        bracket_profit_triggers=0,
        bracket_manual_exits=0,
        bracket_stop_rate=0.0,
        bracket_profit_rate=0.0,
        best_day=2.0,
        worst_day=-1.0,
        avg_daily_return=0.2,
        skewness=0.1,
        kurtosis=0.2,
        final_equity=1100.0,
        initial_equity=1000.0,
        peak_equity=1120.0,
        total_days=10.0,
        trading_days=10,
    )
    engine._trades = []
    engine.calculate_external_benchmarks = lambda: {}
    engine.generate_report = lambda: "report"
    engine.generate_all_orders_report = lambda: "all orders"
    engine._log_analysis_engine_artifacts = lambda mlflow, chart_dpi=150: None

    fake_client = _FakeClient()
    cfg_dir = Path("scratch") / "test_analysis_engine_mlflow_parity"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "edited.yaml"
    cfg_path.write_text("mode: hpo\n", encoding="utf-8")

    monkeypatch.setattr("utils.mlflow_client.MLflowClient.from_config", lambda experiment_name=None: fake_client)
    monkeypatch.setattr("trading.analysis.portfolio_analyzer.PortfolioAnalyzer", _FakeAnalyzer)

    engine.log_to_mlflow(
        experiment_name="test-exp",
        log_signals=False,
        artifact_paths=[str(cfg_path)],
    )

    artifact_names = {name for name, _ in fake_client.artifacts}
    assert "equity_curve.png" in artifact_names
    assert "drawdown.png" in artifact_names
    assert "technical_analysis.png" in artifact_names
    assert "orders_chart.png" in artifact_names
    assert "lifecycle.png" in artifact_names
    assert "lifecycle_interactive.html" in artifact_names
    assert "performance_report.txt" in artifact_names
    assert "signals_orders.csv" in artifact_names
    assert "trades.csv" in artifact_names
    assert ("edited.yaml", "config") in fake_client.artifacts
    assert "analysis_engine_performance_report.txt" in fake_client.texts
    assert "all_orders_report.txt" in fake_client.texts
    cfg_path.unlink()
    cfg_dir.rmdir()
