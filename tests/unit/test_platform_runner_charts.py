from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from trading.platform.runner import _equity_chart_points, _write_chart_artifacts


def test_write_chart_artifacts_from_portfolio_value_history(monkeypatch):
    written: dict[str, str] = {}

    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, text, **kwargs: written.__setitem__(self.as_posix(), text),
    )

    start = datetime(2026, 1, 1)
    portfolio = SimpleNamespace(
        value_history={
            start: 100000.0,
            start + timedelta(days=1): 101000.0,
            start + timedelta(days=2): 99000.0,
        }
    )

    manifest = _write_chart_artifacts(Path("output"), "smoke", portfolio)

    assert manifest["charts"][0]["artifact"] == "charts/equity_curve.json"
    assert "output/chart_manifest.json" in written
    assert "output/charts/equity_curve.json" in written


def test_equity_chart_points_include_drawdown():
    start = datetime(2026, 1, 1)
    portfolio = SimpleNamespace(
        value_history={
            start: 100.0,
            start + timedelta(days=1): 120.0,
            start + timedelta(days=2): 90.0,
        }
    )

    points = _equity_chart_points(portfolio)

    assert points[-1]["strategy"] == 90.0
    assert points[-1]["benchmark"] == 100.0
    assert points[-1]["drawdown"] == -25.0
