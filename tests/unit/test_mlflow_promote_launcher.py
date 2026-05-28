from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from trading.launchers import mlflow_promote_launcher as launcher
from trading.launchers.mlflow_hpo_launcher import SourceRunContext


def test_minimal_state_store_section_uses_configured_live_database(monkeypatch):
    class FakeConfigManager:
        def get(self, key, default=None):
            return {"state_store.default_live_database": "configured_live"}.get(key, default)

    monkeypatch.setattr(launcher, "ConfigManager", FakeConfigManager)
    section = launcher._minimal_state_store_section(
        {"state_store": {"connection_uri": "mongodb://localhost:27017", "database": "trading_test"}}
    )

    assert section == {
        "enabled": True,
        "session_id": "",
        "database": "configured_live",
        "connection_uri": "mongodb://localhost:27017",
    }


def test_promote_run_writes_live_bundle(monkeypatch):
    tmp_dir = Path.cwd() / ".tmp" / "test_promote_run_writes_live_bundle"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    algo_source = tmp_dir / "AlgoImpl.py"
    algo_source.write_text(
        "class DemoAlgo:\n"
        "    def __init__(self, cfg=None, history_length=0):\n"
        "        self.cfg = cfg or {}\n",
        encoding="utf-8",
    )
    portfolio_source = tmp_dir / "PortfolioImpl.py"
    portfolio_source.write_text(
        "class DemoPortfolio:\n"
        "    def __init__(self, cfg=None, order_manager=None):\n"
        "        self.cfg = cfg or {}\n",
        encoding="utf-8",
    )

    context = SourceRunContext(
        run_id="abc123def456",
        run_name="Source Run",
        tracking_uri="http://localhost:5000",
        source_url="http://localhost:5000/#/experiments/1/runs/abc123def456",
        raw_config={
            "mode": "backtest",
            "algorithm": {
                "implementation": "pkg.algos.DemoAlgo",
                "class_name": "DemoAlgo",
                "source_path": str(algo_source),
                "params": {"lookback": 10, "history_length": 20},
            },
            "portfolio": {
                "implementation": "pkg.portfolios.DemoPortfolio",
                "class_name": "DemoPortfolio",
                "source_path": str(portfolio_source),
                "params": {"symbol": "SPY", "cash": 50000},
            },
            "order_manager": {
                "implementation": "trading.core.om.backtesting_om.BacktestingOrderManager",
                "params": {"paper": False},
            },
            "analysis": {"enabled": True, "log_to_mlflow": True},
            "state_store": {"enabled": False, "connection_uri": "mongodb://localhost:27017", "database": "trading_test"},
            "alpaca": {},
            "aggregation": {"enabled": True, "aggregation_period_minutes": 5},
            "optimization": {"enabled": True},
            "mlflow": {"tracking_uri": "http://localhost:5000"},
            "logging": {},
        },
        config_source="artifact:config/runtime_config.yaml",
    )

    monkeypatch.setattr(launcher, "load_source_run_context", lambda run_url, tracking_uri=None: context)

    promoted_dir = Path.cwd() / "trading" / "promoted" / "demo_promotion"
    config_path = promoted_dir / "demo_promotion.yaml"

    try:
        bundle = launcher.promote_run(context.source_url, name="demo_promotion")

        cfg_path = Path(bundle.config_path)
        manifest_path = Path(bundle.manifest_path)
        promoted_dir = Path(bundle.promoted_dir)

        assert cfg_path.exists()
        assert manifest_path.exists()
        assert promoted_dir.exists()

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["mode"] == "live"
        assert cfg["algorithm"]["algorithm"] == "pkg.algos.DemoAlgo"
        assert cfg["algorithm"]["lookback"] == 10
        assert cfg["algorithm"]["history_length"] == 20
        assert cfg["analysis"]["enabled"] is True
        assert cfg["analysis"]["log_to_mlflow"] is True
        assert cfg["state_store"]["enabled"] is True
        assert cfg["state_store"]["session_id"] == ""
        assert cfg["state_store"]["database"] == "live_trading"
        assert cfg["state_store"]["connection_uri"] == "mongodb://localhost:27017"
        assert cfg["order_manager"]["order_manager"] == "trading.core.om.alpaca_om.AlpacaOrderManager"
        assert cfg["order_manager"]["paper"] is False
        assert cfg["alpaca"]["symbols_to_subscribe"] == ["SPY"]
        assert "override_url" in cfg["alpaca"]
        assert cfg["alpaca"]["override_url"] == "ws://hp.lan:8765"
        assert cfg["alpaca"]["warmup"]["symbols"] == ["SPY"]
        assert cfg["aggregation"]["enabled"] is True
        assert cfg["aggregation"]["aggregation_period_minutes"] == 5
        assert cfg["algorithm"]["source_path"].startswith("trading/promoted/demo_promotion/")
        assert cfg["portfolio"]["source_path"].startswith("trading/promoted/demo_promotion/")
        assert (Path.cwd() / cfg["algorithm"]["source_path"]).exists()
        assert (Path.cwd() / cfg["portfolio"]["source_path"]).exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_run_id"] == "abc123def456"
        assert manifest["config_path"] == "trading/promoted/demo_promotion/demo_promotion.yaml"
        assert "run.py live" in manifest["launch_example"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(promoted_dir, ignore_errors=True)
        if config_path.exists():
            config_path.unlink()
