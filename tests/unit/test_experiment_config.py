from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from trading.commands.common import adapt_live_config_to_mongo_backtest, apply_cli_overrides
from trading.config import component_loader
from trading.config import ExperimentService
from trading.experiments import ExperimentRequest, build_runtime, describe_experiment, load_experiment


def _legacy_backtest_config() -> dict:
    return {
        "mode": "backtest",
        "algorithm": {
            "algorithm": "trading.algorithms.spy_trend_macd_algorithm.SpyTrendMACDAlgorithm",
            "spy_symbol": "SPY",
            "upro_symbol": "UPRO",
            "spxu_symbol": "SPXU",
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "strength_scale": 20.0,
            "history_length": 35,
        },
        "portfolio": {
            "portfolio": "trading.core.pf.dual_symbol_switch_portfolio.DualSymbolSwitchPortfolio",
            "upro_symbol": "UPRO",
            "spxu_symbol": "SPXU",
            "cash": 10000,
            "keep_history": True,
            "min_signal_strength": 0,
            "stop_pct": 10.0,
            "profit_pct": 10.0,
            "holding_period_hours": 0,
            "tx_cost": 0.0,
        },
        "order_manager": {
            "order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager",
        },
        "data_provider": {
            "provider": "trading.data_providers.test_data_provider.TestDataProvider",
            "path": "../data/SPY_SPXU_UPRO_1min.csv",
        },
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "aggregation": {"enabled": True, "aggregation_period_minutes": 5},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    }


def test_normalize_legacy_config_and_describe():
    config = ExperimentService.from_dict(_legacy_backtest_config())

    assert config.algorithm.implementation.endswith("SpyTrendMACDAlgorithm")
    assert config.algorithm.params["history_length"] == 35
    assert config.portfolio.params["cash"] == 10000
    assert config.execution_mode() == "simulation"

    descriptor = ExperimentService.describe(config)
    assert descriptor.mode == "backtest"
    assert descriptor.execution_mode == "simulation"
    assert descriptor.algorithm == config.algorithm.implementation
    assert descriptor.data_provider == config.data_provider.implementation
    assert len(descriptor.config_hash) == 16


def test_normalize_new_style_config():
    config = ExperimentService.from_dict({
        "mode": "live",
        "algorithm": {
            "implementation": "trading.algorithms.spy_trend_macd_algorithm.SpyTrendMACDAlgorithm",
            "params": {"macd_fast_period": 8, "macd_slow_period": 21, "macd_signal_period": 5},
        },
        "portfolio": {
            "implementation": "trading.core.pf.dual_symbol_switch_portfolio.DualSymbolSwitchPortfolio",
            "params": {"upro_symbol": "UPRO", "spxu_symbol": "SPXU", "cash": 5000},
        },
        "order_manager": {
            "implementation": "trading.core.om.backtesting_om.BacktestingOrderManager",
            "params": {"market_hours_only": True},
        },
        "alpaca": {"symbols_to_subscribe": ["SPY", "UPRO", "SPXU"]},
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    })

    assert config.mode == "live"
    assert config.execution_mode() == "broker"
    assert config.algorithm.params["macd_fast_period"] == 8
    assert config.order_manager.params["market_hours_only"] is True


def test_walk_forward_window_hpo_section_is_allowed():
    config = ExperimentService.from_dict({
        "mode": "walk-forward",
        "algorithm": {
            "implementation": "trading.algorithms.spy_trend_macd_algorithm.SpyTrendMACDAlgorithm",
            "params": {"macd_fast_period": 8, "macd_slow_period": 21, "macd_signal_period": 5},
        },
        "portfolio": {
            "implementation": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {"symbol": "SPY", "cash": 5000},
        },
        "order_manager": {
            "implementation": "trading.core.om.backtesting_om.BacktestingOrderManager",
            "params": {"market_hours_only": True},
        },
        "data_provider": {
            "implementation": "trading.data_providers.test_data_provider.TestDataProvider",
            "params": {"path": "../data/SPY_5min.csv"},
        },
        "walk_forward": {
            "optimization_window_days": 180,
            "validation_window_days": 30,
            "trading_window_days": 30,
        },
        "walk_forward_window_hpo": {
            "num_samples": 8,
            "objective_metric": "wf_annualized_return",
            "search_space": {
                "optimization_window_days": {"type": "choice", "values": [90, 180]},
                "validation_window_days": {"type": "choice", "values": [20, 30]},
                "trading_window_days": {"type": "choice", "values": [10, 30]},
            },
        },
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    })

    assert config.walk_forward_window_hpo.objective_metric == "wf_annualized_return"


def test_debug_on_sigint_top_level_flag_is_allowed():
    config = ExperimentService.from_dict({
        "mode": "hpo",
        "debug_on_sigint": False,
        "algorithm": {
            "implementation": "tests.fixtures.custom_components.CustomAlgorithm",
            "params": {"lookback": 25, "threshold": 1.5, "history_length": 40},
        },
        "portfolio": {
            "implementation": "tests.fixtures.custom_components.CustomPortfolio",
            "params": {},
        },
        "order_manager": {
            "implementation": "tests.fixtures.custom_components.CustomOrderManager",
            "params": {},
        },
        "data_provider": {
            "implementation": "tests.fixtures.custom_components.CustomDataProvider",
            "params": {},
        },
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    })

    assert config.debug_on_sigint is False


def test_known_component_validation_rejects_bad_param_type():
    cfg = _legacy_backtest_config()
    cfg["algorithm"]["macd_fast_period"] = "not-an-int"

    with pytest.raises(Exception):
        ExperimentService.from_dict(cfg)


def test_external_component_config_model_is_honored():
    raw = {
        "mode": "backtest",
        "algorithm": {
            "implementation": "tests.fixtures.custom_components.CustomAlgorithm",
            "params": {"lookback": 25, "threshold": 1.5, "history_length": 40},
        },
        "portfolio": {
            "implementation": "tests.fixtures.custom_components.CustomPortfolio",
            "params": {},
        },
        "order_manager": {
            "implementation": "tests.fixtures.custom_components.CustomOrderManager",
            "params": {},
        },
        "data_provider": {
            "implementation": "tests.fixtures.custom_components.CustomDataProvider",
            "params": {},
        },
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    }

    config = ExperimentService.from_dict(raw)
    built = build_runtime(config)

    assert built.algorithm.cfg["lookback"] == 25
    assert built.algorithm.history_length == 40
    assert built.portfolio.order_manager is built.order_manager

    raw["algorithm"]["params"]["lookback"] = "bad"
    with pytest.raises(Exception):
        ExperimentService.from_dict(raw)


def test_apply_cli_overrides_supports_new_style_sections():
    raw = {
        "portfolio": {
            "implementation": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {"symbol": "SPY", "cash": 1000},
        },
        "algorithm": {
            "implementation": "trading.algorithms.spy_trend_macd_algorithm.SpyTrendMACDAlgorithm",
            "params": {},
        },
        "aggregation": {},
        "analysis": {},
        "state_store": {},
    }

    class Args:
        symbol = "QQQ"
        cash = 5000.0
        algorithm = "tests.fixtures.custom_components.CustomAlgorithm"
        portfolio = "tests.fixtures.custom_components.CustomPortfolio"
        algorithm_url = "https://example.com/remote_algorithm.py"
        portfolio_url = "https://example.com/remote_portfolio.py"
        data = None
        no_mlflow = True
        run_name = "override-run"
        alpaca_override_url = None
        session_id = "sess-123"
        agg_period = 15

    updated = apply_cli_overrides(raw, Args())
    assert updated["portfolio"]["params"]["symbol"] == "QQQ"
    assert updated["portfolio"]["params"]["cash"] == 5000.0
    assert updated["portfolio"]["implementation"] == "tests.fixtures.custom_components.CustomPortfolio"
    assert updated["portfolio"]["source_url"] == "https://example.com/remote_portfolio.py"
    assert updated["algorithm"]["implementation"] == "tests.fixtures.custom_components.CustomAlgorithm"
    assert updated["algorithm"]["source_url"] == "https://example.com/remote_algorithm.py"
    assert updated["analysis"]["log_to_mlflow"] is False
    assert updated["analysis"]["run_name"] == "override-run"
    assert updated["state_store"]["session_id"] == "sess-123"
    assert updated["aggregation"]["aggregation_period_minutes"] == 15
    assert updated["aggregation"]["enabled"] is True


def test_adapt_live_config_to_mongo_backtest_builds_runtime_sections():
    raw = {
        "mode": "live",
        "algorithm": {
            "algorithm": "tests.fixtures.custom_components.CustomAlgorithm",
            "source_path": "trading/promoted/demo/algorithm_demo.py",
        },
        "portfolio": {
            "portfolio": "tests.fixtures.custom_components.CustomPortfolio",
            "source_path": "trading/promoted/demo/portfolio_demo.py",
            "cash": 2500,
        },
        "order_manager": {
            "order_manager": "trading.core.om.alpaca_om.AlpacaOrderManager",
            "paper": True,
        },
        "state_store": {
            "enabled": True,
            "session_id": "live-demo-1",
            "connection_uri": "mongodb://example:27017",
            "database": "live_trading",
        },
    }

    adapted = adapt_live_config_to_mongo_backtest(raw)

    assert adapted["mode"] == "backtest"
    assert adapted["algorithm"]["source_path"] == "trading/promoted/demo/algorithm_demo.py"
    assert adapted["portfolio"]["source_path"] == "trading/promoted/demo/portfolio_demo.py"
    assert adapted["order_manager"]["order_manager"] == "trading.core.om.backtesting_om.BacktestingOrderManager"
    assert adapted["order_manager"]["market_hours_only"] is True
    assert adapted["data_provider"]["provider"] == "trading.data_providers.mongodb_data_provider.MongoDBDataProvider"
    assert adapted["data_provider"]["session_id"] == "live-demo-1"
    assert adapted["data_provider"]["connection_uri"] == "mongodb://example:27017"
    assert adapted["data_provider"]["database"] == "live_trading"


def test_adapt_live_config_to_mongo_backtest_leaves_regular_backtest_unchanged():
    raw = _legacy_backtest_config()
    adapted = adapt_live_config_to_mongo_backtest(raw)

    assert adapted == raw


def test_adapt_live_config_to_mongo_backtest_force_rewrites_existing_runtime_sections():
    raw = _legacy_backtest_config()
    raw["state_store"] = {
        "enabled": True,
        "session_id": "mongo-force-1",
        "connection_uri": "mongodb://example:27017",
        "database": "live_trading",
    }

    adapted = adapt_live_config_to_mongo_backtest(raw, force=True)

    assert adapted["mode"] == "backtest"
    assert adapted["order_manager"]["order_manager"] == "trading.core.om.backtesting_om.BacktestingOrderManager"
    assert adapted["data_provider"]["provider"] == "trading.data_providers.mongodb_data_provider.MongoDBDataProvider"
    assert adapted["data_provider"]["session_id"] == "mongo-force-1"


def test_remote_component_config_and_runtime(monkeypatch):
    remote_sources = {
        "https://example.com/remote_algorithm.py": """
from pydantic import BaseModel

class RemoteAlgorithmConfig(BaseModel):
    lookback: int

class RemoteAlgorithm:
    def __init__(self, cfg=None, history_length: int = 0):
        self.cfg = cfg or {}
        self.history_length = history_length

    @classmethod
    def config_model(cls):
        return RemoteAlgorithmConfig
""",
        "https://example.com/remote_portfolio.py": """
from pydantic import BaseModel

class RemotePortfolioConfig(BaseModel):
    cash: float

class RemotePortfolio:
    def __init__(self, cfg=None, order_manager=None):
        self.cfg = cfg or {}
        self.order_manager = order_manager

    @classmethod
    def config_model(cls):
        return RemotePortfolioConfig
""",
    }

    component_loader._download_remote_source.cache_clear()
    component_loader._load_remote_module.cache_clear()
    monkeypatch.setattr(
        component_loader,
        "_download_remote_source",
        lambda url: remote_sources[url],
    )

    raw = {
        "mode": "backtest",
        "algorithm": {
            "algorithm": "RemoteAlgorithm",
            "source_url": "https://example.com/remote_algorithm.py",
            "lookback": 25,
            "history_length": 40,
        },
        "portfolio": {
            "portfolio": "RemotePortfolio",
            "source_url": "https://example.com/remote_portfolio.py",
            "cash": 15000,
        },
        "order_manager": {
            "implementation": "tests.fixtures.custom_components.CustomOrderManager",
            "params": {},
        },
        "data_provider": {
            "implementation": "tests.fixtures.custom_components.CustomDataProvider",
            "params": {},
        },
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    }

    config = ExperimentService.from_dict(raw)
    built = build_runtime(config)

    assert config.algorithm.source_url == "https://example.com/remote_algorithm.py"
    assert config.portfolio.source_url == "https://example.com/remote_portfolio.py"
    assert built.algorithm.cfg["lookback"] == 25
    assert built.algorithm.history_length == 40
    assert built.portfolio.cfg["cash"] == 15000
    assert built.portfolio.order_manager is built.order_manager


def test_new_style_remote_component_preserves_params(monkeypatch):
    remote_source = """
from pydantic import BaseModel

class UrlAlgoConfig(BaseModel):
    threshold: float

class UrlAlgo:
    def __init__(self, cfg=None, history_length: int = 0):
        self.cfg = cfg or {}
        self.history_length = history_length

    @classmethod
    def config_model(cls):
        return UrlAlgoConfig
"""

    component_loader._download_remote_source.cache_clear()
    component_loader._load_remote_module.cache_clear()
    monkeypatch.setattr(component_loader, "_download_remote_source", lambda url: remote_source)

    config = ExperimentService.from_dict({
        "mode": "backtest",
        "algorithm": {
            "implementation": "UrlAlgo",
            "source_url": "https://example.com/url_algo.py",
            "params": {"threshold": 1.25, "history_length": 22},
        },
        "portfolio": {
            "implementation": "tests.fixtures.custom_components.CustomPortfolio",
            "params": {},
        },
        "order_manager": {
            "implementation": "tests.fixtures.custom_components.CustomOrderManager",
            "params": {},
        },
        "data_provider": {
            "implementation": "tests.fixtures.custom_components.CustomDataProvider",
            "params": {},
        },
        "analysis": {"enabled": False, "log_to_mlflow": False},
        "state_store": {"enabled": False},
        "mlflow": {"enabled": False},
        "logging": {},
    })

    assert config.algorithm.params["threshold"] == 1.25
    assert config.algorithm.params["history_length"] == 22


def test_local_source_path_component_runtime():
    source_path = Path("scratch") / f"local_algo_{uuid.uuid4().hex}.py"
    source_path.write_text(
        """
from pydantic import BaseModel

class LocalAlgoConfig(BaseModel):
    threshold: float

class LocalAlgo:
    def __init__(self, cfg=None, history_length: int = 0):
        self.cfg = cfg or {}
        self.history_length = history_length

    @classmethod
    def config_model(cls):
        return LocalAlgoConfig
""",
        encoding="utf-8",
    )
    try:
        config = ExperimentService.from_dict({
            "mode": "backtest",
            "algorithm": {
                "implementation": "LocalAlgo",
                "source_path": str(source_path),
                "params": {"threshold": 2.5, "history_length": 18},
            },
            "portfolio": {
                "implementation": "tests.fixtures.custom_components.CustomPortfolio",
                "params": {},
            },
            "order_manager": {
                "implementation": "tests.fixtures.custom_components.CustomOrderManager",
                "params": {},
            },
            "data_provider": {
                "implementation": "tests.fixtures.custom_components.CustomDataProvider",
                "params": {},
            },
            "analysis": {"enabled": False, "log_to_mlflow": False},
            "state_store": {"enabled": False},
            "mlflow": {"enabled": False},
            "logging": {},
        })
        built = build_runtime(config)

        assert config.algorithm.source_path == str(source_path)
        assert built.algorithm.cfg["threshold"] == 2.5
        assert built.algorithm.history_length == 18
    finally:
        if source_path.exists():
            source_path.unlink()


def test_local_source_path_component_runtime_with_windows_style_relative_path():
    source_path = Path("scratch") / f"local_algo_{uuid.uuid4().hex}.py"
    source_path.write_text(
        """
from pydantic import BaseModel

class LocalAlgoConfig(BaseModel):
    threshold: float

class LocalAlgo:
    def __init__(self, cfg=None, history_length: int = 0):
        self.cfg = cfg or {}
        self.history_length = history_length

    @classmethod
    def config_model(cls):
        return LocalAlgoConfig
""",
        encoding="utf-8",
    )
    try:
        windows_style_path = str(source_path).replace("/", "\\")
        config = ExperimentService.from_dict({
            "mode": "backtest",
            "algorithm": {
                "implementation": "LocalAlgo",
                "source_path": windows_style_path,
                "params": {"threshold": 3.5, "history_length": 12},
            },
            "portfolio": {
                "implementation": "tests.fixtures.custom_components.CustomPortfolio",
                "params": {},
            },
            "order_manager": {
                "implementation": "tests.fixtures.custom_components.CustomOrderManager",
                "params": {},
            },
            "data_provider": {
                "implementation": "tests.fixtures.custom_components.CustomDataProvider",
                "params": {},
            },
            "analysis": {"enabled": False, "log_to_mlflow": False},
            "state_store": {"enabled": False},
            "mlflow": {"enabled": False},
            "logging": {},
        })
        built = build_runtime(config)

        assert built.algorithm.cfg["threshold"] == 3.5
        assert built.algorithm.history_length == 12
    finally:
        if source_path.exists():
            source_path.unlink()


def test_external_request_interface_from_file():
    config_path = Path("scratch") / f"experiment_{uuid.uuid4().hex}.yaml"
    config_path.write_text(
        """
mode: backtest
algorithm:
  implementation: tests.fixtures.custom_components.CustomAlgorithm
  params:
    lookback: 10
portfolio:
  implementation: tests.fixtures.custom_components.CustomPortfolio
  params: {}
order_manager:
  implementation: tests.fixtures.custom_components.CustomOrderManager
  params: {}
data_provider:
  implementation: tests.fixtures.custom_components.CustomDataProvider
  params: {}
analysis:
  enabled: false
  log_to_mlflow: false
state_store:
  enabled: false
mlflow:
  enabled: false
logging: {}
""",
        encoding="utf-8",
    )
    try:
        request = ExperimentRequest(config_path=str(config_path))
        config = load_experiment(request)
        descriptor = describe_experiment(config)

        assert config.algorithm.implementation == "tests.fixtures.custom_components.CustomAlgorithm"
        assert descriptor.execution_mode == "simulation"
    finally:
        if config_path.exists():
            config_path.unlink()
