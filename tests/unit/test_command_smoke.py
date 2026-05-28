from __future__ import annotations

from types import SimpleNamespace

from trading.commands import backtest as backtest_cmd
from trading.commands import live as live_cmd
from trading.commands import walk_forward as walk_forward_cmd


class StubEngine:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.ran = False

    def run(self):
        self.ran = True


def test_cmd_backtest_smoke(monkeypatch):
    raw_cfg = {
        "mode": "backtest",
        "analysis": {"enabled": False},
        "aggregation": {"enabled": False},
        "state_store": {"enabled": False},
        "data_provider": {"provider": "dummy.Provider"},
    }
    built = SimpleNamespace(
        data_provider=object(),
        algorithm=object(),
        order_manager=object(),
        portfolio=SimpleNamespace(total_value=1000.0, cash=500.0, positions={}),
    )
    captured = {}

    monkeypatch.setattr(backtest_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(backtest_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(backtest_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(backtest_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(backtest_cmd, "adapt_live_config_to_mongo_backtest", lambda cfg, force=False: cfg)
    monkeypatch.setattr(backtest_cmd, "load_account_creds", lambda account: {"api_key": "x", "secret_key": "y"})
    monkeypatch.setattr(backtest_cmd, "build_experiment_config", lambda cfg: "normalized-config")
    monkeypatch.setattr(backtest_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(
        backtest_cmd.ExperimentService,
        "describe",
        lambda cfg: SimpleNamespace(config_hash="hash1234"),
    )

    def fake_engine(*args, **kwargs):
        engine = StubEngine(*args, **kwargs)
        captured["engine"] = engine
        return engine

    monkeypatch.setattr(backtest_cmd, "BacktestingEngine", fake_engine)
    monkeypatch.setattr(backtest_cmd, "run_analysis", lambda cfg, pf, om, config_path=None: None)

    args = SimpleNamespace(config="cfg.yaml", account="paper")
    backtest_cmd.cmd_backtest(args)

    assert captured["engine"].ran is True


def test_cmd_mongo_backtest_sets_flag_and_reuses_backtest(monkeypatch):
    captured = {}

    def fake_cmd_backtest(args):
        captured["mongo_backtest"] = args.mongo_backtest
        captured["session_id"] = args.session_id

    monkeypatch.setattr(backtest_cmd, "cmd_backtest", fake_cmd_backtest)

    args = SimpleNamespace(
        config="cfg.yaml",
        account="paper",
        session_id="sess-1",
        mongo_backtest=False,
    )
    backtest_cmd.cmd_mongo_backtest(args)

    assert captured["mongo_backtest"] is True
    assert captured["session_id"] == "sess-1"


def test_cmd_live_smoke(monkeypatch):
    raw_cfg = {
        "mode": "live",
        "alpaca": {
            "api_key": "key",
            "secret_key": "secret",
            "symbols_to_subscribe": ["SPY"],
        },
        "analysis": {"enabled": False},
        "aggregation": {"enabled": False},
        "optimization": {"enabled": False},
        "state_store": {"enabled": True, "session_id": "sess-1"},
        "order_manager": {"order_manager": "dummy.OM"},
    }
    built = SimpleNamespace(
        data_provider=None,
        algorithm=object(),
        order_manager=object(),
        portfolio=object(),
    )
    captured = {}

    monkeypatch.setattr(live_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(live_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(live_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(live_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(live_cmd, "load_account_creds", lambda account: {"api_key": "key", "secret_key": "secret"})
    monkeypatch.setattr(live_cmd, "resolve_alpaca_credentials", lambda cfg, creds: cfg)
    monkeypatch.setattr(live_cmd, "build_experiment_config", lambda cfg: "normalized-config")
    monkeypatch.setattr(live_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(
        live_cmd.ExperimentService,
        "describe",
        lambda cfg: SimpleNamespace(config_hash="hash5678"),
    )

    def fake_engine(*args, **kwargs):
        engine = StubEngine(*args, **kwargs)
        captured["engine"] = engine
        return engine

    monkeypatch.setattr(live_cmd, "AlpacaRealTimeEngine", fake_engine)

    args = SimpleNamespace(config="cfg.yaml", account="paper", session_id="sess-1")
    live_cmd.cmd_live(args)

    assert captured["engine"].ran is True


def test_cmd_live_overwrites_stale_order_manager_credentials(monkeypatch):
    raw_cfg = {
        "mode": "live",
        "alpaca": {"symbols_to_subscribe": ["SPY"]},
        "analysis": { "enabled": False },
        "aggregation": { "enabled": False },
        "optimization": { "enabled": False },
        "state_store": { "enabled": True, "session_id": "sess-1" },
        "order_manager": {
            "order_manager": "dummy.OM",
            "api_key": "stale-key",
            "secret_key": "stale-secret",
        },
    }
    captured = {}
    built = SimpleNamespace(data_provider=None, algorithm=object(), order_manager=object(), portfolio=object())

    monkeypatch.setattr(live_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(live_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(live_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(live_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(live_cmd, "load_account_creds", lambda account: {"api_key": "trade-key", "secret_key": "trade-secret"})
    monkeypatch.setattr(live_cmd, "build_experiment_config", lambda cfg: captured.setdefault("cfg", cfg) or "normalized-config")
    monkeypatch.setattr(live_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(live_cmd.ExperimentService, "describe", lambda cfg: SimpleNamespace(config_hash="hash"))
    monkeypatch.setattr(live_cmd, "AlpacaRealTimeEngine", lambda *args, **kwargs: StubEngine(*args, **kwargs))

    live_cmd.cmd_live(SimpleNamespace(config="cfg.yaml", account="paper", session_id="sess-1", alpaca_override_url=None))

    om = captured["cfg"]["order_manager"]
    assert om["api_key"] == "trade-key"
    assert om["secret_key"] == "trade-secret"
    assert captured["cfg"]["state_store"]["database"] == "live_trading"


def test_cmd_live_overwrites_stale_new_style_order_manager_credentials(monkeypatch):
    raw_cfg = {
        "mode": "live",
        "alpaca": {"symbols_to_subscribe": ["SPY"]},
        "analysis": {"enabled": False},
        "aggregation": {"enabled": False},
        "optimization": {"enabled": False},
        "state_store": {"enabled": True, "session_id": "sess-1"},
        "order_manager": {
            "implementation": "dummy.OM",
            "api_key": "stale-top-key",
            "secret_key": "stale-top-secret",
            "params": {"api_key": "stale-param-key", "secret_key": "stale-param-secret"},
        },
    }
    captured = {}
    built = SimpleNamespace(data_provider=None, algorithm=object(), order_manager=object(), portfolio=object())

    monkeypatch.setattr(live_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(live_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(live_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(live_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(live_cmd, "load_account_creds", lambda account: {"api_key": "trade-key", "secret_key": "trade-secret"})
    monkeypatch.setattr(live_cmd, "build_experiment_config", lambda cfg: captured.setdefault("cfg", cfg) or "normalized-config")
    monkeypatch.setattr(live_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(live_cmd.ExperimentService, "describe", lambda cfg: SimpleNamespace(config_hash="hash"))
    monkeypatch.setattr(live_cmd, "AlpacaRealTimeEngine", lambda *args, **kwargs: StubEngine(*args, **kwargs))

    live_cmd.cmd_live(SimpleNamespace(config="cfg.yaml", account="paper", session_id="sess-1", alpaca_override_url=None))

    om = captured["cfg"]["order_manager"]
    assert om["api_key"] == "trade-key"
    assert om["secret_key"] == "trade-secret"
    assert om["params"]["api_key"] == "trade-key"
    assert om["params"]["secret_key"] == "trade-secret"


def test_cmd_live_walk_forward_mode(monkeypatch):
    raw_cfg = {
        "mode": "live",
        "alpaca": {
            "api_key": "key",
            "secret_key": "secret",
            "symbols_to_subscribe": ["SPY"],
        },
        "analysis": {"enabled": False},
        "aggregation": {"enabled": False},
        "optimization": {
            "enabled": True,
            "mode": "walk_forward_live",
            "historical_data_provider": {"provider": "dummy.Provider"},
        },
        "state_store": {"enabled": True, "session_id": "sess-1"},
        "order_manager": {"order_manager": "dummy.OM"},
    }
    built = SimpleNamespace(
        data_provider=None,
        algorithm=object(),
        order_manager=object(),
        portfolio=object(),
    )
    captured = {}

    monkeypatch.setattr(live_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(live_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(live_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(live_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(live_cmd, "load_account_creds", lambda account: {"api_key": "key", "secret_key": "secret"})
    monkeypatch.setattr(live_cmd, "resolve_alpaca_credentials", lambda cfg, creds: cfg)
    monkeypatch.setattr(live_cmd, "build_experiment_config", lambda cfg: "normalized-config")
    monkeypatch.setattr(live_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(live_cmd.ExperimentService, "describe", lambda cfg: SimpleNamespace(config_hash="hashwalk"))

    def fake_live_engine(*args, **kwargs):
        return SimpleNamespace(run=lambda: None)

    def fake_wrapper(inner, cfg):
        captured["cfg"] = cfg
        return StubEngine(inner, cfg)

    monkeypatch.setattr(live_cmd, "AlpacaRealTimeEngine", fake_live_engine)
    monkeypatch.setattr("trading.engines.live_walk_forward_engine.LiveWalkForwardEngine", fake_wrapper)

    args = SimpleNamespace(config="cfg.yaml", account="paper", session_id="sess-1")
    live_cmd.cmd_live(args)

    assert captured["cfg"]["mode"] == "walk_forward_live"


def test_cmd_walk_forward_passes_mlflow_settings(monkeypatch):
    raw_cfg = {
        "mode": "walk-forward",
        "analysis": {
            "experiment_name": "WF Experiment",
            "run_name": "WF Run",
            "description": "WF Desc",
            "log_to_mlflow": False,
        },
        "mlflow": {"tracking_uri": "http://mlflow.local"},
        "walk_forward": {"optimization_window_days": 30},
        "state_store": {"enabled": False},
        "data_provider": {"provider": "dummy.Provider"},
    }
    built = SimpleNamespace(
        data_provider=object(),
        algorithm=object(),
        order_manager=object(),
        portfolio=SimpleNamespace(total_value=1000.0, cash=500.0, positions={}),
    )
    captured = {}

    monkeypatch.setattr(walk_forward_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(walk_forward_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(walk_forward_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(walk_forward_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(walk_forward_cmd, "load_account_creds", lambda account: {"api_key": "x", "secret_key": "y"})
    monkeypatch.setattr(walk_forward_cmd, "flatten_config", lambda cfg: {"config.mode": cfg["mode"]})
    monkeypatch.setattr(walk_forward_cmd, "_collect_config_artifact_paths", lambda cfg, config_path=None: [config_path, "runtime.yaml"])
    monkeypatch.setattr(walk_forward_cmd, "get_git_info", lambda: {"git.commit": "abc123"})
    monkeypatch.setattr(walk_forward_cmd, "build_experiment_config", lambda cfg: "normalized-config")
    monkeypatch.setattr(walk_forward_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(
        walk_forward_cmd.ExperimentService,
        "describe",
        lambda cfg: SimpleNamespace(config_hash="hash9012"),
    )

    def fake_engine(*args, **kwargs):
        engine = StubEngine(*args, **kwargs)
        engine.run = lambda: {"aggregate": {}}
        captured["engine"] = engine
        return engine

    monkeypatch.setattr(walk_forward_cmd, "WalkForwardEngine", fake_engine)

    args = SimpleNamespace(
        config="cfg.yaml",
        account="paper",
        session_id=None,
        walk_forward_num_trials_override=20,
        walk_forward_max_concurrent_trials_override=4,
    )
    walk_forward_cmd.cmd_walk_forward(args)

    assert captured["engine"].kwargs["cfg"]["log_to_mlflow"] is False
    assert captured["engine"].kwargs["cfg"]["tracking_uri"] == "http://mlflow.local"
    assert captured["engine"].kwargs["cfg"]["mlflow_parameters"] == {"config.mode": "walk-forward"}
    assert captured["engine"].kwargs["cfg"]["mlflow_artifact_paths"] == ["cfg.yaml", "runtime.yaml"]
    assert captured["engine"].kwargs["cfg"]["mlflow_tags"] == {"git.commit": "abc123"}
    assert captured["engine"].kwargs["cfg"]["walk_forward"]["num_trials"] == 20
    assert captured["engine"].kwargs["cfg"]["walk_forward"]["max_concurrent_trials"] == 4


def test_cmd_walk_forward_hpo_passes_window_hpo_settings(monkeypatch):
    raw_cfg = {
        "mode": "walk-forward",
        "analysis": {
            "experiment_name": "WF Experiment",
            "run_name": "WF Window HPO",
            "description": "WF HPO Desc",
            "log_to_mlflow": True,
        },
        "mlflow": {"tracking_uri": "http://mlflow.local"},
        "walk_forward": {"optimization_window_days": 30},
        "walk_forward_window_hpo": {
            "num_samples": 2,
            "search_space": {
                "optimization_window_days": {"type": "choice", "values": [30]},
                "validation_window_days": {"type": "choice", "values": [5]},
                "trading_window_days": {"type": "choice", "values": [10]},
            },
        },
        "state_store": {"enabled": False},
        "data_provider": {"provider": "dummy.Provider"},
    }
    built = SimpleNamespace(
        data_provider=object(),
        algorithm=object(),
        order_manager=object(),
        portfolio=SimpleNamespace(total_value=1000.0, cash=500.0, positions={}),
    )
    captured = {}

    monkeypatch.setattr(walk_forward_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(walk_forward_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(walk_forward_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(walk_forward_cmd, "validate_session_id", lambda cfg: None)
    monkeypatch.setattr(walk_forward_cmd, "load_account_creds", lambda account: {"api_key": "x", "secret_key": "y"})
    monkeypatch.setattr(walk_forward_cmd, "flatten_config", lambda cfg: {"config.mode": cfg["mode"]})
    monkeypatch.setattr(walk_forward_cmd, "_collect_config_artifact_paths", lambda cfg, config_path=None: [config_path, "runtime.yaml"])
    monkeypatch.setattr(walk_forward_cmd, "get_git_info", lambda: {"git.commit": "abc123"})
    monkeypatch.setattr(walk_forward_cmd, "build_experiment_config", lambda cfg: "normalized-config")
    monkeypatch.setattr(walk_forward_cmd.ExperimentService, "build", lambda cfg: built)
    monkeypatch.setattr(
        walk_forward_cmd.ExperimentService,
        "describe",
        lambda cfg: SimpleNamespace(config_hash="hash9012"),
    )

    class FakeWindowHPO:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def run(self):
            return {
                "best_windows": {
                    "optimization_window_days": 30,
                    "validation_window_days": 5,
                    "trading_window_days": 10,
                },
                "best_metric": 1.23,
                "final_result": {"aggregate": {}},
            }

    monkeypatch.setattr(walk_forward_cmd, "WalkForwardWindowHPO", FakeWindowHPO)

    args = SimpleNamespace(config="cfg.yaml", account="paper", session_id=None)
    walk_forward_cmd.cmd_walk_forward_hpo(args)

    engine_cfg = captured["kwargs"]["engine_cfg"]
    assert engine_cfg["walk_forward_window_hpo"]["num_samples"] == 2
    assert engine_cfg["tracking_uri"] == "http://mlflow.local"
    assert engine_cfg["experiment_name"] == "WF Experiment"
    assert engine_cfg["mlflow_parameters"] == {"config.mode": "walk-forward"}
    assert engine_cfg["mlflow_artifact_paths"] == ["cfg.yaml", "runtime.yaml"]
    assert engine_cfg["mlflow_tags"] == {"git.commit": "abc123"}
