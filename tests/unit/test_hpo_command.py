from __future__ import annotations

import argparse
from types import SimpleNamespace

from trading.commands import hpo as hpo_cmd


def test_run_hpo_from_raw_config_passes_ray_worker_log_flag(monkeypatch):
    raw_cfg = {
        "mode": "hpo",
        "algorithm": {"implementation": "dummy.Algo", "params": {"alpha": 1}},
        "portfolio": {"implementation": "dummy.Portfolio", "params": {"symbol": "SPY", "cash": 10000.0}},
        "order_manager": {"implementation": "dummy.OM", "params": {}},
        "data_provider": {"implementation": "dummy.Provider", "params": {"path": "data.csv"}},
        "analysis": {"experiment_name": "exp", "run_name": "run", "description": "desc", "log_to_mlflow": False},
        "hpo": {
            "search_space": {},
            "algorithm_param_keys": ["alpha"],
            "portfolio_param_keys": [],
            "num_samples": 3,
            "max_concurrent_trials": 2,
            "log_trials_to_mlflow": False,
            "log_ray_worker_output": False,
        },
    }
    captured = {}

    monkeypatch.setattr(hpo_cmd, "build_experiment_config", lambda cfg: SimpleNamespace(
        algorithm=SimpleNamespace(implementation="dummy.Algo", params={"alpha": 1}),
        portfolio=SimpleNamespace(implementation="dummy.Portfolio", params={"symbol": "SPY", "cash": 10000.0}),
        order_manager=SimpleNamespace(implementation="dummy.OM", params={}),
        data_provider=SimpleNamespace(implementation="dummy.Provider", params={"path": "data.csv"}),
        model_dump=lambda exclude_none=True: {
            "analysis": {"experiment_name": "exp", "run_name": "run", "description": "desc", "log_to_mlflow": False},
            "mlflow": {},
        },
    ))
    monkeypatch.setattr(hpo_cmd, "import_component_class", lambda component: object)
    monkeypatch.setattr(hpo_cmd, "parse_search_space", lambda cfg: cfg)
    monkeypatch.setattr(hpo_cmd, "get_git_info", lambda: {})

    def fake_tune_backtest_hyperparameters(**kwargs):
        captured.update(kwargs)
        return {"alpha": 2}

    monkeypatch.setattr(
        "trading.launchers.run_backtest_ray.tune_backtest_hyperparameters",
        fake_tune_backtest_hyperparameters,
    )

    best = hpo_cmd.run_hpo_from_raw_config(raw_cfg)

    assert best == {"alpha": 2}
    assert captured["log_ray_worker_output"] is False


def test_resolve_hpo_split_dates_uses_config_range():
    start, train_end, val_start, end = hpo_cmd._resolve_hpo_split_dates(
        {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        validation_period_days=5,
    )

    assert start == "2024-01-01"
    assert train_end == "2024-01-26"
    assert val_start == "2024-01-27"
    assert end == "2024-01-31"


def test_cmd_hpo_split_logs_train_and_validation_with_prefixes(monkeypatch):
    raw_cfg = {
        "mode": "hpo",
        "algorithm": {"implementation": "dummy.Algo", "params": {"alpha": 1}},
        "portfolio": {"implementation": "dummy.Portfolio", "params": {"symbol": "SPY", "cash": 10000.0}},
        "order_manager": {"implementation": "dummy.OM", "params": {}},
        "data_provider": {"implementation": "dummy.Provider", "params": {"path": "data.csv"}},
        "analysis": {"experiment_name": "exp", "run_name": "run", "description": "desc", "log_to_mlflow": True},
        "mlflow": {"tracking_uri": "http://mlflow.local"},
        "hpo": {
            "validation_period_days": 10,
            "search_space": {},
            "algorithm_param_keys": ["alpha"],
            "portfolio_param_keys": [],
            "num_samples": 3,
            "max_concurrent_trials": 2,
            "log_trials_to_mlflow": False,
            "log_ray_worker_output": False,
        },
    }
    calls = []
    mlflow_events = []

    monkeypatch.setattr(hpo_cmd, "load_raw_config", lambda path: dict(raw_cfg))
    monkeypatch.setattr(hpo_cmd, "apply_cli_overrides", lambda cfg, args: cfg)
    monkeypatch.setattr(hpo_cmd, "apply_session_log_file", lambda cfg, args: None)
    monkeypatch.setattr(hpo_cmd, "load_account_creds", lambda account: {"api_key": "x", "secret_key": "y"})
    monkeypatch.setattr(hpo_cmd, "build_experiment_config", lambda cfg: SimpleNamespace(
        algorithm=SimpleNamespace(implementation="dummy.Algo", params={"alpha": 1}),
        portfolio=SimpleNamespace(implementation="dummy.Portfolio", params={"symbol": "SPY", "cash": 10000.0}),
        order_manager=SimpleNamespace(implementation="dummy.OM", params={}),
        data_provider=SimpleNamespace(implementation="dummy.Provider", params={"path": "data.csv"}),
        model_dump=lambda exclude_none=True: {
            "analysis": {"experiment_name": "exp", "run_name": "run", "description": "desc", "log_to_mlflow": True},
            "mlflow": {"tracking_uri": "http://mlflow.local"},
        },
    ))
    monkeypatch.setattr(hpo_cmd, "import_component_class", lambda component: object)
    monkeypatch.setattr(hpo_cmd, "parse_search_space", lambda cfg: cfg)
    monkeypatch.setattr(hpo_cmd, "get_git_info", lambda: {})
    monkeypatch.setattr(hpo_cmd, "_resolve_hpo_split_dates", lambda cfg, validation_period_days: ("2024-01-01", "2024-03-21", "2024-03-22", "2024-03-31"))
    monkeypatch.setattr(hpo_cmd, "_collect_config_artifact_paths", lambda cfg, config_path=None: ["cfg.yaml"])

    def fake_tune_backtest_hyperparameters(**kwargs):
        return {"alpha": 2}, [{"config": {"alpha": 2}, "metric": 9.9}]

    def fake_run_backtest_analysis(**kwargs):
        calls.append(kwargs)
        return {"metrics": SimpleNamespace(annualized_return=1.23)}

    class FakeMLflowClient:
        enabled = True

        def __init__(self, experiment_name=None, tracking_uri=None):
            self.experiment_name = experiment_name
            self.tracking_uri = tracking_uri

        def start_run(self, run_name=None, description=None, tags=None):
            mlflow_events.append(("start_run", run_name, description, tags))

            class _Ctx:
                def __enter__(self_nonlocal):
                    return self

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    return False

            return _Ctx()

        def log_json(self, data, filename):
            mlflow_events.append(("log_json", filename, data))

    monkeypatch.setattr(
        "trading.launchers.run_backtest_ray.tune_backtest_hyperparameters",
        fake_tune_backtest_hyperparameters,
    )
    monkeypatch.setattr(hpo_cmd, "_run_backtest_analysis", fake_run_backtest_analysis)
    monkeypatch.setattr(hpo_cmd, "MLflowClient", FakeMLflowClient)

    hpo_cmd.cmd_hpo_split(
        argparse.Namespace(
            config="cfg.yaml",
            account="paper",
            num_samples=None,
            max_concurrent_trials=None,
            validation_period_days=15,
        )
    )

    assert len(calls) == 3
    assert calls[0]["log_to_mlflow"] is False
    assert calls[1]["metric_prefix"] == "trn_"
    assert calls[1].get("artifact_prefix", "") == ""
    assert calls[2]["metric_prefix"] == "val_"
    assert calls[2]["artifact_prefix"] == "val_"
    assert calls[1]["parameters"]["hpo.validation_period_days"] == 15
    assert calls[1]["parameters"]["hpo.objective_metric"] == "val_annualized_return"
    assert any(event[0] == "log_json" and event[1] == "hpo_best_config.json" for event in mlflow_events)
