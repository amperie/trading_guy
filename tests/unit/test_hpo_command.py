from __future__ import annotations

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

