from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

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
    assert train_end == "2024-01-26 23:59:59.999999"
    assert val_start == "2024-01-27"
    assert end == "2024-01-31 23:59:59.999999"


def test_resolve_data_path_prefers_existing_cwd_relative_path(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "prices.csv"
    csv_path.write_text("timestamp\n2024-01-01\n")
    monkeypatch.chdir(tmp_path)

    assert hpo_cmd._resolve_data_path("data/prices.csv") == csv_path.resolve()


def test_build_minimal_warmup_dp_cfg_uses_limit_for_alpaca():
    class DummyAlgo:
        def __init__(self, cfg, history_length=0):
            self.required_warmup_bars = history_length

    class DummyAlpacaProvider:
        __module__ = "trading.data_providers.alpaca_data_provider"
        __name__ = "AlpacaDataProvider"

    warmup_cfg = hpo_cmd._build_minimal_warmup_dp_cfg(
        alg_cfg={"history_length": 260},
        validation_dp_cfg={"start_date": "2024-03-22"},
        training_dp_cfg={
            "symbols": ["UPRO"],
            "timeframe": "Minute",
            "start_date": "2024-01-01",
            "end_date": "2024-03-21 23:59:59.999999",
        },
        algorithm_class=DummyAlgo,
        data_provider_class=DummyAlpacaProvider,
    )

    assert warmup_cfg["limit"] == 260
    assert "start_date" not in warmup_cfg
    assert warmup_cfg["end_date"] == "2024-03-21 23:59:59.999999"


def test_build_minimal_warmup_dp_cfg_uses_reduced_start_for_non_alpaca():
    class DummyAlgo:
        def __init__(self, cfg, history_length=0):
            self.required_warmup_bars = history_length

    warmup_cfg = hpo_cmd._build_minimal_warmup_dp_cfg(
        alg_cfg={"history_length": 100},
        validation_dp_cfg={"start_date": "2024-03-22"},
        training_dp_cfg={
            "path": "data.csv",
            "timeframe": "Day",
            "start_date": "2024-01-01",
            "end_date": "2024-03-21 23:59:59.999999",
        },
        algorithm_class=DummyAlgo,
        data_provider_class=object,
    )

    assert warmup_cfg["end_date"] == "2024-03-21 23:59:59.999999"
    assert warmup_cfg["start_date"] != "2024-01-01"


def test_select_best_split_config_rejects_empty_trial_summaries():
    with pytest.raises(RuntimeError, match="no completed trial metrics"):
        hpo_cmd._select_best_split_config(
            trial_summaries=[],
            objective_metric="val_annualized_return",
            base_backtest_cfg={},
            base_al_cfg={},
            base_pf_cfg={},
            train_dp_cfg={},
            val_dp_cfg={},
            algorithm_class=object,
            portfolio_class=object,
            data_provider_class=object,
            order_manager_class=object,
            algorithm_param_keys=[],
            portfolio_param_keys=[],
        )


def test_select_best_split_config_rejects_non_finite_training_metrics():
    with pytest.raises(RuntimeError, match="no finite training trial metrics"):
        hpo_cmd._select_best_split_config(
            trial_summaries=[{"config": {"alpha": 2}, "metric": float("nan")}],
            objective_metric="trn_annualized_return",
            base_backtest_cfg={},
            base_al_cfg={},
            base_pf_cfg={},
            train_dp_cfg={},
            val_dp_cfg={},
            algorithm_class=object,
            portfolio_class=object,
            data_provider_class=object,
            order_manager_class=object,
            algorithm_param_keys=[],
            portfolio_param_keys=[],
        )


def test_select_best_split_config_rejects_non_finite_validation_metrics(monkeypatch):
    monkeypatch.setattr(hpo_cmd, "_score_split_validation_trials", lambda **kwargs: [])

    with pytest.raises(RuntimeError, match="no finite validation trial metrics"):
        hpo_cmd._select_best_split_config(
            trial_summaries=[{"config": {"alpha": 2}, "metric": 1.0}],
            objective_metric="val_annualized_return",
            base_backtest_cfg={},
            base_al_cfg={},
            base_pf_cfg={},
            train_dp_cfg={},
            val_dp_cfg={},
            algorithm_class=object,
            portfolio_class=object,
            data_provider_class=object,
            order_manager_class=object,
            algorithm_param_keys=["alpha"],
            portfolio_param_keys=[],
        )


def test_score_split_validation_trials_skips_worker_failures(monkeypatch):
    remote_calls = []

    class _Remote:
        def remote(self, **kwargs):
            remote_calls.append(kwargs["trial_config"])
            return f"ref-{kwargs['trial_config']['alpha']}"

    monkeypatch.setattr(hpo_cmd, "_score_split_validation_trial_remote", _Remote())
    monkeypatch.setattr(hpo_cmd.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(hpo_cmd.ray, "wait", lambda refs, num_returns=1: ([refs[0]], refs[1:]))

    def fake_get(ref):
        if ref == "ref-1":
            raise ConnectionError("network reset")
        return {"score": 2.5, "config": {"alpha": 2}, "alg_cfg": {"alpha": 2}, "pf_cfg": {"symbol": "SPY"}}

    monkeypatch.setattr(hpo_cmd.ray, "get", fake_get)

    scored = hpo_cmd._score_split_validation_trials(
        trial_summaries=[{"config": {"alpha": 1}}, {"config": {"alpha": 2}}],
        base_backtest_cfg={},
        base_al_cfg={},
        base_pf_cfg={},
        train_dp_cfg={},
        val_dp_cfg={},
        algorithm_class=object,
        portfolio_class=object,
        data_provider_class=object,
        order_manager_class=object,
        algorithm_param_keys=["alpha"],
        portfolio_param_keys=[],
        validation_metric="sharpe_ratio",
        max_concurrent_trials=2,
    )

    assert remote_calls == [{"alpha": 1}, {"alpha": 2}]
    assert scored == [(2.5, {"alpha": 2}, {"alpha": 2}, {"symbol": "SPY"})]


def test_select_best_split_config_uses_parallel_validation_scores(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        hpo_cmd,
        "_score_split_validation_trials",
        lambda **kwargs: (
            captured.setdefault("validation_metric", kwargs["validation_metric"]),
            (1.5, {"alpha": 3}, {"alpha": 3}, {"symbol": "SPY"}),
            (2.5, {"alpha": 4}, {"alpha": 4}, {"symbol": "SPY"}),
        )[1:],
    )

    best_config, best_al_cfg, best_pf_cfg, best_score = hpo_cmd._select_best_split_config(
        trial_summaries=[
            {"config": {"alpha": 3}, "metric": 1.0},
            {"config": {"alpha": 4}, "metric": 2.0},
        ],
        objective_metric="val_annualized_return",
        base_backtest_cfg={},
        base_al_cfg={},
        base_pf_cfg={},
        train_dp_cfg={},
        val_dp_cfg={},
        algorithm_class=object,
        portfolio_class=object,
        data_provider_class=object,
        order_manager_class=object,
        algorithm_param_keys=["alpha"],
        portfolio_param_keys=[],
        max_concurrent_trials=4,
    )

    assert captured["validation_metric"] == "annualized_return"
    assert best_config == {"alpha": 4}
    assert best_al_cfg == {"alpha": 4}
    assert best_pf_cfg == {"symbol": "SPY"}
    assert best_score == pytest.approx(2.5)


def test_normalize_split_objective_metric_accepts_legacy_aliases():
    assert hpo_cmd._normalize_split_objective_metric("annualized_return") == "val_annualized_return"
    assert hpo_cmd._normalize_split_objective_metric("train_annualized_return") == "trn_annualized_return"
    assert hpo_cmd._normalize_split_objective_metric("sharpe_ratio") == "val_sharpe_ratio"
    assert hpo_cmd._normalize_split_objective_metric("train_sharpe_ratio") == "trn_sharpe_ratio"
    assert hpo_cmd._normalize_split_objective_metric("validation_sharpe_ratio") == "val_sharpe_ratio"


def test_normalize_split_objective_metric_rejects_unknown_value():
    with pytest.raises(ValueError, match="Split HPO objective_metric must be a metric name"):
        hpo_cmd._normalize_split_objective_metric("val_")


def test_normalize_split_objective_metric_defaults_to_validation_metric():
    assert hpo_cmd._normalize_split_objective_metric(None) == "val_annualized_return"
    assert hpo_cmd._normalize_split_objective_metric("") == "val_annualized_return"


def test_select_best_split_config_uses_training_metric_prefix_for_non_annualized_metric():
    best_config, best_al_cfg, best_pf_cfg, best_score = hpo_cmd._select_best_split_config(
        trial_summaries=[
            {"config": {"alpha": 3}, "metric": 1.0},
            {"config": {"alpha": 4}, "metric": 2.0},
        ],
        objective_metric="trn_sharpe_ratio",
        base_backtest_cfg={},
        base_al_cfg={},
        base_pf_cfg={},
        train_dp_cfg={},
        val_dp_cfg={},
        algorithm_class=object,
        portfolio_class=object,
        data_provider_class=object,
        order_manager_class=object,
        algorithm_param_keys=["alpha"],
        portfolio_param_keys=[],
    )

    assert best_config == {"alpha": 4}
    assert best_al_cfg == {"alpha": 4}
    assert best_pf_cfg == {}
    assert best_score == pytest.approx(2.0)


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
    monkeypatch.setattr(
        hpo_cmd,
        "_build_minimal_warmup_dp_cfg",
        lambda **kwargs: {"end_date": "2024-03-21 23:59:59.999999", "limit": 42},
    )

    def fake_tune_backtest_hyperparameters(**kwargs):
        return {"alpha": 2}, [{"config": {"alpha": 2}, "metric": 9.9}]

    def fake_run_backtest_analysis(**kwargs):
        calls.append(kwargs)
        return {"metrics": SimpleNamespace(annualized_return=1.23)}

    def fake_score_split_validation_trials(**kwargs):
        calls.append(kwargs)
        return [(1.23, {"alpha": 2}, {"alpha": 2}, {"symbol": "SPY"})]

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
    monkeypatch.setattr(hpo_cmd, "_score_split_validation_trials", fake_score_split_validation_trials)
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
    assert calls[0]["max_concurrent_trials"] == 2
    assert calls[0]["train_dp_cfg"]["start_date"] == "2024-01-01"
    assert calls[0]["train_dp_cfg"]["end_date"] == "2024-03-21"
    assert calls[1]["metric_prefix"] == "trn_"
    assert calls[1].get("artifact_prefix", "") == ""
    assert calls[2]["metric_prefix"] == "val_"
    assert calls[2]["artifact_prefix"] == "val_"
    assert "warmup_dp_cfg" not in calls[1]
    assert calls[2]["warmup_dp_cfg"]["limit"] == 42
    assert calls[2]["warmup_dp_cfg"]["end_date"] == "2024-03-21 23:59:59.999999"
    assert calls[1]["parameters"]["hpo.validation_period_days"] == 15
    assert calls[1]["parameters"]["hpo.objective_metric"] == "val_annualized_return"
    assert any(event[0] == "log_json" and event[1] == "hpo_best_config.json" for event in mlflow_events)
