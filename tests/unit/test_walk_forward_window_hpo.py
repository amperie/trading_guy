from __future__ import annotations

import pytest

from types import SimpleNamespace

from trading.engines.walk_forward_window_hpo import WalkForwardWindowHPO, WindowHPOCandidate


class DummyDP:
    def __init__(self, cfg):
        self.cfg = cfg


class DummyAlgorithm:
    def __init__(self, cfg, history_length=0):
        self.cfg = cfg
        self.history_length = history_length


class DummyOM:
    pass


class DummyPortfolio:
    def __init__(self, cfg, om, cash, positions, keep_history):
        self.cfg = cfg
        self.om = om
        self.cash = cash
        self.keep_history = keep_history


def _build_optimizer(log_to_mlflow=True):
    return WalkForwardWindowHPO(
        engine_cfg={
            "walk_forward": {},
            "walk_forward_window_hpo": {
                "staging_experiment_name": "tmp-exp",
                "final_experiment_name": "final-exp",
                "staging_artifact_location": "s3://bucket/tmp/group",
                "final_artifact_location": "s3://bucket/final",
                "search_space": {
                    "optimization_window_days": {"type": "choice", "values": [30]},
                    "validation_window_days": {"type": "choice", "values": [5]},
                    "trading_window_days": {"type": "choice", "values": [10]},
                },
            },
            "experiment_name": "base-exp",
            "run_name": "base-run",
            "log_to_mlflow": log_to_mlflow,
            "tracking_uri": "http://mlflow.local",
        },
        dp=DummyDP({"path": "ignored"}),
        al=DummyAlgorithm({"alpha": 1}, history_length=7),
        om=DummyOM(),
        pf=DummyPortfolio({"risk": 2}, DummyOM(), 1000.0, {}, True),
    )


def test_candidate_cfg_uses_staging_experiment_and_tags():
    optimizer = _build_optimizer()

    cfg = optimizer._engine_cfg_for_candidate(
        {
            "optimization_window_days": 30,
            "validation_window_days": 5,
            "trading_window_days": 10,
        },
        trial_number=3,
    )

    assert cfg["experiment_name"] == "tmp-exp"
    assert cfg["artifact_location"] == "s3://bucket/tmp/group"
    assert cfg["walk_forward"]["optimization_window_days"] == 30
    assert cfg["mlflow_tags"]["run_type"] == "walk_forward_window_candidate"
    assert cfg["mlflow_tags"]["window_hpo_group_id"] == optimizer.group_id


def test_winner_cfg_uses_final_experiment():
    optimizer = _build_optimizer()
    windows = {
        "optimization_window_days": 30,
        "validation_window_days": 5,
        "trading_window_days": 10,
    }

    cfg = optimizer._engine_cfg_for_winner(
        windows,
        summary={
            "group_id": optimizer.group_id,
            "objective_metric": "wf_annualized_return",
            "best_windows": windows,
            "best_metric": 4.2,
            "best_trial_number": 1,
            "candidates": [{"valid": True}],
        },
    )

    assert cfg["experiment_name"] == "final-exp"
    assert cfg["artifact_location"] == "s3://bucket/final"
    assert cfg["mlflow_tags"]["run_type"] == "walk_forward_window_winner"
    assert cfg["extra_mlflow_json_artifacts"]["walk_forward_window_hpo_best_windows.json"] == windows
    assert cfg["extra_mlflow_json_artifacts"]["walk_forward_window_hpo_summary.json"]["best_metric"] == 4.2
    assert cfg["mlflow_parameters"]["walk_forward_window_hpo.best_metric"] == 4.2
    assert cfg["mlflow_parameters"]["walk_forward_window_hpo.best_optimization_window_days"] == 30


def test_no_mlflow_flag_is_honored_for_candidates_and_winner():
    optimizer = _build_optimizer(log_to_mlflow=False)
    windows = {
        "optimization_window_days": 30,
        "validation_window_days": 5,
        "trading_window_days": 10,
    }

    assert optimizer._engine_cfg_for_candidate(windows, trial_number=0)["log_to_mlflow"] is False
    assert optimizer._engine_cfg_for_winner(windows)["log_to_mlflow"] is False


def test_metric_reads_aggregate_before_metrics_object():
    optimizer = _build_optimizer()

    metric = optimizer._metric_from_result(
        {
            "aggregate": {"wf_annualized_return": 12.5},
            "metrics": SimpleNamespace(wf_annualized_return=1.0),
        }
    )

    assert metric == 12.5


def test_metric_raises_for_unknown_objective_metric():
    optimizer = _build_optimizer()
    optimizer.objective_metric = "wf_typo_metric"

    with pytest.raises(ValueError, match="Unknown walk_forward_window_hpo.objective_metric"):
        optimizer._metric_from_result(
            {
                "aggregate": {"wf_annualized_return": 12.5},
                "metrics": SimpleNamespace(annualized_return=1.0),
            }
        )


def test_run_raises_clear_error_and_cleans_up_when_all_trials_fail(monkeypatch):
    optimizer = _build_optimizer()
    optimizer.num_samples = 2
    cleanup_calls = []

    class FakeTrialState:
        COMPLETE = "COMPLETE"

    class FakeTrial:
        state = "FAIL"

    class FakeStudy:
        trials = [FakeTrial(), FakeTrial()]

        def optimize(self, objective, n_trials, n_jobs=1, catch=()):
            assert catch == (Exception,)
            return None

    fake_optuna = SimpleNamespace(
        create_study=lambda direction: FakeStudy(),
        trial=SimpleNamespace(TrialState=FakeTrialState),
    )

    monkeypatch.setitem(__import__("sys").modules, "optuna", fake_optuna)
    monkeypatch.setattr(optimizer, "_cleanup_staging", lambda: cleanup_calls.append(True) or {})

    with pytest.raises(RuntimeError, match="All 2 walk-forward window HPO trials failed"):
        optimizer.run()

    assert cleanup_calls == [True]


def test_run_rejects_all_completed_but_min_period_invalid_candidates(monkeypatch):
    optimizer = _build_optimizer()
    optimizer.min_periods = 3
    optimizer.num_samples = 1
    cleanup_calls = []

    class FakeTrialState:
        COMPLETE = "COMPLETE"

    class FakeTrial:
        state = "COMPLETE"
        params = {
            "optimization_window_days": 30,
            "validation_window_days": 5,
            "trading_window_days": 10,
        }

    class FakeStudy:
        trials = [FakeTrial()]

        def optimize(self, objective, n_trials, n_jobs=1, catch=()):
            optimizer.candidates.append(
                WindowHPOCandidate(
                    trial_number=0,
                    metric=-1e18,
                    windows=dict(FakeTrial.params),
                    mlflow_run_id="run-1",
                    num_periods=1,
                    valid=False,
                )
            )

    fake_optuna = SimpleNamespace(
        create_study=lambda direction: FakeStudy(),
        trial=SimpleNamespace(TrialState=FakeTrialState),
    )

    monkeypatch.setitem(__import__("sys").modules, "optuna", fake_optuna)
    monkeypatch.setattr(optimizer, "_cleanup_staging", lambda: cleanup_calls.append(True) or {})

    with pytest.raises(RuntimeError, match="completed walk-forward window HPO candidates were invalid"):
        optimizer.run()

    assert cleanup_calls == [True]
