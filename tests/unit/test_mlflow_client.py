from types import SimpleNamespace

import utils.mlflow_client as mlflow_module


def test_mlflow_client_creates_missing_experiment(monkeypatch):
    created = {}

    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(mlflow_module.mlflow, "set_tracking_uri", lambda uri: created.setdefault("uri", uri))

    def fake_get_experiment_by_name(name):
        created.setdefault("lookup_names", []).append(name)
        return None

    monkeypatch.setattr(mlflow_module.mlflow, "get_experiment_by_name", fake_get_experiment_by_name)
    monkeypatch.setattr(
        mlflow_module.mlflow,
        "create_experiment",
        lambda name, artifact_location=None: (
            created.setdefault("created", (name, artifact_location)),
            "exp-123",
        )[1],
    )
    monkeypatch.setattr(
        mlflow_module.mlflow,
        "get_experiment",
        lambda experiment_id: SimpleNamespace(experiment_id=experiment_id, name="wf-exp"),
    )

    client = mlflow_module.MLflowClient(
        experiment_name="wf-exp",
        tracking_uri="http://mlflow.local",
        enabled=True,
        auto_log_system_info=False,
    )

    assert client.experiment_id == "exp-123"
    assert created["created"] == ("wf-exp", None)
    assert created["uri"] == "http://mlflow.local"


def test_mlflow_client_retries_start_run_after_refresh(monkeypatch):
    state = {"lookup": 0, "start_run": 0}

    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(mlflow_module.mlflow, "set_tracking_uri", lambda uri: None)

    def fake_get_experiment_by_name(name):
        state["lookup"] += 1
        if state["lookup"] == 1:
            return SimpleNamespace(experiment_id="stale-exp", name=name)
        return None

    monkeypatch.setattr(mlflow_module.mlflow, "get_experiment_by_name", fake_get_experiment_by_name)
    monkeypatch.setattr(mlflow_module.mlflow, "create_experiment", lambda name, artifact_location=None: "fresh-exp")
    monkeypatch.setattr(
        mlflow_module.mlflow,
        "get_experiment",
        lambda experiment_id: SimpleNamespace(experiment_id=experiment_id, name="wf-exp"),
    )

    def fake_start_run(experiment_id=None, run_name=None, nested=False):
        state["start_run"] += 1
        if state["start_run"] == 1:
            raise RuntimeError("experiment missing")
        return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    monkeypatch.setattr(mlflow_module.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow_module.mlflow, "set_tag", lambda *args, **kwargs: None)

    client = mlflow_module.MLflowClient(
        experiment_name="wf-exp",
        tracking_uri="http://mlflow.local",
        enabled=True,
        auto_log_system_info=False,
    )

    client.start_run(run_name="sanity")

    assert client.experiment_id == "fresh-exp"
    assert client.run_id == "run-123"
    assert state["start_run"] == 2
