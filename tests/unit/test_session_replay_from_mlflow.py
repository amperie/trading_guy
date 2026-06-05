from __future__ import annotations

from argparse import Namespace

from trading.launchers.mlflow_hpo_launcher import SourceRunContext
from trading.commands import session_replay_from_mlflow as cmd


def _source_context() -> SourceRunContext:
    return SourceRunContext(
        run_id="abc123",
        run_name="source-run",
        tracking_uri="http://mlflow.local",
        source_url="http://mlflow.local/#/experiments/1/runs/abc123",
        config_source="artifact:config/runtime.yaml",
        raw_config={
            "mode": "backtest",
            "algorithm": {
                "implementation": "tests.fixtures.custom_components.CustomAlgorithm",
                "source_path": "scratch/CustomAlgorithm.py",
                "class_name": "CustomAlgorithm",
                "params": {"history_length": 25, "lookback": 10},
            },
            "portfolio": {
                "implementation": "tests.fixtures.custom_components.CustomPortfolio",
                "params": {"cash": 10000},
            },
            "order_manager": {"implementation": "tests.fixtures.custom_components.CustomOrderManager"},
            "analysis": {},
            "mlflow": {},
            "state_store": {
                "connection_uri": "mongodb://stale:27017",
                "database": "stale_database",
            },
        },
    )


def test_prepare_replay_config_uses_configured_live_database(monkeypatch):
    class FakeConfigManager:
        def get(self, key):
            assert key == "state_store"
            return {
                "connection_uri": "mongodb://configured:27017",
                "default_live_database": "live_trading",
                "database": "fallback_database",
            }

    monkeypatch.setattr("utils.config_manager.ConfigManager", lambda: FakeConfigManager())

    args = Namespace(
        session_id="live-session-123",
        connection_uri=None,
        database=None,
        mlflow_experiment_name=None,
    )
    prepared = cmd._prepare_replay_config(_source_context(), args)

    assert prepared["mode"] == "session-replay"
    assert prepared["mlflow"]["enabled"] is True
    assert prepared["mlflow"]["tracking_uri"] == "http://mlflow.local"
    assert prepared["analysis"]["experiment_name"] == "Session Replay From MLflow"
    assert "abc123" in prepared["analysis"]["description"]
    assert "live-session-123" in prepared["analysis"]["description"]
    assert prepared["state_store"]["enabled"] is True
    assert prepared["state_store"]["connection_uri"] == "mongodb://configured:27017"
    assert prepared["state_store"]["database"] == "live_trading"
    assert prepared["state_store"]["session_id"] == "live-session-123"


def test_prepare_replay_config_database_arg_overrides_config(monkeypatch):
    class FakeConfigManager:
        def get(self, key):
            return {"connection_uri": "mongodb://configured:27017", "default_live_database": "live_trading"}

    monkeypatch.setattr("utils.config_manager.ConfigManager", lambda: FakeConfigManager())

    args = Namespace(
        session_id="live-session-123",
        connection_uri="mongodb://override:27017",
        database="manual_db",
        mlflow_experiment_name="Replay Audits",
    )
    prepared = cmd._prepare_replay_config(_source_context(), args)

    assert prepared["state_store"]["connection_uri"] == "mongodb://override:27017"
    assert prepared["state_store"]["database"] == "manual_db"
    assert prepared["analysis"]["experiment_name"] == "Replay Audits"


def test_cmd_session_replay_from_mlflow_persists_config_and_delegates(monkeypatch):
    source_context = _source_context()
    captured = {}

    monkeypatch.setattr(cmd, "load_source_run_context", lambda run_url, tracking_uri=None: source_context)

    def fake_persist(source_context, edited_cfg, output_dir_name, filename_prefix):
        captured["persist"] = {
            "cfg": edited_cfg,
            "output_dir_name": output_dir_name,
            "filename_prefix": filename_prefix,
        }
        return "scratch/generated_session_replay_configs/generated.yaml"

    monkeypatch.setattr(cmd, "persist_edited_config", fake_persist)

    def fake_session_replay(args):
        captured["args"] = args
        return {"session_id": args.session_id}

    monkeypatch.setattr(cmd, "cmd_session_replay", fake_session_replay)

    result = cmd.cmd_session_replay_from_mlflow(Namespace(
        account="paper",
        run_url=source_context.source_url,
        tracking_uri="http://mlflow.local",
        session_id="live-session-123",
        no_mlflow=False,
        run_name=None,
        start_date=None,
        timeframe=None,
        cash=None,
        connection_uri=None,
        database=None,
        mlflow_experiment_name="Replay Audits",
    ))

    assert result == {"session_id": "live-session-123"}
    assert captured["persist"]["output_dir_name"] == "generated_session_replay_configs"
    assert captured["persist"]["filename_prefix"] == "session_replay"
    assert captured["persist"]["cfg"]["mode"] == "session-replay"
    assert captured["persist"]["cfg"]["analysis"]["experiment_name"] == "Replay Audits"
    assert captured["args"].config == "scratch/generated_session_replay_configs/generated.yaml"
    assert captured["args"].use_config_components is True
    assert captured["args"].clean_mongo_backtest is True
    assert captured["args"].source_run_id == "abc123"
    assert captured["args"].source_run_url == source_context.source_url
