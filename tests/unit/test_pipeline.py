from argparse import Namespace
from types import SimpleNamespace

import yaml

from trading.commands import pipeline as pipeline_cmd
from trading.pipeline import evaluate_research_gates, evaluate_review_gates


class _Metrics:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_research_gates_pass_with_configured_thresholds():
    raw_cfg = {
        "pipeline": {
            "gates": {
                "research": {
                    "min_val_annualized_return": 5,
                    "max_val_max_drawdown_pct": 20,
                    "min_val_total_trades": 10,
                    "min_wf_annualized_return": 4,
                    "max_wf_max_drawdown_pct": 25,
                }
            }
        }
    }
    report = evaluate_research_gates(
        raw_cfg,
        {"analysis": {"metrics": _Metrics(annualized_return=8, max_drawdown_pct=12)}},
        {"val_results": {"metrics": _Metrics(annualized_return=7, max_drawdown_pct=18, total_trades=14)}},
        {"aggregate": {"wf_annualized_return": 6, "wf_max_drawdown_pct": 19}},
    )
    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_research_gates_fail_when_validation_or_walk_forward_miss():
    raw_cfg = {
        "pipeline": {
            "gates": {
                "research": {
                    "min_val_annualized_return": 5,
                    "max_wf_max_drawdown_pct": 10,
                }
            }
        }
    }
    report = evaluate_research_gates(
        raw_cfg,
        None,
        {"val_results": {"metrics": _Metrics(annualized_return=3, max_drawdown_pct=8, total_trades=20)}},
        {"aggregate": {"wf_max_drawdown_pct": 12}},
    )
    assert report.passed is False
    assert [check.name for check in report.checks if not check.passed] == [
        "val_annualized_return",
        "wf_max_drawdown_pct",
    ]


def test_review_gates_fail_on_large_drift():
    raw_cfg = {
        "pipeline": {
            "gates": {
                "review": {
                    "max_alpaca_live_equity_drift_pct": 2,
                    "max_mongo_live_equity_drift_pct": 1,
                }
            }
        }
    }
    report = evaluate_review_gates(
        raw_cfg,
        {
            "alpaca_live_equity_drift_pct": 3,
            "mongo_live_equity_drift_pct": 0.5,
        },
    )
    assert report.passed is False
    assert [check.name for check in report.checks if not check.passed] == [
        "alpaca_live_equity_drift_pct"
    ]


def test_pipeline_research_mlflow_config_is_edited_and_persisted(monkeypatch, tmp_path):
    source_context = SimpleNamespace(
        run_id="abc123456789",
        run_name="source run",
        raw_config={"mode": "backtest", "analysis": {"run_name": "original"}},
    )
    calls = {}

    def fake_load_source_run_context(run_url, tracking_uri=None):
        calls["run_url"] = run_url
        calls["tracking_uri"] = tracking_uri
        return source_context

    def fake_edit_config_dict(raw_cfg, editor=None, filename=None, label=None):
        calls["editor"] = editor
        calls["filename"] = filename
        calls["label"] = label
        edited = dict(raw_cfg)
        edited["analysis"] = {"run_name": "edited"}
        return edited

    def fake_persist_edited_config(source_context_arg, edited_cfg, *, output_dir_name, filename_prefix):
        calls["persist_source"] = source_context_arg
        calls["persisted_cfg"] = edited_cfg
        calls["output_dir_name"] = output_dir_name
        calls["filename_prefix"] = filename_prefix
        path = tmp_path / "edited_pipeline.yaml"
        path.write_text(yaml.safe_dump(edited_cfg), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(
        "trading.launchers.mlflow_hpo_launcher.load_source_run_context",
        fake_load_source_run_context,
    )
    monkeypatch.setattr(
        "trading.launchers.mlflow_hpo_launcher.edit_config_dict",
        fake_edit_config_dict,
    )
    monkeypatch.setattr(
        "trading.launchers.mlflow_hpo_launcher.persist_edited_config",
        fake_persist_edited_config,
    )

    args = Namespace(
        config="http://localhost:5000/#/experiments/1/runs/abc123456789",
        tracking_uri="http://localhost:5000",
        editor="vim",
    )

    path = pipeline_cmd._materialize_editable_research_config(args)

    assert path.endswith("edited_pipeline.yaml")
    assert calls["run_url"] == args.config
    assert calls["tracking_uri"] == "http://localhost:5000"
    assert calls["editor"] == "vim"
    assert calls["filename"] == "pipeline_research_config.yaml"
    assert calls["label"] == "pipeline research config"
    assert calls["output_dir_name"] == "generated_pipeline_configs"
    assert calls["filename_prefix"] == "pipeline_research"
    assert calls["persisted_cfg"]["analysis"]["run_name"] == "edited"


def test_pipeline_research_uses_edited_config_path_for_logged_artifacts(monkeypatch):
    edited_path = "scratch/generated_pipeline_configs/pipeline_research_source_abc123.yaml"
    calls = {}

    monkeypatch.setattr(
        pipeline_cmd,
        "_materialize_editable_research_config",
        lambda args: edited_path,
    )
    def fake_load_raw_config(path):
        calls["load_raw_config"] = path
        return {"pipeline": {"auto_promote_research": False}}

    def fake_apply_cli_overrides(raw_cfg, args):
        calls["override_config"] = args.config
        return raw_cfg

    def fake_apply_session_log_file(raw_cfg, args):
        calls["session_log_config"] = args.config

    def fake_cmd_backtest(args):
        calls["backtest_config"] = args.config
        return {"analysis": {}}

    monkeypatch.setattr(pipeline_cmd, "load_raw_config", fake_load_raw_config)
    monkeypatch.setattr(pipeline_cmd, "apply_cli_overrides", fake_apply_cli_overrides)
    monkeypatch.setattr(pipeline_cmd, "apply_session_log_file", fake_apply_session_log_file)
    monkeypatch.setattr(pipeline_cmd, "_preflight_pipeline_research", lambda raw_cfg, **kwargs: None)
    monkeypatch.setattr(pipeline_cmd, "cmd_backtest", fake_cmd_backtest)

    def fake_hpo(raw_cfg, *, config_artifact_path, **kwargs):
        calls["hpo_config_artifact_path"] = config_artifact_path
        return {"val_results": {"metrics": _Metrics(annualized_return=0, max_drawdown_pct=0, total_trades=0)}}

    monkeypatch.setattr(pipeline_cmd, "run_hpo_split_from_raw_config", fake_hpo)
    def fake_cmd_walk_forward(args):
        calls["walk_forward_config"] = args.config
        return {"aggregate": {}}

    monkeypatch.setattr(pipeline_cmd, "cmd_walk_forward", fake_cmd_walk_forward)
    monkeypatch.setattr(
        pipeline_cmd,
        "evaluate_research_gates",
        lambda raw_cfg, backtest_result, hpo_result, walk_forward_result: SimpleNamespace(
            passed=False,
            checks=[],
            to_dict=lambda: {"passed": False},
        ),
    )

    args = Namespace(
        config="http://localhost:5000/#/experiments/1/runs/abc123",
        account="paper",
        num_samples=None,
        max_concurrent_trials=None,
        validation_period_days=None,
        name=None,
    )

    result = pipeline_cmd.cmd_pipeline_research(args)

    assert result["gates"] == {"passed": False}
    assert calls["load_raw_config"] == edited_path
    assert calls["override_config"] == edited_path
    assert calls["session_log_config"] == edited_path
    assert calls["backtest_config"] == edited_path
    assert calls["hpo_config_artifact_path"] == edited_path
    assert calls["walk_forward_config"] == edited_path


def test_pipeline_research_fills_alpaca_data_provider_creds_before_hpo(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline_cmd, "_materialize_editable_research_config", lambda args: args.config)
    monkeypatch.setattr(
        pipeline_cmd,
        "load_raw_config",
        lambda path: {"data_provider": {"provider": "trading.data_providers.alpaca_data_provider.AlpacaDataProvider"}},
    )
    monkeypatch.setattr(pipeline_cmd, "apply_cli_overrides", lambda raw_cfg, args: raw_cfg)
    monkeypatch.setattr(pipeline_cmd, "apply_session_log_file", lambda raw_cfg, args: None)
    monkeypatch.setattr(pipeline_cmd, "_preflight_pipeline_research", lambda raw_cfg, **kwargs: None)
    monkeypatch.setattr(
        pipeline_cmd,
        "load_account_creds",
        lambda account: {"api_key": "key-123", "secret_key": "secret-456"},
    )
    monkeypatch.setattr(pipeline_cmd, "cmd_backtest", lambda args: {"analysis": {}})
    monkeypatch.setattr(pipeline_cmd, "cmd_walk_forward", lambda args: {"aggregate": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "evaluate_research_gates",
        lambda raw_cfg, backtest_result, hpo_result, walk_forward_result: SimpleNamespace(
            passed=False,
            checks=[],
            to_dict=lambda: {"passed": False},
        ),
    )

    def fake_hpo(raw_cfg, **kwargs):
        calls["hpo_raw_cfg"] = raw_cfg
        return {"val_results": {"metrics": _Metrics(annualized_return=0, max_drawdown_pct=0, total_trades=0)}}

    monkeypatch.setattr(pipeline_cmd, "run_hpo_split_from_raw_config", fake_hpo)

    args = SimpleNamespace(
        config="configs/example_hpo_split.yaml",
        account="paper3",
        num_samples=None,
        max_concurrent_trials=None,
        validation_period_days=30,
        name=None,
    )

    pipeline_cmd.cmd_pipeline_research(args)

    assert calls["hpo_raw_cfg"]["data_provider"]["api_key"] == "key-123"
    assert calls["hpo_raw_cfg"]["data_provider"]["secret_key"] == "secret-456"


def test_pipeline_research_runs_preflight_before_backtest(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline_cmd, "_materialize_editable_research_config", lambda args: args.config)
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {"auto_promote_research": False}})
    monkeypatch.setattr(pipeline_cmd, "apply_cli_overrides", lambda raw_cfg, args: raw_cfg)
    monkeypatch.setattr(pipeline_cmd, "apply_session_log_file", lambda raw_cfg, args: None)
    monkeypatch.setattr(
        pipeline_cmd,
        "_preflight_pipeline_research",
        lambda raw_cfg, **kwargs: calls.setdefault("preflight", True),
    )
    monkeypatch.setattr(pipeline_cmd, "cmd_backtest", lambda args: {"analysis": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "run_hpo_split_from_raw_config",
        lambda raw_cfg, **kwargs: {"val_results": {"metrics": _Metrics(annualized_return=0, max_drawdown_pct=0, total_trades=0)}},
    )
    monkeypatch.setattr(pipeline_cmd, "cmd_walk_forward", lambda args: {"aggregate": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "evaluate_research_gates",
        lambda raw_cfg, backtest_result, hpo_result, walk_forward_result: SimpleNamespace(
            passed=False,
            checks=[],
            to_dict=lambda: {"passed": False},
        ),
    )

    args = SimpleNamespace(
        config="configs/example_hpo_split.yaml",
        account="paper3",
        num_samples=None,
        max_concurrent_trials=None,
        validation_period_days=30,
        name=None,
    )

    pipeline_cmd.cmd_pipeline_research(args)
    assert calls["preflight"] is True


def test_pipeline_research_passes_cli_validation_period_days_to_preflight(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline_cmd, "_materialize_editable_research_config", lambda args: args.config)
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {"auto_promote_research": False}})
    monkeypatch.setattr(pipeline_cmd, "apply_cli_overrides", lambda raw_cfg, args: raw_cfg)
    monkeypatch.setattr(pipeline_cmd, "apply_session_log_file", lambda raw_cfg, args: None)
    monkeypatch.setattr(
        pipeline_cmd,
        "_preflight_pipeline_research",
        lambda raw_cfg, **kwargs: calls.setdefault("validation_period_days_override", kwargs.get("validation_period_days_override")),
    )
    monkeypatch.setattr(pipeline_cmd, "cmd_backtest", lambda args: {"analysis": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "run_hpo_split_from_raw_config",
        lambda raw_cfg, **kwargs: {"val_results": {"metrics": _Metrics(annualized_return=0, max_drawdown_pct=0, total_trades=0)}},
    )
    monkeypatch.setattr(pipeline_cmd, "cmd_walk_forward", lambda args: {"aggregate": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "evaluate_research_gates",
        lambda raw_cfg, backtest_result, hpo_result, walk_forward_result: SimpleNamespace(
            passed=False,
            checks=[],
            to_dict=lambda: {"passed": False},
        ),
    )

    args = SimpleNamespace(
        config="configs/example_hpo_split.yaml",
        account="paper3",
        num_samples=None,
        max_concurrent_trials=None,
        validation_period_days=30,
        name=None,
    )

    pipeline_cmd.cmd_pipeline_research(args)
    assert calls["validation_period_days_override"] == 30
