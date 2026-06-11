from argparse import Namespace
from types import SimpleNamespace

import pytest
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


def test_pipeline_research_routes_stage_mlflow_experiments(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline_cmd, "_materialize_editable_research_config", lambda args: args.config)
    monkeypatch.setattr(
        pipeline_cmd,
        "load_raw_config",
        lambda path: {
            "analysis": {"experiment_name": "Base Experiment"},
            "hpo": {
                "search_space": {"alpha": {"type": "uniform", "low": 0.0, "high": 1.0}},
                "algorithm_param_keys": ["alpha"],
                "portfolio_param_keys": ["risk"],
            },
            "pipeline": {
                "auto_promote_research": False,
                "experiments": {
                    "backtest": "Pipeline HPO",
                    "hpo_split": "Pipeline HPO Split",
                    "walk_forward": "Pipeline Walk-Forward",
                },
            },
        },
    )
    monkeypatch.setattr(pipeline_cmd, "apply_cli_overrides", lambda raw_cfg, args: raw_cfg)
    monkeypatch.setattr(pipeline_cmd, "apply_session_log_file", lambda raw_cfg, args: None)
    monkeypatch.setattr(pipeline_cmd, "_preflight_pipeline_research", lambda raw_cfg, **kwargs: None)

    def fake_cmd_backtest(args):
        calls["backtest_experiment"] = args.mlflow_experiment_name_override
        return {"analysis": {}}

    def fake_hpo(raw_cfg, **kwargs):
        calls["hpo_experiment"] = raw_cfg["analysis"]["experiment_name"]
        calls["hpo_num_samples_override"] = kwargs["num_samples_override"]
        calls["hpo_max_concurrent_override"] = kwargs["max_concurrent_override"]
        return {"val_results": {"metrics": _Metrics(annualized_return=0, max_drawdown_pct=0, total_trades=0)}}

    def fake_cmd_walk_forward(args):
        calls["walk_forward_experiment"] = args.mlflow_experiment_name_override
        calls["walk_forward_num_trials_override"] = args.walk_forward_num_trials_override
        calls["walk_forward_max_concurrent_trials_override"] = args.walk_forward_max_concurrent_trials_override
        calls["walk_forward_search_space"] = args.walk_forward_raw_cfg_override["walk_forward"]["search_space"]
        calls["walk_forward_algorithm_param_keys"] = args.walk_forward_raw_cfg_override["walk_forward"]["algorithm_param_keys"]
        calls["walk_forward_portfolio_param_keys"] = args.walk_forward_raw_cfg_override["walk_forward"]["portfolio_param_keys"]
        return {"aggregate": {}}

    monkeypatch.setattr(pipeline_cmd, "cmd_backtest", fake_cmd_backtest)
    monkeypatch.setattr(pipeline_cmd, "run_hpo_split_from_raw_config", fake_hpo)
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

    pipeline_cmd.cmd_pipeline_research(
        SimpleNamespace(
            config="configs/example_hpo_split.yaml",
            account="paper3",
            num_samples=20,
            max_concurrent_trials=4,
            validation_period_days=None,
            name=None,
        )
    )

    assert calls == {
        "backtest_experiment": "Pipeline HPO",
        "hpo_experiment": "Pipeline HPO Split",
        "hpo_num_samples_override": 20,
        "hpo_max_concurrent_override": 4,
        "walk_forward_experiment": "Pipeline Walk-Forward",
        "walk_forward_num_trials_override": 20,
        "walk_forward_max_concurrent_trials_override": 4,
        "walk_forward_search_space": {"alpha": {"type": "uniform", "low": 0.0, "high": 1.0}},
        "walk_forward_algorithm_param_keys": ["alpha"],
        "walk_forward_portfolio_param_keys": ["risk"],
    }


def test_pipeline_paper_logs_bundle_to_paper_experiment(monkeypatch):
    calls = {}
    bundle = SimpleNamespace(config_path="bundle.yaml", manifest_path="manifest.json")
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {}})
    monkeypatch.setattr(pipeline_cmd, "materialize_bundle", lambda *args, **kwargs: bundle)

    def fake_log(raw_cfg, bundle_arg, **kwargs):
        calls["experiment"] = raw_cfg["pipeline"]["experiments"]["bundle_registry"]
        calls["source_run_url"] = kwargs["source_run_url"]
        return {"run_url": "paper-bundle-url"}

    monkeypatch.setattr(pipeline_cmd, "log_registered_bundle", fake_log)
    monkeypatch.setattr(pipeline_cmd, "cmd_live", lambda args: {"session_id": args.session_id})

    pipeline_cmd.cmd_pipeline_paper(
        SimpleNamespace(run_url="http://mlflow/#/experiments/1/runs/source", session_id="paper-1", account="paper")
    )

    assert calls == {
        "experiment": "Pipeline Paper Bundles",
        "source_run_url": "http://mlflow/#/experiments/1/runs/source",
    }


def test_pipeline_paper_from_session_reuses_source_session_by_default(monkeypatch):
    calls = {}

    class FakeStore:
        def get_session(self, session_id):
            calls["lookup_session_id"] = session_id
            return {"metadata": {"source_run_url": "http://mlflow/#/experiments/1/runs/abc"}}

        def update_session(self, session_id, update):
            calls["source_session_update"] = (session_id, update)

    monkeypatch.setattr(pipeline_cmd, "_resolve_session_store", lambda **kwargs: FakeStore())
    monkeypatch.setattr(
        pipeline_cmd,
        "materialize_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            config_path="trading/promoted/bundle/bundle.yaml",
            manifest_path="trading/promoted/bundle/promotion_manifest.json",
        ),
    )
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "log_registered_bundle",
        lambda raw_cfg, bundle, **kwargs: (
            calls.update(
                bundle_metadata=kwargs["metadata"],
                bundle_experiment=raw_cfg["pipeline"]["experiments"]["bundle_registry"],
            )
            or {"run_url": "bundle-url"}
        ),
    )

    def fake_cmd_live(args):
        calls["live_session_id"] = args.session_id
        calls["live_source_session_id"] = args.source_session_id
        return {"session_id": args.session_id, "config_hash": "hash"}

    monkeypatch.setattr(pipeline_cmd, "cmd_live", fake_cmd_live)

    pipeline_cmd.cmd_pipeline_paper_from_session(
        SimpleNamespace(
            source_session_id="adx_momentum_filter_ema_crossover",
            session_id=None,
            account="paper3",
            name=None,
            tracking_uri=None,
            connection_uri=None,
            database=None,
            run_name=None,
            agg_period=None,
            alpaca_override_url=None,
            symbol=None,
            cash=None,
            algorithm=None,
            algorithm_url=None,
            portfolio=None,
            portfolio_url=None,
            no_mlflow=False,
        )
    )

    assert calls["lookup_session_id"] == "adx_momentum_filter_ema_crossover"
    assert calls["live_session_id"] == "adx_momentum_filter_ema_crossover"
    assert calls["live_source_session_id"] == "adx_momentum_filter_ema_crossover"
    assert calls["bundle_metadata"]["session_id"] == "adx_momentum_filter_ema_crossover"
    assert calls["bundle_experiment"] == "Pipeline Paper Bundles"
    assert calls["source_session_update"] == (
        "adx_momentum_filter_ema_crossover",
        {"metadata.account_name": "paper3"},
    )


def test_pipeline_paper_from_session_allows_explicit_target_session(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        pipeline_cmd,
        "_resolve_session_store",
        lambda **kwargs: SimpleNamespace(
            get_session=lambda session_id: {"metadata": {"launch_config_ref": "trading/promoted/bundle/bundle.yaml"}},
            update_session=lambda session_id, update: calls.setdefault("source_session_update", (session_id, update)),
        ),
    )
    monkeypatch.setattr(
        pipeline_cmd,
        "materialize_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            config_path="trading/promoted/bundle/bundle.yaml",
            manifest_path="trading/promoted/bundle/promotion_manifest.json",
        ),
    )
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {}})
    monkeypatch.setattr(pipeline_cmd, "log_registered_bundle", lambda *args, **kwargs: {"run_url": "bundle-url"})

    def fake_cmd_live(args):
        calls["live_session_id"] = args.session_id
        return {"session_id": args.session_id}

    monkeypatch.setattr(pipeline_cmd, "cmd_live", fake_cmd_live)

    pipeline_cmd.cmd_pipeline_paper_from_session(
        SimpleNamespace(
            source_session_id="source-session",
            session_id="paper-rerun-1",
            account="paper3",
            name=None,
            tracking_uri=None,
            connection_uri=None,
            database=None,
            run_name=None,
            agg_period=None,
            alpaca_override_url=None,
            symbol=None,
            cash=None,
            algorithm=None,
            algorithm_url=None,
            portfolio=None,
            portfolio_url=None,
            no_mlflow=False,
        )
    )

    assert calls["live_session_id"] == "paper-rerun-1"
    assert calls["source_session_update"] == ("source-session", {"metadata.account_name": "paper3"})


def test_pipeline_paper_from_session_keeps_existing_source_account_name(monkeypatch):
    calls = {}

    class FakeStore:
        def get_session(self, session_id):
            return {
                "metadata": {
                    "account_name": "paper2",
                    "launch_config_ref": "trading/promoted/bundle/bundle.yaml",
                }
            }

        def update_session(self, session_id, update):
            calls["source_session_update"] = (session_id, update)

    monkeypatch.setattr(pipeline_cmd, "_resolve_session_store", lambda **kwargs: FakeStore())
    monkeypatch.setattr(
        pipeline_cmd,
        "materialize_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            config_path="trading/promoted/bundle/bundle.yaml",
            manifest_path="trading/promoted/bundle/promotion_manifest.json",
        ),
    )
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {}})
    monkeypatch.setattr(pipeline_cmd, "log_registered_bundle", lambda *args, **kwargs: {"run_url": "bundle-url"})
    monkeypatch.setattr(pipeline_cmd, "cmd_live", lambda args: {"session_id": args.session_id})

    pipeline_cmd.cmd_pipeline_paper_from_session(
        SimpleNamespace(
            source_session_id="source-session",
            session_id=None,
            account="paper3",
            name=None,
            tracking_uri=None,
            connection_uri=None,
            database=None,
            run_name=None,
            agg_period=None,
            alpaca_override_url=None,
            symbol=None,
            cash=None,
            algorithm=None,
            algorithm_url=None,
            portfolio=None,
            portfolio_url=None,
            no_mlflow=False,
        )
    )

    assert "source_session_update" not in calls


@pytest.mark.parametrize(
    ("config", "loader"),
    [
        ("trading/promoted/bundle/bundle.yaml", "local"),
        ("http://mlflow/#/experiments/1/runs/approved", "mlflow"),
    ],
)
def test_pipeline_live_logs_bundle_to_live_experiment(monkeypatch, config, loader):
    calls = {}
    bundle = SimpleNamespace(
        config_path="trading/promoted/bundle/bundle.yaml",
        source_run_url="http://mlflow/#/experiments/1/runs/source",
    )
    monkeypatch.setattr(pipeline_cmd, "load_raw_config", lambda path: {"pipeline": {}})
    monkeypatch.setattr(
        pipeline_cmd,
        "load_local_bundle",
        lambda path: calls.setdefault("loader", "local") and bundle,
    )
    monkeypatch.setattr(
        pipeline_cmd,
        "materialize_bundle",
        lambda *args, **kwargs: calls.setdefault("loader", "mlflow") and bundle,
    )

    def fake_log(raw_cfg, bundle_arg, **kwargs):
        calls["experiment"] = raw_cfg["pipeline"]["experiments"]["bundle_registry"]
        calls["source_run_url"] = kwargs["source_run_url"]
        calls["session_id"] = kwargs["metadata"]["session_id"]
        return {"run_url": "live-bundle-url"}

    def fake_live(args):
        calls["live_config"] = args.config
        calls["live_source_run_url"] = args.source_run_url
        return {"session_id": args.session_id}

    monkeypatch.setattr(pipeline_cmd, "log_registered_bundle", fake_log)
    monkeypatch.setattr(pipeline_cmd, "cmd_live", fake_live)

    result = pipeline_cmd.cmd_pipeline_live(
        SimpleNamespace(config=config, session_id="live-1", account="live")
    )

    expected_source = config if loader == "mlflow" else bundle.source_run_url
    assert calls == {
        "loader": loader,
        "experiment": "Pipeline Live Bundles",
        "source_run_url": expected_source,
        "session_id": "live-1",
        "live_config": bundle.config_path,
        "live_source_run_url": expected_source,
    }
    assert result["bundle"]["run_url"] == "live-bundle-url"
    assert result["live"]["session_id"] == "live-1"


def test_pipeline_research_preflight_rejects_invalid_split_objective_metric(monkeypatch):
    monkeypatch.setattr(
        pipeline_cmd,
        "build_experiment_config",
        lambda raw_cfg: SimpleNamespace(data_provider=SimpleNamespace(params={"path": "data/test.csv"})),
    )
    monkeypatch.setattr(pipeline_cmd, "_resolve_hpo_split_dates", lambda cfg, validation_period_days: ("a", "b", "c", "d"))

    with pytest.raises(ValueError, match="Split HPO objective_metric must be a metric name"):
        pipeline_cmd._preflight_pipeline_research(
            {"hpo": {"objective_metric": "val_"}},
            validation_period_days_override=30,
        )
