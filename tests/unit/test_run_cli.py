from __future__ import annotations

import pytest

from run import build_parser


def test_build_parser_backtest_args():
    parser = build_parser()
    args = parser.parse_args([
        "backtest",
        "--config", "configs/example_backtest.yaml",
        "--account", "paper",
        "--cash", "5000",
        "--algorithm-url", "https://example.com/remote_algorithm.py",
        "--portfolio-url", "https://example.com/remote_portfolio.py",
    ])

    assert args.command == "backtest"
    assert args.cash == 5000.0
    assert args.algorithm_url == "https://example.com/remote_algorithm.py"
    assert args.portfolio_url == "https://example.com/remote_portfolio.py"
    assert args.func is not None


def test_build_parser_mongo_backtest_args():
    parser = build_parser()
    args = parser.parse_args([
        "mongo-backtest",
        "--config", "trading/promoted/winner_v1/winner_v1.yaml",
        "--account", "paper",
        "--session-id", "live-20260513-winner",
    ])

    assert args.command == "mongo-backtest"
    assert args.session_id == "live-20260513-winner"
    assert args.mongo_backtest is True
    assert args.func is not None


def test_build_parser_session_replay_args():
    parser = build_parser()
    args = parser.parse_args([
        "session-replay",
        "--config", "configs/example_session_replay.yaml",
        "--account", "paper",
        "--session-id", "abc123",
        "--start-date", "2026-01-01",
    ])

    assert args.command == "session-replay"
    assert args.session_id == "abc123"
    assert args.start_date == "2026-01-01"


def test_build_parser_hpo_from_mlflow_args():
    parser = build_parser()
    args = parser.parse_args([
        "hpo-from-mlflow",
        "--account", "paper",
        "--run-url", "http://localhost:5000/#/experiments/1/runs/abc123",
        "--editor", "vim",
    ])

    assert args.command == "hpo-from-mlflow"
    assert args.account == "paper"
    assert args.run_url == "http://localhost:5000/#/experiments/1/runs/abc123"
    assert args.editor == "vim"


def test_build_parser_hpo_split_from_mlflow_args():
    parser = build_parser()
    args = parser.parse_args([
        "hpo-split-from-mlflow",
        "--account", "paper",
        "--run-url", "http://localhost:5000/#/experiments/1/runs/abc123",
        "--editor", "vim",
    ])

    assert args.command == "hpo-split-from-mlflow"
    assert args.account == "paper"
    assert args.run_url == "http://localhost:5000/#/experiments/1/runs/abc123"
    assert args.editor == "vim"


def test_build_parser_promote_args():
    parser = build_parser()
    args = parser.parse_args([
        "promote",
        "--run-url", "http://localhost:5000/#/experiments/1/runs/abc123",
        "--tracking-uri", "http://localhost:5000",
        "--name", "spy-live",
    ])

    assert args.command == "promote"
    assert args.run_url == "http://localhost:5000/#/experiments/1/runs/abc123"
    assert args.tracking_uri == "http://localhost:5000"
    assert args.name == "spy-live"


def test_build_parser_hpo_split_args():
    parser = build_parser()
    args = parser.parse_args([
        "hpo-split",
        "--config", "configs/example_hpo_split.yaml",
        "--account", "paper",
        "--validation-period-days", "30",
        "--num-samples", "5",
    ])

    assert args.command == "hpo-split"
    assert args.validation_period_days == 30
    assert args.num_samples == 5


def test_root_help_mentions_hpo_from_mlflow(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-h"])
    help_text = capsys.readouterr().out
    assert "hpo-from-mlflow" in help_text
    assert "Recreate an HPO search from an MLflow run" in help_text
    assert "Top-level command summary:" in help_text
    assert "--algorithm-url" in help_text
    assert "--portfolio-url" in help_text
    assert "--tracking-uri" in help_text
    assert "--editor vim" in help_text
    assert "promote" in help_text
    assert "trading/promoted/my_live_bundle/my_live_bundle.yaml" in help_text
    assert "mongo-backtest" in help_text
    assert "configs/example_live_walk_forward.yaml" in help_text
    assert "hpo-split" in help_text
    assert "hpo-split-from-mlflow" in help_text


def test_backtest_help_mentions_remote_component_options(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backtest", "-h"])
    help_text = capsys.readouterr().out
    assert "--algorithm-url" in help_text
    assert "--portfolio-url" in help_text


def test_live_help_mentions_walk_forward_live_mode(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["live", "-h"])
    help_text = capsys.readouterr().out
    assert "walk_forward_live" in help_text
    assert "configs/example_live_walk_forward.yaml" in help_text


def test_walk_forward_help_mentions_validation_window(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["walk-forward", "-h"])
    help_text = capsys.readouterr().out
    assert "validation" in help_text.lower()
    assert "challenger" in help_text.lower()


def test_build_parser_walk_forward_hpo_args():
    parser = build_parser()
    args = parser.parse_args([
        "walk-forward-hpo",
        "--config", "configs/example_walk_forward_hpo.yaml",
        "--account", "paper",
    ])

    assert args.command == "walk-forward-hpo"
    assert args.func is not None


def test_walk_forward_hpo_help_mentions_staging_experiment(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["walk-forward-hpo", "-h"])
    help_text = capsys.readouterr().out
    assert "staging" in help_text
    assert "MLflow" in help_text
    assert "winning windows" in help_text


def test_mongo_backtest_help_mentions_session_id(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["mongo-backtest", "-h"])
    help_text = capsys.readouterr().out
    assert "--session-id" in help_text
    assert "MongoDBDataProvider" in help_text
    assert "BacktestingOrderManager" in help_text


def test_hpo_from_mlflow_help_mentions_editor(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["hpo-from-mlflow", "-h"])
    help_text = capsys.readouterr().out
    assert "--run-url" in help_text
    assert "--tracking-uri" in help_text
    assert "--editor" in help_text
    assert "notepad.exe" in help_text
    assert "vim" in help_text


def test_hpo_split_from_mlflow_help_mentions_validation_period_days(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["hpo-split-from-mlflow", "-h"])
    help_text = capsys.readouterr().out
    assert "--run-url" in help_text
    assert "--tracking-uri" in help_text
    assert "--editor" in help_text
    assert "hpo.validation_period_days" in help_text


def test_hpo_split_help_mentions_validation_period_days(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["hpo-split", "-h"])
    help_text = capsys.readouterr().out
    assert "--validation-period-days" in help_text
    assert "hpo.validation_period_days" in help_text


def test_promote_help_mentions_name(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["promote", "-h"])
    help_text = capsys.readouterr().out
    assert "--run-url" in help_text
    assert "--tracking-uri" in help_text
    assert "--name" in help_text
