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


def test_backtest_help_mentions_remote_component_options(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backtest", "-h"])
    help_text = capsys.readouterr().out
    assert "--algorithm-url" in help_text
    assert "--portfolio-url" in help_text


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
