from __future__ import annotations

import io
import pytest

import run
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


def test_build_parser_split_backtest_args():
    parser = build_parser()
    args = parser.parse_args([
        "split-backtest",
        "--config", "configs/example_backtest.yaml",
        "--account", "paper",
        "--train-start", "2021-01-01",
        "--train-end", "2023-12-31",
        "--val-start", "2024-01-01",
        "--val-end", "2024-06-30",
        "--split-name", "2024_h1",
    ])

    assert args.command == "split-backtest"
    assert args.train_start == "2021-01-01"
    assert args.train_end == "2023-12-31"
    assert args.val_start == "2024-01-01"
    assert args.val_end == "2024-06-30"
    assert args.split_name == "2024_h1"
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


def test_build_parser_session_replay_from_mlflow_args():
    parser = build_parser()
    args = parser.parse_args([
        "session-replay-from-mlflow",
        "--account", "paper",
        "--run-url", "http://localhost:5000/#/experiments/1/runs/abc123",
        "--session-id", "live-1",
        "--tracking-uri", "http://localhost:5000",
        "--database", "live_trading",
        "--mlflow-experiment-name", "Replay Audits",
        "--start-date", "2026-01-01",
    ])

    assert args.command == "session-replay-from-mlflow"
    assert args.account == "paper"
    assert args.run_url == "http://localhost:5000/#/experiments/1/runs/abc123"
    assert args.session_id == "live-1"
    assert args.tracking_uri == "http://localhost:5000"
    assert args.database == "live_trading"
    assert args.mlflow_experiment_name == "Replay Audits"
    assert args.start_date == "2026-01-01"
    assert args.func is not None


def test_build_parser_hpo_from_mlflow_args():
    parser = build_parser()
    args = parser.parse_args([
        "hpo-from-mlflow",
        "--account", "paper",
        "--run-url", "http://localhost:5000/#/experiments/1/runs/abc123",
        "--editor", "vim",
        "--algorithm-param", "momentum_lookback=1200",
        "--algorithm-param", "risk.threshold=1.5",
    ])

    assert args.command == "hpo-from-mlflow"
    assert args.account == "paper"
    assert args.run_url == "http://localhost:5000/#/experiments/1/runs/abc123"
    assert args.editor == "vim"
    assert args.algorithm_param == ["momentum_lookback=1200", "risk.threshold=1.5"]


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


def test_build_parser_pipeline_research_mlflow_editor_args():
    parser = build_parser()
    args = parser.parse_args([
        "pipeline",
        "research",
        "--config", "http://localhost:5000/#/experiments/1/runs/abc123",
        "--account", "paper",
        "--validation-period-days", "30",
        "--tracking-uri", "http://localhost:5000",
        "--editor", "vim",
    ])

    assert args.command == "pipeline"
    assert args.pipeline_stage == "research"
    assert args.config == "http://localhost:5000/#/experiments/1/runs/abc123"
    assert args.validation_period_days == 30
    assert args.tracking_uri == "http://localhost:5000"
    assert args.editor == "vim"


def test_build_parser_pipeline_research_requires_validation_period_days():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "pipeline",
            "research",
            "--config", "configs/example_hpo_split.yaml",
            "--account", "paper",
        ])


def test_main_exits_130_on_keyboard_interrupt(monkeypatch, capsys):
    class FakeParser:
        def parse_args(self):
            class Args:
                @staticmethod
                def func(_args):
                    raise KeyboardInterrupt()
            return Args()

    monkeypatch.setattr(run, "build_parser", lambda: FakeParser())

    with pytest.raises(SystemExit) as exc:
        run.main()

    assert exc.value.code == 130
    assert "Cancelled by user." in capsys.readouterr().err


def test_print_help_uses_color_for_tty(monkeypatch):
    class TtyBuffer(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    parser = build_parser()
    buf = TtyBuffer()
    parser.print_help(file=buf)
    help_text = buf.getvalue()

    assert "\x1b[" in help_text
    assert "usage:" in help_text
    assert "--config" in help_text


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
    assert "split-backtest" in help_text
    assert "configs/example_live_walk_forward.yaml" in help_text
    assert "hpo-split" in help_text
    assert "hpo-split-from-mlflow" in help_text
    assert "session-replay-from-mlflow" in help_text
    assert "Recover config and component code from an MLflow run" in help_text
    assert "MLflow configs are opened in an editor" in help_text


def test_backtest_help_mentions_remote_component_options(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backtest", "-h"])
    help_text = capsys.readouterr().out
    assert "--algorithm-url" in help_text
    assert "--portfolio-url" in help_text
    assert "topology_promoted_from_space_search_trailing_stop_backtest.yaml" in help_text
    assert "bars come from Alpaca, not MongoDB" in help_text


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


def test_session_replay_from_mlflow_help_mentions_recovered_config(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["session-replay-from-mlflow", "-h"])
    help_text = capsys.readouterr().out
    assert "--run-url" in help_text
    assert "--tracking-uri" in help_text
    assert "--session-id" in help_text
    assert "--database" in help_text
    assert "--mlflow-experiment-name" in help_text
    assert "scratch/generated_session_replay_configs" in help_text
    assert "specified MongoDB session's bars" in help_text
    assert "clean backtest" in help_text
    assert "warms up through its normal on_data() gate" in help_text


def test_hpo_split_help_mentions_validation_period_days(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["hpo-split", "-h"])
    help_text = capsys.readouterr().out
    assert "--validation-period-days" in help_text
    assert "hpo.validation_period_days" in help_text


def test_pipeline_research_help_mentions_mlflow_editor_and_logged_config(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pipeline", "research", "-h"])
    help_text = capsys.readouterr().out
    assert "--tracking-uri" in help_text
    assert "--editor" in help_text
    assert "MLflow run URL" in help_text
    assert "scratch/generated_pipeline_configs" in help_text
    assert "logged later reflects the run that actually executed" in help_text


def test_promote_help_mentions_name(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["promote", "-h"])
    help_text = capsys.readouterr().out
    assert "--run-url" in help_text
    assert "--tracking-uri" in help_text
    assert "--name" in help_text
