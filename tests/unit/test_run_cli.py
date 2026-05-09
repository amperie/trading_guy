from __future__ import annotations

from run import build_parser


def test_build_parser_backtest_args():
    parser = build_parser()
    args = parser.parse_args([
        "backtest",
        "--config", "configs/example_backtest.yaml",
        "--account", "paper",
        "--cash", "5000",
    ])

    assert args.command == "backtest"
    assert args.cash == 5000.0
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
