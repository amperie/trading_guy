"""Manage and launch restart-enabled paper sessions in tmux."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config_manager import ConfigManager
from utils.trading_state_store import TradingStateStore


def _store(connection_uri: str | None, database: str | None) -> TradingStateStore:
    cfg = ConfigManager().get("state_store") or {}
    return TradingStateStore(
        connection_uri=connection_uri or cfg.get("connection_uri", "mongodb://localhost:27017"),
        database=database or cfg.get("default_live_database") or cfg.get("database", "trading"),
    )


def _command(session: dict, python: str) -> list[str]:
    return [
        python,
        str(ROOT / "run.py"),
        "pipeline",
        "paper-from-session",
        "--source-session-id",
        str(session["_id"]),
    ]


def launch(store: TradingStateStore, tmux_session: str, python: str, dry_run: bool = False) -> int:
    sessions = store.list_sessions(autostart=True)
    if not sessions:
        print("No sessions have autostart=true.")
        return 0

    commands = [_command(session, python) for session in sessions]
    for session, command in zip(sessions, commands):
        print(f"{session['_id']}: {shlex.join(command)}")
    if dry_run:
        return len(commands)

    if subprocess.run(
        ["tmux", "has-session", "-t", tmux_session],
        capture_output=True,
        check=False,
    ).returncode == 0:
        raise RuntimeError(
            f"tmux session '{tmux_session}' already exists; refusing to launch duplicates"
        )

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", tmux_session, "-n", "paper", shlex.join(commands[0])],
        cwd=ROOT,
        check=True,
    )
    for command in commands[1:]:
        subprocess.run(
            ["tmux", "split-window", "-t", f"{tmux_session}:paper", "-c", str(ROOT), shlex.join(command)],
            check=True,
        )
        subprocess.run(
            ["tmux", "select-layout", "-t", f"{tmux_session}:paper", "tiled"],
            check=True,
        )
    print(f"Started {len(commands)} session(s). Attach with: tmux attach -t {tmux_session}")
    return len(commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-uri")
    parser.add_argument("--database")
    sub = parser.add_subparsers(dest="action", required=True)

    for action in ("enable", "disable"):
        command = sub.add_parser(action)
        command.add_argument("session_id")

    sub.add_parser("list")
    start = sub.add_parser("start")
    start.add_argument("--tmux-session", default="paper-sessions")
    start.add_argument("--python", default=sys.executable)
    start.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = _store(args.connection_uri, args.database)
    if args.action in {"enable", "disable"}:
        if not store.get_session(args.session_id):
            raise ValueError(f"Session '{args.session_id}' not found")
        store.update_session(args.session_id, {"autostart": args.action == "enable"})
        print(f"{args.session_id}: autostart={args.action == 'enable'}")
    elif args.action == "list":
        for session in store.list_sessions(autostart=True):
            print(f"{session['_id']}\t{(session.get('metadata') or {}).get('account_name', '')}")
    else:
        launch(store, args.tmux_session, args.python, args.dry_run)


if __name__ == "__main__":
    main()
