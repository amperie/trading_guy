from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import paper_session_autostart as autostart


class FakeStore:
    def __init__(self, sessions):
        self.sessions = sessions

    def list_sessions(self, autostart=None):
        assert autostart is True
        return self.sessions


def test_launch_builds_one_tiled_tmux_pane_per_session(monkeypatch):
    calls = []
    store = FakeStore([
        {"_id": "paper-1", "metadata": {"account_name": "paper"}},
        {"_id": "paper-2", "metadata": {"account_name": "paper2"}},
    ])

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)

    assert autostart.launch(store, "papers", "python3") == 2
    assert calls[0][0] == ["tmux", "has-session", "-t", "papers"]
    assert calls[1][0][:6] == ["tmux", "new-session", "-d", "-s", "papers", "-n"]
    assert "--source-session-id paper-1" in calls[1][0][-1]
    assert calls[2][0][:4] == ["tmux", "split-window", "-t", "papers:paper"]
    assert "--source-session-id paper-2" in calls[2][0][-1]
    assert calls[3][0] == ["tmux", "select-layout", "-t", "papers:paper", "tiled"]


def test_launch_refuses_existing_tmux_session(monkeypatch):
    monkeypatch.setattr(
        autostart.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(RuntimeError, match="refusing to launch duplicates"):
        autostart.launch(
            FakeStore([{"_id": "paper-1", "metadata": {"account_name": "paper"}}]),
            "papers",
            "python3",
        )


def test_command_reads_account_from_session():
    command = autostart._command({"_id": "paper-1"}, "python3")
    assert "--account" not in command
