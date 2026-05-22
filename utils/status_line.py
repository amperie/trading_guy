from __future__ import annotations

import shutil
import sys


class StatusLine:
    """Single-line TTY status renderer with a no-op fallback."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = sys.stdout.isatty() if enabled is None else bool(enabled)
        self._last_len = 0

    def update(self, text: str) -> None:
        if not self.enabled:
            return
        width = shutil.get_terminal_size((120, 20)).columns
        clipped = text[: max(0, width - 1)]
        padding = " " * max(0, self._last_len - len(clipped))
        sys.stdout.write("\r" + clipped + padding)
        sys.stdout.flush()
        self._last_len = len(clipped)

    def clear(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\r" + (" " * self._last_len) + "\r")
        sys.stdout.flush()
        self._last_len = 0

    def close(self) -> None:
        self.clear()
