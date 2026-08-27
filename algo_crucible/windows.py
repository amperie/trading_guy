from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CrucibleWindow:
    window_id: str
    index: int
    train_start: datetime
    train_end: datetime
    embargo_start: datetime
    embargo_end: datetime
    validation_start: datetime
    validation_end: datetime

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_walk_forward_windows(
    *,
    data_start,
    data_end,
    optimization_window_days: int,
    validation_window_days: int,
    embargo_days: int,
    step_days: int | None = None,
    min_windows: int = 1,
) -> list[CrucibleWindow]:
    data_start = _to_dt(data_start)
    data_end = _to_dt(data_end)
    if optimization_window_days <= 0:
        raise ValueError("optimization_window_days must be > 0")
    if validation_window_days <= 0:
        raise ValueError("validation_window_days must be > 0")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")
    step_days = step_days or validation_window_days
    if step_days <= 0:
        raise ValueError("step_days must be > 0")

    windows: list[CrucibleWindow] = []
    train_start = data_start
    idx = 0
    while True:
        train_end = train_start + timedelta(days=optimization_window_days)
        embargo_start = train_end
        embargo_end = embargo_start + timedelta(days=embargo_days)
        validation_start = embargo_end
        validation_end = validation_start + timedelta(days=validation_window_days)
        if validation_end > data_end:
            break
        windows.append(CrucibleWindow(
            window_id=f"window_{idx:04d}",
            index=idx,
            train_start=train_start,
            train_end=train_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            validation_start=validation_start,
            validation_end=validation_end,
        ))
        idx += 1
        train_start = train_start + timedelta(days=step_days)

    if len(windows) < min_windows:
        raise ValueError(f"Only {len(windows)} walk-forward windows generated; min_windows={min_windows}")
    validate_no_leakage(windows, embargo_days=embargo_days)
    return windows


def validate_no_leakage(windows: list[CrucibleWindow], *, embargo_days: int) -> None:
    required_gap = timedelta(days=embargo_days)
    for window in windows:
        if window.train_start >= window.train_end:
            raise ValueError(f"{window.window_id}: train_start must be before train_end")
        if window.validation_start >= window.validation_end:
            raise ValueError(f"{window.window_id}: validation_start must be before validation_end")
        if window.validation_start < window.train_end + required_gap:
            raise ValueError(f"{window.window_id}: validation starts before train_end + embargo")
        if window.embargo_start != window.train_end:
            raise ValueError(f"{window.window_id}: embargo must start at train_end")
        if window.embargo_end != window.validation_start:
            raise ValueError(f"{window.window_id}: validation must start at embargo_end")


def windows_to_rows(windows: list[CrucibleWindow]) -> list[dict[str, Any]]:
    rows = []
    for window in windows:
        row = window.to_dict()
        for key, value in row.items():
            if isinstance(value, datetime):
                row[key] = value.isoformat()
        rows.append(row)
    return rows


def data_range_from_frame(df: pd.DataFrame) -> tuple[datetime, datetime]:
    timestamps = pd.to_datetime(df["timestamp"])
    return timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime()


def _to_dt(value) -> datetime:
    return pd.Timestamp(value).to_pydatetime()
