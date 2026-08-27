from __future__ import annotations

from datetime import datetime

import pytest

from algo_crucible.windows import CrucibleWindow, generate_walk_forward_windows, validate_no_leakage, windows_to_rows


def test_generate_walk_forward_windows_with_embargo():
    windows = generate_walk_forward_windows(
        data_start=datetime(2024, 1, 1),
        data_end=datetime(2024, 5, 1),
        optimization_window_days=30,
        validation_window_days=10,
        embargo_days=5,
        step_days=20,
        min_windows=3,
    )

    assert len(windows) == 4
    assert windows[0].train_end == datetime(2024, 1, 31)
    assert windows[0].validation_start == datetime(2024, 2, 5)
    assert all(window.validation_start >= window.train_end for window in windows)


def test_generate_walk_forward_windows_enforces_min_windows():
    with pytest.raises(ValueError, match="min_windows"):
        generate_walk_forward_windows(
            data_start=datetime(2024, 1, 1),
            data_end=datetime(2024, 2, 1),
            optimization_window_days=30,
            validation_window_days=10,
            embargo_days=5,
            min_windows=1,
        )


def test_validate_no_leakage_rejects_bad_embargo_boundary():
    bad = CrucibleWindow(
        window_id="window_bad",
        index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 2, 1),
        embargo_start=datetime(2024, 2, 1),
        embargo_end=datetime(2024, 2, 6),
        validation_start=datetime(2024, 2, 5),
        validation_end=datetime(2024, 2, 20),
    )

    with pytest.raises(ValueError, match="train_end \\+ embargo"):
        validate_no_leakage([bad], embargo_days=5)


def test_windows_to_rows_serializes_datetimes():
    windows = generate_walk_forward_windows(
        data_start=datetime(2024, 1, 1),
        data_end=datetime(2024, 3, 15),
        optimization_window_days=30,
        validation_window_days=10,
        embargo_days=5,
    )

    rows = windows_to_rows(windows)

    assert rows[0]["window_id"] == "window_0000"
    assert rows[0]["train_start"] == "2024-01-01T00:00:00"
