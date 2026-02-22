"""Rolling performance tracker for live trading engines.

Tracks portfolio value over a configurable rolling window and provides
annualized return calculations for comparison with HPO results.
"""
from collections import deque
from datetime import datetime, timedelta


class PerformanceTracker:
    """Track rolling portfolio performance over a sliding time window.

    Used by SelfOptimizingLiveEngine to compare current live performance
    against HPO backtest results on the same rolling window.

    Args:
        window_days: Number of days to keep in the rolling window.
    """

    def __init__(self, window_days: int):
        self.window_days = window_days
        self._values: deque[tuple[datetime, float]] = deque()

    def update(self, timestamp: datetime, portfolio_value: float):
        """Record a portfolio value snapshot, pruning old entries.

        Args:
            timestamp: Current timestamp.
            portfolio_value: Current total portfolio value.
        """
        cutoff = timestamp - timedelta(days=self.window_days)
        while self._values and self._values[0][0] < cutoff:
            self._values.popleft()
        self._values.append((timestamp, portfolio_value))

    def annualized_return(self) -> float | None:
        """Calculate annualized return over the current window.

        Returns:
            Annualized return as a decimal (e.g. 0.15 = 15%), or None
            if insufficient data (fewer than 2 data points or zero days).
        """
        if len(self._values) < 2:
            return None
        start_val = self._values[0][1]
        end_val = self._values[-1][1]
        days = (self._values[-1][0] - self._values[0][0]).total_seconds() / 86400
        if days <= 0 or start_val <= 0:
            return None
        total_return = (end_val - start_val) / start_val
        return ((1 + total_return) ** (365.0 / days)) - 1

    @property
    def num_observations(self) -> int:
        """Number of value observations currently in the window."""
        return len(self._values)

    @property
    def latest_value(self) -> float | None:
        """Most recent portfolio value, or None if empty."""
        return self._values[-1][1] if self._values else None
