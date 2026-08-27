from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from math import ceil
from math import sqrt

from trading.core.classes import PriceData


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    symbol: str
    timestamp: object
    bars_seen: int
    close: float
    trend_regime: str
    volatility_regime: str
    composite_regime: str
    trend_return: float | None
    trend_strength: float | None
    realized_volatility: float | None
    annualized_volatility: float | None
    volatility_percentile: float | None
    drawdown: float | None
    distance_from_baseline: float | None
    inferred_bar_minutes: float | None
    bars_per_day: float | None
    is_ready: bool

    def to_dict(self) -> dict:
        return asdict(self)


class MarketRegimeDetector:
    """Online market-regime classifier that only uses bars seen so far."""

    default_cfg = {
        "trend_lookback_days": 50.0,
        "trend_threshold": 0.03,
        "baseline_ma_window_days": 200.0,
        "volatility_lookback_days": 50.0,
        "volatility_percentile_window_days": 252.0,
        "low_vol_percentile": 30.0,
        "high_vol_percentile": 70.0,
        "drawdown_lookback_days": 252.0,
        "annualization_days": 252.0,
        "market_hours_per_day": 6.5,
        "default_bar_minutes": 1.0,
        "require_full_windows": True,
    }

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**self.default_cfg, **(cfg or {})}
        self._default_bars_per_day = self._bars_per_day_from_minutes(float(self.cfg["default_bar_minutes"]))
        self._max_history = max(
            self._configured_bars("trend_lookback", "trend_lookback_days", "trend_lookback_hours") + 1,
            self._configured_bars("baseline_ma_window", "baseline_ma_window_days", "baseline_ma_window_hours"),
            self._configured_bars("volatility_lookback", "volatility_lookback_days", "volatility_lookback_hours") + 1,
            self._configured_bars("drawdown_lookback", "drawdown_lookback_days", "drawdown_lookback_hours"),
        )
        self._bars: dict[str, deque[PriceData]] = defaultdict(lambda: deque(maxlen=self._max_history))
        self._bar_minutes: dict[str, float] = {}
        self._vol_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._configured_bars(
                "volatility_percentile_window",
                "volatility_percentile_window_days",
                "volatility_percentile_window_hours",
            ))
        )
        self._snapshots: dict[str, MarketRegimeSnapshot] = {}

    @property
    def required_bars(self) -> int:
        if not self.cfg["require_full_windows"]:
            return 2
        return max(
            self._configured_bars("trend_lookback", "trend_lookback_days", "trend_lookback_hours") + 1,
            self._configured_bars("baseline_ma_window", "baseline_ma_window_days", "baseline_ma_window_hours"),
            self._configured_bars("volatility_lookback", "volatility_lookback_days", "volatility_lookback_hours") + 1,
        )

    def update(self, tick: list[PriceData]) -> dict[str, MarketRegimeSnapshot]:
        updated = {}
        for bar in tick:
            bars = self._bars[bar.symbol]
            if bars:
                self._update_bar_minutes(bar.symbol, bars[-1].timestamp, bar.timestamp)
            bars.append(bar)
            snapshot = self._classify(bar.symbol)
            self._snapshots[bar.symbol] = snapshot
            updated[bar.symbol] = snapshot
        return updated

    def get(self, symbol: str) -> MarketRegimeSnapshot | None:
        return self._snapshots.get(symbol)

    def snapshot(self) -> dict[str, dict]:
        return {symbol: snap.to_dict() for symbol, snap in self._snapshots.items()}

    def _classify(self, symbol: str) -> MarketRegimeSnapshot:
        bars = self._bars[symbol]
        current = bars[-1]
        closes = [bar.close for bar in bars]
        bars_per_day = self._current_bars_per_day(symbol)
        trend_lookback = self._configured_bars("trend_lookback", "trend_lookback_days", "trend_lookback_hours", bars_per_day)
        baseline_window = self._configured_bars("baseline_ma_window", "baseline_ma_window_days", "baseline_ma_window_hours", bars_per_day)
        volatility_lookback = self._configured_bars("volatility_lookback", "volatility_lookback_days", "volatility_lookback_hours", bars_per_day)
        volatility_percentile_window = self._configured_bars(
            "volatility_percentile_window",
            "volatility_percentile_window_days",
            "volatility_percentile_window_hours",
            bars_per_day,
        )
        drawdown_lookback = self._configured_bars("drawdown_lookback", "drawdown_lookback_days", "drawdown_lookback_hours", bars_per_day)
        returns = self._returns(closes)
        realized_vol = self._realized_volatility(returns, volatility_lookback)
        vol_percentile = None
        if realized_vol is not None:
            vol_hist = self._vol_history[symbol]
            vol_hist.append(realized_vol)
            vol_percentile = self._percentile_rank(list(vol_hist)[-volatility_percentile_window:], realized_vol)

        trend_return = self._trend_return(closes, trend_lookback)
        baseline_distance = self._distance_from_ma(closes, baseline_window)
        drawdown = self._drawdown(closes, drawdown_lookback)
        ready = len(bars) >= self._required_bars(bars_per_day)
        trend_regime, trend_strength = self._trend_regime(trend_return, baseline_distance, ready)
        vol_regime = self._volatility_regime(vol_percentile, ready)
        annualization_bars = float(self.cfg["annualization_days"]) * bars_per_day
        annualized_vol = None if realized_vol is None else realized_vol * sqrt(annualization_bars)

        return MarketRegimeSnapshot(
            symbol=symbol,
            timestamp=current.timestamp,
            bars_seen=len(bars),
            close=current.close,
            trend_regime=trend_regime,
            volatility_regime=vol_regime,
            composite_regime=f"{trend_regime}_{vol_regime}",
            trend_return=trend_return,
            trend_strength=trend_strength,
            realized_volatility=realized_vol,
            annualized_volatility=annualized_vol,
            volatility_percentile=vol_percentile,
            drawdown=drawdown,
            distance_from_baseline=baseline_distance,
            inferred_bar_minutes=self._bar_minutes.get(symbol),
            bars_per_day=bars_per_day,
            is_ready=ready,
        )

    def _required_bars(self, bars_per_day: float) -> int:
        if not self.cfg["require_full_windows"]:
            return 2
        return max(
            self._configured_bars("trend_lookback", "trend_lookback_days", "trend_lookback_hours", bars_per_day) + 1,
            self._configured_bars("baseline_ma_window", "baseline_ma_window_days", "baseline_ma_window_hours", bars_per_day),
            self._configured_bars("volatility_lookback", "volatility_lookback_days", "volatility_lookback_hours", bars_per_day) + 1,
        )

    def _configured_bars(
        self,
        bars_key: str,
        days_key: str,
        hours_key: str,
        bars_per_day: float | None = None,
    ) -> int:
        if bars_key in self.cfg:
            return max(1, int(self.cfg[bars_key]))
        if bars_per_day is None:
            bars_per_day = self._default_bars_per_day
        if hours_key in self.cfg:
            hours_per_day = float(self.cfg["market_hours_per_day"])
            return max(1, ceil(float(self.cfg[hours_key]) / hours_per_day * bars_per_day))
        return max(1, ceil(float(self.cfg[days_key]) * bars_per_day))

    def _update_bar_minutes(self, symbol: str, previous: object, current: object) -> None:
        if not isinstance(previous, datetime) or not isinstance(current, datetime):
            return
        minutes = (current - previous).total_seconds() / 60.0
        if minutes <= 0:
            return
        self._bar_minutes[symbol] = minutes

    def _current_bars_per_day(self, symbol: str) -> float:
        return self._bars_per_day_from_minutes(self._bar_minutes.get(symbol, float(self.cfg["default_bar_minutes"])))

    def _bars_per_day_from_minutes(self, minutes: float) -> float:
        if minutes >= 18 * 60:
            return 1.0
        return max(1.0, (float(self.cfg["market_hours_per_day"]) * 60.0) / max(minutes, 1e-9))

    def _trend_regime(
        self,
        trend_return: float | None,
        baseline_distance: float | None,
        ready: bool,
    ) -> tuple[str, float | None]:
        if not ready or trend_return is None:
            return "UNKNOWN", None
        threshold = float(self.cfg["trend_threshold"])
        if trend_return >= threshold and (baseline_distance is None or baseline_distance >= 0):
            return "UPTREND", trend_return / threshold
        if trend_return <= -threshold and (baseline_distance is None or baseline_distance <= 0):
            return "DOWNTREND", abs(trend_return) / threshold
        return "RANGE", abs(trend_return) / threshold if threshold else 0.0

    def _volatility_regime(self, percentile: float | None, ready: bool) -> str:
        if not ready or percentile is None:
            return "UNKNOWN"
        if percentile <= float(self.cfg["low_vol_percentile"]):
            return "LOW_VOL"
        if percentile >= float(self.cfg["high_vol_percentile"]):
            return "HIGH_VOL"
        return "NORMAL_VOL"

    @staticmethod
    def _returns(closes: list[float]) -> list[float]:
        return [
            (current / previous) - 1.0
            for previous, current in zip(closes, closes[1:])
            if previous
        ]

    @staticmethod
    def _realized_volatility(returns: list[float], lookback: int) -> float | None:
        if len(returns) < lookback:
            return None
        sample = returns[-lookback:]
        mean = sum(sample) / len(sample)
        variance = sum((ret - mean) ** 2 for ret in sample) / len(sample)
        return variance ** 0.5

    @staticmethod
    def _trend_return(closes: list[float], lookback: int) -> float | None:
        if len(closes) < lookback + 1 or not closes[-lookback - 1]:
            return None
        return (closes[-1] / closes[-lookback - 1]) - 1.0

    @staticmethod
    def _distance_from_ma(closes: list[float], window: int) -> float | None:
        if len(closes) < window:
            return None
        ma = sum(closes[-window:]) / window
        if ma == 0:
            return None
        return (closes[-1] / ma) - 1.0

    @staticmethod
    def _drawdown(closes: list[float], lookback: int) -> float | None:
        sample = closes[-lookback:]
        if not sample:
            return None
        peak = max(sample)
        if peak == 0:
            return None
        return (sample[-1] / peak) - 1.0

    @staticmethod
    def _percentile_rank(values: list[float], value: float) -> float | None:
        if not values:
            return None
        below = sum(1 for item in values if item < value)
        equal = sum(1 for item in values if item == value)
        return 100.0 * (below + 0.5 * equal) / len(values)


def classify_ticks(
    ticks: list[list[PriceData]],
    cfg: dict | None = None,
) -> list[dict[str, MarketRegimeSnapshot]]:
    detector = MarketRegimeDetector(cfg)
    return [detector.update(tick) for tick in ticks]
