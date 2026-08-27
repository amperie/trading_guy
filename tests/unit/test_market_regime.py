from datetime import datetime, timedelta

from trading.analysis.market_regime import MarketRegimeDetector, classify_ticks
from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData
from trading.engines.base_engine import BaseEngine


def bar(close: float, index: int = 0, symbol: str = "SPY") -> PriceData:
    return PriceData(
        symbol=symbol,
        timestamp=datetime(2026, 1, 1, 9, 30) + timedelta(minutes=index),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
    )


class NoSignalAlgorithm(Algorithm):
    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        return []


class IndicatorAlgorithm(NoSignalAlgorithm):
    def get_indicator_snapshot(self, data: list[PriceData] | None = None) -> dict | None:
        return {"custom": 1}


class StubEngine(BaseEngine):
    def run(self):
        pass

    def on_tick(self, tick):
        pass

    def finalize(self):
        pass


def detector_cfg() -> dict:
    return {
        "trend_lookback_hours": 3 / 60,
        "trend_threshold": 0.02,
        "baseline_ma_window_hours": 3 / 60,
        "volatility_lookback_hours": 3 / 60,
        "volatility_percentile_window_hours": 5 / 60,
        "drawdown_lookback_hours": 5 / 60,
        "default_bar_minutes": 1,
        "require_full_windows": True,
    }


def test_detector_uses_only_seen_bars_for_current_snapshot():
    detector = MarketRegimeDetector(detector_cfg())
    snapshots = []
    for idx, close in enumerate([100, 101, 102, 103, 80]):
        snapshots.append(detector.update([bar(close, idx)])["SPY"])

    assert snapshots[2].is_ready is False
    assert snapshots[3].trend_regime == "UPTREND"
    assert snapshots[3].close == 103
    assert snapshots[4].trend_regime == "DOWNTREND"


def test_algorithm_updates_regime_before_strategy_logic():
    class RegimeReadingAlgorithm(NoSignalAlgorithm):
        def __init__(self):
            super().__init__(
                cfg={"market_regime": {"enabled": True, **detector_cfg()}},
                history_length=0,
            )
            self.seen = []

        def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
            snapshot = self.get_regime("SPY")
            self.seen.append(None if snapshot is None else snapshot.close)
            return []

    algo = RegimeReadingAlgorithm()
    for idx, close in enumerate([100, 101, 102, 103]):
        algo.on_data([bar(close, idx)])

    assert algo.seen == [100, 101, 102, 103]
    assert algo.get_regime("SPY").trend_regime == "UPTREND"


def test_engine_persistence_merges_indicator_and_regime_snapshots():
    algo = IndicatorAlgorithm(
        cfg={"market_regime": {"enabled": True, **detector_cfg()}},
        history_length=0,
    )
    for idx, close in enumerate([100, 101, 102, 103]):
        algo.on_data([bar(close, idx)])

    snapshot = StubEngine(al=algo)._get_indicator_snapshot_for_persistence([bar(103, 3)])

    assert snapshot["custom"] == 1
    assert snapshot["market_regime"]["SPY"]["trend_regime"] == "UPTREND"


def test_classify_ticks_runs_detector_over_historical_ticks():
    snapshots = classify_ticks([[bar(close, idx)] for idx, close in enumerate([100, 101, 102, 103])], detector_cfg())

    assert len(snapshots) == 4
    assert snapshots[-1]["SPY"].trend_regime == "UPTREND"


def test_duration_config_accounts_for_data_granularity():
    detector = MarketRegimeDetector({
        "trend_lookback_hours": 1,
        "trend_threshold": 0.02,
        "baseline_ma_window_hours": 1,
        "volatility_lookback_hours": 1,
        "volatility_percentile_window_hours": 2,
        "drawdown_lookback_hours": 2,
        "default_bar_minutes": 1,
    })

    five_minute_bars = [
        PriceData("SPY", datetime(2026, 1, 1, 9, 30) + timedelta(minutes=5 * idx), close, close, close, close, 1000)
        for idx, close in enumerate(range(100, 114))
    ]
    snapshots = [detector.update([item])["SPY"] for item in five_minute_bars]

    assert snapshots[12].is_ready is True
    assert snapshots[12].bars_per_day == 78
    assert snapshots[12].trend_regime == "UPTREND"
