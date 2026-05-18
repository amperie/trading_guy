from __future__ import annotations

from datetime import datetime, timedelta

from trading.core.classes import PriceData
from trading.promoted.volatility_filtered_rsi_mean_reversion_backtest_c5f0894c.VolatilityFilteredRsiMeanReversionAlgorithm import (
    VolatilityFilteredRsiMeanReversionAlgorithm,
)


def _build_bar(ts: datetime, close: float) -> PriceData:
    return PriceData(
        symbol="SPY",
        timestamp=ts,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000,
    )


def test_promoted_algo_coerces_float_period_params():
    algo = VolatilityFilteredRsiMeanReversionAlgorithm(
        {
            "symbol": "SPY",
            "regime_detection": {
                "ma_short_period": 50.7,
                "ma_long_period": 200.2,
                "ma_proximity_tolerance": 0.02,
                "atr_period": 14.1,
                "atr_percentile_window": 20.4,
                "atr_percentile_level": 49.6,
            },
            "rsi_config": {
                "rsi_period": 13.9,
                "rsi_oversold_threshold": 29.7,
                "rsi_overbought_threshold": 69.8,
            },
            "price_confirmation": {"reversal_bar_lookback": 2.2},
        },
        history_length=260,
    )

    ts = datetime(2026, 1, 1, 9, 30)
    for i in range(260):
        algo.on_data([_build_bar(ts + timedelta(minutes=i), 100.0 + (i % 7) * 0.1)])

    assert isinstance(algo.ma_short_period, int)
    assert isinstance(algo.ma_long_period, int)
    assert isinstance(algo.atr_period, int)
    assert isinstance(algo.atr_percentile_window, int)
    assert isinstance(algo.atr_percentile_level, int)
    assert isinstance(algo.rsi_period, int)
    assert isinstance(algo.rsi_oversold_threshold, int)
    assert isinstance(algo.rsi_overbought_threshold, int)
    assert isinstance(algo.reversal_bar_lookback, int)

