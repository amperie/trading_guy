from dataclasses import asdict
from datetime import timedelta

import numpy as np
import pytest

from trading.analysis.market_regime import MarketRegimeDetector, classify_ticks
from tests.unit.test_market_regime import bar


@pytest.mark.parametrize('kind', ['random', 'constant', 'zeros', 'gaps', 'repeating'])
@pytest.mark.parametrize('full', [True, False])
def test_batch_matches_online(kind, full):
    cfg = dict(trend_lookback_days=.03, baseline_ma_window_days=.05,
        volatility_lookback_days=.02, volatility_percentile_window_days=.08,
        drawdown_lookback_days=.1, require_full_windows=full)
    prices = 100*np.exp(np.random.default_rng(13).normal(0, .01, (250, 2)).cumsum(axis=0))
    if kind == 'constant':
        prices[:] = 100
    if kind == 'zeros':
        prices[::19] = 0
    if kind == 'repeating':
        prices[:] = (100+np.arange(250) % 5)[:, None]
    ticks = [[bar(float(price), i, str(symbol)) for symbol, price in enumerate(row)] for i, row in enumerate(prices)]
    if kind == 'gaps':
        for i, tick in enumerate(ticks):
            for item in tick:
                item.timestamp += timedelta(days=i//50)
    detector = MarketRegimeDetector(cfg)
    expected = [detector.update(tick) for tick in ticks]
    actual = classify_ticks(ticks, cfg)
    for left, right in zip(expected, actual):
        for symbol in left:
            a, b = asdict(left[symbol]), asdict(right[symbol])
            for key in a:
                if isinstance(a[key], float):
                    assert b[key] == pytest.approx(a[key], rel=1e-8, abs=1e-10), key
                else:
                    assert b[key] == a[key], key


def test_batch_does_not_look_ahead():
    ticks = [[bar(100+i % 9, i)] for i in range(90)]
    cfg = dict(volatility_lookback=5, volatility_percentile_window=20, baseline_ma_window=5, trend_lookback=5)
    assert classify_ticks(ticks[:40], cfg) == classify_ticks(ticks, cfg)[:40]


@pytest.mark.parametrize('period', [5, 10, 20])
def test_near_tied_variances_preserve_regime_labels(period):
    cfg = dict(volatility_lookback=period, volatility_percentile_window=20, baseline_ma_window=5, trend_lookback=5)
    ticks = [[bar(100+i % 5, i)] for i in range(300)]
    detector = MarketRegimeDetector(cfg)
    expected = [detector.update(tick)['SPY'].composite_regime for tick in ticks]
    assert [item['SPY'].composite_regime for item in classify_ticks(ticks, cfg)] == expected
