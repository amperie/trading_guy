from datetime import datetime, timezone

import pytest

from trading.core.classes import PriceData, TickResults
from trading.engines.tick_aggregation_passthrough_engine import TickAggregationPassthroughEngine


class _Downstream:
    def __init__(self):
        self.calls = []

    def on_tick(self, tick):
        self.calls.append(tick)
        return TickResults(orders=[])


def _pd(symbol: str, ts: datetime, price: float, volume: float = 1.0):
    return PriceData(
        symbol=symbol,
        timestamp=ts,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price + 0.1,
        volume=volume,
    )


def test_tick_aggregation_passthrough_engine_per_symbol():
    downstream = _Downstream()
    cfg = {
        "downstream_engine": downstream,
        "aggregation_period_minutes": 5,
        "use_market_open": True,
        "market_open_hour": 9,
        "market_open_minute": 30,
    }
    engine = TickAggregationPassthroughEngine(cfg)

    # Minutes 09:31 -> 09:35 should aggregate into a bar ending at 09:35.
    base = datetime(2026, 1, 1, 9, 31, 0)
    for i in range(5):
        ts = base.replace(minute=31 + i)
        tick = [_pd("AAA", ts, 100 + i, volume=10 + i)]
        # Only include BBB for the first three minutes
        if i < 3:
            tick.append(_pd("BBB", ts, 200 + i, volume=5 + i))
        result = engine.on_tick(tick)

        # Only emit at 09:35 for AAA, never for BBB (missing at boundary)
        if ts.minute < 35:
            assert result.orders == []
            assert len(downstream.calls) == 0

    assert len(downstream.calls) == 1
    aggregated_tick = downstream.calls[0]
    assert len(aggregated_tick) == 1
    agg = aggregated_tick[0]
    assert agg.symbol == "AAA"
    assert agg.timestamp == datetime(2026, 1, 1, 9, 35, 0)
    assert agg.open == 100
    assert agg.high == 104 + 0.5
    assert agg.low == 100 - 0.5
    assert agg.close == 104 + 0.1
    assert agg.volume == sum(10 + i for i in range(5))


def test_tick_aggregation_emits_on_exact_boundary():
    downstream = _Downstream()
    cfg = {
        "downstream_engine": downstream,
        "aggregation_period_minutes": 5,
        "use_market_open": True,
        "market_open_hour": 9,
        "market_open_minute": 30,
    }
    engine = TickAggregationPassthroughEngine(cfg)

    # Start exactly on boundary 09:30 and should emit at 09:30 immediately.
    ts = datetime(2026, 1, 1, 9, 30, 0)
    engine.on_tick([_pd("AAA", ts, 50, volume=2)])

    assert len(downstream.calls) == 1
    aggregated_tick = downstream.calls[0]
    assert len(aggregated_tick) == 1
    agg = aggregated_tick[0]
    assert agg.timestamp == ts
    assert agg.open == 50
    assert agg.high == 50 + 0.5
    assert agg.low == 50 - 0.5
    assert agg.close == 50 + 0.1
    assert agg.volume == 2


def test_tick_aggregation_respects_timezone_alignment():
    downstream = _Downstream()
    cfg = {
        "downstream_engine": downstream,
        "aggregation_period_minutes": 7,
        "use_market_open": True,
        "market_open_hour": 9,
        "market_open_minute": 30,
    }
    engine = TickAggregationPassthroughEngine(cfg)

    # Use timezone-aware timestamps; boundary should preserve tzinfo
    base = datetime(2026, 1, 1, 9, 31, 0, tzinfo=timezone.utc)
    for i in range(7):
        ts = base.replace(minute=31 + i)
        engine.on_tick([_pd("AAA", ts, 10 + i)])

    assert len(downstream.calls) == 1
    agg = downstream.calls[0][0]
    assert agg.timestamp.tzinfo == timezone.utc
    # 7-min windows aligned to market open (09:30) end at 09:37, 09:44, ...
    assert agg.timestamp == datetime(2026, 1, 1, 9, 37, 0, tzinfo=timezone.utc)


def test_tick_aggregation_skips_empty_windows():
    downstream = _Downstream()
    cfg = {
        "downstream_engine": downstream,
        "aggregation_period_minutes": 5,
        "use_market_open": True,
        "market_open_hour": 9,
        "market_open_minute": 30,
    }
    engine = TickAggregationPassthroughEngine(cfg)

    # AAA appears, then no AAA for next window; should not emit empty bar.
    engine.on_tick([_pd("AAA", datetime(2026, 1, 1, 9, 31, 0), 10)])
    engine.on_tick([_pd("AAA", datetime(2026, 1, 1, 9, 35, 0), 11)])
    assert len(downstream.calls) == 1

    # Next window has only BBB; AAA should not emit.
    engine.on_tick([_pd("BBB", datetime(2026, 1, 1, 9, 36, 0), 20)])
    engine.on_tick([_pd("BBB", datetime(2026, 1, 1, 9, 40, 0), 21)])

    assert len(downstream.calls) == 2
    symbols = [b.symbol for b in downstream.calls[1]]
    assert symbols == ["BBB"]
