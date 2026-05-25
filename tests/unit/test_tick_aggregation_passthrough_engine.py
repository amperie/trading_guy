import time
from datetime import datetime, timezone
from unittest.mock import patch

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


def _cfg(downstream, **overrides):
    cfg = {
        "downstream_engine": downstream,
        "state_store": {"enabled": False},
    }
    cfg.update(overrides)
    return cfg


def test_tick_aggregation_passthrough_engine_per_symbol():
    downstream = _Downstream()
    cfg = _cfg(
        downstream,
        aggregation_period_minutes=5,
        use_market_open=True,
        market_open_hour=9,
        market_open_minute=30,
    )
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
    cfg = _cfg(
        downstream,
        aggregation_period_minutes=5,
        use_market_open=True,
        market_open_hour=9,
        market_open_minute=30,
    )
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
    cfg = _cfg(
        downstream,
        aggregation_period_minutes=7,
        use_market_open=True,
        market_open_hour=9,
        market_open_minute=30,
    )
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
    cfg = _cfg(
        downstream,
        aggregation_period_minutes=5,
        use_market_open=True,
        market_open_hour=9,
        market_open_minute=30,
    )
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


# --- Cross-symbol batching tests ---


def _batch_cfg(downstream, **overrides):
    cfg = _cfg(
        downstream,
        aggregation_period_minutes=5,
        use_market_open=True,
        market_open_hour=9,
        market_open_minute=30,
        batch_symbols=True,
        expected_symbols=["AAA", "BBB"],
        batch_timeout_seconds=2.0,
    )
    cfg.update(overrides)
    return cfg


def test_batch_groups_symbols_into_single_downstream_call():
    """Two symbols completing the same window in separate on_tick calls
    get batched into one downstream call."""
    downstream = _Downstream()
    engine = TickAggregationPassthroughEngine(_batch_cfg(downstream))

    # Feed AAA bars for 09:31-09:35
    for i in range(5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("AAA", ts, 100 + i)])

    # AAA window completed but BBB hasn't — no downstream call yet
    assert len(downstream.calls) == 0

    # Feed BBB bars for 09:31-09:35
    for i in range(5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("BBB", ts, 200 + i)])

    # Now both symbols completed the 09:35 window — should flush
    assert len(downstream.calls) == 1
    symbols = sorted([pd.symbol for pd in downstream.calls[0]])
    assert symbols == ["AAA", "BBB"]
    for pd in downstream.calls[0]:
        assert pd.timestamp == datetime(2026, 1, 1, 9, 35, 0)


def test_batch_flushes_on_timeout():
    """If one symbol never arrives, batch flushes after timeout."""
    downstream = _Downstream()
    engine = TickAggregationPassthroughEngine(
        _batch_cfg(downstream, batch_timeout_seconds=1.0)
    )

    # Feed only AAA bars for 09:31-09:35
    for i in range(5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("AAA", ts, 100 + i)])

    # AAA window done, BBB missing — not flushed yet
    assert len(downstream.calls) == 0

    # Simulate timeout by advancing monotonic clock
    real_monotonic = time.monotonic
    with patch("trading.engines.tick_aggregation_passthrough_engine.time") as mock_time:
        mock_time.monotonic.return_value = real_monotonic() + 2.0
        # Next on_tick triggers the timeout check
        engine.on_tick([_pd("AAA", datetime(2026, 1, 1, 9, 36, 0), 110)])

    assert len(downstream.calls) == 1
    assert len(downstream.calls[0]) == 1
    assert downstream.calls[0][0].symbol == "AAA"


def test_batch_immediate_flush_when_all_symbols_present():
    """When all expected symbols arrive in the same on_tick, flush immediately."""
    downstream = _Downstream()
    engine = TickAggregationPassthroughEngine(_batch_cfg(downstream))

    # Feed both symbols together at the boundary
    for i in range(5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("AAA", ts, 100 + i), _pd("BBB", ts, 200 + i)])

    assert len(downstream.calls) == 1
    symbols = sorted([pd.symbol for pd in downstream.calls[0]])
    assert symbols == ["AAA", "BBB"]


def test_batch_backward_compatible_when_disabled():
    """batch_symbols=False preserves original behavior (no batching)."""
    downstream = _Downstream()
    cfg = _batch_cfg(downstream, batch_symbols=False)
    engine = TickAggregationPassthroughEngine(cfg)

    # Feed AAA only for 09:31-09:35 — should flush immediately without waiting for BBB
    for i in range(5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("AAA", ts, 100 + i)])

    assert len(downstream.calls) == 1
    assert downstream.calls[0][0].symbol == "AAA"


def test_batch_inferred_symbols():
    """When expected_symbols is not configured, engine infers from seen symbols."""
    downstream = _Downstream()
    cfg = _batch_cfg(downstream, expected_symbols=[])
    engine = TickAggregationPassthroughEngine(cfg)

    # First tick introduces AAA — engine now expects {AAA}
    engine.on_tick([_pd("AAA", datetime(2026, 1, 1, 9, 31, 0), 100)])

    # Second tick introduces BBB — engine now expects {AAA, BBB}
    engine.on_tick([_pd("BBB", datetime(2026, 1, 1, 9, 31, 0), 200)])

    # Complete AAA window at 09:35
    for i in range(1, 5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("AAA", ts, 100 + i)])

    # AAA done but BBB hasn't completed its window — no flush
    assert len(downstream.calls) == 0

    # Complete BBB window at 09:35
    for i in range(1, 5):
        ts = datetime(2026, 1, 1, 9, 31 + i, 0)
        engine.on_tick([_pd("BBB", ts, 200 + i)])

    # Both done — flush
    assert len(downstream.calls) == 1
    symbols = sorted([pd.symbol for pd in downstream.calls[0]])
    assert symbols == ["AAA", "BBB"]
