"""Tests for MultiTimeframeAlgorithm base class."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading.core.classes import MarketSignal, PriceData, SignalType
from trading.core.multi_timeframe_algorithm import MultiTimeframeAlgorithm

# ── Helpers ───────────────────────────────────────────────────────────────────

# Market opens at 9:30 on 2026-01-02.
OPEN = datetime(2026, 1, 2, 9, 30)


def _bar(symbol: str, ts: datetime, close: float = 100.0, volume: float = 100.0) -> PriceData:
    return PriceData(
        symbol=symbol, timestamp=ts,
        open=close, high=close + 0.5, low=close - 0.5,
        close=close, volume=volume,
    )


def _at(minutes: int, symbol: str = "SPY", close: float = 100.0) -> list[PriceData]:
    return [_bar(symbol, OPEN + timedelta(minutes=minutes), close=close)]


# ── Minimal concrete subclass ─────────────────────────────────────────────────

class RecordingMTFAlgo(MultiTimeframeAlgorithm):
    """Records every on_mtf_data call; emits no signals."""

    def __init__(self, cfg=None, history_length: int = 0):
        super().__init__(cfg, history_length)
        self.calls: list[tuple[list[PriceData], dict[int, list[PriceData]]]] = []

    def on_mtf_data(self, tick, new_bars):
        self.calls.append((tick, dict(new_bars)))
        return []


def _warmed(cfg: dict, history_length: int = 10) -> RecordingMTFAlgo:
    """Return a RecordingMTFAlgo with the warmup gate already open."""
    algo = RecordingMTFAlgo(cfg=cfg, history_length=history_length)
    algo._ticks_seen = algo.required_warmup_bars
    return algo


# ── Init and config ───────────────────────────────────────────────────────────

def test_raises_if_timeframes_missing():
    with pytest.raises(ValueError, match="timeframes"):
        RecordingMTFAlgo(cfg={"history_length": 10})


def test_raises_if_timeframes_empty():
    with pytest.raises(ValueError, match="timeframes"):
        RecordingMTFAlgo(cfg={"timeframes": [], "history_length": 10})


def test_timeframes_are_sorted():
    algo = RecordingMTFAlgo(cfg={"timeframes": [60, 5, 15], "history_length": 10})
    assert algo.timeframes == [5, 15, 60]


def test_bar_history_has_key_per_timeframe():
    algo = RecordingMTFAlgo(cfg={"timeframes": [5, 15], "history_length": 10})
    assert set(algo.bar_history.keys()) == {5, 15}


# ── Warmup gate ───────────────────────────────────────────────────────────────

def test_required_warmup_bars_is_history_length_times_max_timeframe():
    algo = RecordingMTFAlgo(cfg={"timeframes": [5, 60]}, history_length=200)
    assert algo.required_warmup_bars == 200 * 60


def test_required_warmup_bars_zero_history_length():
    algo = RecordingMTFAlgo(cfg={"timeframes": [5, 60]}, history_length=0)
    assert algo.required_warmup_bars == 0


def test_required_warmup_bars_override_respected():
    algo = RecordingMTFAlgo(cfg={"timeframes": [5, 60], "history_length": 10})
    algo._required_warmup_bars_override = 42
    assert algo.required_warmup_bars == 42


def test_warmup_gate_suppresses_signals():
    class BullishAlgo(MultiTimeframeAlgorithm):
        def on_mtf_data(self, tick, new_bars):
            return [MarketSignal(SignalType.BUY, "SPY", 80)]

    algo = BullishAlgo(cfg={"timeframes": [5]}, history_length=10)
    # required_warmup_bars = 10 * 5 = 50; only 1 tick seen → gate still closed
    signals = algo.on_data(_at(0))
    assert signals == []


def test_signals_emitted_once_warmed_up():
    class BullishAlgo(MultiTimeframeAlgorithm):
        def on_mtf_data(self, tick, new_bars):
            return [MarketSignal(SignalType.BUY, "SPY", 80)]

    algo = BullishAlgo(cfg={"timeframes": [5], "history_length": 10})
    algo._ticks_seen = algo.required_warmup_bars
    signals = algo.on_data(_at(0))
    assert len(signals) == 1


# ── Bar completion: single timeframe ─────────────────────────────────────────

def test_boundary_tick_flushes_immediately():
    """A tick at exactly 9:30 on a 5-min boundary is its own bar."""
    algo = _warmed({"timeframes": [5]})
    algo.on_data(_at(0))  # 9:30 — boundary
    _, new_bars = algo.calls[-1]
    assert 5 in new_bars
    assert new_bars[5][0].timestamp == OPEN


def test_no_bar_mid_window():
    algo = _warmed({"timeframes": [5]})
    algo.on_data(_at(1))  # 9:31 — window ends at 9:35, not done
    _, new_bars = algo.calls[-1]
    assert 5 not in new_bars


def test_5min_bar_completes_at_window_end():
    """Ticks 9:31–9:35 produce one 5-min bar timestamped at 9:35."""
    algo = _warmed({"timeframes": [5]})
    for i in range(1, 6):
        algo.on_data(_at(i))
    _, new_bars = algo.calls[-1]
    assert 5 in new_bars
    assert new_bars[5][0].timestamp == OPEN + timedelta(minutes=5)


def test_bar_not_in_new_bars_between_windows():
    """After 9:35 flushes, ticks in the next window do not produce a bar until 9:40."""
    algo = _warmed({"timeframes": [5]})
    for i in range(1, 8):  # up to 9:37
        algo.on_data(_at(i))
    _, new_bars = algo.calls[-1]  # last tick was 9:37
    assert 5 not in new_bars


def test_gap_in_ticks_flushes_stale_window():
    """If the data stream skips past a window end, the stale window is flushed."""
    algo = _warmed({"timeframes": [5]})
    algo.on_data(_at(1))   # 9:31, window ends at 9:35
    algo.on_data(_at(10))  # 9:40 — skips past 9:35, should flush the 9:31 bar
    _, new_bars = algo.calls[-1]
    assert 5 in new_bars
    assert new_bars[5][0].timestamp == OPEN + timedelta(minutes=5)  # bar ts = 9:35


# ── Bar completion: multiple timeframes ──────────────────────────────────────

def test_two_timeframes_complete_independently():
    """5-min bars fire more often than 15-min bars over a 17-tick window."""
    algo = _warmed({"timeframes": [5, 15]})
    completed: dict[int, list[datetime]] = {5: [], 15: []}

    for i in range(17):  # 9:30 … 9:46
        algo.on_data(_at(i))
        _, new_bars = algo.calls[-1]
        for tf in [5, 15]:
            if tf in new_bars:
                completed[tf].append(new_bars[tf][0].timestamp)

    # 5-min bars: 9:30, 9:35, 9:40, 9:45
    assert len(completed[5]) == 4
    # 15-min bars: 9:30, 9:45
    assert len(completed[15]) == 2


def test_only_completing_timeframes_appear_in_new_bars():
    """When only the 5-min bar completes, the 15-min key must be absent."""
    algo = _warmed({"timeframes": [5, 15]})
    for i in range(1, 6):
        algo.on_data(_at(i))  # 9:35 — 5-min fires, 15-min does not
    _, new_bars = algo.calls[-1]
    assert 5 in new_bars
    assert 15 not in new_bars


# ── OHLCV accumulation ────────────────────────────────────────────────────────

def test_ohlcv_accumulation():
    """Verify open=first, high=max, low=min, close=last, volume=sum."""
    algo = _warmed({"timeframes": [5]})
    # Feed ticks 9:31–9:35 with distinct prices.
    closes = [100.0, 105.0, 95.0, 102.0, 98.0]
    for i, c in enumerate(closes, start=1):
        algo.on_data([_bar("SPY", OPEN + timedelta(minutes=i), close=c, volume=10.0)])

    bar = algo.bar_history[5]["SPY"][-1]
    assert bar.open == 100.0                      # open of first raw tick
    assert bar.close == 98.0                      # close of last raw tick
    assert bar.high == pytest.approx(105.5)       # max(close + 0.5) = 105.5
    assert bar.low == pytest.approx(94.5)         # min(close - 0.5) = 94.5
    assert bar.volume == pytest.approx(50.0)      # 5 × 10.0


# ── bar_history accumulation and maxlen ──────────────────────────────────────

def test_bar_history_grows_as_bars_complete():
    algo = _warmed({"timeframes": [5]}, history_length=10)
    for i in range(11):  # 9:30..9:40 — produces 3 bars (9:30, 9:35, 9:40)
        algo.on_data(_at(i))
    assert len(algo.bar_history[5]["SPY"]) == 3


def test_bar_history_respects_maxlen():
    """With history_length=2, only the 2 most recent bars are kept."""
    algo = _warmed({"timeframes": [5]}, history_length=2)
    for i in range(16):  # 9:30..9:45 — produces 4 bars
        algo.on_data(_at(i))
    assert len(algo.bar_history[5]["SPY"]) == 2


def test_bar_history_stores_pricedata_objects():
    algo = _warmed({"timeframes": [5]})
    algo.on_data(_at(0))  # boundary → immediate flush
    bar = algo.bar_history[5]["SPY"][-1]
    assert isinstance(bar, PriceData)
    assert bar.symbol == "SPY"


# ── Multiple symbols ──────────────────────────────────────────────────────────

def test_multiple_symbols_aggregated_independently():
    """Two symbols in the same tick each produce their own completed bar."""
    algo = _warmed({"timeframes": [5]})
    tick = [
        _bar("SPY", OPEN, close=400.0),
        _bar("QQQ", OPEN, close=300.0),
    ]
    algo.on_data(tick)
    _, new_bars = algo.calls[-1]
    assert 5 in new_bars
    symbols = {b.symbol for b in new_bars[5]}
    assert symbols == {"SPY", "QQQ"}


def test_symbols_tracked_separately_in_bar_history():
    algo = _warmed({"timeframes": [5]})
    tick = [_bar("SPY", OPEN, close=400.0), _bar("QQQ", OPEN, close=300.0)]
    algo.on_data(tick)
    assert algo.bar_history[5]["SPY"][-1].close == 400.0
    assert algo.bar_history[5]["QQQ"][-1].close == 300.0


# ── Raw tick passthrough ──────────────────────────────────────────────────────

def test_on_mtf_data_receives_original_tick():
    """The raw tick passed to on_mtf_data is the same object as what on_data received."""
    algo = _warmed({"timeframes": [5]})
    raw = _at(0)
    algo.on_data(raw)
    received_tick, _ = algo.calls[-1]
    assert received_tick is raw
