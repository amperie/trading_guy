"""
Unit tests for the algorithm and engine warm-up functionality.

Tests that Algorithm.warm_up() correctly pre-populates price_history
and price_data_history without generating signals, and that
BaseEngine.warm_up() correctly delegates to the algorithm.
"""
import pytest
from datetime import datetime
from collections import deque

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType, TickResults
from trading.engines.base_engine import BaseEngine
from trading.data_providers.data_provider import DataProvider


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_tick(symbol: str, close: float, minute: int = 0) -> list[PriceData]:
    """Create a single-symbol tick with the given close price."""
    return [PriceData(
        symbol=symbol,
        timestamp=datetime(2026, 2, 17, 9, 30 + minute),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000,
    )]


def make_multi_tick(prices: dict[str, float], minute: int = 0) -> list[PriceData]:
    """Create a multi-symbol tick. prices = {"AAPL": 150.0, "SPY": 500.0}."""
    return [
        PriceData(
            symbol=sym,
            timestamp=datetime(2026, 2, 17, 9, 30 + minute),
            open=close - 1, high=close + 1, low=close - 2,
            close=close, volume=1000,
        )
        for sym, close in prices.items()
    ]


class DummyAlgorithm(Algorithm):
    """Concrete algorithm that records every on_data_logic call."""

    def __init__(self, history_length: int = 0, full_history: bool = False):
        super().__init__(
            cfg={"history_length": history_length, "full_history": full_history},
            history_length=history_length,
        )
        self.on_data_calls = []

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        self.on_data_calls.append(data)
        return []


class StubDataProvider(DataProvider):
    """DataProvider backed by an in-memory list of ticks."""

    def __init__(self, ticks: list[list[PriceData]]):
        # Bypass the config-file-reading super().__init__
        self.cfg = {}
        self.data = None
        self._ticks = ticks

    def load_data(self):
        # iterate() is overridden so this is unused, but required by ABC
        pass

    def iterate(self):
        for tick in self._ticks:
            yield tick


class StubEngine(BaseEngine):
    """Minimal concrete engine for testing warm_up."""

    def run(self):
        pass

    def on_tick(self, tick):
        pass

    def finalize(self):
        pass


# ---------------------------------------------------------------------------
# Algorithm.warm_up tests
# ---------------------------------------------------------------------------

class TestAlgorithmWarmUp:

    def test_warm_up_populates_price_history(self):
        algo = DummyAlgorithm(history_length=10)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(5)]
        algo.warm_up(ticks)

        assert len(algo.price_history["AAPL"]) == 5
        assert list(algo.price_history["AAPL"]) == [100, 101, 102, 103, 104]

    def test_warm_up_populates_price_data_history(self):
        algo = DummyAlgorithm(history_length=10)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(3)]
        algo.warm_up(ticks)

        assert len(algo.price_data_history["AAPL"]) == 3
        assert all(
            isinstance(pd, PriceData)
            for pd in algo.price_data_history["AAPL"]
        )

    def test_warm_up_does_not_call_on_data_logic(self):
        algo = DummyAlgorithm(history_length=10)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(5)]
        algo.warm_up(ticks)

        assert algo.on_data_calls == []

    def test_warm_up_respects_deque_maxlen(self):
        algo = DummyAlgorithm(history_length=5)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(10)]
        algo.warm_up(ticks)

        assert len(algo.price_history["AAPL"]) == 5
        # Deque should keep the last 5 values
        assert list(algo.price_history["AAPL"]) == [105, 106, 107, 108, 109]

    def test_warm_up_with_multiple_symbols(self):
        algo = DummyAlgorithm(history_length=10)
        ticks = [
            make_multi_tick({"AAPL": 150 + i, "SPY": 500 + i}, i)
            for i in range(4)
        ]
        algo.warm_up(ticks)

        assert len(algo.price_history["AAPL"]) == 4
        assert len(algo.price_history["SPY"]) == 4
        assert list(algo.price_history["AAPL"]) == [150, 151, 152, 153]
        assert list(algo.price_history["SPY"]) == [500, 501, 502, 503]

    def test_warm_up_with_empty_ticks(self):
        algo = DummyAlgorithm(history_length=10)
        algo.warm_up([])

        assert len(algo.price_history) == 0

    def test_warm_up_with_zero_history_length(self):
        """When history_length is 0, warm_up runs but history stays empty."""
        algo = DummyAlgorithm(history_length=0)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(5)]
        algo.warm_up(ticks)

        # price_history is a plain dict (not defaultdict) when history_length=0
        assert len(algo.price_history) == 0

    def test_warm_up_then_on_data_continues_history(self):
        """After warm_up, on_data should append to the same history."""
        algo = DummyAlgorithm(history_length=10)

        warmup_ticks = [make_tick("AAPL", 100 + i, i) for i in range(3)]
        algo.warm_up(warmup_ticks)
        assert len(algo.price_history["AAPL"]) == 3

        # Now simulate a live tick via on_data
        live_tick = make_tick("AAPL", 200, minute=10)
        algo.on_data(live_tick)

        assert len(algo.price_history["AAPL"]) == 4
        assert list(algo.price_history["AAPL"]) == [100, 101, 102, 200]
        # on_data_logic should have been called once (for the live tick)
        assert len(algo.on_data_calls) == 1

    def test_warm_up_populates_full_history_when_enabled(self):
        algo = DummyAlgorithm(history_length=10, full_history=True)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(3)]
        algo.warm_up(ticks)

        assert len(algo.full_history) == 3

    def test_warm_up_does_not_populate_full_history_when_disabled(self):
        algo = DummyAlgorithm(history_length=10, full_history=False)
        ticks = [make_tick("AAPL", 100 + i, i) for i in range(3)]
        algo.warm_up(ticks)

        assert len(algo.full_history) == 0


# ---------------------------------------------------------------------------
# BaseEngine.warm_up tests
# ---------------------------------------------------------------------------

class TestBaseEngineWarmUp:

    def test_engine_warm_up_delegates_to_algorithm(self):
        algo = DummyAlgorithm(history_length=10)
        engine = StubEngine(al=algo)

        ticks = [make_tick("AAPL", 100 + i, i) for i in range(5)]
        dp = StubDataProvider(ticks)
        engine.warm_up(dp)

        assert len(algo.price_history["AAPL"]) == 5
        assert algo.on_data_calls == []

    def test_engine_warm_up_with_no_algorithm(self):
        """warm_up should not raise when algorithm is None."""
        engine = StubEngine(al=None)
        ticks = [make_tick("AAPL", 100, 0)]
        dp = StubDataProvider(ticks)
        # Should log a warning but not raise
        engine.warm_up(dp)

    def test_engine_warm_up_iterates_full_provider(self):
        algo = DummyAlgorithm(history_length=100)
        engine = StubEngine(al=algo)

        ticks = [make_tick("SPY", 500 + i, i) for i in range(20)]
        dp = StubDataProvider(ticks)
        engine.warm_up(dp)

        assert len(algo.price_history["SPY"]) == 20
