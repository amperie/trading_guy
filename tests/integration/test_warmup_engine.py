"""
Integration tests for the engine warm-up pipeline.

Tests the full warm-up flow: engine reads warmup config, instantiates a
DataProvider, iterates it, and pre-populates the algorithm's history.
Uses mocks for the Alpaca-specific components to avoid API calls.
"""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, TickResults
from trading.data_providers.data_provider import DataProvider


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_tick(symbol: str, close: float, minute: int = 0) -> list[PriceData]:
    return [PriceData(
        symbol=symbol,
        timestamp=datetime(2026, 2, 17, 9, 30 + minute),
        open=close - 1, high=close + 1, low=close - 2,
        close=close, volume=1000,
    )]


class DummyAlgorithm(Algorithm):
    def __init__(self, history_length: int = 0):
        super().__init__(
            cfg={"history_length": history_length},
            history_length=history_length,
        )
        self.on_data_calls = []

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        self.on_data_calls.append(data)
        return []


class FakeDataProvider(DataProvider):
    """DataProvider that yields pre-built ticks without file/API access."""

    def __init__(self, cfg: dict = None):
        # Bypass ConfigManager entirely
        self.cfg = cfg or {}
        self.data = None
        self._tick_count = self.cfg.get("limit", 10)
        self._symbol = self.cfg.get("symbols", ["FAKE"])[0]

    def load_data(self):
        pass

    def iterate(self):
        for i in range(self._tick_count):
            yield make_tick(self._symbol, 100.0 + i, i)


# ---------------------------------------------------------------------------
# AlpacaRealTimeEngine warmup config logic tests
#
# These tests exercise the warmup block from _connect() by extracting the
# same logic and running it synchronously, avoiding the need for
# pytest-asyncio or a running event loop.
# ---------------------------------------------------------------------------

def _run_warmup_logic(cfg: dict, algo: Algorithm):
    """Reproduce the warmup block from AlpacaRealTimeEngine._connect().

    This mirrors the exact logic in _connect() so we can test it
    without async infrastructure or Alpaca SDK imports.
    """
    from utils.utils import instantiate_from_string
    from trading.engines.base_engine import BaseEngine

    warmup_cfg = cfg.get("warmup", None)
    if warmup_cfg is not None and algo is not None:
        warmup_cfg = dict(warmup_cfg)
        provider_path = warmup_cfg.pop("provider")
        if "limit" not in warmup_cfg and algo.history_length > 0:
            warmup_cfg["limit"] = algo.history_length
        dp = instantiate_from_string(provider_path, cfg=warmup_cfg)

        # Use BaseEngine.warm_up as an unbound call via a stub
        ticks = list(dp.iterate())
        algo.warm_up(ticks)


@pytest.mark.integration
class TestAlpacaEngineWarmUpConfig:
    """Test the warmup config parsing and limit auto-detection logic."""

    def test_warmup_with_explicit_limit(self):
        algo = DummyAlgorithm(history_length=20)
        cfg = {
            "warmup": {
                "provider": "tests.integration.test_warmup_engine.FakeDataProvider",
                "symbols": ["AAPL"],
                "limit": 7,
            }
        }
        _run_warmup_logic(cfg, algo)

        assert len(algo.price_history["AAPL"]) == 7
        assert algo.on_data_calls == []

    def test_warmup_auto_limit_from_history_length(self):
        algo = DummyAlgorithm(history_length=5)
        cfg = {
            "warmup": {
                "provider": "tests.integration.test_warmup_engine.FakeDataProvider",
                "symbols": ["SPY"],
                # no "limit" — should auto-set to history_length=5
            }
        }
        _run_warmup_logic(cfg, algo)

        assert len(algo.price_history["SPY"]) == 5
        assert algo.on_data_calls == []

    def test_warmup_explicit_limit_overrides_history_length(self):
        algo = DummyAlgorithm(history_length=20)
        cfg = {
            "warmup": {
                "provider": "tests.integration.test_warmup_engine.FakeDataProvider",
                "symbols": ["AAPL"],
                "limit": 3,
            }
        }
        _run_warmup_logic(cfg, algo)

        assert len(algo.price_history["AAPL"]) == 3

    def test_no_warmup_config_skips_warmup(self):
        algo = DummyAlgorithm(history_length=10)
        cfg = {}
        _run_warmup_logic(cfg, algo)

        assert len(algo.price_history) == 0

    def test_warmup_does_not_mutate_original_config(self):
        warmup_section = {
            "provider": "tests.integration.test_warmup_engine.FakeDataProvider",
            "symbols": ["AAPL"],
            "limit": 5,
        }
        cfg = {"warmup": warmup_section}
        algo = DummyAlgorithm(history_length=10)

        _run_warmup_logic(cfg, algo)

        # Original dict should be untouched
        assert "provider" in warmup_section

    def test_warmup_then_live_tick_continues_history(self):
        algo = DummyAlgorithm(history_length=10)
        cfg = {
            "warmup": {
                "provider": "tests.integration.test_warmup_engine.FakeDataProvider",
                "symbols": ["AAPL"],
                "limit": 3,
            }
        }
        _run_warmup_logic(cfg, algo)

        assert len(algo.price_history["AAPL"]) == 3

        # Simulate a live tick (minute=50 to stay within 0..59)
        live_tick = make_tick("AAPL", 999.0, minute=20)
        algo.on_data(live_tick)

        assert len(algo.price_history["AAPL"]) == 4
        assert algo.price_history["AAPL"][-1] == 999.0
        assert len(algo.on_data_calls) == 1

    def test_warmup_with_history_length_zero_no_auto_limit(self):
        """When history_length=0, limit is not auto-set."""
        algo = DummyAlgorithm(history_length=0)
        cfg = {
            "warmup": {
                "provider": "tests.integration.test_warmup_engine.FakeDataProvider",
                "symbols": ["AAPL"],
                # no limit, history_length=0 => no auto-limit, default=10
            }
        }
        _run_warmup_logic(cfg, algo)

        # FakeDataProvider defaults to 10 ticks, but history_length=0
        # means nothing is stored
        assert len(algo.price_history) == 0


# ---------------------------------------------------------------------------
# AlpacaDataProvider.load_data parameter passing tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAlpacaDataProviderLimit:
    """Test that AlpacaDataProvider passes limit/start/end correctly."""

    def test_load_data_with_limit_only(self):
        from trading.data_providers.alpaca_data_provider import AlpacaDataProvider

        with patch.object(AlpacaDataProvider, "__init__", lambda self, cfg: None):
            dp = AlpacaDataProvider.__new__(AlpacaDataProvider)
            dp.cfg = {"limit": 50, "symbols": ["AAPL"]}
            dp.data = None

            with patch.object(dp, "fetch_bars", return_value="mock_df") as mock_fetch:
                dp.load_data()
                mock_fetch.assert_called_once_with(start=None, end=None, limit=50)

    def test_load_data_with_all_params(self):
        from trading.data_providers.alpaca_data_provider import AlpacaDataProvider

        with patch.object(AlpacaDataProvider, "__init__", lambda self, cfg: None):
            dp = AlpacaDataProvider.__new__(AlpacaDataProvider)
            dp.cfg = {
                "start_date": "2026-02-14",
                "end_date": "2026-02-17",
                "limit": 100,
                "symbols": ["AAPL"],
            }
            dp.data = None

            with patch.object(dp, "fetch_bars", return_value="mock_df") as mock_fetch:
                dp.load_data()
                args = mock_fetch.call_args
                assert args.kwargs["limit"] == 100
                assert args.kwargs["start"] is not None
                assert args.kwargs["end"] is not None

    def test_load_data_with_no_optional_params(self):
        from trading.data_providers.alpaca_data_provider import AlpacaDataProvider

        with patch.object(AlpacaDataProvider, "__init__", lambda self, cfg: None):
            dp = AlpacaDataProvider.__new__(AlpacaDataProvider)
            dp.cfg = {"symbols": ["AAPL"]}
            dp.data = None

            with patch.object(dp, "fetch_bars", return_value="mock_df") as mock_fetch:
                dp.load_data()
                mock_fetch.assert_called_once_with(start=None, end=None, limit=None)

    def test_fetch_bars_passes_limit_to_request(self):
        """fetch_bars includes limit in the StockBarsRequest."""
        from trading.data_providers.alpaca_data_provider import AlpacaDataProvider

        import pandas as pd

        with patch("trading.data_providers.alpaca_data_provider.StockHistoricalDataClient"):
            dp = AlpacaDataProvider.__new__(AlpacaDataProvider)
            dp.symbols = ["AAPL"]
            dp.timeframe = MagicMock()
            dp.adjustment = "split"
            dp.client = MagicMock()

            mock_df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2026-02-17 14:30:00+00:00"]),
                "symbol": ["AAPL"],
                "open": [150.0], "high": [151.0], "low": [149.0],
                "close": [150.5], "volume": [1000],
                "trade_count": [10], "vwap": [150.2],
            })
            mock_bars = MagicMock()
            mock_bars.df = mock_df.set_index(["symbol", "timestamp"])
            dp.client.get_stock_bars.return_value = mock_bars

            with patch("trading.data_providers.alpaca_data_provider.StockBarsRequest") as MockReq:
                dp.fetch_bars(limit=50)
                MockReq.assert_called_once()
                call_kwargs = MockReq.call_args.kwargs
                assert call_kwargs["limit"] == 50
                assert call_kwargs["start"] is None
                assert call_kwargs["end"] is None
