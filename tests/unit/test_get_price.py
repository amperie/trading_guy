"""
Tests for Portfolio.get_price() — the tick-or-cache price lookup method.
"""
from datetime import datetime
from trading.core.classes import PriceData, MarketSignal, SignalType
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio


def _make_tick(*symbols_prices, ts=None):
    """Helper: build a tick list from (symbol, price) pairs."""
    if ts is None:
        ts = datetime(2024, 1, 1, 10, 0)
    return [
        PriceData(sym, ts, p, p, p, p, 100)
        for sym, p in symbols_prices
    ]


def _make_pf(cash=10000.0):
    om = BacktestingOrderManager()
    return SingleSymbolPortfolio(
        {"symbol": "AAPL", "stop_pct": 10.0, "profit_pct": 20.0},
        om, cash, {}, False,
    )


class TestGetPriceFromTick:
    """get_price returns close from the current tick when symbol is present."""

    def test_symbol_in_tick(self):
        pf = _make_pf()
        tick = _make_tick(("AAPL", 150.0))
        assert pf.get_price("AAPL", tick) == 150.0

    def test_correct_symbol_selected(self):
        pf = _make_pf()
        tick = _make_tick(("AAPL", 150.0), ("MSFT", 400.0))
        assert pf.get_price("MSFT", tick) == 400.0

    def test_returns_close_not_other_ohlc(self):
        pf = _make_pf()
        ts = datetime(2024, 1, 1, 10, 0)
        tick = [PriceData("AAPL", ts, 100.0, 110.0, 90.0, 105.0, 100)]
        assert pf.get_price("AAPL", tick) == 105.0


class TestGetPriceFallback:
    """get_price falls back to previous_price when symbol missing from tick."""

    def test_fallback_to_previous_price(self):
        pf = _make_pf()
        pf.previous_price["UPRO"] = 55.0
        tick = _make_tick(("SPY", 450.0))  # UPRO not in tick
        assert pf.get_price("UPRO", tick) == 55.0

    def test_returns_none_when_never_seen(self):
        pf = _make_pf()
        tick = _make_tick(("SPY", 450.0))
        assert pf.get_price("UNKNOWN", tick) is None

    def test_tick_price_preferred_over_cache(self):
        """When symbol is in both tick and cache, tick wins."""
        pf = _make_pf()
        pf.previous_price["AAPL"] = 100.0  # stale
        tick = _make_tick(("AAPL", 200.0))  # fresh
        assert pf.get_price("AAPL", tick) == 200.0

    def test_empty_tick_uses_cache(self):
        pf = _make_pf()
        pf.previous_price["AAPL"] = 99.0
        assert pf.get_price("AAPL", []) == 99.0

    def test_empty_tick_no_cache_returns_none(self):
        pf = _make_pf()
        assert pf.get_price("AAPL", []) is None


class TestGetPriceCachePopulatedByTicks:
    """previous_price dict is populated by processing ticks (via _update_pf_value)."""

    def test_previous_price_populated_after_position_tick(self):
        pf = _make_pf()
        tick = _make_tick(("AAPL", 100.0))
        signal = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal], tick)

        # After buying AAPL, _update_pf_value should cache the price
        assert pf.previous_price.get("AAPL") == 100.0

    def test_cache_updates_on_subsequent_ticks(self):
        pf = _make_pf()
        # Buy first
        tick1 = _make_tick(("AAPL", 100.0), ts=datetime(2024, 1, 1, 10, 0))
        signal = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal], tick1)

        # New tick with different price
        tick2 = _make_tick(("AAPL", 110.0), ts=datetime(2024, 1, 1, 10, 1))
        pf.process_market_signals_for_tick([], tick2)

        assert pf.previous_price["AAPL"] == 110.0

    def test_cache_survives_missing_tick(self):
        """If a position's symbol disappears from tick, cache retains last price."""
        pf = _make_pf()
        # Buy AAPL
        tick1 = _make_tick(("AAPL", 100.0), ts=datetime(2024, 1, 1, 10, 0))
        signal = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal], tick1)

        # Next tick has only SPY (not AAPL) — _update_pf_value uses cache
        tick2 = _make_tick(("SPY", 450.0), ts=datetime(2024, 1, 1, 10, 1))
        pf.process_market_signals_for_tick([], tick2)

        # Portfolio should still value AAPL at 100
        assert pf.previous_price["AAPL"] == 100.0
        expected_value = pf.cash + pf.positions["AAPL"].quantity * 100.0
        assert abs(pf.total_value - expected_value) < 0.01
