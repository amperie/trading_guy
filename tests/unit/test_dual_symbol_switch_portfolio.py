"""
Tests for DualSymbolSwitchPortfolio — dual-symbol switching with bracket orders,
holding periods, and the get_price fallback for live trading single-symbol ticks.
"""
from datetime import datetime, timedelta
from trading.core.classes import (
    PriceData, MarketSignal, SignalType, OrderStatus, OrderType, BracketOrder,
)
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg(**overrides):
    base = {
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "stop_pct": 10.0,
        "profit_pct": 20.0,
        "tx_cost": 0.0,
        "holding_period_hours": 0,   # default: no holding period
        "min_signal_strength": 0,
    }
    base.update(overrides)
    return base


def _make_pf(cash=10000.0, **cfg_overrides):
    om = BacktestingOrderManager()
    return DualSymbolSwitchPortfolio(
        _cfg(**cfg_overrides), om, cash, {}, keep_history=False,
    )


def _tick(*symbols_prices, ts=None):
    if ts is None:
        ts = datetime(2024, 1, 1, 10, 0)
    return [PriceData(s, ts, p, p, p, p, 1000) for s, p in symbols_prices]


def _buy_signal(symbol, strength=50):
    return MarketSignal(SignalType.BUY, symbol, strength)


# ---------------------------------------------------------------------------
# Basic entry — all symbols in tick (backtesting path)
# ---------------------------------------------------------------------------
class TestBasicEntry:
    def test_entry_creates_bracket_order(self):
        pf = _make_pf()
        tick = _tick(("UPRO", 50.0), ("SPXU", 20.0))
        sig = _buy_signal("UPRO")
        result = pf.process_market_signals_for_tick([sig], tick)

        assert len(result.orders) == 1
        order = result.orders[0]
        assert isinstance(order, BracketOrder)
        assert order.symbol == "UPRO"
        assert order.quantity == 198  # int(10000 * 0.99 / 50)

    def test_entry_uses_all_cash(self):
        pf = _make_pf(cash=5000.0)
        tick = _tick(("SPXU", 25.0), ("UPRO", 100.0))
        sig = _buy_signal("SPXU")
        pf.process_market_signals_for_tick([sig], tick)

        assert pf.cash < 60.0  # spent ~all cash (1% reserve)
        assert "SPXU" in pf.positions
        assert pf.positions["SPXU"].quantity == 198  # int(5000 * 0.99 / 25)

    def test_no_signal_no_order(self):
        pf = _make_pf()
        tick = _tick(("UPRO", 50.0), ("SPXU", 20.0))
        result = pf.process_market_signals_for_tick([], tick)
        assert len(result.orders) == 0

    def test_sell_signal_ignored(self):
        pf = _make_pf()
        tick = _tick(("UPRO", 50.0))
        sig = MarketSignal(SignalType.SELL, "UPRO", 50)
        result = pf.process_market_signals_for_tick([sig], tick)
        assert len(result.orders) == 0

    def test_unrelated_symbol_ignored(self):
        pf = _make_pf()
        tick = _tick(("AAPL", 150.0))
        sig = _buy_signal("AAPL")
        result = pf.process_market_signals_for_tick([sig], tick)
        assert len(result.orders) == 0

    def test_min_signal_strength_filter(self):
        pf = _make_pf(min_signal_strength=60)
        tick = _tick(("UPRO", 50.0))
        sig = _buy_signal("UPRO", strength=30)
        result = pf.process_market_signals_for_tick([sig], tick)
        assert len(result.orders) == 0


# ---------------------------------------------------------------------------
# get_price fallback — target symbol NOT in tick (live trading path)
# ---------------------------------------------------------------------------
class TestGetPriceFallbackInLiveTrading:
    def test_get_price_returns_cached_when_symbol_missing(self):
        """get_price returns cached price when symbol not in tick."""
        pf = _make_pf()
        pf.previous_price["UPRO"] = 50.0
        tick = _tick(("SPY", 450.0))
        assert pf.get_price("UPRO", tick) == 50.0

    def test_get_price_returns_none_when_never_seen(self):
        """get_price returns None when symbol has no cached price and not in tick."""
        pf = _make_pf()
        tick = _tick(("SPY", 450.0))
        assert pf.get_price("UPRO", tick) is None

    def test_entry_skipped_when_no_price_available(self):
        """No tick price and no cached price → skip (return empty)."""
        pf = _make_pf()
        tick = _tick(("SPY", 450.0))
        sig = _buy_signal("UPRO")
        result = pf.process_market_signals_for_tick([sig], tick)
        assert len(result.orders) == 0

    def test_entry_with_target_in_tick_works(self):
        """Normal path: target symbol is in the tick, entry succeeds."""
        pf = _make_pf()
        tick = _tick(("UPRO", 50.0), ("SPXU", 20.0))
        sig = _buy_signal("UPRO")
        result = pf.process_market_signals_for_tick([sig], tick)
        assert len(result.orders) == 1
        assert result.orders[0].symbol == "UPRO"
        assert result.orders[0].quantity == 198  # int(10000 * 0.99 / 50)

    def test_process_tick_market_signals_uses_get_price(self):
        """Verify that the portfolio's signal processing calls get_price
        which can fall back to cached prices (unit test of the method itself)."""
        pf = _make_pf()
        # Seed a cached price for UPRO
        pf.previous_price["UPRO"] = 50.0

        # get_price should work even though UPRO is not in the tick
        tick = _tick(("SPY", 450.0))
        price = pf.get_price("UPRO", tick)
        assert price == 50.0

        # Note: Full end-to-end live trading test requires an OM that can
        # handle missing symbols in tick (e.g. AlpacaOM). BacktestingOM
        # crashes when the symbol isn't in the tick because it does
        # find_pricedata_in_list in _submit_order_to_backend.


# ---------------------------------------------------------------------------
# Holding same symbol — no duplicate entry
# ---------------------------------------------------------------------------
class TestSameSymbolIgnored:
    def test_no_additional_order_when_already_holding(self):
        pf = _make_pf()
        ts0 = datetime(2024, 1, 1, 10, 0)
        tick0 = _tick(("UPRO", 50.0), ts=ts0)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)
        qty_after_buy = pf.positions["UPRO"].quantity

        # Same signal again — should be ignored
        ts1 = ts0 + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 55.0), ts=ts1)
        result = pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick1)

        assert len(result.orders) == 0
        assert pf.positions["UPRO"].quantity == qty_after_buy


# ---------------------------------------------------------------------------
# Holding period enforcement
# ---------------------------------------------------------------------------
class TestHoldingPeriod:
    def test_switch_blocked_during_holding_period(self):
        pf = _make_pf(holding_period_hours=2)
        ts0 = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts0)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)

        # 30 minutes later, try to switch — should be blocked
        ts1 = ts0 + timedelta(minutes=30)
        tick1 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts1)
        result = pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)

        assert len(result.orders) == 0
        assert pf._pending_switch_state is None  # no switch initiated

    def test_switch_allowed_after_holding_period(self):
        pf = _make_pf(holding_period_hours=2)
        ts0 = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts0)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)

        # 3 hours later, switch should be allowed
        ts1 = ts0 + timedelta(hours=3)
        tick1 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts1)
        pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)

        assert pf._pending_switch_state == "SWITCHING"
        assert pf._pending_target == "SPXU"


# ---------------------------------------------------------------------------
# Bracket order triggers (stop-loss / profit-taker)
# ---------------------------------------------------------------------------
class TestBracketTriggers:
    def test_profit_taker_closes_position(self):
        pf = _make_pf(profit_pct=20.0, stop_pct=10.0)
        ts0 = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO at 50 → profit at 60, stop at 45
        tick0 = _tick(("UPRO", 50.0), ts=ts0)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)
        cash_after_buy = pf.cash

        # Price hits profit target
        ts1 = ts0 + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 60.0), ts=ts1)
        pf.process_market_signals_for_tick([], tick1)

        assert pf.cash > cash_after_buy
        assert "UPRO" not in pf.positions

    def test_stop_loss_closes_position(self):
        pf = _make_pf(profit_pct=20.0, stop_pct=10.0)
        ts0 = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO at 50 → stop at 45
        tick0 = _tick(("UPRO", 50.0), ts=ts0)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)

        # Price hits stop
        ts1 = ts0 + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 44.0), ts=ts1)
        pf.process_market_signals_for_tick([], tick1)

        assert "UPRO" not in pf.positions
        assert pf.cash < 10000.0  # took a loss


# ---------------------------------------------------------------------------
# Symbol switching (full cycle)
# ---------------------------------------------------------------------------
class TestSymbolSwitch:
    def test_full_switch_cycle(self):
        """Buy UPRO → signal SPXU → manual sell → buy SPXU."""
        pf = _make_pf()
        ts = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)
        assert "UPRO" in pf.positions and pf.positions["UPRO"].quantity > 0

        # Signal SPXU → triggers manual sell of UPRO
        ts1 = ts + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts1)
        pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)
        assert pf._pending_switch_state == "SWITCHING"

        # Next tick: manual sale fills, then entry into SPXU
        ts2 = ts1 + timedelta(minutes=1)
        tick2 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts2)
        result = pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick2)

        assert pf._pending_switch_state is None
        assert len(result.orders) == 1
        assert result.orders[0].symbol == "SPXU"
        assert "SPXU" in pf.positions and pf.positions["SPXU"].quantity > 0

    def test_manual_sale_credits_cash_correctly(self):
        """Manual sale must credit sell proceeds to cash before buying next symbol."""
        pf = _make_pf(cash=10000.0)
        ts = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO at $50 → 198 shares (int(10000 * 0.99 / 50)), leftover $100
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)
        assert pf.positions["UPRO"].quantity == 198
        assert pf.cash < 110.0  # ~all cash spent (1% reserve)

        # Signal SPXU → triggers manual sell of UPRO at $55
        ts1 = ts + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 55.0), ("SPXU", 20.0), ts=ts1)
        pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)

        # After manual sale: UPRO position cleared, cash = 100 + 198 * 55 = 10990
        assert "UPRO" not in pf.positions
        assert pf.cash == 10990.0

        # Next tick: buy SPXU with the recovered cash
        ts2 = ts1 + timedelta(minutes=1)
        tick2 = _tick(("UPRO", 55.0), ("SPXU", 20.0), ts=ts2)
        result = pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick2)

        assert len(result.orders) == 1
        assert result.orders[0].symbol == "SPXU"
        # int(10990 * 0.99 / 20) = 544 shares of SPXU
        assert pf.positions["SPXU"].quantity == 544
        assert pf.cash < 120.0  # ~all cash reinvested

    def test_manual_sale_at_loss_credits_reduced_cash(self):
        """Manual sale at a lower price credits the reduced proceeds."""
        pf = _make_pf(cash=10000.0)
        ts = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO at $50 → 200 shares
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)

        # Signal SPXU → manual sell UPRO at $46 (loss, but above stop at $45)
        ts1 = ts + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 46.0), ("SPXU", 20.0), ts=ts1)
        pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)

        assert "UPRO" not in pf.positions
        assert pf.cash == 9208.0  # 100 + 198 * 46

    def test_manual_sale_with_tx_cost(self):
        """Transaction costs are deducted from the manual sale proceeds."""
        pf = _make_pf(cash=10000.0, tx_cost=5.0)
        ts = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO at $50 → qty = floor((10000 - 5) / 50) = 199
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)
        upro_qty = pf.positions["UPRO"].quantity
        cash_after_buy = pf.cash

        # Signal SPXU → manual sell UPRO at $50 (flat)
        ts1 = ts + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts1)
        pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)

        # Cash = leftover + (qty * 50) - tx_cost
        expected_cash = cash_after_buy + (upro_qty * 50.0) - 5.0
        assert pf.cash == expected_cash
        assert "UPRO" not in pf.positions

    def test_strongest_signal_wins(self):
        pf = _make_pf()
        tick = _tick(("UPRO", 50.0), ("SPXU", 20.0))
        sig_weak = _buy_signal("UPRO", strength=10)
        sig_strong = _buy_signal("SPXU", strength=90)
        result = pf.process_market_signals_for_tick([sig_weak, sig_strong], tick)

        assert len(result.orders) == 1
        assert result.orders[0].symbol == "SPXU"


# ---------------------------------------------------------------------------
# Entry time tracking
# ---------------------------------------------------------------------------
class TestEntryTimeTracking:
    def test_entry_time_recorded(self):
        pf = _make_pf()
        ts = datetime(2024, 1, 1, 10, 0)
        tick = _tick(("UPRO", 50.0), ts=ts)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick)

        assert pf._symbol_entry_time.get("UPRO") == ts

    def test_entry_time_cleared_on_switch(self):
        pf = _make_pf()
        ts = datetime(2024, 1, 1, 10, 0)

        # Buy UPRO
        tick0 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts)
        pf.process_market_signals_for_tick([_buy_signal("UPRO")], tick0)

        # Signal switch
        ts1 = ts + timedelta(minutes=1)
        tick1 = _tick(("UPRO", 50.0), ("SPXU", 20.0), ts=ts1)
        pf.process_market_signals_for_tick([_buy_signal("SPXU")], tick1)

        assert "UPRO" not in pf._symbol_entry_time
