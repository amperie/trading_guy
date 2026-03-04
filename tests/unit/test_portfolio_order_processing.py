"""
Tests for Portfolio._update_pf_from_order(), _update_pf_sell(), and
_update_pf_value() error and edge-case paths.

Covers:
  - processed_by_portfolio=True  → skip without any cash/position changes
  - MARKET FILLED BUY/SELL       → correct cash and position updates
  - MARKET CANCELED              → marks processed, no cash change
  - MARKET PENDING (unexpected)  → NotImplementedError
  - BRACKET PENDING_SALE         → initial buy applied; flag reset to False
  - BRACKET FILLED               → exit sell applied from SOLD_ORDER
  - BRACKET PENDING (unexpected) → NotImplementedError
  - STOP_LOSS / PROFIT_TAKER     → NotImplementedError (unhandled at portfolio level)
  - _update_pf_sell: no position           → raises Exception
  - _update_pf_sell: insufficient quantity → raises Exception
  - _update_pf_sell: quantity hits zero    → position entry deleted
  - _update_pf_sell: partial sell          → position quantity reduced
  - _update_pf_sell: cash increased by proceeds minus tx_cost
  - _update_pf_value: held position with no tick price and no cache → ValueError
  - _update_pf_value: cached price used when symbol missing from tick
  - _update_pf_value: all tick symbols are cached in previous_price
"""
import pytest
from datetime import datetime

from trading.core.classes import (
    BracketOrder, Order, OrderAction, OrderStatus, OrderType, Position, PriceData,
)
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio

TS = datetime(2024, 1, 1, 10, 0)


def _pd(symbol, price, ts=None):
    ts = ts or TS
    return PriceData(symbol, ts, price, price, price, price, 1000)


def _make_pf(cash=10000.0):
    om = BacktestingOrderManager()
    pf = SingleSymbolPortfolio(
        {'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0},
        om,
        cash,
    )
    return pf, om


def _inject(om, order):
    """Directly insert an order into the OM's tracking dict, bypassing submit logic."""
    om._all_orders[order.order_id] = order


def _filled_buy(symbol, price, qty, tx_cost=0.0):
    return Order(
        placed_datetime=TS, action=OrderAction.BUY, type=OrderType.MARKET,
        symbol=symbol, price=price, quantity=qty, cash=price * qty,
        tx_cost=tx_cost, status=OrderStatus.FILLED, executed_datetime=TS,
    )


def _filled_sell(symbol, price, qty, tx_cost=0.0):
    return Order(
        placed_datetime=TS, action=OrderAction.SELL, type=OrderType.MARKET,
        symbol=symbol, price=price, quantity=qty, cash=price * qty,
        tx_cost=tx_cost, status=OrderStatus.FILLED, executed_datetime=TS,
    )


# ---------------------------------------------------------------------------
# processed_by_portfolio flag guard
# ---------------------------------------------------------------------------

class TestAlreadyProcessedSkipped:

    def test_already_processed_order_does_not_change_cash(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10)
        order.processed_by_portfolio = True
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert pf.cash == 10000.0

    def test_already_processed_order_does_not_create_position(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10)
        order.processed_by_portfolio = True
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert "AAPL" not in pf.positions


# ---------------------------------------------------------------------------
# MARKET order processing
# ---------------------------------------------------------------------------

class TestMarketOrderProcessing:

    def test_market_filled_buy_reduces_cash_by_cost(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10)  # cost = 500
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert pf.cash == 9500.0

    def test_market_filled_buy_creates_position(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10)
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].quantity == 10

    def test_market_filled_buy_deducts_tx_cost(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10, tx_cost=5.0)  # cost = 500 + 5
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert pf.cash == 10000.0 - 500.0 - 5.0

    def test_market_filled_sell_increases_cash_by_proceeds(self):
        pf, om = _make_pf(cash=5000.0)
        pf.positions["AAPL"] = Position("AAPL", 20)
        order = _filled_sell("AAPL", 60.0, 10)  # proceeds = 600
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert pf.cash == 5600.0

    def test_market_filled_sell_reduces_position_quantity(self):
        pf, om = _make_pf()
        pf.positions["AAPL"] = Position("AAPL", 20)
        order = _filled_sell("AAPL", 60.0, 10)
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert pf.positions["AAPL"].quantity == 10

    def test_market_canceled_marks_processed(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10)
        order.status = OrderStatus.CANCELED
        order.cash = 0.0
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert order.processed_by_portfolio is True

    def test_market_canceled_does_not_change_cash(self):
        pf, om = _make_pf(cash=10000.0)
        order = _filled_buy("AAPL", 50.0, 10)
        order.status = OrderStatus.CANCELED
        order.cash = 0.0
        _inject(om, order)
        pf._update_pf_from_order(order.order_id)
        assert pf.cash == 10000.0

    def test_market_pending_raises_not_implemented(self):
        pf, om = _make_pf()
        order = _filled_buy("AAPL", 50.0, 10)
        order.status = OrderStatus.PENDING
        _inject(om, order)
        with pytest.raises(NotImplementedError):
            pf._update_pf_from_order(order.order_id)


# ---------------------------------------------------------------------------
# BRACKET order processing
# ---------------------------------------------------------------------------

class TestBracketOrderProcessing:

    def _pending_sale_bracket(self, symbol="AAPL", price=50.0, qty=100):
        bo = BracketOrder.create_bracket_order(
            symbol=symbol, high_sell_price=price * 1.2, low_sell_price=price * 0.9,
            quantity=qty, tx_cost=0.0, current_tick=[_pd(symbol, price)]
        )
        bo.status = OrderStatus.PENDING_SALE
        bo.price = price
        bo.quantity = qty
        bo.cash = price * qty
        return bo

    def test_bracket_pending_sale_reduces_cash_by_buy_cost(self):
        pf, om = _make_pf(cash=10000.0)
        bo = self._pending_sale_bracket(price=50.0, qty=100)  # cost = 5000
        _inject(om, bo)
        pf._update_pf_from_order(bo.order_id)
        assert pf.cash == 5000.0

    def test_bracket_pending_sale_creates_position(self):
        pf, om = _make_pf(cash=10000.0)
        bo = self._pending_sale_bracket(price=50.0, qty=100)
        _inject(om, bo)
        pf._update_pf_from_order(bo.order_id)
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].quantity == 100

    def test_bracket_pending_sale_resets_processed_flag_to_false(self):
        """Flag must be False after initial buy so stop/profit can apply the exit later."""
        pf, om = _make_pf(cash=10000.0)
        bo = self._pending_sale_bracket()
        _inject(om, bo)
        pf._update_pf_from_order(bo.order_id)
        assert bo.processed_by_portfolio is False

    def test_bracket_filled_credits_cash_from_sold_order(self):
        pf, om = _make_pf(cash=0.0)
        pf.positions["AAPL"] = Position("AAPL", 100)
        bo = self._pending_sale_bracket(price=50.0, qty=100)
        bo.status = OrderStatus.FILLED
        stop = Order(
            placed_datetime=TS, action=OrderAction.SELL, type=OrderType.STOP_LOSS,
            symbol="AAPL", price=42.0, quantity=100, cash=4200.0,
            status=OrderStatus.FILLED, executed_datetime=TS,
        )
        bo.SOLD_ORDER = stop
        _inject(om, bo)
        pf._update_pf_from_order(bo.order_id)
        assert pf.cash == 4200.0

    def test_bracket_filled_removes_position(self):
        pf, om = _make_pf(cash=0.0)
        pf.positions["AAPL"] = Position("AAPL", 100)
        bo = self._pending_sale_bracket(price=50.0, qty=100)
        bo.status = OrderStatus.FILLED
        stop = Order(
            placed_datetime=TS, action=OrderAction.SELL, type=OrderType.STOP_LOSS,
            symbol="AAPL", price=42.0, quantity=100, cash=4200.0,
            status=OrderStatus.FILLED, executed_datetime=TS,
        )
        bo.SOLD_ORDER = stop
        _inject(om, bo)
        pf._update_pf_from_order(bo.order_id)
        assert "AAPL" not in pf.positions

    def test_bracket_pending_raises_not_implemented(self):
        pf, om = _make_pf()
        bo = self._pending_sale_bracket()
        bo.status = OrderStatus.PENDING  # unexpected state for portfolio
        _inject(om, bo)
        with pytest.raises(NotImplementedError):
            pf._update_pf_from_order(bo.order_id)


# ---------------------------------------------------------------------------
# Unimplemented order types at portfolio level
# ---------------------------------------------------------------------------

class TestUnimplementedOrderTypes:

    def test_stop_loss_type_raises_not_implemented(self):
        pf, om = _make_pf()
        order = Order(
            placed_datetime=TS, action=OrderAction.SELL, type=OrderType.STOP_LOSS,
            symbol="AAPL", price=50.0, quantity=10, cash=500.0,
            status=OrderStatus.FILLED,
        )
        _inject(om, order)
        with pytest.raises(NotImplementedError):
            pf._update_pf_from_order(order.order_id)

    def test_profit_taker_type_raises_not_implemented(self):
        pf, om = _make_pf()
        order = Order(
            placed_datetime=TS, action=OrderAction.SELL, type=OrderType.PROFIT_TAKER,
            symbol="AAPL", price=60.0, quantity=10, cash=600.0,
            status=OrderStatus.FILLED,
        )
        _inject(om, order)
        with pytest.raises(NotImplementedError):
            pf._update_pf_from_order(order.order_id)


# ---------------------------------------------------------------------------
# _update_pf_sell error paths
# ---------------------------------------------------------------------------

class TestUpdatePfSell:

    def test_raises_when_no_position_for_symbol(self):
        pf, om = _make_pf()
        order = _filled_sell("AAPL", 50.0, 10)
        with pytest.raises(Exception):
            pf._update_pf_sell(order)

    def test_raises_when_sell_quantity_exceeds_held(self):
        pf, om = _make_pf()
        pf.positions["AAPL"] = Position("AAPL", 5)
        order = _filled_sell("AAPL", 50.0, 10)  # trying to sell 10, only hold 5
        with pytest.raises(Exception):
            pf._update_pf_sell(order)

    def test_position_deleted_when_quantity_reaches_zero(self):
        pf, om = _make_pf()
        pf.positions["AAPL"] = Position("AAPL", 10)
        order = _filled_sell("AAPL", 50.0, 10)  # sell all 10
        pf._update_pf_sell(order)
        assert "AAPL" not in pf.positions

    def test_partial_sell_leaves_remaining_position(self):
        pf, om = _make_pf()
        pf.positions["AAPL"] = Position("AAPL", 20)
        order = _filled_sell("AAPL", 50.0, 8)
        pf._update_pf_sell(order)
        assert pf.positions["AAPL"].quantity == 12

    def test_cash_increased_by_proceeds_minus_tx_cost(self):
        pf, om = _make_pf(cash=1000.0)
        pf.positions["AAPL"] = Position("AAPL", 10)
        order = _filled_sell("AAPL", 50.0, 10, tx_cost=5.0)
        pf._update_pf_sell(order)
        assert pf.cash == 1000.0 + 500.0 - 5.0

    def test_marks_order_as_processed(self):
        pf, om = _make_pf()
        pf.positions["AAPL"] = Position("AAPL", 10)
        order = _filled_sell("AAPL", 50.0, 10)
        pf._update_pf_sell(order)
        assert order.processed_by_portfolio is True


# ---------------------------------------------------------------------------
# _update_pf_value error paths and caching
# ---------------------------------------------------------------------------

class TestUpdatePfValue:

    def test_raises_when_held_position_has_no_price_and_no_cache(self):
        pf, om = _make_pf()
        pf.positions["AAPL"] = Position("AAPL", 10)
        # Tick contains only SPY — AAPL absent, cache empty
        tick = [_pd("SPY", 450.0)]
        with pytest.raises(ValueError):
            pf._update_pf_value(tick)

    def test_uses_cached_price_when_symbol_missing_from_tick(self):
        pf, om = _make_pf(cash=10000.0)
        pf.positions["AAPL"] = Position("AAPL", 10)
        pf.previous_price["AAPL"] = 55.0  # prime the cache
        tick = [_pd("SPY", 450.0)]  # no AAPL in tick
        value = pf._update_pf_value(tick)
        assert value == 10000.0 + 10 * 55.0

    def test_caches_every_symbol_present_in_tick(self):
        pf, om = _make_pf()
        tick = [_pd("AAPL", 50.0), _pd("SPY", 450.0)]
        pf._update_pf_value(tick)
        assert pf.previous_price.get("AAPL") == 50.0
        assert pf.previous_price.get("SPY") == 450.0

    def test_caches_symbols_not_currently_held(self):
        """Symbols in the tick that are not yet held should still be cached
        (needed so get_price() works for symbols we're about to enter)."""
        pf, om = _make_pf()
        # UPRO in tick but not in positions
        tick = [_pd("UPRO", 80.0)]
        pf._update_pf_value(tick)
        assert pf.previous_price.get("UPRO") == 80.0

    def test_total_value_updated_correctly(self):
        pf, om = _make_pf(cash=5000.0)
        pf.positions["AAPL"] = Position("AAPL", 10)
        tick = [_pd("AAPL", 60.0)]
        value = pf._update_pf_value(tick)
        assert value == 5000.0 + 10 * 60.0
        assert pf.total_value == value
