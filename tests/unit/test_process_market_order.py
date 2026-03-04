"""
Tests for _process_market_order() helper in backtesting_om.py and the
_update_orders_statuses_from_backend() bulk update method.

Covers:
  - BUY: quantity capping by available cash, CANCELED when zero quantity
  - SELL: quantity capping by position, CANCELED when no position
  - Price / cash / timestamp field population
  - Routing through _update_order_status_from_backend for MARKET orders
  - Bulk update: returns only changed order IDs, error validation
"""
from datetime import datetime

import pytest

from trading.core.classes import (
    BracketOrder, Order, OrderAction, OrderStatus, OrderType, Position, PriceData,
)
from trading.core.om.backtesting_om import BacktestingOrderManager, _process_market_order

TS = datetime(2024, 1, 1, 10, 0)


def _pd(symbol, price, ts=None):
    ts = ts or TS
    return PriceData(symbol, ts, price, price, price, price, 1000)


def _market_order(symbol, action, quantity, tx_cost=0.0):
    return Order(
        placed_datetime=TS,
        action=action,
        type=OrderType.MARKET,
        symbol=symbol,
        price=0.0,
        quantity=quantity,
        cash=0.0,
        tx_cost=tx_cost,
    )


def _positions(symbol, qty):
    return {symbol: Position(symbol, qty)}


# ---------------------------------------------------------------------------
# BUY order cases
# ---------------------------------------------------------------------------

class TestProcessMarketOrderBuy:

    def test_buy_filled_with_sufficient_cash(self):
        order = _market_order("AAPL", OrderAction.BUY, 10)
        result = _process_market_order(order, _pd("AAPL", 50.0), 1000.0, 10, {})
        assert result.status == OrderStatus.FILLED
        assert result.quantity == 10

    def test_buy_quantity_capped_by_available_cash(self):
        # cash=500, price=50 → int(500/50)=10; requested 100 → capped to 10
        order = _market_order("AAPL", OrderAction.BUY, 100)
        result = _process_market_order(order, _pd("AAPL", 50.0), 500.0, 100, {})
        assert result.status == OrderStatus.FILLED
        assert result.quantity == 10

    def test_buy_canceled_when_price_exceeds_cash(self):
        # cash=30, price=50 → int(30/50)=0 → CANCELED
        order = _market_order("AAPL", OrderAction.BUY, 10)
        result = _process_market_order(order, _pd("AAPL", 50.0), 30.0, 10, {})
        assert result.status == OrderStatus.CANCELED
        assert result.quantity == 0

    def test_buy_price_set_to_tick_close(self):
        order = _market_order("AAPL", OrderAction.BUY, 5)
        result = _process_market_order(order, _pd("AAPL", 42.0), 1000.0, 5, {})
        assert result.price == 42.0

    def test_buy_cash_is_price_times_quantity(self):
        order = _market_order("AAPL", OrderAction.BUY, 5)
        result = _process_market_order(order, _pd("AAPL", 42.0), 1000.0, 5, {})
        assert result.cash == 5 * 42.0

    def test_buy_cash_computed_from_capped_quantity(self):
        # cash=500, price=50, requested=100 → qty=10, cash should be 10*50=500
        order = _market_order("AAPL", OrderAction.BUY, 100)
        result = _process_market_order(order, _pd("AAPL", 50.0), 500.0, 100, {})
        assert result.cash == result.quantity * 50.0

    def test_buy_executed_datetime_set_to_tick_timestamp(self):
        ts = datetime(2024, 6, 1, 9, 35)
        order = _market_order("AAPL", OrderAction.BUY, 5)
        result = _process_market_order(order, _pd("AAPL", 42.0, ts), 1000.0, 5, {})
        assert result.executed_datetime == ts

    def test_buy_placed_datetime_set_to_tick_timestamp(self):
        ts = datetime(2024, 6, 1, 9, 35)
        order = _market_order("AAPL", OrderAction.BUY, 5)
        result = _process_market_order(order, _pd("AAPL", 42.0, ts), 1000.0, 5, {})
        assert result.placed_datetime == ts

    def test_buy_returns_order_unchanged_when_pd_is_none(self):
        order = _market_order("AAPL", OrderAction.BUY, 10)
        result = _process_market_order(order, None, 1000.0, 10, {})
        assert result.status == OrderStatus.PENDING  # unchanged


# ---------------------------------------------------------------------------
# SELL order cases
# ---------------------------------------------------------------------------

class TestProcessMarketOrderSell:

    def test_sell_filled_with_sufficient_position(self):
        order = _market_order("AAPL", OrderAction.SELL, 10)
        result = _process_market_order(order, _pd("AAPL", 50.0), 0.0, 10, _positions("AAPL", 50))
        assert result.status == OrderStatus.FILLED
        assert result.quantity == 10

    def test_sell_quantity_capped_by_position(self):
        # hold 30, requested 100 → tx_quantity = min(100, 100, 30) = 30
        order = _market_order("AAPL", OrderAction.SELL, 100)
        result = _process_market_order(order, _pd("AAPL", 50.0), 0.0, 100, _positions("AAPL", 30))
        assert result.status == OrderStatus.FILLED
        assert result.quantity == 30

    def test_sell_canceled_when_no_position(self):
        # symbol not in positions → position_quantity=0 → tx_quantity=0 → CANCELED
        order = _market_order("AAPL", OrderAction.SELL, 10)
        result = _process_market_order(order, _pd("AAPL", 50.0), 0.0, 10, {})
        assert result.status == OrderStatus.CANCELED
        assert result.quantity == 0

    def test_sell_cash_reflects_capped_quantity(self):
        order = _market_order("AAPL", OrderAction.SELL, 100)
        result = _process_market_order(order, _pd("AAPL", 50.0), 0.0, 100, _positions("AAPL", 30))
        assert result.cash == 30 * 50.0

    def test_sell_price_set_to_tick_close(self):
        order = _market_order("AAPL", OrderAction.SELL, 10)
        result = _process_market_order(order, _pd("AAPL", 77.5), 0.0, 10, _positions("AAPL", 50))
        assert result.price == 77.5

    def test_sell_executed_datetime_set_to_tick_timestamp(self):
        ts = datetime(2024, 6, 1, 9, 40)
        order = _market_order("AAPL", OrderAction.SELL, 10)
        result = _process_market_order(order, _pd("AAPL", 50.0, ts), 0.0, 10, _positions("AAPL", 50))
        assert result.executed_datetime == ts

    def test_sell_quantity_also_capped_by_quantity_param(self):
        # quantity_param=5 < position=50 and < order.quantity=10 → tx_quantity=5
        order = _market_order("AAPL", OrderAction.SELL, 10)
        result = _process_market_order(order, _pd("AAPL", 50.0), 0.0, 5, _positions("AAPL", 50))
        assert result.quantity == 5
        assert result.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# MARKET order routing via _update_order_status_from_backend
# ---------------------------------------------------------------------------

class TestMarketOrderViaUpdateStatus:

    def test_market_buy_routes_through_update_status(self):
        om = BacktestingOrderManager()
        order = _market_order("AAPL", OrderAction.BUY, 10)
        result = om._update_order_status_from_backend(order, [_pd("AAPL", 50.0)], {}, 1000.0)
        assert result.status == OrderStatus.FILLED
        assert result.price == 50.0

    def test_market_sell_routes_through_update_status(self):
        om = BacktestingOrderManager()
        order = _market_order("AAPL", OrderAction.SELL, 5)
        tick = [_pd("AAPL", 50.0)]
        result = om._update_order_status_from_backend(order, tick, _positions("AAPL", 20), 0.0)
        assert result.status == OrderStatus.FILLED
        assert result.quantity == 5

    def test_already_filled_order_returned_unchanged(self):
        om = BacktestingOrderManager()
        order = _market_order("AAPL", OrderAction.BUY, 10)
        order.status = OrderStatus.FILLED
        order.price = 99.0
        result = om._update_order_status_from_backend(order, [_pd("AAPL", 50.0)], {}, 1000.0)
        assert result.price == 99.0  # not overwritten

    def test_market_order_stays_pending_when_symbol_not_in_tick(self):
        om = BacktestingOrderManager()
        order = _market_order("AAPL", OrderAction.BUY, 10)
        tick = [_pd("SPY", 450.0)]  # no AAPL
        result = om._update_order_status_from_backend(order, tick, {}, 1000.0)
        assert result.status == OrderStatus.PENDING

    def test_update_status_raises_when_tick_is_none(self):
        om = BacktestingOrderManager()
        order = _market_order("AAPL", OrderAction.BUY, 10)
        with pytest.raises(ValueError):
            om._update_order_status_from_backend(order, None, {}, 1000.0)


# ---------------------------------------------------------------------------
# Bulk update: _update_orders_statuses_from_backend
# ---------------------------------------------------------------------------

class TestBulkOrderStatusUpdate:

    def _pending_sale_bracket(self, symbol, entry, stop, profit, qty=10):
        """Return a bracket already in PENDING_SALE (initial buy done)."""
        bo = BracketOrder.create_bracket_order(
            symbol=symbol, high_sell_price=profit, low_sell_price=stop,
            quantity=qty, tx_cost=0.0, current_tick=[_pd(symbol, entry)]
        )
        bo.status = OrderStatus.PENDING_SALE
        bo.price = entry
        bo.quantity = qty
        bo.cash = entry * qty
        return bo

    def test_only_changed_order_ids_returned(self):
        om = BacktestingOrderManager()
        b1 = self._pending_sale_bracket("AAPL", 50.0, 40.0, 60.0)
        b2 = self._pending_sale_bracket("GOOG", 100.0, 90.0, 110.0)
        orders = {b1.order_id: b1, b2.order_id: b2}
        # AAPL drops below stop, GOOG stays flat
        tick = [_pd("AAPL", 38.0), _pd("GOOG", 100.0)]
        positions = {"AAPL": Position("AAPL", 10), "GOOG": Position("GOOG", 10)}
        changed = om._update_orders_statuses_from_backend(orders, tick, positions, 0.0)
        assert b1.order_id in changed
        assert b2.order_id not in changed

    def test_unchanged_orders_not_in_return(self):
        om = BacktestingOrderManager()
        b1 = self._pending_sale_bracket("AAPL", 50.0, 40.0, 60.0)
        orders = {b1.order_id: b1}
        tick = [_pd("AAPL", 50.0)]  # price in range, no trigger
        changed = om._update_orders_statuses_from_backend(
            orders, tick, {"AAPL": Position("AAPL", 10)}, 0.0
        )
        assert changed == []

    def test_empty_orders_returns_empty_list(self):
        om = BacktestingOrderManager()
        changed = om._update_orders_statuses_from_backend({}, [_pd("AAPL", 50.0)], {}, 0.0)
        assert changed == []

    def test_both_changed_when_both_trigger(self):
        om = BacktestingOrderManager()
        b1 = self._pending_sale_bracket("AAPL", 50.0, 40.0, 60.0)
        b2 = self._pending_sale_bracket("GOOG", 100.0, 90.0, 110.0)
        orders = {b1.order_id: b1, b2.order_id: b2}
        tick = [_pd("AAPL", 62.0), _pd("GOOG", 112.0)]  # both hit profit
        positions = {"AAPL": Position("AAPL", 10), "GOOG": Position("GOOG", 10)}
        changed = om._update_orders_statuses_from_backend(orders, tick, positions, 0.0)
        assert b1.order_id in changed
        assert b2.order_id in changed
        assert len(changed) == 2

    def test_raises_when_tick_is_none(self):
        om = BacktestingOrderManager()
        with pytest.raises(ValueError):
            om._update_orders_statuses_from_backend({}, None, {}, 0.0)

    def test_raises_when_positions_is_none(self):
        om = BacktestingOrderManager()
        with pytest.raises(ValueError):
            om._update_orders_statuses_from_backend({}, [_pd("AAPL", 50.0)], None, 0.0)
