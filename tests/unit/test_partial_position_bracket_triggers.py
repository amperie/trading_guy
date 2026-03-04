"""
Tests for bracket stop-loss and profit-taker triggers when the actual portfolio
position is smaller than the bracket order quantity.

When a bracket was created for N shares but the portfolio now holds fewer (e.g.
due to partial fills, manual sells, or live-trading reconciliation), the OM must
cap tx_quantity to what is actually held.

  stop-loss branch:  tx_quantity = min(so.quantity, position_quantity)
  profit-taker:      tx_quantity = min(po.quantity, position_quantity)

Also covers: symbol completely absent from positions dict (position_quantity=0).
"""
from datetime import datetime

from trading.core.classes import BracketOrder, OrderStatus, Position, PriceData
from trading.core.om.backtesting_om import BacktestingOrderManager

TS = datetime(2024, 1, 1, 10, 0)


def _pd(symbol, price, ts=None):
    ts = ts or TS
    return PriceData(symbol, ts, price, price, price, price, 1000)


def _make_pending_sale_bracket(om, symbol, entry, stop, profit, qty=100):
    """Submit a bracket and advance it to PENDING_SALE (initial buy done)."""
    bo = BracketOrder.create_bracket_order(
        symbol=symbol, high_sell_price=profit, low_sell_price=stop,
        quantity=qty, tx_cost=0.0, current_tick=[_pd(symbol, entry)]
    )
    om.submit_order(bo, [_pd(symbol, entry)], {}, qty * entry * 10)
    assert bo.status == OrderStatus.PENDING_SALE, "Setup: bracket should be in PENDING_SALE"
    return bo


# ---------------------------------------------------------------------------
# Stop-loss partial position
# ---------------------------------------------------------------------------

class TestStopLossPartialPosition:

    def test_stop_caps_quantity_to_held_position(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 30)}  # only 30 of the 100 still held
        om.update_order_status(bo, [_pd("AAPL", 38.0)], positions, 0.0)
        assert bo.get_child_order("STOP").quantity == 30

    def test_stop_cash_reflects_capped_quantity(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 30)}
        om.update_order_status(bo, [_pd("AAPL", 38.0)], positions, 0.0)
        so = bo.get_child_order("STOP")
        assert so.cash == 30 * 38.0

    def test_stop_bracket_is_filled_with_partial_position(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 30)}
        om.update_order_status(bo, [_pd("AAPL", 38.0)], positions, 0.0)
        assert bo.status == OrderStatus.FILLED

    def test_stop_trigger_with_zero_position(self):
        """Symbol absent from positions → position_quantity=0 → tx_quantity=0."""
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        om.update_order_status(bo, [_pd("AAPL", 38.0)], {}, 0.0)
        so = bo.get_child_order("STOP")
        assert so.quantity == 0
        assert so.cash == 0.0
        assert bo.status == OrderStatus.FILLED

    def test_stop_sold_order_is_stop_child(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 30)}
        om.update_order_status(bo, [_pd("AAPL", 38.0)], positions, 0.0)
        assert bo.SOLD_ORDER is bo.get_child_order("STOP")

    def test_stop_profit_child_canceled_on_stop_trigger(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 30)}
        om.update_order_status(bo, [_pd("AAPL", 38.0)], positions, 0.0)
        assert bo.get_child_order("PROFIT").status == OrderStatus.CANCELED


# ---------------------------------------------------------------------------
# Profit-taker partial position
# ---------------------------------------------------------------------------

class TestProfitTakerPartialPosition:

    def test_profit_caps_quantity_to_held_position(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 45)}  # only 45 of the 100 held
        om.update_order_status(bo, [_pd("AAPL", 62.0)], positions, 0.0)
        assert bo.get_child_order("PROFIT").quantity == 45

    def test_profit_cash_reflects_capped_quantity(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 45)}
        om.update_order_status(bo, [_pd("AAPL", 62.0)], positions, 0.0)
        po = bo.get_child_order("PROFIT")
        assert po.cash == 45 * 62.0

    def test_profit_bracket_is_filled_with_partial_position(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 45)}
        om.update_order_status(bo, [_pd("AAPL", 62.0)], positions, 0.0)
        assert bo.status == OrderStatus.FILLED

    def test_profit_trigger_with_zero_position(self):
        """Symbol absent from positions → position_quantity=0 → tx_quantity=0."""
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        om.update_order_status(bo, [_pd("AAPL", 62.0)], {}, 0.0)
        po = bo.get_child_order("PROFIT")
        assert po.quantity == 0
        assert po.cash == 0.0
        assert bo.status == OrderStatus.FILLED

    def test_profit_sold_order_is_profit_child(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 45)}
        om.update_order_status(bo, [_pd("AAPL", 62.0)], positions, 0.0)
        assert bo.SOLD_ORDER is bo.get_child_order("PROFIT")

    def test_profit_stop_child_canceled_on_profit_trigger(self):
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=100)
        positions = {"AAPL": Position("AAPL", 45)}
        om.update_order_status(bo, [_pd("AAPL", 62.0)], positions, 0.0)
        assert bo.get_child_order("STOP").status == OrderStatus.CANCELED

    def test_profit_uses_profit_child_quantity_not_stop_quantity(self):
        """
        Regression for the so.quantity → po.quantity bug fix.
        Both children start at qty=10. Position only holds 7.
        The fix: profit branch must use po.quantity (not so.quantity) before
        capping against position_quantity. Result must be min(10, 7)=7.
        """
        om = BacktestingOrderManager()
        bo = _make_pending_sale_bracket(om, "AAPL", 50.0, 40.0, 60.0, qty=10)
        positions = {"AAPL": Position("AAPL", 7)}
        om.update_order_status(bo, [_pd("AAPL", 62.0)], positions, 0.0)
        po = bo.get_child_order("PROFIT")
        assert po.quantity == 7  # min(po.quantity=10, position=7)
        assert po.cash == 7 * 62.0
