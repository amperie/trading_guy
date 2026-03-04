"""
Tests for BacktestingOM deferred BRACKET order behavior.

When a BRACKET order is submitted but the symbol has no price data in the current
tick, the order stays PENDING (not CANCELED) and retries on the next tick when
price data becomes available.

Changes covered:
  1. _submit_order_to_backend:  pd is None → stays PENDING, not CANCELED
  2. _update_order_status_from_backend: PENDING BRACKET + pd found → fills to PENDING_SALE
  3. After deferred fill, stop/profit monitoring works normally
"""
from datetime import datetime
from trading.core.classes import (
    PriceData, OrderStatus, BracketOrder, Position
)
from trading.core.om.backtesting_om import BacktestingOrderManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(symbol, price, ts=None):
    if ts is None:
        ts = datetime(2024, 1, 1, 10, 0)
    return [PriceData(symbol, ts, price, price, price, price, 1000)]


def _spy_tick(ts=None):
    """Tick that contains only SPY — symbol UPRO is absent."""
    if ts is None:
        ts = datetime(2024, 1, 1, 10, 0)
    return [PriceData("SPY", ts, 450.0, 450.0, 450.0, 450.0, 10000)]


def _make_bracket(symbol="UPRO", quantity=100):
    return BracketOrder.create_bracket_order(
        symbol=symbol,
        high_sell_price=55.0,
        low_sell_price=45.0,
        quantity=quantity,
        tx_cost=0.0,
        current_tick=_tick(symbol, 50.0),
    )


# ---------------------------------------------------------------------------
# Submit with no price data → PENDING, not CANCELED
# ---------------------------------------------------------------------------

class TestBracketDeferOnMissingPriceData:

    def test_bracket_stays_pending_when_no_price_data(self):
        om = BacktestingOrderManager()
        bo = _make_bracket()
        result = om.submit_order(bo, _spy_tick(), {}, 10000.0)
        assert result.status == OrderStatus.PENDING

    def test_bracket_not_canceled_when_no_price_data(self):
        om = BacktestingOrderManager()
        bo = _make_bracket()
        result = om.submit_order(bo, _spy_tick(), {}, 10000.0)
        assert result.status != OrderStatus.CANCELED

    def test_deferred_bracket_stays_in_pending_orders(self):
        """Deferred order must remain in pending_orders_by_id so it gets retried."""
        om = BacktestingOrderManager()
        bo = _make_bracket()
        om.submit_order(bo, _spy_tick(), {}, 10000.0)
        assert bo.order_id in om.pending_orders_by_id

    def test_normal_bracket_submit_unaffected(self):
        """Normal submission with price data in tick still fills to PENDING_SALE immediately."""
        om = BacktestingOrderManager()
        bo = _make_bracket()
        result = om.submit_order(bo, _tick("UPRO", 50.0), {}, 10000.0)
        assert result.status == OrderStatus.PENDING_SALE


# ---------------------------------------------------------------------------
# Retry on next tick: PENDING → PENDING_SALE
# ---------------------------------------------------------------------------

class TestBracketRetryOnNextTick:

    def test_pending_bracket_transitions_to_pending_sale_on_retry(self):
        om = BacktestingOrderManager()
        bo = _make_bracket(quantity=100)

        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)
        assert bo.status == OrderStatus.PENDING

        om.update_pending_orders(_tick("UPRO", 50.0, ts=datetime(2024, 1, 1, 10, 1)), {}, 10000.0)

        assert bo.status == OrderStatus.PENDING_SALE

    def test_retry_sets_price_to_tick_close(self):
        om = BacktestingOrderManager()
        bo = _make_bracket(quantity=100)

        om.submit_order(bo, _spy_tick(), {}, 10000.0)
        om.update_pending_orders(_tick("UPRO", 52.0), {}, 10000.0)

        assert bo.price == 52.0

    def test_retry_sets_cash(self):
        om = BacktestingOrderManager()
        bo = _make_bracket(quantity=100)

        om.submit_order(bo, _spy_tick(), {}, 10000.0)
        om.update_pending_orders(_tick("UPRO", 52.0), {}, 10000.0)

        assert bo.cash == 52.0 * bo.quantity

    def test_retry_sets_executed_datetime(self):
        om = BacktestingOrderManager()
        bo = _make_bracket()

        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)
        retry_ts = datetime(2024, 1, 1, 10, 5)
        om.update_pending_orders(_tick("UPRO", 50.0, ts=retry_ts), {}, 10000.0)

        assert bo.executed_datetime == retry_ts

    def test_retry_quantity_capped_by_available_cash(self):
        """If cash at retry time is less than original order cost, quantity is reduced."""
        om = BacktestingOrderManager()
        bo = _make_bracket(quantity=100)  # 100 shares @ $50 = $5000

        om.submit_order(bo, _spy_tick(), {}, 10000.0)
        # Only $200 cash available → can afford 4 shares at $50
        om.update_pending_orders(_tick("UPRO", 50.0), {}, 200.0)

        assert bo.quantity == 4
        assert bo.status == OrderStatus.PENDING_SALE

    def test_retry_order_stays_in_pending_awaiting_trigger(self):
        """After PENDING → PENDING_SALE retry, order stays in pending_orders_by_id
        (stop/profit still need monitoring)."""
        om = BacktestingOrderManager()
        bo = _make_bracket()

        om.submit_order(bo, _spy_tick(), {}, 10000.0)
        om.update_pending_orders(_tick("UPRO", 50.0), {}, 10000.0)

        assert bo.order_id in om.pending_orders_by_id

    def test_stays_pending_if_price_still_absent_on_retry_tick(self):
        """Order stays PENDING if the symbol is still absent from the tick on retry."""
        om = BacktestingOrderManager()
        bo = _make_bracket()

        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)
        # Second tick also has no UPRO
        om.update_pending_orders(_spy_tick(ts=datetime(2024, 1, 1, 10, 1)), {}, 10000.0)

        assert bo.status == OrderStatus.PENDING

    def test_multiple_deferred_ticks_then_fills(self):
        """Order deferred for several ticks before price data appears."""
        om = BacktestingOrderManager()
        bo = _make_bracket(quantity=50)

        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)

        for i in range(1, 5):
            om.update_pending_orders(_spy_tick(ts=datetime(2024, 1, 1, 10, i)), {}, 10000.0)
            assert bo.status == OrderStatus.PENDING, f"Should still be PENDING on tick {i}"

        # Tick 5: price data available
        om.update_pending_orders(_tick("UPRO", 48.0, ts=datetime(2024, 1, 1, 10, 5)), {}, 10000.0)
        assert bo.status == OrderStatus.PENDING_SALE
        assert bo.price == 48.0


# ---------------------------------------------------------------------------
# Full lifecycle: defer → fill → stop/profit monitoring
# ---------------------------------------------------------------------------

class TestBracketRetryFullLifecycle:

    def test_stop_loss_triggers_after_deferred_fill(self):
        """BRACKET deferred tick 1, fills tick 2, stop-loss triggers tick 3."""
        om = BacktestingOrderManager()
        bo = BracketOrder.create_bracket_order(
            symbol="UPRO",
            high_sell_price=55.0,
            low_sell_price=45.0,
            quantity=100,
            tx_cost=0.0,
            current_tick=_tick("UPRO", 50.0),
        )

        # Tick 1: deferred (no UPRO)
        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)
        assert bo.status == OrderStatus.PENDING

        # Tick 2: fills at $50
        om.update_pending_orders(_tick("UPRO", 50.0, ts=datetime(2024, 1, 1, 10, 1)), {}, 10000.0)
        assert bo.status == OrderStatus.PENDING_SALE

        positions = {"UPRO": Position("UPRO", bo.quantity)}

        # Tick 3: price drops to $44 → stop triggers
        om.update_pending_orders(_tick("UPRO", 44.0, ts=datetime(2024, 1, 1, 10, 2)), positions, 0.0)

        assert bo.status == OrderStatus.FILLED
        assert bo.get_child_order("STOP").status == OrderStatus.FILLED
        assert bo.get_child_order("PROFIT").status == OrderStatus.CANCELED

    def test_profit_taker_triggers_after_deferred_fill(self):
        """BRACKET deferred tick 1, fills tick 2, profit-taker triggers tick 3."""
        om = BacktestingOrderManager()
        bo = BracketOrder.create_bracket_order(
            symbol="UPRO",
            high_sell_price=55.0,
            low_sell_price=45.0,
            quantity=100,
            tx_cost=0.0,
            current_tick=_tick("UPRO", 50.0),
        )

        # Tick 1: deferred
        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)

        # Tick 2: fills
        om.update_pending_orders(_tick("UPRO", 50.0, ts=datetime(2024, 1, 1, 10, 1)), {}, 10000.0)

        positions = {"UPRO": Position("UPRO", bo.quantity)}

        # Tick 3: price rises to $56 → profit triggers
        om.update_pending_orders(_tick("UPRO", 56.0, ts=datetime(2024, 1, 1, 10, 2)), positions, 0.0)

        assert bo.status == OrderStatus.FILLED
        assert bo.get_child_order("PROFIT").status == OrderStatus.FILLED
        assert bo.get_child_order("STOP").status == OrderStatus.CANCELED

    def test_sell_proceeds_correct_after_deferred_fill_and_stop(self):
        """Verify sell cash is calculated from the actual trigger price, not the deferred price."""
        om = BacktestingOrderManager()
        bo = BracketOrder.create_bracket_order(
            symbol="UPRO",
            high_sell_price=55.0,
            low_sell_price=45.0,
            quantity=100,
            tx_cost=0.0,
            current_tick=_tick("UPRO", 50.0),
        )

        om.submit_order(bo, _spy_tick(ts=datetime(2024, 1, 1, 10, 0)), {}, 10000.0)
        om.update_pending_orders(_tick("UPRO", 50.0, ts=datetime(2024, 1, 1, 10, 1)), {}, 10000.0)

        positions = {"UPRO": Position("UPRO", bo.quantity)}
        om.update_pending_orders(_tick("UPRO", 43.0, ts=datetime(2024, 1, 1, 10, 2)), positions, 0.0)

        stop_order = bo.get_child_order("STOP")
        assert stop_order.price == 43.0
        assert stop_order.cash == 43.0 * stop_order.quantity
