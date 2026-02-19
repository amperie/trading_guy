"""Tests for AlpacaOrderManager enum handling and sync_orders_from_broker.

Verifies that alpaca-py enum attributes (which stringify as 'ClassName.VALUE')
are correctly parsed using _enum_val() so that bracket orders, statuses, and
sides are detected properly.
"""
from __future__ import annotations

import enum
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from trading.core.classes import (
    BracketOrder, Order, OrderAction, OrderStatus, OrderTimeInForce, OrderType, Position,
)


# ---------------------------------------------------------------------------
# Fake Alpaca enums that mimic alpaca-py behaviour:
#   str(FakeOrderClass.BRACKET) -> "FakeOrderClass.BRACKET"
#   FakeOrderClass.BRACKET.value -> "bracket"
# ---------------------------------------------------------------------------

class FakeOrderStatus(enum.Enum):
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    ACCEPTED = "accepted"

class FakeOrderSide(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class FakeOrderClass(enum.Enum):
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"

class FakeOrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class FakeTimeInForce(enum.Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


def _make_alpaca_order(
    *,
    id="remote-123",
    symbol="AAPL",
    status=FakeOrderStatus.FILLED,
    side=FakeOrderSide.BUY,
    order_class=FakeOrderClass.SIMPLE,
    qty="10",
    filled_avg_price="150.0",
    filled_qty="10",
    submitted_at=None,
    filled_at=None,
    legs=None,
    order_type=FakeOrderType.MARKET,
    time_in_force=FakeTimeInForce.GTC,
):
    """Build a mock Alpaca order object with enum attributes."""
    obj = MagicMock()
    obj.id = id
    obj.symbol = symbol
    obj.status = status
    obj.side = side
    obj.order_class = order_class
    obj.qty = qty
    obj.filled_avg_price = filled_avg_price
    obj.filled_qty = filled_qty
    obj.submitted_at = submitted_at or datetime(2025, 1, 1, 10, 0, 0)
    obj.filled_at = filled_at or datetime(2025, 1, 1, 10, 0, 1)
    obj.legs = legs
    obj.order_type = order_type
    obj.time_in_force = time_in_force
    return obj


def _make_alpaca_leg(
    *,
    id="leg-1",
    order_type=FakeOrderType.STOP,
    status=FakeOrderStatus.NEW,
    filled_avg_price=None,
    filled_qty=None,
    filled_at=None,
    stop_price=None,
    limit_price=None,
):
    leg = MagicMock()
    leg.id = id
    leg.order_type = order_type
    leg.status = status
    leg.filled_avg_price = filled_avg_price
    leg.filled_qty = filled_qty
    leg.filled_at = filled_at
    leg.stop_price = stop_price
    leg.limit_price = limit_price
    return leg


# Patch the alpaca import guard so AlpacaOrderManager can be imported
# even without alpaca-py installed.
@pytest.fixture(autouse=True)
def _patch_alpaca_imports():
    """Make AlpacaOrderManager importable without alpaca-py."""
    import trading.core.om.alpaca_om as mod
    original = mod.ALPACA_AVAILABLE
    mod.ALPACA_AVAILABLE = True
    yield
    mod.ALPACA_AVAILABLE = original


def _create_om():
    """Create an AlpacaOrderManager with a mocked TradingClient."""
    with patch("trading.core.om.alpaca_om.TradingClient"):
        from trading.core.om.alpaca_om import AlpacaOrderManager
        om = AlpacaOrderManager({"api_key": "test", "secret_key": "test", "paper": True})
    return om


# ---- _enum_val tests ----

class TestEnumVal:
    def test_handles_enum_with_value(self):
        from trading.core.om.alpaca_om import _enum_val
        assert _enum_val(FakeOrderStatus.FILLED) == "filled"
        assert _enum_val(FakeOrderClass.BRACKET) == "bracket"
        assert _enum_val(FakeOrderSide.BUY) == "buy"

    def test_handles_plain_string(self):
        from trading.core.om.alpaca_om import _enum_val
        assert _enum_val("FILLED") == "filled"
        assert _enum_val("bracket") == "bracket"
        assert _enum_val("") == ""

    def test_handles_empty_and_none_like(self):
        from trading.core.om.alpaca_om import _enum_val
        assert _enum_val("") == ""
        assert _enum_val(None) == "none"


# ---- sync_orders_from_broker tests ----

class TestSyncOrdersFromBroker:
    def test_simple_filled_market_order(self):
        om = _create_om()
        alpaca_order = _make_alpaca_order(
            id="remote-1",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.SIMPLE,
        )
        om.client.get_orders.return_value = [alpaca_order]

        result = om.sync_orders_from_broker()
        assert result["new"] == 1

        # Find the created order
        assert len(om._all_orders) == 1
        order = list(om._all_orders.values())[0]
        assert order.type == OrderType.MARKET
        assert order.status == OrderStatus.FILLED
        assert order.action == OrderAction.BUY

    def test_simple_canceled_order(self):
        om = _create_om()
        alpaca_order = _make_alpaca_order(
            id="remote-2",
            status=FakeOrderStatus.CANCELED,
            side=FakeOrderSide.SELL,
            order_class=FakeOrderClass.SIMPLE,
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]
        assert order.status == OrderStatus.CANCELED
        assert order.action == OrderAction.SELL

    def test_simple_pending_order(self):
        om = _create_om()
        alpaca_order = _make_alpaca_order(
            id="remote-3",
            status=FakeOrderStatus.ACCEPTED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.SIMPLE,
            filled_avg_price=None,
            filled_qty=None,
            filled_at=None,
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]
        assert order.status == OrderStatus.PENDING
        assert order.order_id in om._pending_orders_by_id

    def test_bracket_order_detected(self):
        """Bracket orders must be detected when order_class is an enum."""
        om = _create_om()

        stop_leg = _make_alpaca_leg(
            id="leg-stop",
            order_type=FakeOrderType.STOP,
            status=FakeOrderStatus.NEW,
            stop_price=140.0,
        )
        profit_leg = _make_alpaca_leg(
            id="leg-profit",
            order_type=FakeOrderType.LIMIT,
            status=FakeOrderStatus.NEW,
            limit_price=160.0,
        )

        alpaca_order = _make_alpaca_order(
            id="remote-bracket",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.BRACKET,
            legs=[stop_leg, profit_leg],
        )
        om.client.get_orders.return_value = [alpaca_order]

        result = om.sync_orders_from_broker()
        assert result["new"] == 1

        order = list(om._all_orders.values())[0]
        assert order.type == OrderType.BRACKET
        assert order.status == OrderStatus.PENDING_SALE  # entry filled, no exit yet

        # Children should exist
        stop_child = order.get_child_order("STOP")
        profit_child = order.get_child_order("PROFIT")
        assert stop_child is not None
        assert profit_child is not None
        assert stop_child.type == OrderType.STOP_LOSS
        assert profit_child.type == OrderType.PROFIT_TAKER

    def test_bracket_order_with_filled_stop_leg(self):
        """When a stop leg is filled, SOLD_ORDER should be set and status FILLED."""
        om = _create_om()

        stop_leg = _make_alpaca_leg(
            id="leg-stop",
            order_type=FakeOrderType.STOP,
            status=FakeOrderStatus.FILLED,
            stop_price=140.0,
            filled_avg_price=139.5,
            filled_qty=10,
            filled_at=datetime(2025, 1, 1, 11, 0, 0),
        )
        profit_leg = _make_alpaca_leg(
            id="leg-profit",
            order_type=FakeOrderType.LIMIT,
            status=FakeOrderStatus.CANCELED,
            limit_price=160.0,
        )

        alpaca_order = _make_alpaca_order(
            id="remote-bracket-filled",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.BRACKET,
            legs=[stop_leg, profit_leg],
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]

        assert order.type == OrderType.BRACKET
        assert order.status == OrderStatus.FILLED
        assert order.SOLD_ORDER is not None
        assert order.SOLD_ORDER.type == OrderType.STOP_LOSS
        assert order.SOLD_ORDER.status == OrderStatus.FILLED
        assert order.SOLD_ORDER.price == 139.5

    def test_bracket_all_children_canceled_sets_filled(self):
        """When all child orders are canceled, bracket status should be FILLED."""
        om = _create_om()

        stop_leg = _make_alpaca_leg(
            id="leg-stop",
            order_type=FakeOrderType.STOP,
            status=FakeOrderStatus.CANCELED,
            stop_price=140.0,
        )
        profit_leg = _make_alpaca_leg(
            id="leg-profit",
            order_type=FakeOrderType.LIMIT,
            status=FakeOrderStatus.CANCELED,
            limit_price=160.0,
        )

        alpaca_order = _make_alpaca_order(
            id="remote-bracket-all-canceled",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.BRACKET,
            legs=[stop_leg, profit_leg],
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]

        assert order.type == OrderType.BRACKET
        assert order.status == OrderStatus.FILLED

    def test_oco_order_class_treated_as_bracket(self):
        """OCO order class should also be treated as bracket."""
        om = _create_om()

        stop_leg = _make_alpaca_leg(
            id="leg-stop", order_type=FakeOrderType.STOP,
            status=FakeOrderStatus.NEW, stop_price=140.0,
        )
        alpaca_order = _make_alpaca_order(
            id="remote-oco",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.OCO,
            legs=[stop_leg],
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]
        assert order.type == OrderType.BRACKET


# ---- _update_order_status_from_backend tests ----

class TestUpdateOrderStatusFromBackend:
    def test_market_order_filled(self):
        om = _create_om()
        order = Order(
            action=OrderAction.BUY, type=OrderType.MARKET, symbol="AAPL",
            price=0.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
            status=OrderStatus.PENDING, platform_id="remote-1",
        )
        om._local_to_remote[order.order_id] = "remote-1"

        alpaca_resp = _make_alpaca_order(
            id="remote-1", status=FakeOrderStatus.FILLED,
            filled_avg_price="155.0", filled_qty="10",
        )
        om.client.get_order_by_id.return_value = alpaca_resp

        om._update_order_status_from_backend(order)
        assert order.status == OrderStatus.FILLED
        assert order.price == 155.0

    def test_bracket_entry_fill_moves_to_pending_sale(self):
        om = _create_om()
        order = BracketOrder(
            action=OrderAction.BUY, type=OrderType.BRACKET, symbol="AAPL",
            price=0.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
            status=OrderStatus.PENDING, platform_id="remote-bracket",
        )
        order.add_child_order("STOP", Order(
            action=OrderAction.SELL, type=OrderType.STOP_LOSS, symbol="AAPL",
            price=140.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
        ))
        order.add_child_order("PROFIT", Order(
            action=OrderAction.SELL, type=OrderType.PROFIT_TAKER, symbol="AAPL",
            price=160.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
        ))

        # Alpaca says entry is filled, no legs filled yet
        alpaca_resp = _make_alpaca_order(
            id="remote-bracket", status=FakeOrderStatus.FILLED,
            filled_avg_price="150.0", filled_qty="10",
            legs=[
                _make_alpaca_leg(id="leg-s", order_type=FakeOrderType.STOP, status=FakeOrderStatus.NEW),
                _make_alpaca_leg(id="leg-p", order_type=FakeOrderType.LIMIT, status=FakeOrderStatus.NEW),
            ],
        )
        om.client.get_order_by_id.return_value = alpaca_resp

        om._update_order_status_from_backend(order)
        assert order.status == OrderStatus.PENDING_SALE
        assert order.price == 150.0

    def test_bracket_exit_fill_sets_filled(self):
        om = _create_om()
        order = BracketOrder(
            action=OrderAction.BUY, type=OrderType.BRACKET, symbol="AAPL",
            price=150.0, quantity=10, cash=1500.0, placed_datetime=datetime.now(),
            status=OrderStatus.PENDING_SALE, platform_id="remote-bracket",
        )
        stop_child = Order(
            action=OrderAction.SELL, type=OrderType.STOP_LOSS, symbol="AAPL",
            price=140.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
        )
        order.add_child_order("STOP", stop_child)
        order.add_child_order("PROFIT", Order(
            action=OrderAction.SELL, type=OrderType.PROFIT_TAKER, symbol="AAPL",
            price=160.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
        ))

        alpaca_resp = _make_alpaca_order(
            id="remote-bracket", status=FakeOrderStatus.FILLED,
            filled_avg_price="150.0", filled_qty="10",
            legs=[
                _make_alpaca_leg(
                    id="leg-s", order_type=FakeOrderType.STOP,
                    status=FakeOrderStatus.FILLED,
                    filled_avg_price=139.0, filled_qty=10,
                    filled_at=datetime(2025, 1, 1, 12, 0, 0),
                ),
                _make_alpaca_leg(
                    id="leg-p", order_type=FakeOrderType.LIMIT,
                    status=FakeOrderStatus.CANCELED,
                ),
            ],
        )
        om.client.get_order_by_id.return_value = alpaca_resp

        om._update_order_status_from_backend(order)
        assert order.status == OrderStatus.FILLED
        assert order.SOLD_ORDER is not None
        assert order.SOLD_ORDER.price == 139.0


# ---- TimeInForce tests ----

class TestTimeInForce:
    def test_sync_sets_tif_on_market_order(self):
        """Synced market orders should have time_in_force from Alpaca."""
        om = _create_om()
        alpaca_order = _make_alpaca_order(
            id="remote-tif",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.SIMPLE,
            time_in_force=FakeTimeInForce.DAY,
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]
        assert order.time_in_force == OrderTimeInForce.DAY

    def test_sync_sets_tif_on_bracket_and_children(self):
        """Bracket orders and their children should get TIF from Alpaca."""
        om = _create_om()

        stop_leg = _make_alpaca_leg(
            id="leg-stop", order_type=FakeOrderType.STOP,
            status=FakeOrderStatus.NEW, stop_price=140.0,
        )
        profit_leg = _make_alpaca_leg(
            id="leg-profit", order_type=FakeOrderType.LIMIT,
            status=FakeOrderStatus.NEW, limit_price=160.0,
        )
        alpaca_order = _make_alpaca_order(
            id="remote-bracket-tif",
            status=FakeOrderStatus.FILLED,
            side=FakeOrderSide.BUY,
            order_class=FakeOrderClass.BRACKET,
            legs=[stop_leg, profit_leg],
            time_in_force=FakeTimeInForce.IOC,
        )
        om.client.get_orders.return_value = [alpaca_order]

        om.sync_orders_from_broker()
        order = list(om._all_orders.values())[0]
        assert order.time_in_force == OrderTimeInForce.IOC
        assert order.get_child_order("STOP").time_in_force == OrderTimeInForce.IOC
        assert order.get_child_order("PROFIT").time_in_force == OrderTimeInForce.IOC

    def test_update_sets_tif_on_existing_order(self):
        """_update_order_status_from_backend should set TIF from Alpaca."""
        om = _create_om()
        order = Order(
            action=OrderAction.BUY, type=OrderType.MARKET, symbol="AAPL",
            price=0.0, quantity=10, cash=0.0, placed_datetime=datetime.now(),
            time_in_force=OrderTimeInForce.GTC,
            status=OrderStatus.PENDING, platform_id="remote-1",
        )
        om._local_to_remote[order.order_id] = "remote-1"

        alpaca_resp = _make_alpaca_order(
            id="remote-1", status=FakeOrderStatus.FILLED,
            filled_avg_price="155.0", filled_qty="10",
            time_in_force=FakeTimeInForce.DAY,
        )
        om.client.get_order_by_id.return_value = alpaca_resp

        om._update_order_status_from_backend(order)
        assert order.time_in_force == OrderTimeInForce.DAY

    def test_submit_uses_order_tif(self):
        """_submit_order_to_backend should use the order's time_in_force."""
        om = _create_om()
        order = Order.create_market_order("AAPL", OrderAction.BUY, 10)
        order.time_in_force = OrderTimeInForce.IOC

        # Mock the submit response
        mock_resp = MagicMock()
        mock_resp.id = "remote-submitted"
        mock_resp.legs = None
        om.client.submit_order.return_value = mock_resp

        om._submit_order_to_backend(order)

        # Verify the request was built with IOC
        call_args = om.client.submit_order.call_args
        req = call_args.kwargs.get("order_data") or call_args[1].get("order_data") or call_args[0][0]
        # The TIF on the request object should map to IOC
        from trading.core.om.alpaca_om import _ALPACA_TIF_MAP
        expected_tif = _ALPACA_TIF_MAP[OrderTimeInForce.IOC]
        assert req.time_in_force == expected_tif
