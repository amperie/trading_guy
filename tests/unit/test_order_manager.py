"""
Unit tests for OrderManager implementations
"""
import pytest
from datetime import datetime

from core.order_manager import OrderManager
from core.om.backtesting_om import BacktestingOM
from core.classes import (
    Order, PriceData, OrderAction, OrderType, OrderStatus
)


class TestPerfectOm:
    """Tests for PerfectOm (backtesting order manager)"""

    def test_perfect_om_initialization(self):
        """Test PerfectOm initialization"""
        om = BacktestingOM()
        assert om.cfg == {}

    def test_perfect_om_initialization_with_config(self):
        """Test PerfectOm with config"""
        cfg = {"test_key": "test_value"}
        om = BacktestingOM(cfg)
        assert om.cfg == cfg

    def test_perfect_om_buy_order(self, sample_pricedata_list):
        """Test creating a buy order"""
        om = BacktestingOM()

        order = om.buy("AAPL", 100, sample_pricedata_list)

        assert order.action == OrderAction.BUY
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.price == 151.0  # Close price from fixture
        assert order.cash == 151.0 * 100
        assert order.status == OrderStatus.FILLED
        assert order.type == OrderType.MARKET
        assert order.tx_cost == 0.0

    def test_perfect_om_sell_order(self, sample_pricedata_list):
        """Test creating a sell order"""
        om = BacktestingOM()

        order = om.sell("GOOGL", 50, sample_pricedata_list)

        assert order.action == OrderAction.SELL
        assert order.symbol == "GOOGL"
        assert order.quantity == 50
        assert order.price == 2810.0  # Close price from fixture
        assert order.cash == 2810.0 * 50
        assert order.status == OrderStatus.FILLED

    def test_perfect_om_order_filled_immediately(self, sample_pricedata_list):
        """Test that PerfectOm fills orders immediately"""
        om = BacktestingOM()

        order = om.buy("AAPL", 100, sample_pricedata_list)

        assert order.status == OrderStatus.FILLED
        assert order.placed_datetime is not None
        assert order.executed_datetime is not None

    def test_perfect_om_order_has_unique_id(self, sample_pricedata_list):
        """Test that orders have unique IDs"""
        om = BacktestingOM()

        order1 = om.buy("AAPL", 100, sample_pricedata_list)
        order2 = om.buy("AAPL", 100, sample_pricedata_list)

        assert order1.order_id != order2.order_id

    def test_perfect_om_get_order_status(self, sample_pending_order):
        """Test get_order_status marks order as filled"""
        om = BacktestingOM()

        # PerfectOm always marks orders as filled
        updated_order = om.get_order_status(sample_pending_order)

        assert updated_order.status == OrderStatus.FILLED

    def test_perfect_om_update_all_orders(self):
        """Test update_all_orders returns empty list"""
        om = BacktestingOM()
        result = om.update_all_orders()
        # PerfectOm doesn't track orders, so this returns empty
        assert result is None or result == []

    def test_perfect_om_update_order_statuses(self, sample_order):
        """Test update_order_statuses"""
        om = BacktestingOM()
        result = om.update_order_statuses([sample_order])
        # PerfectOm doesn't need to update statuses
        assert result is None or result == []

    def test_perfect_om_buy_uses_close_price(self, sample_pricedata):
        """Test that buy order uses close price"""
        om = BacktestingOM()

        order = om.buy("AAPL", 100, [sample_pricedata])

        assert order.price == sample_pricedata.close

    def test_perfect_om_sell_uses_close_price(self, sample_pricedata):
        """Test that sell order uses close price"""
        om = BacktestingOM()

        order = om.sell("AAPL", 100, [sample_pricedata])

        assert order.price == sample_pricedata.close

    def test_perfect_om_calculates_cash_correctly(self, sample_pricedata_list):
        """Test that cash calculation is correct"""
        om = BacktestingOM()

        # Buy order
        buy_order = om.buy("AAPL", 100, sample_pricedata_list)
        assert buy_order.cash == buy_order.price * buy_order.quantity

        # Sell order
        sell_order = om.sell("GOOGL", 50, sample_pricedata_list)
        assert sell_order.cash == sell_order.price * sell_order.quantity

    def test_perfect_om_zero_quantity_order(self, sample_pricedata_list):
        """Test order with zero quantity"""
        om = BacktestingOM()

        order = om.buy("AAPL", 0, sample_pricedata_list)

        assert order.quantity == 0
        assert order.cash == 0.0

    def test_perfect_om_large_quantity_order(self, sample_pricedata_list):
        """Test order with large quantity"""
        om = BacktestingOM()

        order = om.buy("AAPL", 1000000, sample_pricedata_list)

        assert order.quantity == 1000000
        assert order.cash == 151.0 * 1000000
        assert order.status == OrderStatus.FILLED


class TestOrderManagerInterface:
    """Tests for OrderManager abstract base class"""

    def test_order_manager_is_abstract(self):
        """Test that OrderManager cannot be instantiated directly"""
        with pytest.raises(TypeError):
            om = OrderManager()

    def test_order_manager_requires_implementation(self):
        """Test that subclasses must implement abstract methods"""

        class IncompleteOM(OrderManager):
            """Missing required methods"""
            pass

        with pytest.raises(TypeError):
            om = IncompleteOM()

    def test_order_manager_complete_implementation(self, sample_pricedata_list):
        """Test that complete implementation works"""

        class CompleteOM(OrderManager):
            def buy(self, ticker: str, shares: int, tick: list[PriceData]) -> Order:
                return Order(
                    placed_datetime=datetime.now(),
                    executed_datetime=datetime.now(),
                    action=OrderAction.BUY,
                    type=OrderType.MARKET,
                    symbol=ticker,
                    price=100.0,
                    quantity=shares,
                    cash=100.0 * shares,
                    status=OrderStatus.FILLED
                )

            def sell(self, ticker: str, shares: int, tick: list[PriceData]) -> Order:
                return Order(
                    placed_datetime=datetime.now(),
                    executed_datetime=datetime.now(),
                    action=OrderAction.SELL,
                    type=OrderType.MARKET,
                    symbol=ticker,
                    price=100.0,
                    quantity=shares,
                    cash=100.0 * shares,
                    status=OrderStatus.FILLED
                )

            def update_order_statuses(self, orders: list[Order]):
                return []

            def update_all_orders(self) -> list[Order]:
                return []

            def get_order_status(self, order: Order) -> Order:
                return order

        # Should not raise
        om = CompleteOM()
        order = om.buy("AAPL", 100, sample_pricedata_list)
        assert order.quantity == 100
