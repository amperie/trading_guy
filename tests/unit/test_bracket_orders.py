"""
Unit tests for bracket order functionality
Tests bracket order creation, lifecycle, and execution with BacktestingOM
"""
import pytest
from datetime import datetime, timezone

from core.classes import (
    Order, PriceData, OrderAction, OrderType, OrderStatus
)
from core.om.backtesting_om import BacktestingOM


class TestBracketOrderCreation:
    """Tests for creating bracket orders"""

    def test_bracket_order_basic_creation(self):
        """Test creating a basic bracket order"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100,
            tx_cost=1.0
        )

        # Verify main order
        assert main_order.action == OrderAction.BUY
        assert main_order.type == OrderType.BRACKET
        assert main_order.symbol == "AAPL"
        assert main_order.price == 100.0
        assert main_order.quantity == 100
        assert main_order.status == OrderStatus.PENDING
        assert main_order.cash == 100.0 * 100
        assert main_order.tx_cost == 1.0

        # Verify child orders exist
        assert len(main_order._child_orders_dict) == 2
        assert "STOP" in main_order._child_orders_dict
        assert "PROFIT_TAKER" in main_order._child_orders_dict

    def test_bracket_order_child_relationships(self):
        """Test parent-child relationships in bracket order"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Verify child orders list
        assert len(main_order.child_orders) == 2
        assert stop_order.order_id in main_order.child_orders
        assert profit_order.order_id in main_order.child_orders

        # Verify child orders dict
        assert "STOP" in main_order._child_orders_dict
        assert "PROFIT_TAKER" in main_order._child_orders_dict
        assert main_order._child_orders_dict["STOP"] == stop_order
        assert main_order._child_orders_dict["PROFIT_TAKER"] == profit_order

        # Verify parent references
        assert stop_order.parent_id == main_order.order_id
        assert profit_order.parent_id == main_order.order_id

    def test_bracket_order_stop_loss_properties(self):
        """Test stop-loss order properties"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100,
            tx_cost=1.0
        )

        stop_order = main_order._child_orders_dict["STOP"]

        assert stop_order.action == OrderAction.SELL
        assert stop_order.type == OrderType.STOP_LOSS
        assert stop_order.price == 95.0
        assert stop_order.quantity == 100
        assert stop_order.status == OrderStatus.PENDING
        assert stop_order.cash == 0  # Child orders start with 0 cash
        assert stop_order.tx_cost == 1.0

    def test_bracket_order_profit_taker_properties(self):
        """Test profit-taker order properties"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100,
            tx_cost=1.0
        )

        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        assert profit_order.action == OrderAction.SELL
        assert profit_order.type == OrderType.PROFIT_TAKER
        assert profit_order.price == 110.0
        assert profit_order.quantity == 100
        assert profit_order.status == OrderStatus.PENDING
        assert profit_order.cash == 0  # Child orders start with 0 cash
        assert profit_order.tx_cost == 1.0

    def test_bracket_order_unique_ids(self):
        """Test that each order in bracket has unique ID"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # All IDs should be unique
        assert main_order.order_id != stop_order.order_id
        assert main_order.order_id != profit_order.order_id
        assert stop_order.order_id != profit_order.order_id

        # All should start with "local-"
        assert main_order.order_id.startswith("local-")
        assert stop_order.order_id.startswith("local-")
        assert profit_order.order_id.startswith("local-")

    def test_bracket_order_with_zero_tx_cost(self):
        """Test bracket order with no transaction cost"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100,
            tx_cost=0
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        assert main_order.tx_cost == 0
        assert stop_order.tx_cost == 0
        assert profit_order.tx_cost == 0

    def test_bracket_order_different_quantities(self):
        """Test bracket orders with various quantities"""
        for quantity in [1, 10, 100, 1000, 10000]:
            main_order = Order.create_bracket_order(
                symbol="AAPL",
                price=100.0,
                high_sell_price=110.0,
                low_sell_price=95.0,
                quantity=quantity
            )

            stop_order = main_order._child_orders_dict["STOP"]
            profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

            assert main_order.quantity == quantity
            assert stop_order.quantity == quantity
            assert profit_order.quantity == quantity
            assert main_order.cash == 100.0 * quantity


class TestBracketOrderBacktestingOM:
    """Tests for bracket order execution with BacktestingOM"""

    @pytest.fixture
    def om(self):
        """Returns a BacktestingOM instance"""
        return BacktestingOM()

    @pytest.fixture
    def initial_tick(self):
        """Price data at entry: $100"""
        return [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            open=99.0,
            high=101.0,
            low=98.0,
            close=100.0,
            volume=1000000.0
        )]

    @pytest.fixture
    def profit_tick(self):
        """Price data triggering profit-taker: $110+"""
        return [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            open=109.0,
            high=112.0,
            low=108.0,
            close=111.0,
            volume=1200000.0
        )]

    @pytest.fixture
    def stop_loss_tick(self):
        """Price data triggering stop-loss: $95 or below"""
        return [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            open=96.0,
            high=97.0,
            low=93.0,
            close=94.0,
            volume=1500000.0
        )]

    @pytest.fixture
    def neutral_tick(self):
        """Price data between stop-loss and profit-taker: $102"""
        return [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            open=101.0,
            high=103.0,
            low=100.0,
            close=102.0,
            volume=1100000.0
        )]

    def test_bracket_order_pending_to_pending_sale_transition(self, om, initial_tick):
        """Test bracket order transitions from PENDING to PENDING_SALE"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Initially all orders are PENDING
        assert main_order.status == OrderStatus.PENDING
        assert stop_order.status == OrderStatus.PENDING
        assert profit_order.status == OrderStatus.PENDING

        # Process the order
        updated_order = om.get_order_status(main_order, initial_tick)

        # Main order should transition to PENDING_SALE (entry filled)
        assert updated_order.status == OrderStatus.PENDING_SALE
        assert updated_order.executed_datetime == initial_tick[0].timestamp
        assert updated_order.processed_by_portfolio == False

        # Child orders should still be PENDING
        assert stop_order.status == OrderStatus.PENDING
        assert profit_order.status == OrderStatus.PENDING

    def test_bracket_order_entry_price_filled_at_market(self, om, initial_tick):
        """Test that entry order gets filled at current market price"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,  # Limit price
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        updated_order = om.get_order_status(main_order, initial_tick)

        # Should be filled at close price
        assert updated_order.price == 100.0
        assert updated_order.cash == 100.0 * 100

    def test_bracket_order_no_trigger_in_neutral_range(self, om, initial_tick, neutral_tick):
        """Test that bracket order doesn't trigger when price is between stop/profit"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Fill the entry
        updated_order = om.get_order_status(main_order, initial_tick)
        assert updated_order.status == OrderStatus.PENDING_SALE

        # Process with neutral tick (price = 102)
        updated_order = om.get_order_status(updated_order, neutral_tick)

        # Should still be PENDING_SALE (neither child triggered)
        assert updated_order.status == OrderStatus.PENDING_SALE
        assert stop_order.status == OrderStatus.PENDING
        assert profit_order.status == OrderStatus.PENDING

    def test_bracket_order_profit_taker_triggers(self, om, initial_tick, profit_tick):
        """Test that profit-taker triggers when price reaches high_sell_price"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Fill the entry
        main_order = om.get_order_status(main_order, initial_tick)
        assert main_order.status == OrderStatus.PENDING_SALE

        # Trigger profit-taker (price = 111.0 >= 110.0)
        main_order = om.get_order_status(main_order, profit_tick)

        # Main order should be FILLED
        assert main_order.status == OrderStatus.FILLED
        assert main_order.executed_datetime == profit_tick[0].timestamp
        assert main_order.processed_by_portfolio == False

        # Profit order should be FILLED
        assert profit_order.status == OrderStatus.FILLED
        assert profit_order.price == 111.0
        assert profit_order.cash == 111.0 * 100
        assert profit_order.executed_datetime == profit_tick[0].timestamp

        # Stop order should be CANCELED
        assert stop_order.status == OrderStatus.CANCELED

    def test_bracket_order_stop_loss_triggers(self, om, initial_tick, stop_loss_tick):
        """Test that stop-loss triggers when price drops to low_sell_price"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Fill the entry
        main_order = om.get_order_status(main_order, initial_tick)
        assert main_order.status == OrderStatus.PENDING_SALE

        # Trigger stop-loss (price = 94.0 <= 95.0)
        main_order = om.get_order_status(main_order, stop_loss_tick)

        # Main order should be FILLED
        assert main_order.status == OrderStatus.FILLED
        assert main_order.executed_datetime == stop_loss_tick[0].timestamp
        assert main_order.processed_by_portfolio == False

        # Stop order should be FILLED
        assert stop_order.status == OrderStatus.FILLED
        assert stop_order.price == 94.0
        assert stop_order.cash == 94.0 * 100
        assert stop_order.executed_datetime == stop_loss_tick[0].timestamp

        # Profit order should be CANCELED
        assert profit_order.status == OrderStatus.CANCELED

    def test_bracket_order_entry_price_preservation_on_profit(self, om, initial_tick, profit_tick):
        """Test that entry price is preserved when profit-taker triggers"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        # Fill the entry at $100
        main_order = om.get_order_status(main_order, initial_tick)
        entry_price = main_order.price
        assert entry_price == 100.0

        # Trigger profit-taker at $111
        main_order = om.get_order_status(main_order, profit_tick)

        # CRITICAL: Entry price should still be $100, not $111
        # This tests for the bug where main_order.price gets overwritten with exit price
        assert main_order.price == entry_price, \
            f"Entry price corrupted! Expected {entry_price}, got {main_order.price}"

    def test_bracket_order_entry_price_preservation_on_stop_loss(self, om, initial_tick, stop_loss_tick):
        """Test that entry price is preserved when stop-loss triggers"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        # Fill the entry at $100
        main_order = om.get_order_status(main_order, initial_tick)
        entry_price = main_order.price
        assert entry_price == 100.0

        # Trigger stop-loss at $94
        main_order = om.get_order_status(main_order, stop_loss_tick)

        # CRITICAL: Entry price should still be $100, not $94
        # This tests for the bug where main_order.price gets overwritten with exit price
        assert main_order.price == entry_price, \
            f"Entry price corrupted! Expected {entry_price}, got {main_order.price}"

    def test_bracket_order_profit_loss_calculation(self, om, initial_tick, profit_tick, stop_loss_tick):
        """Test that P&L can be calculated correctly from bracket orders"""
        # Test profitable trade
        main_profit = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        profit_order = main_profit._child_orders_dict["PROFIT_TAKER"]

        main_profit = om.get_order_status(main_profit, initial_tick)
        main_profit = om.get_order_status(main_profit, profit_tick)

        # Calculate P&L: (exit_price - entry_price) * quantity
        entry_price = main_profit.price
        exit_price = profit_order.price
        expected_profit = (exit_price - entry_price) * 100

        assert expected_profit == 1100.0, "Should profit $11 * 100 shares = $1100"

        # Test losing trade
        main_loss = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_loss._child_orders_dict["STOP"]

        main_loss = om.get_order_status(main_loss, initial_tick)
        main_loss = om.get_order_status(main_loss, stop_loss_tick)

        # Calculate P&L
        entry_price = main_loss.price
        exit_price = stop_order.price
        expected_loss = (exit_price - entry_price) * 100

        assert expected_loss == -600.0, "Should lose $6 * 100 shares = -$600"

    def test_bracket_order_exact_profit_trigger_price(self, om, initial_tick):
        """Test profit-taker triggers exactly at high_sell_price"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Fill the entry
        main_order = om.get_order_status(main_order, initial_tick)

        # Price exactly at profit target
        exact_profit_tick = [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            open=109.0,
            high=110.5,
            low=108.0,
            close=110.0,  # Exactly at profit target
            volume=1200000.0
        )]

        main_order = om.get_order_status(main_order, exact_profit_tick)

        # Should trigger (price >= high_sell_price)
        assert profit_order.status == OrderStatus.FILLED
        assert stop_order.status == OrderStatus.CANCELED

    def test_bracket_order_exact_stop_loss_trigger_price(self, om, initial_tick):
        """Test stop-loss triggers exactly at low_sell_price"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        # Fill the entry
        main_order = om.get_order_status(main_order, initial_tick)

        # Price exactly at stop-loss target
        exact_stop_tick = [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            open=96.0,
            high=97.0,
            low=93.0,
            close=95.0,  # Exactly at stop-loss target
            volume=1500000.0
        )]

        main_order = om.get_order_status(main_order, exact_stop_tick)

        # Should trigger (price <= low_sell_price)
        assert stop_order.status == OrderStatus.FILLED
        assert profit_order.status == OrderStatus.CANCELED

    def test_bracket_order_already_filled_no_reprocessing(self, om, initial_tick, profit_tick):
        """Test that already FILLED bracket orders are not reprocessed"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        # Fill the entry and trigger profit
        main_order = om.get_order_status(main_order, initial_tick)
        main_order = om.get_order_status(main_order, profit_tick)

        assert main_order.status == OrderStatus.FILLED
        original_price = main_order.price

        # Try to process again (should not change anything)
        main_order = om.get_order_status(main_order, profit_tick)

        assert main_order.status == OrderStatus.FILLED
        assert main_order.price == original_price

    def test_bracket_order_canceled_no_reprocessing(self, om, initial_tick):
        """Test that CANCELED orders are not reprocessed"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]

        # Manually cancel
        stop_order.status = OrderStatus.CANCELED

        # Try to process (should return immediately)
        result = om.get_order_status(stop_order, initial_tick)

        assert result.status == OrderStatus.CANCELED

    def test_bracket_order_multiple_symbols(self, om):
        """Test bracket orders for multiple symbols simultaneously"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        all_orders = {}

        for symbol in symbols:
            main_order = Order.create_bracket_order(
                symbol=symbol,
                price=100.0,
                high_sell_price=110.0,
                low_sell_price=95.0,
                quantity=100
            )
            all_orders[symbol] = main_order

        # Verify all orders are unique and correct
        assert len(all_orders) == 3
        for symbol, order in all_orders.items():
            assert order.symbol == symbol
            assert order.type == OrderType.BRACKET
            assert len(order.child_orders) == 2


class TestBracketOrderEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_bracket_order_very_tight_spread(self):
        """Test bracket order with very small spread"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=100.01,  # Only 1 cent profit target
            low_sell_price=99.99,    # Only 1 cent stop
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        assert stop_order.price == 99.99
        assert profit_order.price == 100.01

    def test_bracket_order_very_wide_spread(self):
        """Test bracket order with very wide spread"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=200.0,   # 100% profit target
            low_sell_price=50.0,     # 50% stop
            quantity=100
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        assert stop_order.price == 50.0
        assert profit_order.price == 200.0

    def test_bracket_order_single_share(self):
        """Test bracket order with single share"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=1
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        assert main_order.quantity == 1
        assert stop_order.quantity == 1
        assert profit_order.quantity == 1
        assert main_order.cash == 100.0

    def test_bracket_order_high_tx_cost(self):
        """Test bracket order with high transaction costs"""
        main_order = Order.create_bracket_order(
            symbol="AAPL",
            price=100.0,
            high_sell_price=110.0,
            low_sell_price=95.0,
            quantity=100,
            tx_cost=50.0  # $50 commission
        )

        stop_order = main_order._child_orders_dict["STOP"]
        profit_order = main_order._child_orders_dict["PROFIT_TAKER"]

        assert main_order.tx_cost == 50.0
        assert stop_order.tx_cost == 50.0
        assert profit_order.tx_cost == 50.0
