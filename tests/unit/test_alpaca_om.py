"""
Unit tests for AlpacaOrderManager with mocked Alpaca API.

These tests use a mocked Alpaca TradingClient to test the AlpacaOrderManager logic
without making real API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import os

from trading.core.classes import (
    Order, PriceData, BracketOrder, OrderStatus,
    OrderAction, OrderType, Position
)
from trading.core.om.alpaca_om import AlpacaOrderManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca TradingClient with realistic responses."""
    with patch('trading.core.om.alpaca_om.TradingClient') as MockClient:
        client = Mock()

        # Mock account
        account = Mock()
        account.account_number = "PA123456789"
        account.cash = "100000.00"
        account.buying_power = "100000.00"
        account.portfolio_value = "100000.00"
        account.equity = "100000.00"
        client.get_account.return_value = account

        # Mock order submission
        def mock_submit_order(order_data):
            order = Mock()
            order.id = f"alpaca-{id(order_data)}"
            order.status = "filled"
            order.filled_avg_price = "150.50"
            order.filled_qty = str(order_data.qty)
            order.submitted_at = datetime.now()
            order.filled_at = datetime.now()
            order.legs = None  # No legs for market orders
            return order

        client.submit_order.side_effect = mock_submit_order

        # Mock order status query
        def mock_get_order_by_id(order_id):
            order = Mock()
            order.id = order_id
            order.status = "filled"
            order.filled_avg_price = "150.50"
            order.filled_qty = "10"
            order.submitted_at = datetime.now()
            order.filled_at = datetime.now()
            order.legs = None
            return order

        client.get_order_by_id.side_effect = mock_get_order_by_id

        # Mock cancel
        client.cancel_order_by_id.return_value = Mock()

        MockClient.return_value = client
        yield client


@pytest.fixture
def alpaca_om(mock_alpaca_client):
    """AlpacaOrderManager with mocked client."""
    cfg = {
        "api_key": "test_key",
        "secret_key": "test_secret",
        "paper": True,
        "time_in_force": "day"
    }
    return AlpacaOrderManager(cfg)


@pytest.fixture
def sample_tick():
    """Sample price data for testing."""
    return [PriceData(
        symbol="SPY",
        timestamp=datetime(2024, 1, 15, 10, 0),
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=100000
    )]


# ============================================================================
# Test Class 1: Initialization & Configuration
# ============================================================================

class TestInitialization:
    """Test AlpacaOrderManager initialization and configuration."""

    def test_init_from_config(self, mock_alpaca_client):
        """Load credentials from config dict."""
        cfg = {"api_key": "test_key", "secret_key": "test_secret", "paper": True}
        om = AlpacaOrderManager(cfg)

        # Verify client was created
        assert om.client is not None
        assert om.time_in_force == "day"


    def test_init_missing_api_key(self, mock_alpaca_client):
        """Raise error if api_key is missing."""
        cfg = {"secret_key": "test_secret"}

        with pytest.raises(ValueError, match="api_key and secret_key"):
            AlpacaOrderManager(cfg)


    def test_init_missing_secret_key(self, mock_alpaca_client):
        """Raise error if secret_key is missing."""
        cfg = {"api_key": "test_key"}

        with pytest.raises(ValueError, match="api_key and secret_key"):
            AlpacaOrderManager(cfg)


    def test_init_custom_time_in_force(self, mock_alpaca_client):
        """Support custom time_in_force configuration."""
        cfg = {
            "api_key": "test_key",
            "secret_key": "test_secret",
            "time_in_force": "gtc"
        }
        om = AlpacaOrderManager(cfg)

        assert om.time_in_force == "gtc"


# ============================================================================
# Test Class 2: Market Order Submission
# ============================================================================

class TestMarketOrderSubmission:
    """Test market order submission to Alpaca."""

    def test_submit_market_buy_order(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Submit BUY order and verify it's tracked."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 10, tx_cost=1.0, current_tick=sample_tick
        )

        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)

        # Verify order submitted
        assert result.status == OrderStatus.PENDING
        assert result.platform_id is not None
        assert result.order_id in alpaca_om._local_to_remote

        # Verify Alpaca API called
        mock_alpaca_client.submit_order.assert_called_once()


    def test_submit_market_sell_order(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Submit SELL order."""
        order = Order.create_market_order(
            "SPY", OrderAction.SELL, 5, tx_cost=1.0, current_tick=sample_tick
        )

        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=0.0)

        assert result.status == OrderStatus.PENDING
        assert result.platform_id is not None
        mock_alpaca_client.submit_order.assert_called_once()


    def test_market_order_id_mapping(self, alpaca_om, sample_tick):
        """Verify local<->Alpaca ID mapping stored."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 1, tx_cost=0.0, current_tick=sample_tick
        )
        local_id = order.order_id

        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)

        # Check mapping exists
        assert local_id in alpaca_om._local_to_remote
        alpaca_id = alpaca_om._local_to_remote[local_id]
        assert alpaca_id is not None
        assert result.platform_id == alpaca_id


# ============================================================================
# Test Class 3: Bracket Order Submission
# ============================================================================

class TestBracketOrderSubmission:
    """Test bracket order submission (native Alpaca brackets)."""

    def test_submit_bracket_order(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Submit bracket order with stop and profit legs."""
        # Mock bracket order response with legs
        def mock_submit_bracket(order_data):
            main_order = Mock()
            main_order.id = "main-123"
            main_order.status = "accepted"
            main_order.filled_avg_price = None
            main_order.filled_qty = None
            main_order.submitted_at = datetime.now()
            main_order.filled_at = None

            # Create mock legs
            stop_leg = Mock()
            stop_leg.id = "stop-456"
            stop_leg.order_type = "stop"
            stop_leg.status = "held"

            profit_leg = Mock()
            profit_leg.id = "profit-789"
            profit_leg.order_type = "limit"
            profit_leg.status = "held"

            main_order.legs = [stop_leg, profit_leg]
            return main_order

        mock_alpaca_client.submit_order.side_effect = mock_submit_bracket

        # Create bracket order
        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=160.0,
            low_sell_price=145.0,
            quantity=10,
            tx_cost=1.0,
            current_tick=sample_tick
        )

        result = alpaca_om.submit_order(bracket, sample_tick, {}, pf_cash=10000.0)

        # Verify main order submitted
        assert result.status == OrderStatus.PENDING
        assert result.platform_id == "main-123"

        # Verify child leg IDs tracked
        assert result.order_id in alpaca_om._bracket_child_map
        child_map = alpaca_om._bracket_child_map[result.order_id]
        assert child_map["STOP"] == "stop-456"
        assert child_map["PROFIT"] == "profit-789"

        # Verify child orders have platform IDs
        assert result.get_child_order("STOP").platform_id == "stop-456"
        assert result.get_child_order("PROFIT").platform_id == "profit-789"


    def test_bracket_buy_only(self, alpaca_om, sample_tick):
        """Bracket orders only support BUY entries."""
        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=160.0,
            low_sell_price=145.0,
            quantity=10,
            tx_cost=1.0,
            current_tick=sample_tick
        )

        # Change action to SELL
        bracket.action = OrderAction.SELL

        with pytest.raises(NotImplementedError, match="BUY entries only"):
            alpaca_om.submit_order(bracket, sample_tick, {}, pf_cash=10000.0)


# ============================================================================
# Test Class 4: Order Status Updates
# ============================================================================

class TestOrderStatusUpdates:
    """Test order status updates from Alpaca."""

    def test_update_market_order_to_filled(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Update market order status from PENDING to FILLED."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 10, tx_cost=0.0, current_tick=sample_tick
        )

        # Submit order (status = PENDING)
        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)
        assert result.status == OrderStatus.PENDING

        # Mock filled status - use side_effect to override fixture's side_effect
        def mock_filled_order(order_id):
            return Mock(
                id=order_id,
                status="filled",
                filled_avg_price="150.75",
                filled_qty="10",
                filled_at=datetime.now(),
                legs=None  # Important: market orders have no legs
            )

        mock_alpaca_client.get_order_by_id.side_effect = mock_filled_order

        # Update status
        updated = alpaca_om.update_order_status(result, sample_tick, {}, 0.0)

        assert updated.status == OrderStatus.FILLED
        assert updated.price == 150.75
        assert updated.quantity == 10


    def test_update_bracket_entry_fill(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Update bracket order when entry fills."""
        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=160.0,
            low_sell_price=145.0,
            quantity=10,
            tx_cost=1.0,
            current_tick=sample_tick
        )

        # Submit (PENDING)
        result = alpaca_om.submit_order(bracket, sample_tick, {}, pf_cash=10000.0)

        # Mock entry filled
        mock_alpaca_client.get_order_by_id.return_value = Mock(
            id=result.platform_id,
            status="filled",
            filled_avg_price="150.50",
            filled_qty="10",
            filled_at=datetime.now(),
            legs=[
                Mock(id="stop-1", status="held", order_type="stop"),
                Mock(id="profit-1", status="held", order_type="limit")
            ]
        )

        # Update status
        updated = alpaca_om.update_order_status(result, sample_tick, {}, 0.0)

        # Should transition to PENDING_SALE (entry filled, awaiting exit)
        assert updated.status == OrderStatus.PENDING_SALE
        assert updated.price == 150.50
        assert updated.quantity == 10


    def test_update_bracket_stop_loss_filled(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Update bracket when stop-loss leg fills."""
        # Mock submit with legs
        def mock_submit_bracket(order_data):
            main_order = Mock()
            main_order.id = "main-bracket-123"
            main_order.status = "filled"
            main_order.filled_avg_price = "150.50"
            main_order.filled_qty = "10"
            main_order.submitted_at = datetime.now()
            main_order.filled_at = datetime.now()
            main_order.legs = [
                Mock(id="stop-456", order_type="stop", status="held"),
                Mock(id="profit-789", order_type="limit", status="held")
            ]
            return main_order

        mock_alpaca_client.submit_order.side_effect = mock_submit_bracket

        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=160.0,
            low_sell_price=145.0,
            quantity=10,
            tx_cost=1.0,
            current_tick=sample_tick
        )

        # Submit - entry fills immediately in this mock
        result = alpaca_om.submit_order(bracket, sample_tick, {}, pf_cash=10000.0)

        # Create side_effect for get_order_by_id that returns different results on each call
        call_count = [0]  # Use list to allow modification in nested function

        def mock_get_bracket_status(order_id):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: entry filled, legs held
                return Mock(
                    id=order_id,
                    status="filled",
                    filled_avg_price="150.50",
                    filled_qty="10",
                    filled_at=datetime.now(),
                    legs=[
                        Mock(id="stop-456", status="held", order_type="stop"),
                        Mock(id="profit-789", status="held", order_type="limit")
                    ]
                )
            else:
                # Second call: stop-loss filled
                return Mock(
                    id=order_id,
                    status="filled",
                    filled_avg_price="150.50",
                    filled_qty="10",
                    filled_at=datetime.now(),
                    legs=[
                        Mock(
                            id="stop-456",
                            status="filled",
                            order_type="stop",
                            filled_avg_price="145.00",
                            filled_qty="10",
                            filled_at=datetime.now()
                        ),
                        Mock(id="profit-789", status="canceled", order_type="limit")
                    ]
                )

        mock_alpaca_client.get_order_by_id.side_effect = mock_get_bracket_status

        # First update - entry filled, transition to PENDING_SALE
        updated = alpaca_om.update_order_status(result, sample_tick, {}, 0.0)
        assert updated.status == OrderStatus.PENDING_SALE

        # Second update - stop-loss filled
        final = alpaca_om.update_order_status(updated, sample_tick, {}, 0.0)

        # Should be FILLED with SOLD_ORDER set
        assert final.status == OrderStatus.FILLED
        assert final.SOLD_ORDER is not None
        assert final.SOLD_ORDER.type == OrderType.STOP_LOSS
        assert final.SOLD_ORDER.price == 145.00


    def test_batch_status_update(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Update multiple orders in batch."""
        # Create 3 orders
        orders = {}
        for i in range(3):
            order = Order.create_market_order(
                "SPY", OrderAction.BUY, 1, tx_cost=0.0, current_tick=sample_tick
            )
            result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)
            orders[result.order_id] = result

        # Mock all filled
        mock_alpaca_client.get_order_by_id.return_value = Mock(
            status="filled",
            filled_avg_price="150.50",
            filled_qty="1",
            filled_at=datetime.now()
        )

        # Batch update
        changed_ids = alpaca_om.update_pending_orders(sample_tick, {}, 0.0)

        # All 3 should have changed status
        assert len(changed_ids) == 3
        for order in orders.values():
            assert order.status == OrderStatus.FILLED


# ============================================================================
# Test Class 5: Order Cancellation
# ============================================================================

class TestOrderCancellation:
    """Test order cancellation via Alpaca API."""

    def test_cancel_pending_order(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Cancel a pending order via Alpaca."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 10, tx_cost=0.0, current_tick=sample_tick
        )

        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)

        # Cancel
        alpaca_om.cancel_all_pending_orders()

        # Verify Alpaca cancel called
        mock_alpaca_client.cancel_order_by_id.assert_called_with(result.platform_id)


    def test_cancel_handles_api_error(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Handle API errors gracefully when canceling."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 10, tx_cost=0.0, current_tick=sample_tick
        )

        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)

        # Mock API error
        mock_alpaca_client.cancel_order_by_id.side_effect = Exception("API error")

        # Should not raise, just log warning
        alpaca_om.cancel_all_pending_orders()  # Should complete without error


# ============================================================================
# Test Class 6: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_update_order_without_platform_id(self, alpaca_om, sample_tick):
        """Skip update if order has no platform_id."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 10, tx_cost=0.0, current_tick=sample_tick
        )
        # Don't submit, so no platform_id

        # Update should return order unchanged
        result = alpaca_om.update_order_status(order, sample_tick, {}, 0.0)
        assert result == order


    def test_update_handles_api_error(self, alpaca_om, mock_alpaca_client, sample_tick):
        """Handle API errors during status update."""
        order = Order.create_market_order(
            "SPY", OrderAction.BUY, 10, tx_cost=0.0, current_tick=sample_tick
        )

        result = alpaca_om.submit_order(order, sample_tick, {}, pf_cash=10000.0)

        # Mock API error
        mock_alpaca_client.get_order_by_id.side_effect = Exception("Network error")

        # Should return order unchanged, not raise
        updated = alpaca_om.update_order_status(result, sample_tick, {}, 0.0)
        assert updated.status == OrderStatus.PENDING  # Unchanged


    def test_bracket_without_stop_order(self, alpaca_om, sample_tick):
        """Raise error if bracket missing STOP child."""
        # Create a bracket manually without children
        bracket = BracketOrder(
            action=OrderAction.BUY,
            type=OrderType.BRACKET,
            symbol="SPY",
            price=150.0,
            quantity=10,
            cash=1500.0,
            placed_datetime=datetime.now()
        )
        # Add only PROFIT child, not STOP
        profit_order = Order(
            action=OrderAction.SELL,
            type=OrderType.PROFIT_TAKER,
            symbol="SPY",
            price=160.0,
            quantity=10,
            cash=0.0,
            placed_datetime=datetime.now(),
            parent_id=bracket.order_id
        )
        bracket.add_child_order("PROFIT", profit_order)
        # Missing STOP child

        # AlpacaOM catches KeyError from get_child_order and raises ValueError
        with pytest.raises(ValueError, match="STOP and PROFIT child orders"):
            alpaca_om.submit_order(bracket, sample_tick, {}, pf_cash=10000.0)


# ============================================================================
# Summary
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
