"""
Integration tests for AlpacaOrderManager with real Alpaca paper account.

CRITICAL VALIDATION PATTERN:
Every test that submits an order MUST independently verify it exists in Alpaca by:
1. Submit order via AlpacaOrderManager
2. Get Alpaca ID from mapping
3. INDEPENDENTLY query Alpaca API for the order
4. Assert order details match (symbol, qty, side, status)

Setup Requirements:
- Set environment variables: ALPACA_API_KEY and ALPACA_SECRET_KEY
- Alpaca paper account must have funds
- Run during market hours for best results (or use extended hours)

Run with: pytest tests/integration/test_alpaca_om_integration.py -v -m integration
"""

import pytest
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from trading.core.classes import (
    Order, PriceData, BracketOrder, OrderStatus,
    OrderAction, OrderType, Position
)
from trading.core.om.alpaca_om import AlpacaOrderManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def alpaca_om_live():
    """
    Real AlpacaOrderManager connected to paper account.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        pytest.skip("Alpaca API credentials not found in environment")

    cfg = {
        "api_key": api_key,
        "secret_key": secret_key,
        "paper": True,
        "time_in_force": "day"
    }

    om = AlpacaOrderManager(cfg)

    # Pre-test cleanup
    print("\n[Setup] Canceling existing pending orders...")
    om.cancel_all_pending_orders()

    yield om

    # Post-test cleanup
    print("\n[Teardown] Canceling remaining pending orders...")
    om.cancel_all_pending_orders()


@pytest.fixture
def sample_tick():
    """Sample price data for testing."""
    return [PriceData(
        symbol="SPY",
        timestamp=datetime.now(),
        open=450.0,
        high=451.0,
        low=449.0,
        close=450.5,
        volume=100000
    )]


# ============================================================================
# Test Class 1: Connection & Setup Tests
# ============================================================================

@pytest.mark.integration
class TestAlpacaConnection:
    """Test basic Alpaca API connectivity."""

    def test_connect_to_alpaca(self, alpaca_om_live):
        """Verify can connect to Alpaca paper account."""
        account = alpaca_om_live.client.get_account()

        assert account.account_number is not None
        assert str(account.account_number).startswith("PA")  # Paper account
        assert float(account.buying_power) > 0

        print(f"\n[OK] Connected to paper account: {account.account_number}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")


    def test_list_existing_positions(self, alpaca_om_live):
        """List any existing positions in paper account."""
        positions = alpaca_om_live.client.get_all_positions()

        print(f"\n[OK] Found {len(positions)} existing positions:")
        for pos in positions[:5]:  # Show first 5
            print(f"  - {pos.symbol}: {pos.qty} @ ${float(pos.current_price):.2f}")


# ============================================================================
# Test Class 2: Market Orders with Alpaca Verification
# ============================================================================

@pytest.mark.integration
class TestMarketOrdersRealAlpaca:
    """Test market orders with real Alpaca paper account."""

    def test_submit_market_buy_and_verify_in_alpaca(self, alpaca_om_live, sample_tick):
        """
        CRITICAL TEST: Submit BUY order, verify it exists in Alpaca.

        Steps:
        1. Submit market BUY via AlpacaOrderManager
        2. Independently query Alpaca API for the order
        3. Verify order details match (symbol, qty, side, status)
        4. Clean up by selling position or canceling order
        """
        # Step 1: Submit order
        order = Order.create_market_order("SPY", OrderAction.BUY, 1, 0.0, sample_tick)

        result = alpaca_om_live.submit_order(order, sample_tick, {}, pf_cash=100000.0)

        print(f"\n1. Submitted order via AlpacaOM:")
        print(f"   Local ID: {result.order_id}")
        print(f"   Status: {result.status.name}")
        print(f"   Qty: {result.quantity}")

        # Step 2: Get Alpaca ID
        assert result.order_id in alpaca_om_live._local_to_remote
        alpaca_order_id = alpaca_om_live._local_to_remote[result.order_id]
        print(f"   Alpaca ID: {alpaca_order_id}")

        # Step 3: CRITICAL - Verify order exists in Alpaca
        alpaca_order = alpaca_om_live.client.get_order_by_id(alpaca_order_id)

        print(f"\n2. Verified in Alpaca API:")
        print(f"   ID: {alpaca_order.id}")
        print(f"   Symbol: {alpaca_order.symbol}")
        print(f"   Qty: {alpaca_order.qty}")
        print(f"   Side: {alpaca_order.side}")
        print(f"   Status: {alpaca_order.status}")
        if alpaca_order.filled_avg_price:
            print(f"   Filled Price: ${float(alpaca_order.filled_avg_price):.2f}")

        # Assertions - ORDER REALLY EXISTS IN ALPACA
        assert str(alpaca_order.id) == alpaca_order_id
        assert alpaca_order.symbol == "SPY"
        assert int(alpaca_order.qty) == 1
        assert alpaca_order.side.name.lower() == "buy"  # Convert to lowercase
        # Use .value to get just the status string (e.g., "accepted" not "OrderStatus.ACCEPTED")
        assert alpaca_order.status.value.lower() in ["filled", "new", "accepted", "pending_new"]

        # Step 4: Clean up
        print(f"\n3. Cleaning up...")
        if str(alpaca_order.status).lower() == "filled":
            print(f"   Order filled, selling position...")
            sell_order = Order.create_market_order("SPY", OrderAction.SELL, 1, 0.0, sample_tick)
            positions = {"SPY": Position("SPY", 1)}
            alpaca_om_live.submit_order(sell_order, sample_tick, positions, 0.0)
            print(f"   [OK] Position sold")
        else:
            print(f"   Order still pending, canceling...")
            alpaca_om_live.cancel_all_pending_orders()
            print(f"   [OK] Order canceled")


    def test_submit_market_sell_and_verify(self, alpaca_om_live, sample_tick):
        """
        Submit SELL order and verify it exists in Alpaca.
        First buys 1 share, then sells it.
        """
        print("\n1. First, buy 1 share of SPY...")
        buy_order = Order.create_market_order("SPY", OrderAction.BUY, 1, 0.0, sample_tick)
        buy_result = alpaca_om_live.submit_order(buy_order, sample_tick, {}, pf_cash=100000.0)

        # Wait for fill
        time.sleep(2)
        alpaca_om_live.update_order_status(buy_result, sample_tick, {}, 0.0)

        if buy_result.status != OrderStatus.FILLED:
            pytest.skip("Buy order not filled quickly enough")

        print(f"   [OK] Buy filled at ${buy_result.price:.2f}")

        # Now sell
        print("\n2. Submitting SELL order...")
        sell_order = Order.create_market_order("SPY", OrderAction.SELL, 1, 0.0, sample_tick)
        positions = {"SPY": Position("SPY", 1)}

        sell_result = alpaca_om_live.submit_order(sell_order, sample_tick, positions, 0.0)

        print(f"   Local ID: {sell_result.order_id}")
        print(f"   Status: {sell_result.status.name}")

        # Verify in Alpaca
        alpaca_id = alpaca_om_live._local_to_remote[sell_result.order_id]
        alpaca_order = alpaca_om_live.client.get_order_by_id(alpaca_id)

        print(f"\n3. Verified SELL in Alpaca:")
        print(f"   Alpaca ID: {alpaca_order.id}")
        print(f"   Side: {alpaca_order.side}")
        print(f"   Status: {alpaca_order.status}")

        assert alpaca_order.symbol == "SPY"
        assert alpaca_order.side.name.lower() == "sell"  # Convert to lowercase
        assert int(alpaca_order.qty) == 1


    @pytest.mark.slow
    def test_query_all_orders_from_alpaca(self, alpaca_om_live, sample_tick):
        """
        Verify we can query all orders from Alpaca and find our submitted orders.
        """
        # Submit 2 test orders
        print("\n1. Submitting 2 test orders...")

        order1 = Order.create_market_order("SPY", OrderAction.BUY, 1, 0.0, sample_tick)
        result1 = alpaca_om_live.submit_order(order1, sample_tick, {}, pf_cash=100000.0)
        print(f"   Order 1: {result1.order_id}")

        time.sleep(1)

        order2 = Order.create_market_order("SPY", OrderAction.BUY, 1, 0.0, sample_tick)
        result2 = alpaca_om_live.submit_order(order2, sample_tick, {}, pf_cash=100000.0)
        print(f"   Order 2: {result2.order_id}")

        # Query ALL orders from Alpaca
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)
        all_orders = alpaca_om_live.client.get_orders(request)

        print(f"\n2. Queried Alpaca - found {len(all_orders)} total orders")

        # Find our orders in the list
        alpaca_id1 = alpaca_om_live._local_to_remote[result1.order_id]
        alpaca_id2 = alpaca_om_live._local_to_remote[result2.order_id]

        found_ids = [str(order.id) for order in all_orders]  # Convert UUIDs to strings
        assert alpaca_id1 in found_ids
        assert alpaca_id2 in found_ids

        print(f"\n3. Verification:")
        print(f"   [OK] Found order {alpaca_id1} in Alpaca order list")
        print(f"   [OK] Found order {alpaca_id2} in Alpaca order list")

        # Clean up
        alpaca_om_live.cancel_all_pending_orders()


# ============================================================================
# Test Class 3: Bracket Orders with Alpaca Verification
# ============================================================================

@pytest.mark.integration
class TestBracketOrdersRealAlpaca:
    """Test bracket orders with real Alpaca paper account."""

    def test_submit_bracket_verify_in_alpaca(self, alpaca_om_live, sample_tick):
        """
        CRITICAL TEST: Submit bracket order and verify in Alpaca.
        Uses native Alpaca bracket orders (server-side).
        Verifies child orders are captured in local BracketOrder object.
        """
        print("\n1. Creating bracket order with far-from-market prices...")

        # Use prices far from current market to avoid accidental fills
        # SPY typically trades around $400-700, so these won't execute
        profit_price = 999.99  # Far above market
        stop_price = 50.00     # Far below market

        print(f"   Profit target: ${profit_price:.2f} (far above market)")
        print(f"   Stop loss: ${stop_price:.2f} (far below market)")

        print("\n2. Submitting bracket order...")

        # Create bracket order with far-from-market prices
        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=profit_price,  # Won't fill
            low_sell_price=stop_price,     # Won't fill
            quantity=1,
            tx_cost=0.0,
            current_tick=sample_tick
        )

        # Submit
        result = alpaca_om_live.submit_order(bracket, sample_tick, {}, pf_cash=100000.0)

        print(f"   Local Bracket ID: {result.order_id}")
        print(f"   Status: {result.status.name}")

        # CRITICAL: Verify child orders are captured locally
        stop_child = result.get_child_order('STOP')
        profit_child = result.get_child_order('PROFIT')

        assert stop_child is not None, "STOP child order not found in local BracketOrder"
        assert profit_child is not None, "PROFIT child order not found in local BracketOrder"

        print(f"\n3. Local child orders captured:")
        print(f"   STOP order: {stop_child.order_id}")
        print(f"   - Price: ${stop_child.price:.2f}")
        print(f"   - Platform ID: {stop_child.platform_id}")
        print(f"   PROFIT order: {profit_child.order_id}")
        print(f"   - Price: ${profit_child.price:.2f}")
        print(f"   - Platform ID: {profit_child.platform_id}")

        # Verify platform IDs were assigned to child orders
        assert stop_child.platform_id is not None, "STOP child missing platform_id"
        assert profit_child.platform_id is not None, "PROFIT child missing platform_id"

        # Verify initial buy in Alpaca
        assert result.order_id in alpaca_om_live._local_to_remote
        alpaca_order_id = alpaca_om_live._local_to_remote[result.order_id]

        alpaca_order = alpaca_om_live.client.get_order_by_id(alpaca_order_id)

        print(f"\n4. Verified in Alpaca API:")
        print(f"   Alpaca ID: {alpaca_order.id}")
        print(f"   Symbol: {alpaca_order.symbol}")
        print(f"   Side: {alpaca_order.side}")
        print(f"   Order Class: {alpaca_order.order_class}")
        print(f"   Status: {alpaca_order.status}")

        assert alpaca_order.symbol == "SPY"
        assert int(alpaca_order.qty) == 1
        assert alpaca_order.side.name.lower() == "buy"
        assert alpaca_order.order_class.name.lower() == "bracket"

        # Verify legs exist in Alpaca and match local child orders
        if hasattr(alpaca_order, 'legs') and alpaca_order.legs:
            print(f"\n5. Bracket legs in Alpaca:")
            alpaca_leg_ids = set()
            for leg in alpaca_order.legs:
                print(f"   - {leg.order_type}: {str(leg.id)} ({leg.status})")
                alpaca_leg_ids.add(str(leg.id))

            assert len(alpaca_order.legs) == 2, "Expected 2 legs (stop & profit)"

            # CRITICAL: Verify local child orders match Alpaca legs
            assert stop_child.platform_id in alpaca_leg_ids, "STOP leg not found in Alpaca"
            assert profit_child.platform_id in alpaca_leg_ids, "PROFIT leg not found in Alpaca"
            print(f"   [OK] Local child orders match Alpaca legs")

        # Clean up
        print(f"\n6. Cleaning up...")
        alpaca_om_live.cancel_all_pending_orders()
        print(f"   [OK] Bracket order canceled")


    def test_bracket_child_order_platform_ids(self, alpaca_om_live, sample_tick):
        """
        Test that bracket order child orders receive platform IDs from Alpaca legs.
        """
        print("\n1. Submitting bracket order...")

        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=999.99,  # Won't fill
            low_sell_price=50.00,    # Won't fill
            quantity=1,
            tx_cost=0.0,
            current_tick=sample_tick
        )

        result = alpaca_om_live.submit_order(bracket, sample_tick, {}, pf_cash=100000.0)

        # Verify child orders exist
        stop_order = result.get_child_order("STOP")
        profit_order = result.get_child_order("PROFIT")

        print(f"\n2. Child order platform IDs:")
        print(f"   STOP platform_id: {stop_order.platform_id}")
        print(f"   PROFIT platform_id: {profit_order.platform_id}")

        # CRITICAL: Both child orders must have platform IDs
        assert stop_order.platform_id is not None
        assert profit_order.platform_id is not None
        assert stop_order.platform_id != profit_order.platform_id

        # Verify in bracket_child_map
        assert result.order_id in alpaca_om_live._bracket_child_map
        child_map = alpaca_om_live._bracket_child_map[result.order_id]

        print(f"\n3. Bracket child map:")
        print(f"   STOP: {child_map['STOP']}")
        print(f"   PROFIT: {child_map['PROFIT']}")

        # child_map values are UUIDs, platform_id are strings - compare as strings
        assert str(child_map["STOP"]) == stop_order.platform_id
        assert str(child_map["PROFIT"]) == profit_order.platform_id

        print(f"\n[OK] All child orders have correct platform IDs")

        # Clean up
        alpaca_om_live.cancel_all_pending_orders()


    def test_bracket_order_independent_verification(self, alpaca_om_live, sample_tick):
        """
        Test bracket order by independently querying each leg in Alpaca.
        This ensures child orders actually exist as separate orders in Alpaca.
        """
        print("\n1. Submitting bracket order...")

        bracket = BracketOrder.create_bracket_order(
            symbol="SPY",
            high_sell_price=999.99,
            low_sell_price=50.00,
            quantity=1,
            tx_cost=0.0,
            current_tick=sample_tick
        )

        result = alpaca_om_live.submit_order(bracket, sample_tick, {}, pf_cash=100000.0)

        stop_order = result.get_child_order("STOP")
        profit_order = result.get_child_order("PROFIT")

        print(f"   Main order: {result.platform_id}")
        print(f"   STOP leg: {stop_order.platform_id}")
        print(f"   PROFIT leg: {profit_order.platform_id}")

        # CRITICAL: Independently verify STOP leg exists in Alpaca
        print(f"\n2. Independently verifying STOP leg in Alpaca...")
        try:
            stop_leg_alpaca = alpaca_om_live.client.get_order_by_id(stop_order.platform_id)
            print(f"   [OK] STOP leg found: {stop_leg_alpaca.id}")
            print(f"       Type: {stop_leg_alpaca.order_type}")
            print(f"       Status: {stop_leg_alpaca.status}")
            assert str(stop_leg_alpaca.id) == stop_order.platform_id
        except Exception as e:
            pytest.fail(f"STOP leg not found in Alpaca: {e}")

        # CRITICAL: Independently verify PROFIT leg exists in Alpaca
        print(f"\n3. Independently verifying PROFIT leg in Alpaca...")
        try:
            profit_leg_alpaca = alpaca_om_live.client.get_order_by_id(profit_order.platform_id)
            print(f"   [OK] PROFIT leg found: {profit_leg_alpaca.id}")
            print(f"       Type: {profit_leg_alpaca.order_type}")
            print(f"       Status: {profit_leg_alpaca.status}")
            assert str(profit_leg_alpaca.id) == profit_order.platform_id
        except Exception as e:
            pytest.fail(f"PROFIT leg not found in Alpaca: {e}")

        print(f"\n[OK] All bracket order legs independently verified in Alpaca")

        # Clean up
        alpaca_om_live.cancel_all_pending_orders()


# ============================================================================
# Test Class 4: Order Tracking & Status
# ============================================================================

@pytest.mark.integration
class TestOrderTracking:
    """Test order tracking and status synchronization."""

    def test_local_to_alpaca_id_mapping_persists(self, alpaca_om_live, sample_tick):
        """Verify ID mappings maintained throughout order lifecycle."""
        order = Order.create_market_order("SPY", OrderAction.BUY, 1, 0.0, sample_tick)
        local_id = order.order_id

        result = alpaca_om_live.submit_order(order, sample_tick, {}, pf_cash=100000.0)

        # Check mapping exists
        assert local_id in alpaca_om_live._local_to_remote
        alpaca_id = alpaca_om_live._local_to_remote[local_id]
        assert alpaca_id is not None
        assert result.platform_id == alpaca_id

        print(f"\n[OK] ID mapping maintained:")
        print(f"  Local ID: {local_id}")
        print(f"  Alpaca ID: {alpaca_id}")
        print(f"  Platform ID: {result.platform_id}")

        # Clean up
        if result.status != OrderStatus.FILLED:
            alpaca_om_live.cancel_all_pending_orders()
        else:
            # Sell position
            sell = Order.create_market_order("SPY", OrderAction.SELL, 1, 0.0, sample_tick)
            alpaca_om_live.submit_order(sell, sample_tick, {"SPY": Position("SPY", 1)}, 0.0)


    def test_order_status_sync_from_alpaca(self, alpaca_om_live, sample_tick):
        """Verify order status updates from Alpaca."""
        order = Order.create_market_order("SPY", OrderAction.BUY, 1, 0.0, sample_tick)

        result = alpaca_om_live.submit_order(order, sample_tick, {}, pf_cash=100000.0)
        initial_status = result.status

        print(f"\n1. Initial status: {initial_status.name}")

        # Update status from Alpaca
        time.sleep(1)
        updated = alpaca_om_live.update_order_status(result, sample_tick, {}, 0.0)

        print(f"2. Updated status: {updated.status.name}")

        # Verify status changed or is filled
        assert updated.status in [OrderStatus.FILLED, OrderStatus.PENDING]

        # Clean up
        if updated.status != OrderStatus.FILLED:
            alpaca_om_live.cancel_all_pending_orders()
        else:
            sell = Order.create_market_order("SPY", OrderAction.SELL, 1, 0.0, sample_tick)
            alpaca_om_live.submit_order(sell, sample_tick, {"SPY": Position("SPY", 1)}, 0.0)


# ============================================================================
# Summary
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
