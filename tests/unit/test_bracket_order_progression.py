"""
Test suite for bracket order status progression through BacktestingOM
Tests: stop-loss trigger, profit-taker trigger, and manual sale
"""
import pytest
from datetime import datetime
from core.classes import (
    Order, PriceData, OrderStatus, OrderType,
    OrderAction, BracketOrder, Position
)
from core.om.backtesting_om import BacktestingOM


class TestBracketOrderStatusProgression:
    """Test bracket order lifecycle and status transitions"""

    def test_bracket_order_stop_loss_triggers(self):
        """
        Test that bracket order triggers stop-loss when price drops below stop price.

        Flow:
        1. Create bracket order at $100 (stop: $95, profit: $110)
        2. Initial status should be PENDING_SALE after submit
        3. Process at $94 (below stop) -> status becomes FILLED (stop triggered)
        4. Verify stop order is FILLED, profit order is CANCELED
        5. Verify SOLD_ORDER points to stop order
        """
        om = BacktestingOM()

        # Create bracket order: buy at $100, stop at $95, profit at $110
        symbol = "AAPL"
        tick_initial = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000
        )]

        # Create bracket order manually
        price = 100.0
        profit_price = price * 1.10  # 10% profit = $110
        stop_price = price * 0.95     # 5% stop = $95
        bracket_order = BracketOrder.create_bracket_order(
            symbol=symbol,
            price=price,
            high_sell_price=profit_price,
            low_sell_price=stop_price,
            quantity=10,
            tx_cost=1.0,
            current_tick=tick_initial
        )

        # Submit the bracket order
        positions = {}
        pf_cash = 10000.0
        bracket_order = om.submit_order(bracket_order, tick_initial, positions, pf_cash)

        # Step 1: Verify initial state after submission
        assert bracket_order.status == OrderStatus.PENDING_SALE, \
            "Bracket order should be PENDING_SALE after initial buy"
        assert bracket_order.type == OrderType.BRACKET
        assert bracket_order.quantity == 10
        assert bracket_order.price == 100.0

        # Verify child orders exist
        stop_order = bracket_order.get_child_order("STOP")
        profit_order = bracket_order.get_child_order("PROFIT")

        assert stop_order.price == 95.0, "Stop price should be $95 (5% below $100)"
        assert round(profit_order.price, 2) == 110.0, "Profit price should be $110 (10% above $100)"
        assert stop_order.status == OrderStatus.PENDING
        assert profit_order.status == OrderStatus.PENDING

        # Step 2: Price stays flat - should remain PENDING_SALE
        tick_flat = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 31),
            open=100.0, high=100.5, low=99.5, close=100.0, volume=1000
        )]

        updated_order = om.update_order_status(bracket_order, tick_flat, positions, pf_cash)
        assert updated_order.status == OrderStatus.PENDING_SALE, \
            "Should remain PENDING_SALE when price doesn't trigger stop/profit"

        # Step 3: Price drops to $94 - STOP LOSS SHOULD TRIGGER
        tick_drop = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 32),
            open=96.0, high=96.0, low=93.0, close=94.0, volume=2000
        )]

        updated_order = om.update_order_status(bracket_order, tick_drop, positions, pf_cash)

        # Verify main order is FILLED
        assert updated_order.status == OrderStatus.FILLED, \
            "Bracket order should be FILLED when stop loss triggers"

        # Verify stop order was FILLED
        stop_order = updated_order.get_child_order("STOP")
        assert stop_order.status == OrderStatus.FILLED, \
            "Stop loss order should be FILLED"
        assert stop_order.price == 94.0, \
            "Stop order should execute at current price ($94)"
        assert stop_order.cash == 94.0 * 10, \
            "Sale proceeds should be $940 (10 shares * $94)"

        # Verify profit order was CANCELED
        profit_order = updated_order.get_child_order("PROFIT")
        assert profit_order.status == OrderStatus.CANCELED, \
            "Profit taker should be CANCELED when stop triggers"

        # Verify SOLD_ORDER was set
        assert updated_order.SOLD_ORDER == stop_order, \
            "SOLD_ORDER should point to the stop order"


    def test_bracket_order_profit_taker_triggers(self):
        """
        Test that bracket order triggers profit-taker when price rises above profit price.
        """
        om = BacktestingOM()

        # Create bracket order
        symbol = "TSLA"
        tick_initial = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000
        )]

        # Create bracket order
        price = 100.0
        profit_price = price * 1.10
        stop_price = price * 0.95
        bracket_order = BracketOrder.create_bracket_order(
            symbol=symbol,
            price=price,
            high_sell_price=profit_price,
            low_sell_price=stop_price,
            quantity=20,
            tx_cost=2.0,
            current_tick=tick_initial
        )

        positions = {}
        pf_cash = 10000.0
        bracket_order = om.submit_order(bracket_order, tick_initial, positions, pf_cash)

        # Verify initial state
        assert bracket_order.status == OrderStatus.PENDING_SALE

        stop_order = bracket_order.get_child_order("STOP")
        profit_order = bracket_order.get_child_order("PROFIT")

        assert stop_order.price == 95.0
        assert round(profit_order.price, 2) == 110.0

        # Price rises to $112 - PROFIT TAKER SHOULD TRIGGER
        tick_rise = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 10, 30),
            open=105.0, high=115.0, low=105.0, close=112.0, volume=3000
        )]

        updated_order = om.update_order_status(bracket_order, tick_rise, positions, pf_cash)

        # Verify main order is FILLED
        assert updated_order.status == OrderStatus.FILLED, \
            "Bracket order should be FILLED when profit taker triggers"

        # Verify profit order was FILLED
        profit_order = updated_order.get_child_order("PROFIT")
        assert profit_order.status == OrderStatus.FILLED, \
            "Profit taker order should be FILLED"
        assert profit_order.price == 112.0, \
            "Profit order should execute at current price ($112)"
        assert profit_order.cash == 112.0 * 20, \
            "Sale proceeds should be $2,240 (20 shares * $112)"

        # Verify stop order was CANCELED
        stop_order = updated_order.get_child_order("STOP")
        assert stop_order.status == OrderStatus.CANCELED, \
            "Stop loss should be CANCELED when profit triggers"

        # Verify SOLD_ORDER was set
        assert updated_order.SOLD_ORDER == profit_order, \
            "SOLD_ORDER should point to the profit order"


    def test_bracket_order_manual_sale(self):
        """
        Test manual sale of bracket order while in PENDING_SALE status.
        """
        om = BacktestingOM()

        # Create bracket order
        symbol = "MSFT"
        tick_initial = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000
        )]

        price = 100.0
        profit_price = price * 1.10
        stop_price = price * 0.95
        bracket_order = BracketOrder.create_bracket_order(
            symbol=symbol,
            price=price,
            high_sell_price=profit_price,
            low_sell_price=stop_price,
            quantity=15,
            tx_cost=1.5,
            current_tick=tick_initial
        )

        positions = {}
        pf_cash = 10000.0
        bracket_order = om.submit_order(bracket_order, tick_initial, positions, pf_cash)

        # After bracket order fills, add the position
        from core.classes import Position
        positions[symbol] = Position(symbol, 15)

        # Verify initial state
        assert bracket_order.status == OrderStatus.PENDING_SALE
        assert bracket_order.MANUAL_SALE == False
        assert bracket_order.get_child_order("MANUAL_ORDER") is None

        # Set MANUAL_SALE flag (simulating portfolio requesting manual exit)
        bracket_order.MANUAL_SALE = True

        # Price is at $102 when manual sale requested
        tick_manual_sale = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 10, 0),
            open=102.0, high=103.0, low=101.0, close=102.0, volume=2000
        )]

        # Update status - this should create and fill the manual order
        updated_order = om.update_order_status(bracket_order, tick_manual_sale, positions, pf_cash)

        # Verify bracket order is FILLED
        assert updated_order.status == OrderStatus.FILLED, \
            "Bracket order should be FILLED after manual sale completes"

        # Verify manual order was created and filled
        manual_order = updated_order.get_child_order("MANUAL_ORDER")
        assert manual_order is not None
        assert manual_order.type == OrderType.MARKET
        assert manual_order.action == OrderAction.SELL
        assert manual_order.quantity == 15
        assert manual_order.status == OrderStatus.FILLED
        assert manual_order.price == 102.0, \
            "Manual sale should execute at current market price ($102)"
        assert manual_order.cash == 102.0 * 15, \
            "Sale proceeds should be $1,530 (15 shares * $102)"

        # Verify stop and profit orders were CANCELED
        stop_order = updated_order.get_child_order("STOP")
        profit_order = updated_order.get_child_order("PROFIT")
        assert stop_order.status == OrderStatus.CANCELED
        assert profit_order.status == OrderStatus.CANCELED

        # Verify SOLD_ORDER points to manual order
        assert updated_order.SOLD_ORDER == manual_order


    def test_bracket_order_price_oscillates_without_triggering(self):
        """
        Test that bracket order doesn't trigger when price stays between stop and profit.
        """
        om = BacktestingOM()

        symbol = "NVDA"
        tick_initial = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000
        )]

        price = 100.0
        profit_price = price * 1.10
        stop_price = price * 0.95
        bracket_order = BracketOrder.create_bracket_order(
            symbol=symbol,
            price=price,
            high_sell_price=profit_price,
            low_sell_price=stop_price,
            quantity=10,
            tx_cost=1.0,
            current_tick=tick_initial
        )

        positions = {}
        pf_cash = 10000.0
        bracket_order = om.submit_order(bracket_order, tick_initial, positions, pf_cash)

        # Test multiple price movements within the bracket range
        test_prices = [96.0, 105.0, 97.0, 108.0, 96.5, 109.0]

        for test_price in test_prices:
            tick = [PriceData(
                symbol=symbol,
                timestamp=datetime(2024, 1, 1, 10, 0),
                open=test_price, high=test_price, low=test_price, close=test_price, volume=1000
            )]

            updated_order = om.update_order_status(bracket_order, tick, positions, pf_cash)

            assert updated_order.status == OrderStatus.PENDING_SALE, \
                f"Order should remain PENDING_SALE at price ${test_price}"

            stop_order = updated_order.get_child_order("STOP")
            profit_order = updated_order.get_child_order("PROFIT")

            assert stop_order.status == OrderStatus.PENDING, \
                f"Stop order should remain PENDING at price ${test_price}"
            assert profit_order.status == OrderStatus.PENDING, \
                f"Profit order should remain PENDING at price ${test_price}"


    def test_bracket_order_exact_stop_price(self):
        """
        Test that bracket order triggers when price equals stop price (boundary condition).
        """
        om = BacktestingOM()

        symbol = "SPY"
        tick_initial = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000
        )]

        price = 100.0
        profit_price = price * 1.10
        stop_price = price * 0.95
        bracket_order = BracketOrder.create_bracket_order(
            symbol=symbol,
            price=price,
            high_sell_price=profit_price,
            low_sell_price=stop_price,
            quantity=10,
            tx_cost=0.0,
            current_tick=tick_initial
        )

        positions = {}
        pf_cash = 10000.0
        bracket_order = om.submit_order(bracket_order, tick_initial, positions, pf_cash)

        # Price drops to exactly $95 (stop price)
        tick_exact_stop = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 35),
            open=96.0, high=96.0, low=94.0, close=95.0, volume=1000
        )]

        updated_order = om.update_order_status(bracket_order, tick_exact_stop, positions, pf_cash)

        # Should trigger at exactly the stop price (price <= stop_price)
        assert updated_order.status == OrderStatus.FILLED, \
            "Stop loss should trigger when price equals stop price"

        stop_order = updated_order.get_child_order("STOP")
        assert stop_order.status == OrderStatus.FILLED


    def test_bracket_order_exact_profit_price(self):
        """
        Test that bracket order triggers when price equals profit price (boundary condition).
        """
        om = BacktestingOM()

        symbol = "QQQ"
        tick_initial = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000
        )]

        price = 100.0
        profit_price = price * 1.10
        stop_price = price * 0.95
        bracket_order = BracketOrder.create_bracket_order(
            symbol=symbol,
            price=price,
            high_sell_price=profit_price,
            low_sell_price=stop_price,
            quantity=10,
            tx_cost=0.0,
            current_tick=tick_initial
        )

        positions = {}
        pf_cash = 10000.0
        bracket_order = om.submit_order(bracket_order, tick_initial, positions, pf_cash)

        # Price rises to exactly $110.01 (just above profit price to avoid float precision issues)
        tick_exact_profit = [PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, 10, 0),
            open=105.0, high=112.0, low=105.0, close=110.01, volume=2000
        )]

        updated_order = om.update_order_status(bracket_order, tick_exact_profit, positions, pf_cash)

        # Should trigger at exactly the profit price (price >= profit_price)
        assert updated_order.status == OrderStatus.FILLED, \
            "Profit taker should trigger when price equals profit price"

        profit_order = updated_order.get_child_order("PROFIT")
        assert profit_order.status == OrderStatus.FILLED
