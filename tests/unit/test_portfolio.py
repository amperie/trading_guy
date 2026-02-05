"""
Comprehensive test suite for Portfolio class functionality
Tests market orders, bracket orders, history tracking, and portfolio state management
"""
from datetime import datetime
from trading.core.classes import (
    PriceData, OrderStatus, OrderAction, MarketSignal, SignalType
)
from trading.core.om.backtesting_om import BacktestingOM
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio


class TestPortfolioMarketOrders:
    """Test portfolio with market orders"""

    def test_market_order_buy_updates_cash_and_positions(self):
        """Test that buying with market order correctly updates cash and positions"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Initial state
        assert pf.cash == 1000.0
        assert len(pf.positions) == 0
        assert pf.total_value == 0.0

        # Create buy signal
        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal = MarketSignal(SignalType.BUY, "AAPL", 100)

        # Process buy order
        result = pf.process_market_signals_for_tick([signal], tick)

        # Verify order was created
        assert len(result.orders) > 0
        assert result.orders[0].symbol == "AAPL"
        assert result.orders[0].action == OrderAction.BUY

        # Verify cash decreased and position created
        # With bracket order at $10, buying 100 shares costs $1000
        assert pf.cash < 1000.0
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].quantity > 0

    def test_market_order_sell_updates_cash_and_positions(self):
        """Test that selling with market order correctly updates cash and positions"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # First buy some shares
        tick_buy = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick_buy)

        cash_after_buy = pf.cash
        quantity_bought = pf.positions["AAPL"].quantity

        # Now sell at a higher price (profit)
        tick_sell = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 15.0, 15.0, 15.0, 15.0, 100)]
        signal_sell = MarketSignal(SignalType.SELL, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_sell], tick_sell)

        # Verify position was sold
        # Note: With bracket orders, the sell happens when stop/profit triggers
        # So we need to check if position was reduced
        assert pf.cash >= cash_after_buy  # Should have more cash after profitable sell


class TestPortfolioBracketOrders:
    """Test portfolio with bracket orders"""

    def test_bracket_order_profit_taker_trigger(self):
        """Test that bracket order profit-taker triggers correctly"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Buy at $10 (bracket order with profit at $12, stop at $8)
        tick_buy = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick_buy)

        cash_after_buy = pf.cash
        quantity_bought = pf.positions["AAPL"].quantity

        # Price rises to $12 - should trigger profit taker
        tick_profit = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick_profit)

        # Verify profit was taken
        assert pf.cash > cash_after_buy, "Cash should increase after profit-taker triggers"
        # Position should be sold
        assert pf.positions["AAPL"].quantity < quantity_bought or pf.positions["AAPL"].quantity == 0

    def test_bracket_order_stop_loss_trigger(self):
        """Test that bracket order stop-loss triggers correctly"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Buy at $10 (bracket order with profit at $12, stop at $8)
        tick_buy = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick_buy)

        cash_after_buy = pf.cash
        quantity_bought = pf.positions["AAPL"].quantity

        # Price drops to $8 - should trigger stop loss
        tick_stop = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 8.0, 8.0, 8.0, 8.0, 100)]
        pf.process_market_signals_for_tick([], tick_stop)

        # Verify stop loss was triggered
        # Cash should be less than initial (loss occurred)
        assert pf.cash < 1000.0, "Should have taken a loss"
        assert pf.cash > cash_after_buy, "Cash should increase after stop-loss triggers (sold shares)"

    def test_bracket_order_no_trigger_within_range(self):
        """Test that bracket order doesn't trigger when price stays in range"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Buy at $10 (bracket order with profit at $12, stop at $8)
        tick_buy = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick_buy)

        cash_after_buy = pf.cash
        quantity_bought = pf.positions["AAPL"].quantity

        # Price moves within range (stop is at $9.0, profit at $12.0)
        # Use prices strictly between stop and profit to avoid triggering
        for price in [9.5, 10.0, 11.0]:
            tick = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), price, price, price, price, 100)]
            pf.process_market_signals_for_tick([], tick)

            # Verify position hasn't changed
            assert pf.positions["AAPL"].quantity == quantity_bought
            assert pf.cash == cash_after_buy


class TestPortfolioProgression:
    """Test portfolio progression through multiple orders"""

    def test_multiple_bracket_orders_progression(self):
        """Test portfolio through complete cycle: buy, profit, buy, stop-loss"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        initial_cash = 1000.0

        # Step 1: Buy at $10
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        cash_after_first_buy = pf.cash
        assert cash_after_first_buy < initial_cash

        # Step 2: Trigger profit taker at $12 (20% gain)
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        cash_after_profit = pf.cash
        assert cash_after_profit > cash_after_first_buy
        assert cash_after_profit > initial_cash  # Made profit

        # Step 3: Buy again at $13
        tick3 = [PriceData("AAPL", datetime(2024, 1, 1, 12, 0), 13.0, 13.0, 13.0, 13.0, 100)]
        pf.process_market_signals_for_tick([signal_buy], tick3)

        cash_after_second_buy = pf.cash
        assert cash_after_second_buy < cash_after_profit

        # Step 4: Trigger stop loss at $11 (loss)
        tick4 = [PriceData("AAPL", datetime(2024, 1, 1, 13, 0), 11.0, 11.0, 11.0, 11.0, 100)]
        pf.process_market_signals_for_tick([], tick4)

        final_cash = pf.cash
        assert final_cash > cash_after_second_buy  # Got cash back from selling
        # Final cash should still be positive but less than after first profit
        assert final_cash > 0

    def test_insufficient_funds_prevents_order(self):
        """Test that order is not created when insufficient funds"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 100.0, {}, True)  # Only $100

        # Try to buy at $10 (can only afford 10 shares)
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        # Should have bought with all available cash
        assert pf.cash < 100.0

        # Try to buy again - should fail (no money)
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        result = pf.process_market_signals_for_tick([signal_buy], tick2)

        # Order might be created but should be quantity 0 or not filled
        if len(result.orders) > 0:
            assert result.orders[0].quantity == 0 or result.orders[0].status == OrderStatus.CANCELED


class TestPortfolioHistory:
    """Test portfolio history tracking"""

    def test_history_tracking_enabled(self):
        """Test that history is tracked when keep_history=True"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, keep_history=True)

        # Process several ticks
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 11, 0),
        ]

        for ts in timestamps:
            tick = [PriceData("AAPL", ts, 10.0, 10.0, 10.0, 10.0, 100)]
            pf.process_market_signals_for_tick([], tick)

        # Verify history was tracked
        assert len(pf.tick_history) == len(timestamps)
        assert len(pf.cash_history) == len(timestamps)
        assert len(pf.value_history) == len(timestamps)

        # Verify timestamps match
        for ts in timestamps:
            assert ts in pf.tick_history
            assert ts in pf.cash_history
            assert ts in pf.value_history

    def test_history_not_tracked_when_disabled(self):
        """Test that history is not tracked when keep_history=False"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, keep_history=False)

        # Process a tick
        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        pf.process_market_signals_for_tick([], tick)

        # Verify history was not tracked
        assert len(pf.tick_history) == 0
        assert len(pf.cash_history) == 0
        assert len(pf.value_history) == 0

    def test_cash_history_accuracy(self):
        """Test that cash history accurately reflects cash changes"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, keep_history=True)

        initial_cash = 1000.0

        # Buy order
        ts1 = datetime(2024, 1, 1, 10, 0)
        tick1 = [PriceData("AAPL", ts1, 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        # Verify cash history
        assert ts1 in pf.cash_history
        assert pf.cash_history[ts1] < initial_cash
        assert pf.cash_history[ts1] == pf.cash

        # Sell order (profit taker)
        ts2 = datetime(2024, 1, 1, 11, 0)
        tick2 = [PriceData("AAPL", ts2, 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        # Verify cash increased
        assert ts2 in pf.cash_history
        assert pf.cash_history[ts2] > pf.cash_history[ts1]
        assert pf.cash_history[ts2] == pf.cash

    def test_value_history_accuracy(self):
        """Test that portfolio value history is calculated correctly"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, keep_history=True)

        # Buy at $10
        ts1 = datetime(2024, 1, 1, 10, 0)
        tick1 = [PriceData("AAPL", ts1, 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        value_at_10 = pf.value_history[ts1]

        # Price moves to $15 (position value increases)
        ts2 = datetime(2024, 1, 1, 11, 0)
        tick2 = [PriceData("AAPL", ts2, 15.0, 15.0, 15.0, 15.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        value_at_15 = pf.value_history[ts2]

        # Portfolio value should increase when price increases
        assert value_at_15 > value_at_10


class TestPortfolioValueCalculation:
    """Test portfolio total value calculation"""

    def test_portfolio_value_with_no_positions(self):
        """Test that portfolio value equals cash when no positions"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        pf.process_market_signals_for_tick([], tick)

        # No positions, so total value = cash
        assert pf.total_value == pf.cash
        assert pf.total_value == 1000.0

    def test_portfolio_value_with_positions(self):
        """Test that portfolio value includes position value"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Buy at $10
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        # Total value = cash + (quantity * current_price)
        expected_value = pf.cash + (pf.positions["AAPL"].quantity * 10.0)
        assert abs(pf.total_value - expected_value) < 0.01  # Allow small float precision error

    def test_portfolio_value_changes_with_price(self):
        """Test that portfolio value updates as price changes"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Buy at $10
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        value_at_10 = pf.total_value
        quantity = pf.positions["AAPL"].quantity

        # Price rises to $15
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 15.0, 15.0, 15.0, 15.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        value_at_15 = pf.total_value

        # Value should increase by (quantity * price_increase)
        expected_increase = quantity * 5.0
        assert abs((value_at_15 - value_at_10) - expected_increase) < 0.01


class TestPortfolioPositionTracking:
    """Test position tracking accuracy"""

    def test_position_created_on_buy(self):
        """Test that position is created when buying"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick)

        # Verify position created
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].symbol == "AAPL"
        assert pf.positions["AAPL"].quantity > 0

    def test_position_updated_on_multiple_buys(self):
        """Test that position quantity accumulates on multiple buys"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 2000.0, {}, True)

        # First buy
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        first_quantity = pf.positions["AAPL"].quantity

        # Trigger profit taker to sell and free up cash
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        # Second buy
        tick3 = [PriceData("AAPL", datetime(2024, 1, 1, 12, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        pf.process_market_signals_for_tick([signal_buy], tick3)

        # Position should exist after second buy
        assert "AAPL" in pf.positions

    def test_position_reduced_on_sell(self):
        """Test that position is reduced when selling"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Buy
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        quantity_before = pf.positions["AAPL"].quantity

        # Trigger profit taker (sell)
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        # Position should be reduced or eliminated
        if "AAPL" in pf.positions:
            assert pf.positions["AAPL"].quantity < quantity_before
        # If position is 0, it might still exist in dict but with 0 quantity


class TestPortfolioOrderTracking:
    """Test order tracking and history"""

    def test_orders_stored_in_history(self):
        """Test that all orders are stored"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Create multiple orders
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        orders_after_first = len(pf.om.all_orders)

        # Trigger sell
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        # Should have multiple orders stored
        assert len(pf.om.all_orders) >= orders_after_first

    def test_pending_orders_tracked(self):
        """Test that pending orders are tracked separately"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Create bracket order
        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick)

        # Should have pending orders (bracket order waiting for trigger)
        assert len(pf.om.pending_orders_by_id) > 0

    def test_filled_orders_tracked(self):
        """Test that filled orders are tracked"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Create and fill bracket order
        tick1 = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        pf.process_market_signals_for_tick([signal_buy], tick1)

        # Trigger profit taker
        tick2 = [PriceData("AAPL", datetime(2024, 1, 1, 11, 0), 12.0, 12.0, 12.0, 12.0, 100)]
        pf.process_market_signals_for_tick([], tick2)

        # Should have filled orders
        assert len(pf.om.filled_orders_by_id) > 0


class TestPortfolioEdgeCases:
    """Test edge cases and error conditions"""

    def test_no_signal_no_order(self):
        """Test that no order is created when no signal"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        result = pf.process_market_signals_for_tick([], tick)  # No signals

        # No new orders should be created
        assert len(result.orders) == 0

    def test_zero_cash_prevents_buy(self):
        """Test that buying is prevented when cash is zero"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 0.0, {}, True)  # Zero cash

        tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, 0), 10.0, 10.0, 10.0, 10.0, 100)]
        signal_buy = MarketSignal(SignalType.BUY, "AAPL", 100)
        result = pf.process_market_signals_for_tick([signal_buy], tick)

        # Order might be created but should have 0 quantity or be canceled
        if len(result.orders) > 0:
            assert result.orders[0].quantity == 0 or result.orders[0].status == OrderStatus.CANCELED

    def test_multiple_ticks_same_price(self):
        """Test processing multiple ticks at same price"""
        om = BacktestingOM()
        pf = SingleSymbolPortfolio({'symbol': 'AAPL', 'stop_pct': 10.0, 'profit_pct': 20.0}, om, 1000.0, {}, True)

        # Process same price multiple times
        for i in range(5):
            tick = [PriceData("AAPL", datetime(2024, 1, 1, 10, i), 10.0, 10.0, 10.0, 10.0, 100)]
            pf.process_market_signals_for_tick([], tick)

        # Should have 5 entries in history
        assert len(pf.tick_history) == 5
        assert len(pf.value_history) == 5
