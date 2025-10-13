"""
Unit tests for Portfolio base class and SingleSymbolPortfolio
"""
import pytest
from datetime import datetime

from core.portfolio import Portfolio
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from core.classes import (
    Order, MarketSignal, PriceData, Position,
    OrderAction, OrderStatus, SignalType
)
from core.om.backtesting_om import BacktestingOM


class MockPortfolio(Portfolio):
    """Mock portfolio for testing base class"""

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> list[Order]:
        """Mock implementation that returns empty list"""
        return []


class TestPortfolioBase:
    """Tests for Portfolio base class"""

    def test_portfolio_initialization(self, portfolio_config):
        """Test portfolio initialization"""
        pf = MockPortfolio(portfolio_config)

        assert pf.cash == 100000.0
        assert pf.keep_history is True
        assert len(pf.positions) == 0
        assert len(pf.orders) == 0
        assert pf.total_value == 0.0

    def test_portfolio_set_order_manager(self, portfolio_config):
        """Test setting order manager"""
        pf = MockPortfolio(portfolio_config)
        om = BacktestingOM()

        pf.set_order_manager(om)

        assert pf.order_manager == om

    def test_portfolio_process_filled_buy_order(
            self, portfolio_config, sample_order):
        """Test processing a filled buy order"""
        pf = MockPortfolio(portfolio_config)
        initial_cash = pf.cash

        # Process the filled order
        pf._process_filled_orders([sample_order])

        # Check that position was created
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].quantity == 100

        # Check that cash was deducted
        expected_cash = initial_cash - (sample_order.cash + sample_order.tx_cost)
        assert pf.cash == expected_cash

        # Check that order was marked as processed
        assert sample_order.processed_by_portfolio is True

    def test_portfolio_process_filled_sell_order(
            self, portfolio_config, sample_timestamp):
        """Test processing a filled sell order"""
        pf = MockPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # First create a position
        pf.positions["AAPL"] = Position(symbol="AAPL", quantity=200)
        initial_cash = pf.cash

        # Create sell order
        sell_order = Order(
            placed_datetime=sample_timestamp,
            executed_datetime=sample_timestamp,
            action=OrderAction.SELL,
            type=OrderStatus.FILLED,
            symbol="AAPL",
            price=151.0,
            quantity=100,
            cash=15100.0,
            status=OrderStatus.FILLED,
            tx_cost=1.0
        )

        pf._process_filled_orders([sell_order])

        # Check that position was reduced
        assert pf.positions["AAPL"].quantity == 100

        # Check that cash was added
        expected_cash = initial_cash + (sell_order.cash - sell_order.tx_cost)
        assert pf.cash == expected_cash

    def test_portfolio_store_orders(self, portfolio_config, sample_order):
        """Test storing orders"""
        pf = MockPortfolio(portfolio_config)

        pf._store_orders([sample_order])

        assert len(pf.orders) == 1
        assert sample_order.order_id in pf.orders_by_id
        assert pf.orders_by_id[sample_order.order_id] == sample_order

    def test_portfolio_store_pending_order(
            self, portfolio_config, sample_pending_order):
        """Test storing pending orders"""
        pf = MockPortfolio(portfolio_config)

        pf._store_orders([sample_pending_order])

        assert len(pf.orders) == 1
        assert sample_pending_order.order_id in pf.pending_orders_by_id
        assert sample_pending_order.order_id in pf.orders_by_id

    def test_portfolio_close_pending_order(
            self, portfolio_config, sample_pending_order):
        """Test closing a pending order"""
        pf = MockPortfolio(portfolio_config)

        # Store as pending
        pf.pending_orders_by_id[sample_pending_order.order_id] = sample_pending_order

        # Close it
        pf._close_pending_order(sample_pending_order)

        assert sample_pending_order.order_id not in pf.pending_orders_by_id
        assert sample_pending_order.order_id in pf.orders_by_id

    def test_portfolio_update_value(
            self, portfolio_config, sample_pricedata_list):
        """Test portfolio value calculation"""
        pf = MockPortfolio(portfolio_config)

        # Add some positions
        pf.positions["AAPL"] = Position(symbol="AAPL", quantity=100)
        pf.positions["GOOGL"] = Position(symbol="GOOGL", quantity=50)

        # Update value
        pf._update_pf_value(sample_pricedata_list)

        # AAPL: 100 * 151 = 15100
        # GOOGL: 50 * 2810 = 140500
        # Cash: 100000
        # Total: 255600
        expected_value = 100000 + (100 * 151.0) + (50 * 2810.0)
        assert pf.total_value == expected_value

    def test_portfolio_history_tracking(
            self, portfolio_config, sample_pricedata_list,
            sample_market_signals):
        """Test that portfolio tracks history"""
        pf = MockPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # Process tick
        pf.process_tick_market_signals(sample_market_signals, sample_pricedata_list)

        # Check history was recorded
        timestamp = sample_pricedata_list[0].timestamp
        assert timestamp in pf.tick_history
        assert timestamp in pf.cash_history
        assert timestamp in pf.value_history


class TestSingleSymbolPortfolio:
    """Tests for SingleSymbolPortfolio implementation"""

    def test_single_symbol_portfolio_initialization(self, portfolio_config):
        """Test single symbol portfolio initialization"""
        pf = SingleSymbolPortfolio(portfolio_config)

        assert pf.cfg['symbol'] == "AAPL"
        assert pf.cash == 100000.0

    def test_single_symbol_portfolio_buy_signal(
            self, portfolio_config, sample_pricedata_list):
        """Test portfolio handles buy signal"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # Create buy signal
        signals = [MarketSignal(SignalType.BUY, "AAPL", 100)]

        # Process
        orders = pf.process_tick_market_signals(signals, sample_pricedata_list)

        assert len(orders) == 1
        assert orders[0].action == OrderAction.BUY
        assert orders[0].symbol == "AAPL"
        assert orders[0].status == OrderStatus.FILLED

    def test_single_symbol_portfolio_sell_signal(
            self, portfolio_config, sample_pricedata_list):
        """Test portfolio handles sell signal"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # First create a position
        pf.positions["AAPL"] = Position(symbol="AAPL", quantity=100)

        # Create sell signal
        signals = [MarketSignal(SignalType.SELL, "AAPL", 100)]

        # Process
        orders = pf.process_tick_market_signals(signals, sample_pricedata_list)

        assert len(orders) == 1
        assert orders[0].action == OrderAction.SELL
        assert orders[0].symbol == "AAPL"
        assert orders[0].quantity == 100

    def test_single_symbol_portfolio_no_signal(
            self, portfolio_config, sample_pricedata_list):
        """Test portfolio with no matching signal"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # Signal for different symbol
        signals = [MarketSignal(SignalType.BUY, "GOOGL", 100)]

        # Process - should return no orders
        orders = pf.process_tick_market_signals(signals, sample_pricedata_list)

        assert len(orders) == 0

    def test_single_symbol_portfolio_buy_max_quantity(
            self, portfolio_config, sample_pricedata_list):
        """Test that portfolio buys maximum quantity with available cash"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # Create buy signal
        signals = [MarketSignal(SignalType.BUY, "AAPL", 100)]

        # Process
        orders = pf.process_tick_market_signals(signals, sample_pricedata_list)

        # AAPL price is 151, cash is 100000
        # Max quantity = 100000 / 151 = 662
        expected_quantity = int(100000 / 151.0)
        assert orders[0].quantity == expected_quantity

    def test_single_symbol_portfolio_sell_without_position(
            self, portfolio_config, sample_pricedata_list):
        """Test that portfolio doesn't sell if no position exists"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # Create sell signal but no position
        signals = [MarketSignal(SignalType.SELL, "AAPL", 100)]

        # Process - should return no orders
        orders = pf.process_tick_market_signals(signals, sample_pricedata_list)

        assert len(orders) == 0

    def test_single_symbol_portfolio_position_after_buy(
            self, portfolio_config, sample_pricedata_list):
        """Test that position is created after buy"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)
        initial_cash = pf.cash

        # Create buy signal
        signals = [MarketSignal(SignalType.BUY, "AAPL", 100)]

        # Process
        orders = pf.process_tick_market_signals(signals, sample_pricedata_list)

        # Check position was created
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].quantity == orders[0].quantity

        # Check cash was reduced
        assert pf.cash < initial_cash

    def test_single_symbol_portfolio_multiple_ticks(
            self, portfolio_config, sample_pricedata_list):
        """Test portfolio over multiple ticks"""
        pf = SingleSymbolPortfolio(portfolio_config)
        om = BacktestingOM()
        pf.set_order_manager(om)

        # Tick 1: Buy
        signals1 = [MarketSignal(SignalType.BUY, "AAPL", 100)]
        orders1 = pf.process_tick_market_signals(signals1, sample_pricedata_list)

        assert len(orders1) == 1
        position_after_buy = pf.positions["AAPL"].quantity

        # Tick 2: Sell
        signals2 = [MarketSignal(SignalType.SELL, "AAPL", 100)]
        orders2 = pf.process_tick_market_signals(signals2, sample_pricedata_list)

        assert len(orders2) == 1
        assert pf.positions["AAPL"].quantity == 0
