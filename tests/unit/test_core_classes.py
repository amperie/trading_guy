"""
Unit tests for core data classes
Tests: PriceData, Order, Position, MarketSignal, and related enums
"""
import pytest
from datetime import datetime, timezone
import uuid

from core.classes import (
    PriceData, Order, Position, MarketSignal,
    SignalType, OrderType, OrderAction, OrderStatus
)


class TestPriceData:
    """Tests for PriceData dataclass"""

    def test_pricedata_creation(self, sample_pricedata):
        """Test creating a PriceData object"""
        assert sample_pricedata.symbol == "AAPL"
        assert sample_pricedata.close == 151.0
        assert sample_pricedata.volume == 1000000.0
        assert isinstance(sample_pricedata.timestamp, datetime)

    def test_pricedata_from_dict(self, sample_timestamp):
        """Test creating PriceData from dictionary"""
        data_dict = {
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000.0
        }
        pd = PriceData.from_dict(data_dict, sample_timestamp)

        assert pd.symbol == "AAPL"
        assert pd.timestamp == sample_timestamp
        assert pd.close == 151.0

    def test_pricedata_optional_fields(self, sample_timestamp):
        """Test PriceData with optional fields"""
        pd = PriceData(
            symbol="AAPL",
            timestamp=sample_timestamp,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0
        )

        assert pd.trade_count is None
        assert pd.vwap is None
        assert pd.exchange is None


class TestPosition:
    """Tests for Position dataclass"""

    def test_position_creation(self, sample_position):
        """Test creating a Position"""
        assert sample_position.symbol == "AAPL"
        assert sample_position.quantity == 100

    def test_position_modification(self):
        """Test modifying position quantity"""
        pos = Position(symbol="AAPL", quantity=100)
        pos.quantity += 50
        assert pos.quantity == 150

        pos.quantity -= 30
        assert pos.quantity == 120


class TestMarketSignal:
    """Tests for MarketSignal dataclass"""

    def test_signal_creation(self, sample_market_signal):
        """Test creating a MarketSignal"""
        assert sample_market_signal.type == SignalType.BUY
        assert sample_market_signal.symbol == "AAPL"
        assert sample_market_signal.strength == 75

    def test_signal_types(self):
        """Test different signal types"""
        buy_signal = MarketSignal(SignalType.BUY, "AAPL", 80)
        sell_signal = MarketSignal(SignalType.SELL, "GOOGL", 60)

        assert buy_signal.type == SignalType.BUY
        assert sell_signal.type == SignalType.SELL

    def test_signal_strength_range(self):
        """Test that signal strength is within expected range"""
        signal = MarketSignal(SignalType.BUY, "AAPL", 100)
        assert 0 <= signal.strength <= 100


class TestOrder:
    """Tests for Order dataclass"""

    def test_order_creation(self, sample_order):
        """Test creating an Order"""
        assert sample_order.symbol == "AAPL"
        assert sample_order.action == OrderAction.BUY
        assert sample_order.type == OrderType.MARKET
        assert sample_order.quantity == 100
        assert sample_order.price == 151.0
        assert sample_order.status == OrderStatus.FILLED

    def test_order_id_generation(self, sample_timestamp):
        """Test that order_id is automatically generated"""
        order = Order(
            placed_datetime=sample_timestamp,
            executed_datetime=sample_timestamp,
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol="AAPL",
            price=151.0,
            quantity=100,
            cash=15100.0,
            status=OrderStatus.FILLED
        )

        assert order.order_id.startswith("local-")
        assert len(order.order_id) > 10

    def test_order_id_uniqueness(self, sample_timestamp):
        """Test that each order gets a unique ID"""
        order1 = Order(
            placed_datetime=sample_timestamp,
            executed_datetime=sample_timestamp,
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol="AAPL",
            price=151.0,
            quantity=100,
            cash=15100.0,
            status=OrderStatus.FILLED
        )

        order2 = Order(
            placed_datetime=sample_timestamp,
            executed_datetime=sample_timestamp,
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol="AAPL",
            price=151.0,
            quantity=100,
            cash=15100.0,
            status=OrderStatus.FILLED
        )

        assert order1.order_id != order2.order_id

    def test_order_default_fields(self, sample_timestamp):
        """Test order default field values"""
        order = Order(
            placed_datetime=sample_timestamp,
            executed_datetime=sample_timestamp,
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol="AAPL",
            price=151.0,
            quantity=100,
            cash=15100.0,
            status=OrderStatus.FILLED
        )

        assert order.tx_cost == 0.0
        assert order.parent_id is None
        assert order.child_orders == []
        assert order.processed_by_portfolio is False

    def test_pending_order(self, sample_pending_order):
        """Test creating a pending order"""
        assert sample_pending_order.status == OrderStatus.PENDING
        assert sample_pending_order.type == OrderType.STOP
        assert sample_pending_order.executed_datetime is None

    def test_order_parent_child_relationship(self, sample_timestamp):
        """Test parent-child order relationships"""
        parent_order = Order(
            placed_datetime=sample_timestamp,
            executed_datetime=sample_timestamp,
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol="AAPL",
            price=151.0,
            quantity=100,
            cash=15100.0,
            status=OrderStatus.FILLED
        )

        child_order_id = "local-child-123"
        parent_order.child_orders.append(child_order_id)

        assert len(parent_order.child_orders) == 1
        assert parent_order.child_orders[0] == child_order_id


class TestEnums:
    """Tests for enum types"""

    def test_signal_type_enum(self):
        """Test SignalType enum"""
        assert SignalType.BUY.value == 1
        assert SignalType.SELL.value == 2

    def test_order_type_enum(self):
        """Test OrderType enum"""
        assert OrderType.MARKET.value == 1
        assert OrderType.STOP.value == 2
        assert OrderType.STOP_LOSS.value == 3
        assert OrderType.BRACKET.value == 4

    def test_order_action_enum(self):
        """Test OrderAction enum"""
        assert OrderAction.BUY.value == 1
        assert OrderAction.SELL.value == 2

    def test_order_status_enum(self):
        """Test OrderStatus enum"""
        assert OrderStatus.PENDING.value == 0
        assert OrderStatus.FILLED.value == 1
        assert OrderStatus.CANCELED.value == 2
