"""
Tests for TradingStateStore — MongoDB persistence layer.

Uses mongomock to run without a real MongoDB instance.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

try:
    import mongomock
    HAS_MONGOMOCK = True
except ImportError:
    HAS_MONGOMOCK = False

from trading.core.classes import (
    PriceData, MarketSignal, Order, BracketOrder,
    OrderType, OrderAction, OrderStatus, OrderTimeInForce, SignalType,
)
from utils.utils import (
    serialize_order, deserialize_order,
    serialize_pricedata, deserialize_pricedata,
)

pytestmark = pytest.mark.skipif(not HAS_MONGOMOCK, reason="mongomock not installed")


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def store():
    from utils.trading_state_store import TradingStateStore
    client = mongomock.MongoClient()
    return TradingStateStore(client=client, database="test_trading")


@pytest.fixture
def sample_tick():
    ts = datetime(2024, 6, 15, 10, 0, 0)
    return [
        PriceData("AAPL", ts, 150.0, 151.0, 149.5, 150.5, 1000, trade_count=50, vwap=150.3),
        PriceData("TSLA", ts, 250.0, 252.0, 248.0, 251.0, 2000),
    ]


@pytest.fixture
def sample_signals():
    return [
        MarketSignal(SignalType.BUY, "AAPL", 75, metadata={"rsi": 30.5}),
        MarketSignal(SignalType.SELL, "TSLA", 80, metadata={"reason": "overbought"}),
    ]


@pytest.fixture
def sample_market_order():
    return Order(
        placed_datetime=datetime(2024, 6, 15, 10, 0, 0),
        action=OrderAction.BUY,
        type=OrderType.MARKET,
        symbol="AAPL",
        price=150.0,
        quantity=10,
        cash=1500.0,
        executed_datetime=datetime(2024, 6, 15, 10, 0, 1),
        status=OrderStatus.FILLED,
        order_id="local-test-001",
        tx_cost=1.0,
        platform_id="plat-001",
    )


@pytest.fixture
def sample_bracket_order():
    bo = BracketOrder.create_bracket_order(
        symbol="AAPL",
        high_sell_price=160.0,
        low_sell_price=140.0,
        quantity=10,
        tx_cost=1.0,
        current_tick=[PriceData("AAPL", datetime(2024, 6, 15, 10, 0, 0),
                                150.0, 151.0, 149.5, 150.5, 1000)],
    )
    # Simulate fill
    bo.price = 150.0
    bo.cash = 1500.0
    bo.status = OrderStatus.FILLED
    bo.executed_datetime = datetime(2024, 6, 15, 10, 0, 1)

    # Simulate stop-loss trigger
    stop = bo.get_child_order("STOP")
    stop.status = OrderStatus.FILLED
    stop.price = 140.0
    stop.cash = 1400.0
    stop.executed_datetime = datetime(2024, 6, 15, 11, 0, 0)
    bo.SOLD_ORDER = stop

    profit = bo.get_child_order("PROFIT")
    profit.status = OrderStatus.CANCELED

    return bo


# ------------------------------------------------------------------ #
#  Serialization round-trip tests
# ------------------------------------------------------------------ #

class TestPriceDataSerialization:

    def test_round_trip(self, sample_tick):
        pd_obj = sample_tick[0]
        data = serialize_pricedata(pd_obj)
        restored = deserialize_pricedata(data)

        assert restored.symbol == pd_obj.symbol
        assert restored.timestamp == pd_obj.timestamp
        assert restored.open == pd_obj.open
        assert restored.high == pd_obj.high
        assert restored.low == pd_obj.low
        assert restored.close == pd_obj.close
        assert restored.volume == pd_obj.volume
        assert restored.trade_count == pd_obj.trade_count
        assert restored.vwap == pd_obj.vwap

    def test_optional_fields_none(self):
        pd_obj = PriceData("SPY", datetime(2024, 1, 1), 100, 101, 99, 100.5, 500)
        data = serialize_pricedata(pd_obj)
        restored = deserialize_pricedata(data)
        assert restored.trade_count is None
        assert restored.vwap is None
        assert restored.exchange is None


class TestOrderSerialization:

    def test_market_order_round_trip(self, sample_market_order):
        data = serialize_order(sample_market_order)
        restored = deserialize_order(data)

        assert restored.order_id == sample_market_order.order_id
        assert restored.action == sample_market_order.action
        assert restored.type == sample_market_order.type
        assert restored.symbol == sample_market_order.symbol
        assert restored.price == sample_market_order.price
        assert restored.quantity == sample_market_order.quantity
        assert restored.cash == sample_market_order.cash
        assert restored.status == sample_market_order.status
        assert restored.tx_cost == sample_market_order.tx_cost
        assert restored.platform_id == sample_market_order.platform_id
        assert not isinstance(restored, BracketOrder)

    def test_bracket_order_round_trip(self, sample_bracket_order):
        data = serialize_order(sample_bracket_order)
        restored = deserialize_order(data)

        assert isinstance(restored, BracketOrder)
        assert restored.order_id == sample_bracket_order.order_id
        assert restored.status == OrderStatus.FILLED
        assert restored.MANUAL_SALE == False

        # Child orders
        stop = restored.get_child_order("STOP")
        assert stop is not None
        assert stop.type == OrderType.STOP_LOSS
        assert stop.status == OrderStatus.FILLED
        assert stop.price == 140.0

        profit = restored.get_child_order("PROFIT")
        assert profit is not None
        assert profit.type == OrderType.PROFIT_TAKER
        assert profit.status == OrderStatus.CANCELED

        # SOLD_ORDER linked
        assert restored.SOLD_ORDER is not None
        assert restored.SOLD_ORDER.order_id == stop.order_id

    def test_bracket_order_with_no_sold_order(self):
        bo = BracketOrder.create_bracket_order(
            symbol="SPY", high_sell_price=110.0, low_sell_price=90.0,
            quantity=5, tx_cost=0.0,
        )
        bo.status = OrderStatus.PENDING_SALE
        data = serialize_order(bo)
        restored = deserialize_order(data)

        assert isinstance(restored, BracketOrder)
        assert restored.SOLD_ORDER is None
        assert restored.status == OrderStatus.PENDING_SALE


# ------------------------------------------------------------------ #
#  Session management tests
# ------------------------------------------------------------------ #

class TestSessionManagement:

    def test_create_session(self, store):
        sid = store.create_session(name="Test Run", account_id="PA-123")
        assert sid is not None
        session = store.get_session(sid)
        assert session["name"] == "Test Run"
        assert session["account_id"] == "PA-123"

    def test_create_session_explicit_id(self, store):
        sid = store.create_session(session_id="my-session", name="Explicit")
        assert sid == "my-session"
        session = store.get_session("my-session")
        assert session["name"] == "Explicit"

    def test_list_sessions(self, store):
        store.create_session(name="Run 1", account_id="PA-123")
        store.create_session(name="Run 2", account_id="PA-123")
        store.create_session(name="Run 3", account_id="PA-456")

        all_sessions = store.list_sessions()
        assert len(all_sessions) == 3

        filtered = store.list_sessions(account_id="PA-123")
        assert len(filtered) == 2

    def test_get_nonexistent_session(self, store):
        assert store.get_session("nonexistent") is None


# ------------------------------------------------------------------ #
#  Tick persistence tests
# ------------------------------------------------------------------ #

class TestTickPersistence:

    def test_save_and_load_tick(self, store, sample_tick, sample_signals):
        sid = store.create_session(name="Test")
        ts = sample_tick[0].timestamp

        store.save_tick(sid, ts, sample_tick, cash=95000.0, total_value=102500.0,
                        signals=sample_signals)

        data = store.load_portfolio_history(sid)

        # Portfolio snapshots
        assert ts in data["value_history"]
        assert data["value_history"][ts] == 102500.0
        assert data["cash_history"][ts] == 95000.0

        # Tick history
        assert ts in data["tick_history"]
        assert len(data["tick_history"][ts]) == 2
        assert data["tick_history"][ts][0].symbol in ("AAPL", "TSLA")

        # Signals
        assert ts in data["signals_history"]
        assert len(data["signals_history"][ts]) == 2

    def test_save_tick_without_signals(self, store, sample_tick):
        sid = store.create_session(name="Test")
        ts = sample_tick[0].timestamp

        store.save_tick(sid, ts, sample_tick, cash=100000.0, total_value=100000.0)

        data = store.load_portfolio_history(sid)
        assert ts in data["value_history"]
        assert len(data["signals_history"]) == 0

    def test_upsert_overwrites_tick(self, store, sample_tick):
        sid = store.create_session(name="Test")
        ts = sample_tick[0].timestamp

        store.save_tick(sid, ts, sample_tick, cash=100000.0, total_value=100000.0)
        store.save_tick(sid, ts, sample_tick, cash=99000.0, total_value=101000.0)

        data = store.load_portfolio_history(sid)
        assert data["cash_history"][ts] == 99000.0
        assert data["value_history"][ts] == 101000.0

    def test_multiple_timestamps(self, store):
        sid = store.create_session(name="Test")

        for i in range(5):
            ts = datetime(2024, 6, 15, 10, i, 0)
            tick = [PriceData("AAPL", ts, 150 + i, 151 + i, 149, 150.5 + i, 1000)]
            store.save_tick(sid, ts, tick, cash=100000.0 - i * 1000,
                            total_value=100000.0 + i * 500)

        data = store.load_portfolio_history(sid)
        assert len(data["value_history"]) == 5
        assert len(data["tick_history"]) == 5

    def test_save_tick_persists_indicator_snapshot_and_previous_tick_metadata(self, store, sample_tick):
        sid = store.create_session(name="Test")
        ts = sample_tick[0].timestamp
        signals = [
            MarketSignal(
                SignalType.BUY,
                "AAPL",
                75,
                metadata={"current": {"macd_line": 1.5}, "previous_tick_metadata": {"macd_line": 1.2}},
            )
        ]
        snapshot = {"AAPL": {"macd_line": 1.5, "signal_line": 1.1}}

        store.save_tick(
            sid,
            ts,
            sample_tick,
            cash=95000.0,
            total_value=102500.0,
            signals=signals,
            indicator_snapshot=snapshot,
        )

        tick_doc = store._ticks.find_one({"session_id": sid, "timestamp": ts, "symbol": "AAPL"})
        assert tick_doc["indicator_snapshot"] == snapshot

        signal_doc = store._signals.find_one({"session_id": sid, "timestamp": ts, "symbol": "AAPL"})
        assert signal_doc["metadata"]["previous_tick_metadata"] == {"macd_line": 1.2}


# ------------------------------------------------------------------ #
#  Order persistence tests
# ------------------------------------------------------------------ #

class TestOrderPersistence:

    def test_save_and_load_order(self, store, sample_market_order):
        sid = store.create_session(name="Test")
        store.save_order(sid, sample_market_order)

        data = store.load_orders(sid)
        assert sample_market_order.order_id in data["all_orders"]
        assert sample_market_order.order_id in data["filled_orders_by_id"]

        restored = data["all_orders"][sample_market_order.order_id]
        assert restored.symbol == "AAPL"
        assert restored.price == 150.0
        assert restored.status == OrderStatus.FILLED

    def test_save_orders_bulk(self, store):
        sid = store.create_session(name="Test")

        orders = {}
        for i in range(3):
            o = Order(
                placed_datetime=datetime(2024, 6, 15, 10, i, 0),
                action=OrderAction.BUY,
                type=OrderType.MARKET,
                symbol="AAPL",
                price=150.0 + i,
                quantity=10,
                cash=1500.0 + i * 10,
                status=OrderStatus.FILLED,
                order_id=f"local-bulk-{i}",
            )
            orders[o.order_id] = o

        store.save_orders(sid, orders)
        data = store.load_orders(sid)
        assert len(data["all_orders"]) == 3

    def test_upsert_order(self, store, sample_market_order):
        sid = store.create_session(name="Test")

        # First save as pending
        sample_market_order.status = OrderStatus.PENDING
        store.save_order(sid, sample_market_order)

        # Update to filled
        sample_market_order.status = OrderStatus.FILLED
        store.save_order(sid, sample_market_order)

        data = store.load_orders(sid)
        assert len(data["all_orders"]) == 1
        assert data["all_orders"][sample_market_order.order_id].status == OrderStatus.FILLED

    def test_bracket_order_persistence(self, store, sample_bracket_order):
        sid = store.create_session(name="Test")
        store.save_order(sid, sample_bracket_order)

        data = store.load_orders(sid)
        order = data["all_orders"][sample_bracket_order.order_id]
        assert isinstance(order, BracketOrder)
        assert order.SOLD_ORDER is not None

        # SOLD_ORDER should also be in filled_orders (via the load logic)
        assert order.SOLD_ORDER.order_id in data["filled_orders_by_id"]

    def test_pending_orders_separation(self, store):
        sid = store.create_session(name="Test")

        filled = Order(
            placed_datetime=datetime(2024, 6, 15, 10, 0, 0),
            action=OrderAction.BUY, type=OrderType.MARKET,
            symbol="AAPL", price=150.0, quantity=10, cash=1500.0,
            status=OrderStatus.FILLED, order_id="filled-1",
        )
        pending = Order(
            placed_datetime=datetime(2024, 6, 15, 10, 1, 0),
            action=OrderAction.BUY, type=OrderType.MARKET,
            symbol="TSLA", price=250.0, quantity=5, cash=1250.0,
            status=OrderStatus.PENDING, order_id="pending-1",
        )
        store.save_orders(sid, {filled.order_id: filled, pending.order_id: pending})

        data = store.load_orders(sid)
        assert "filled-1" in data["filled_orders_by_id"]
        assert "filled-1" not in data["pending_orders_by_id"]
        assert "pending-1" in data["pending_orders_by_id"]
        assert "pending-1" not in data["filled_orders_by_id"]


# ------------------------------------------------------------------ #
#  Time filtering tests
# ------------------------------------------------------------------ #

class TestTimeFiltering:

    def test_load_portfolio_history_with_range(self, store):
        sid = store.create_session(name="Test")

        for i in range(10):
            ts = datetime(2024, 6, 15, 10, i, 0)
            tick = [PriceData("AAPL", ts, 150, 151, 149, 150.5, 1000)]
            store.save_tick(sid, ts, tick, cash=100000.0, total_value=100000.0)

        start = datetime(2024, 6, 15, 10, 3, 0)
        end = datetime(2024, 6, 15, 10, 7, 0)
        data = store.load_portfolio_history(sid, start=start, end=end)

        assert len(data["value_history"]) == 5  # timestamps 3,4,5,6,7


# ------------------------------------------------------------------ #
#  Multi-session tests
# ------------------------------------------------------------------ #

class TestMultiSession:

    def test_session_isolation(self, store):
        sid1 = store.create_session(name="Session 1")
        sid2 = store.create_session(name="Session 2")

        ts = datetime(2024, 6, 15, 10, 0, 0)
        tick = [PriceData("AAPL", ts, 150, 151, 149, 150.5, 1000)]

        store.save_tick(sid1, ts, tick, cash=100000.0, total_value=100000.0)
        store.save_tick(sid2, ts, tick, cash=200000.0, total_value=200000.0)

        data1 = store.load_portfolio_history(sid1)
        data2 = store.load_portfolio_history(sid2)

        assert data1["cash_history"][ts] == 100000.0
        assert data2["cash_history"][ts] == 200000.0

    def test_multi_session_merge(self, store):
        sid1 = store.create_session(name="Session 1")
        sid2 = store.create_session(name="Session 2")

        ts1 = datetime(2024, 6, 15, 10, 0, 0)
        ts2 = datetime(2024, 6, 16, 10, 0, 0)

        tick1 = [PriceData("AAPL", ts1, 150, 151, 149, 150.5, 1000)]
        tick2 = [PriceData("AAPL", ts2, 155, 156, 154, 155.5, 1200)]

        store.save_tick(sid1, ts1, tick1, cash=100000.0, total_value=100000.0)
        store.save_tick(sid2, ts2, tick2, cash=99000.0, total_value=101000.0)

        data = store.load_portfolio_history_multi([sid1, sid2])
        assert len(data["value_history"]) == 2
        assert ts1 in data["value_history"]
        assert ts2 in data["value_history"]

    def test_multi_session_orders_merge(self, store, sample_market_order):
        sid1 = store.create_session(name="Session 1")
        sid2 = store.create_session(name="Session 2")

        o2 = Order(
            placed_datetime=datetime(2024, 6, 16, 10, 0, 0),
            action=OrderAction.SELL, type=OrderType.MARKET,
            symbol="AAPL", price=155.0, quantity=10, cash=1550.0,
            status=OrderStatus.FILLED, order_id="local-test-002",
        )

        store.save_order(sid1, sample_market_order)
        store.save_order(sid2, o2)

        data = store.load_orders_multi([sid1, sid2])
        assert len(data["all_orders"]) == 2


# ------------------------------------------------------------------ #
#  AnalysisEngine integration tests
# ------------------------------------------------------------------ #

class TestAnalysisEngineIntegration:

    def test_create_analysis_engine(self, store):
        sid = store.create_session(name="Test")

        # Save some ticks
        for i in range(10):
            ts = datetime(2024, 6, 15, 10, i, 0)
            tick = [PriceData("AAPL", ts, 150 + i, 151 + i, 149, 150.5 + i, 1000)]
            store.save_tick(sid, ts, tick, cash=100000.0 - i * 1000,
                            total_value=100000.0 + i * 500)

        # Save a buy and sell order
        buy = Order(
            placed_datetime=datetime(2024, 6, 15, 10, 2, 0),
            executed_datetime=datetime(2024, 6, 15, 10, 2, 0),
            action=OrderAction.BUY, type=OrderType.MARKET,
            symbol="AAPL", price=152.0, quantity=10, cash=1520.0,
            status=OrderStatus.FILLED, order_id="buy-1",
        )
        sell = Order(
            placed_datetime=datetime(2024, 6, 15, 10, 8, 0),
            executed_datetime=datetime(2024, 6, 15, 10, 8, 0),
            action=OrderAction.SELL, type=OrderType.MARKET,
            symbol="AAPL", price=158.0, quantity=10, cash=1580.0,
            status=OrderStatus.FILLED, order_id="sell-1",
        )
        store.save_orders(sid, {buy.order_id: buy, sell.order_id: sell})

        engine = store.create_analysis_engine(sid)

        # Verify portfolio shell
        assert engine.portfolio.keep_history is True
        assert len(engine.portfolio.value_history) == 10
        assert len(engine.portfolio.tick_history) == 10

        # Verify order manager shell
        assert len(engine.order_manager.all_orders) == 2
        assert len(engine.order_manager.filled_orders_by_id) == 2

        # Extract trades
        trades = engine.extract_trades()
        assert len(trades) == 1
        assert trades[0].symbol == "AAPL"
        assert trades[0].entry_price == 152.0
        assert trades[0].exit_price == 158.0

    def test_create_analysis_engine_multi(self, store):
        sid1 = store.create_session(name="Session 1")
        sid2 = store.create_session(name="Session 2")

        # Session 1: buy
        ts1 = datetime(2024, 6, 15, 10, 0, 0)
        tick1 = [PriceData("AAPL", ts1, 150, 151, 149, 150.5, 1000)]
        store.save_tick(sid1, ts1, tick1, cash=100000.0, total_value=100000.0)

        buy = Order(
            placed_datetime=ts1, executed_datetime=ts1,
            action=OrderAction.BUY, type=OrderType.MARKET,
            symbol="AAPL", price=150.0, quantity=10, cash=1500.0,
            status=OrderStatus.FILLED, order_id="buy-cross-1",
        )
        store.save_order(sid1, buy)

        # Session 2: sell
        ts2 = datetime(2024, 6, 16, 10, 0, 0)
        tick2 = [PriceData("AAPL", ts2, 160, 161, 159, 160.5, 1200)]
        store.save_tick(sid2, ts2, tick2, cash=101600.0, total_value=101600.0)

        sell = Order(
            placed_datetime=ts2, executed_datetime=ts2,
            action=OrderAction.SELL, type=OrderType.MARKET,
            symbol="AAPL", price=160.0, quantity=10, cash=1600.0,
            status=OrderStatus.FILLED, order_id="sell-cross-1",
        )
        store.save_order(sid2, sell)

        engine = store.create_analysis_engine_multi([sid1, sid2])

        assert len(engine.portfolio.value_history) == 2
        assert len(engine.order_manager.all_orders) == 2

        trades = engine.extract_trades()
        assert len(trades) == 1
        assert trades[0].pnl > 0  # Profitable trade


class TestOptimizationEvents:

    def test_save_and_list_optimization_events(self, store):
        sid = store.create_session(name="WF Live")
        event_id = store.save_optimization_event(
            sid,
            {
                "event_type": "walk_forward_live",
                "created_at": datetime(2024, 6, 15, 12, 0, 0),
                "objective_metric": "annualized_return",
                "adopted": True,
                "status": "pending_activation",
            },
        )

        store.update_optimization_event(
            sid,
            event_id,
            {"status": "activated", "activation_time": datetime(2024, 6, 15, 12, 5, 0)},
        )

        events = store.list_optimization_events(sid)
        assert len(events) == 1
        assert events[0]["event_id"] == event_id
        assert events[0]["status"] == "activated"
        assert events[0]["objective_metric"] == "annualized_return"


# ------------------------------------------------------------------ #
#  PortfolioShell and OrderManagerShell tests
# ------------------------------------------------------------------ #

class TestShellObjects:

    def test_portfolio_shell_empty(self, store):
        sid = store.create_session(name="Empty")
        engine = store.create_analysis_engine(sid)

        assert engine.portfolio.keep_history is True
        assert engine.portfolio.total_value == 0.0
        assert engine.portfolio.cash == 0.0
        assert len(engine.portfolio.value_history) == 0

    def test_order_manager_shell_properties(self, store, sample_market_order):
        sid = store.create_session(name="Test")
        store.save_order(sid, sample_market_order)

        engine = store.create_analysis_engine(sid)
        om = engine.order_manager

        assert hasattr(om, "all_orders")
        assert hasattr(om, "filled_orders_by_id")
        assert hasattr(om, "pending_orders_by_id")
        assert sample_market_order.order_id in om.all_orders
