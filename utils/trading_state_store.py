"""
MongoDB-backed persistence layer for trading state.

Stores portfolio history, orders, ticks, and signals across sessions so that
AnalysisEngine can analyze long-term performance without relying on in-memory
state that is lost on restart.

Data is partitioned by session_id so that different backtest runs or live
trading accounts can be analyzed independently or combined.

Collections:
    sessions             — Session metadata (name, account_id, timestamps)
    ticks                — PriceData per timestamp per symbol
    portfolio_snapshots  — Cash + total_value per timestamp
    orders               — Full order documents (upserted each tick)
    signals              — MarketSignal documents per timestamp

Requires: pymongo
Optional: mongomock (for testing without a real MongoDB instance)

Quick Start
-----------
**Install dependencies:**

    pip install pymongo

**Enable via config.yaml (zero code changes needed):**

    state_store:
      enabled: true
      connection_uri: "mongodb://localhost:27017"
      database: "trading"
      session_id: null         # null = new UUID each run
      session_name: "my_run"
      account_id: ""
      metadata: {}

    When enabled, the engine automatically creates a session, saves tick
    data / portfolio snapshots / signals / orders on every tick, and logs
    the session_id so you can reference it later.

**Manual Python usage (direct instantiation):**

    from utils.trading_state_store import TradingStateStore

    store = TradingStateStore("mongodb://localhost:27017", database="trading")

    # Create a session (returns a UUID string)
    sid = store.create_session(
        name="SMA Crossover Paper",
        account_id="PA-XXX",
        metadata={"sma_short": 5, "sma_long": 20},
    )

    # Inside your engine loop, each tick:
    store.save_tick(sid, timestamp, tick, cash, total_value, signals)
    store.save_orders(sid, order_manager.all_orders)

**List existing sessions:**

    sessions = store.list_sessions()                        # all sessions
    sessions = store.list_sessions(account_id="PA-XXX")    # filtered

    for s in sessions:
        print(s["_id"], s["name"], s["created_at"])

**Rebuild from stored data — two options:**

Option A — ``PortfolioAnalyzer.from_mongodb()`` (recommended, no store object needed)::

    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

    # Single session (connection read from config.yaml state_store section)
    analyzer = PortfolioAnalyzer.from_mongodb(sid)
    results = analyzer.run_analysis(output_dir="output/session_abc")

    # Multiple sessions merged (live bot restarted several times)
    analyzer = PortfolioAnalyzer.from_mongodb_multi([sid1, sid2, sid3])
    results = analyzer.run_full_analysis(log_to_mlflow=True, run_name="Combined")

    # Date-filtered slice
    from datetime import datetime
    analyzer = PortfolioAnalyzer.from_mongodb(
        sid,
        start=datetime(2024, 6, 1),
        end=datetime(2024, 9, 30),
    )

    # Explicit connection (overrides config.yaml)
    analyzer = PortfolioAnalyzer.from_mongodb(
        sid,
        connection_uri="mongodb://hp.lan:27017",
        database="trading_live",
    )

Option B — ``TradingStateStore.create_analysis_engine()`` (AnalysisEngine path)::

    # Single session
    engine = store.create_analysis_engine(sid)
    results = engine.run_full_analysis(log_to_mlflow=False)

    # Multiple sessions merged (e.g. live trading across restarts)
    engine = store.create_analysis_engine_multi([sid1, sid2, sid3])
    results = engine.run_full_analysis(log_to_mlflow=True, run_name="Combined")

    # Time-filtered slice of a session
    from datetime import datetime
    engine = store.create_analysis_engine(
        sid,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 30),
    )
"""
import uuid
from datetime import datetime
from typing import Union

from pymongo import MongoClient, ASCENDING

from trading.core.classes import (
    PriceData, MarketSignal, Order, BracketOrder,
    OrderStatus, OrderType,
)
from utils.utils import (
    serialize_pricedata, deserialize_pricedata,
    serialize_order, deserialize_order,
    serialize_market_signal, deserialize_market_signal,
    make_json_serializable,
)
from utils.logger import Logger

logger = Logger().get_logger(__name__)


def _order_status_rank(order: Union[Order, BracketOrder]) -> tuple[int, datetime | None, datetime | None]:
    """Rank duplicate broker orders so the most-complete representation wins."""
    status_rank = {
        OrderStatus.FILLED: 3,
        OrderStatus.CANCELED: 3,
        OrderStatus.PENDING_SALE: 2,
        OrderStatus.PENDING: 1,
    }.get(order.status, 0)
    return (
        status_rank,
        order.executed_datetime,
        order.placed_datetime,
    )


def _dedupe_orders_by_platform_id(
    orders: list[Union[Order, BracketOrder]],
) -> list[Union[Order, BracketOrder]]:
    """Collapse duplicate local orders that point at the same broker platform_id."""
    deduped: dict[str, Union[Order, BracketOrder]] = {}
    for order in orders:
        key = order.platform_id or order.order_id
        current = deduped.get(key)
        if current is None or _order_status_rank(order) > _order_status_rank(current):
            deduped[key] = order
    return list(deduped.values())


class TradingStateStore:
    """MongoDB persistence layer with session partitioning.

    Each session represents a single backtest run or live trading session.
    Data is stored in 5 collections (sessions, ticks, portfolio_snapshots,
    orders, signals) with composite indexes for fast queries.

    Usage — Path 1: Config-driven (automatic, via engine)
    -------------------------------------------------------
    Set ``state_store.enabled: true`` in config.yaml and start your engine.
    No code changes required.  The engine will:
      1. Call ``create_session()`` with the config values and log the session_id.
      2. Call ``save_tick()`` and ``save_orders()`` automatically on every tick.

    To resume / append to an existing session across restarts, paste the
    session_id UUID into ``state_store.session_id`` in config.yaml.

    Usage — Path 2: Manual (direct Python)
    ----------------------------------------
    ::

        from utils.trading_state_store import TradingStateStore

        store = TradingStateStore("mongodb://localhost:27017", database="trading")

        # Create a session
        sid = store.create_session(
            name="SMA Crossover Paper",
            account_id="PA-XXX",
            metadata={"sma_short": 5, "sma_long": 20},
        )

        # Engine tick loop
        for timestamp, tick, cash, total_value, signals in my_engine:
            store.save_tick(sid, timestamp, tick, cash, total_value, signals)
            store.save_orders(sid, order_manager.all_orders)

    Single-session analysis
    ------------------------
    ::

        engine = store.create_analysis_engine(sid)
        results = engine.run_full_analysis(log_to_mlflow=False)

    Multi-session analysis (merge multiple runs)
    ---------------------------------------------
    Useful when a live trading bot was restarted several times and each restart
    created a new session::

        engine = store.create_analysis_engine_multi([sid1, sid2, sid3])
        results = engine.run_full_analysis(log_to_mlflow=True, run_name="Combined")

    Time-filtered analysis
    -----------------------
    Slice any session to a specific date range::

        from datetime import datetime
        engine = store.create_analysis_engine(
            sid,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
        )
        metrics = engine.calculate_metrics()
    """

    def __init__(
        self,
        connection_uri: str = "mongodb://localhost:27017",
        database: str = "trading",
        client: MongoClient = None,
    ):
        """Connect to MongoDB and ensure indexes exist.

        Args:
            connection_uri: MongoDB connection string.
            database: Database name.
            client: Optional pre-built MongoClient (useful for mongomock in tests).
        """
        if client is not None:
            self._client = client
        else:
            self._client = MongoClient(connection_uri)
        self._db = self._client[database]

        # Collection references
        self._sessions = self._db["sessions"]
        self._ticks = self._db["ticks"]
        self._snapshots = self._db["portfolio_snapshots"]
        self._orders = self._db["orders"]
        self._signals = self._db["signals"]
        self._optimization_events = self._db["optimization_events"]

        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes if they don't already exist."""
        self._ticks.create_index(
            [("session_id", ASCENDING), ("timestamp", ASCENDING), ("symbol", ASCENDING)],
            unique=True,
        )
        self._snapshots.create_index(
            [("session_id", ASCENDING), ("timestamp", ASCENDING)],
            unique=True,
        )
        self._orders.create_index(
            [("session_id", ASCENDING), ("order_id", ASCENDING)],
            unique=True,
        )
        self._signals.create_index(
            [("session_id", ASCENDING), ("timestamp", ASCENDING)],
        )
        self._optimization_events.create_index(
            [("session_id", ASCENDING), ("created_at", ASCENDING)],
        )
        self._optimization_events.create_index(
            [("session_id", ASCENDING), ("event_id", ASCENDING)],
            unique=True,
        )

    # ------------------------------------------------------------------ #
    #  Session management
    # ------------------------------------------------------------------ #

    def create_session(
        self,
        session_id: str = None,
        name: str = "",
        description: str = "",
        account_id: str = "",
        metadata: dict = None,
    ) -> str:
        """Create a new session document.

        Args:
            session_id: Explicit session ID. Auto-generated UUID if None.
            name: Human-readable session name.
            description: Session description.
            account_id: Alpaca account ID or arbitrary identifier.
            metadata: Arbitrary config/params dict.

        Returns:
            The session_id string.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        doc = {
            "_id": session_id,
            "name": name,
            "description": description,
            "account_id": account_id,
            "created_at": datetime.now(),
            "metadata": metadata or {},
        }
        self._sessions.insert_one(doc)
        logger.info(f"Created session {session_id[:8]}... name='{name}' account='{account_id}'")
        return session_id

    def get_session(self, session_id: str) -> dict:
        """Return the session document or None."""
        return self._sessions.find_one({"_id": session_id})

    def update_session(self, session_id: str, update: dict) -> None:
        """Update fields on an existing session document."""
        self._sessions.update_one({"_id": session_id}, {"$set": update})

    def list_sessions(
        self,
        account_id: str = None,
        autostart: bool = None,
    ) -> list[dict]:
        """List sessions, optionally filtered by account_id or autostart."""
        query = {}
        if account_id is not None:
            query["account_id"] = account_id
        if autostart is not None:
            query["autostart"] = autostart
        return list(self._sessions.find(query).sort("created_at", -1))

    # ------------------------------------------------------------------ #
    #  Write methods (called by engine each tick)
    # ------------------------------------------------------------------ #

    def save_tick(
        self,
        session_id: str,
        timestamp: datetime,
        tick: list[PriceData],
        cash: float,
        total_value: float,
        signals: list[MarketSignal] = None,
        positions: dict = None,
        indicator_snapshot: dict | None = None,
    ):
        """Save tick data, portfolio snapshot, and signals for one timestamp.

        Args:
            session_id: Session to write to.
            timestamp: Tick timestamp.
            tick: List of PriceData objects for this tick.
            cash: Portfolio cash after this tick.
            total_value: Portfolio total value after this tick.
            signals: MarketSignal list from this tick (may be empty/None).
            positions: Portfolio positions dict {symbol: Position} (may be empty/None).
        """
        # Tick data (one document per symbol)
        for pd_obj in tick:
            doc = serialize_pricedata(pd_obj)
            doc["session_id"] = session_id
            if indicator_snapshot is not None:
                doc["indicator_snapshot"] = make_json_serializable(indicator_snapshot)
            self._ticks.replace_one(
                {"session_id": session_id, "timestamp": doc["timestamp"], "symbol": doc["symbol"]},
                doc,
                upsert=True,
            )

        # Build positions data from current tick prices
        positions_data = {}
        if positions:
            prices = {pd_obj.symbol: pd_obj.close for pd_obj in tick}
            for symbol, pos in positions.items():
                price = prices.get(symbol)
                positions_data[symbol] = {
                    "quantity": pos.quantity,
                    "price": price,
                    "market_value": pos.quantity * price if price else 0,
                }

        # Portfolio snapshot
        snap = {
            "session_id": session_id,
            "timestamp": timestamp,
            "cash": cash,
            "total_value": total_value,
            "positions": positions_data,
        }
        self._snapshots.replace_one(
            {"session_id": session_id, "timestamp": timestamp},
            snap,
            upsert=True,
        )

        # Signals
        if signals:
            sig_docs = []
            for sig in signals:
                sig_doc = serialize_market_signal(sig)
                sig_doc["session_id"] = session_id
                sig_doc["timestamp"] = timestamp
                sig_docs.append(sig_doc)
            # Delete old signals for this timestamp then insert new ones
            self._signals.delete_many({"session_id": session_id, "timestamp": timestamp})
            self._signals.insert_many(sig_docs)

    def save_order(self, session_id: str, order: Order):
        """Upsert a single order document.

        Args:
            session_id: Session to write to.
            order: Order or BracketOrder to save.
        """
        doc = serialize_order(order)
        doc["session_id"] = session_id
        self._orders.replace_one(
            {"session_id": session_id, "order_id": doc["order_id"]},
            doc,
            upsert=True,
        )

    def save_orders(self, session_id: str, orders: dict[str, Order]):
        """Bulk upsert all orders via bulk_write.

        Args:
            session_id: Session to write to.
            orders: Dict mapping order_id -> Order.
        """
        if not orders:
            return
        for order in orders.values():
            doc = serialize_order(order)
            doc["session_id"] = session_id
            self._orders.replace_one(
                {"session_id": session_id, "order_id": doc["order_id"]},
                doc,
                upsert=True,
            )

    def save_optimization_event(self, session_id: str, event: dict) -> str:
        """Upsert a walk-forward / live optimization event document."""
        event_id = event.get("event_id") or str(uuid.uuid4())
        doc = {
            "session_id": session_id,
            "event_id": event_id,
            "created_at": event.get("created_at", datetime.now()),
            **{k: v for k, v in event.items() if k != "event_id"},
        }
        self._optimization_events.replace_one(
            {"session_id": session_id, "event_id": event_id},
            doc,
            upsert=True,
        )
        return event_id

    def update_optimization_event(self, session_id: str, event_id: str, update: dict) -> None:
        """Update fields on a persisted optimization event."""
        self._optimization_events.update_one(
            {"session_id": session_id, "event_id": event_id},
            {"$set": update},
        )

    def list_optimization_events(self, session_id: str) -> list[dict]:
        """Return optimization events for a session ordered by creation time."""
        return list(
            self._optimization_events.find({"session_id": session_id}).sort("created_at", ASCENDING)
        )

    # ------------------------------------------------------------------ #
    #  Read methods (reconstruct objects for AnalysisEngine)
    # ------------------------------------------------------------------ #

    def load_portfolio_history(
        self,
        session_id: str,
        start: datetime = None,
        end: datetime = None,
    ) -> dict:
        """Load portfolio history data for a session.

        Returns:
            Dict with keys: tick_history, value_history, cash_history, signals_history.
            Each matches the Portfolio attribute format.
        """
        query = {"session_id": session_id}
        time_filter = self._time_filter(start, end)
        if time_filter:
            query["timestamp"] = time_filter

        # Portfolio snapshots -> value_history and cash_history
        value_history = {}
        cash_history = {}
        projection = {"timestamp": 1, "total_value": 1, "cash": 1}
        for doc in self._snapshots.find(query, projection).sort("timestamp", ASCENDING):
            ts = doc["timestamp"]
            value_history[ts] = doc["total_value"]
            cash_history[ts] = doc["cash"]

        # Ticks -> tick_history
        tick_query = {"session_id": session_id}
        if time_filter:
            tick_query["timestamp"] = time_filter
        tick_history = {}
        for doc in self._ticks.find(tick_query).sort("timestamp", ASCENDING):
            ts = doc["timestamp"]
            pd_obj = deserialize_pricedata(doc)
            if ts not in tick_history:
                tick_history[ts] = []
            tick_history[ts].append(pd_obj)

        # Signals -> signals_history
        sig_query = {"session_id": session_id}
        if time_filter:
            sig_query["timestamp"] = time_filter
        signals_history = {}
        for doc in self._signals.find(sig_query).sort("timestamp", ASCENDING):
            ts = doc["timestamp"]
            sig = deserialize_market_signal(doc)
            if ts not in signals_history:
                signals_history[ts] = []
            signals_history[ts].append(sig)

        return {
            "tick_history": tick_history,
            "value_history": value_history,
            "cash_history": cash_history,
            "signals_history": signals_history,
        }

    def load_equity_history(
        self,
        session_id: str,
        start: datetime = None,
        end: datetime = None,
    ) -> dict:
        """Load only snapshot-derived equity/cash history for a session."""
        query = {"session_id": session_id}
        time_filter = self._time_filter(start, end)
        if time_filter:
            query["timestamp"] = time_filter

        value_history = {}
        cash_history = {}
        projection = {"timestamp": 1, "total_value": 1, "cash": 1}
        for doc in self._snapshots.find(query, projection).sort("timestamp", ASCENDING):
            ts = doc["timestamp"]
            value_history[ts] = doc["total_value"]
            cash_history[ts] = doc["cash"]

        return {
            "value_history": value_history,
            "cash_history": cash_history,
        }

    def load_orders(
        self,
        session_id: str,
        start: datetime = None,
        end: datetime = None,
    ) -> dict:
        """Load orders for a session, returning dicts matching OrderManager attributes.

        Returns:
            Dict with keys: all_orders, filled_orders_by_id, pending_orders_by_id.
        """
        query = {"session_id": session_id}
        if start or end:
            time_filter = self._time_filter(start, end)
            if time_filter:
                query["placed_datetime"] = time_filter

        all_orders = {}
        filled_orders = {}
        pending_orders = {}

        orders = [deserialize_order(doc) for doc in self._orders.find(query)]
        for order in _dedupe_orders_by_platform_id(orders):
            all_orders[order.order_id] = order
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED}:
                filled_orders[order.order_id] = order
            else:
                pending_orders[order.order_id] = order

        # For filled bracket orders, also register the SOLD_ORDER child
        for order in list(filled_orders.values()):
            if (isinstance(order, BracketOrder)
                    and order.status == OrderStatus.FILLED
                    and order.SOLD_ORDER is not None):
                so = order.SOLD_ORDER
                filled_orders[so.order_id] = so
                all_orders[so.order_id] = so

        return {
            "all_orders": all_orders,
            "filled_orders_by_id": filled_orders,
            "pending_orders_by_id": pending_orders,
        }

    def load_portfolio_history_multi(
        self,
        session_ids: list[str],
        start: datetime = None,
        end: datetime = None,
    ) -> dict:
        """Load and merge portfolio history from multiple sessions."""
        merged = {
            "tick_history": {},
            "value_history": {},
            "cash_history": {},
            "signals_history": {},
        }
        for sid in session_ids:
            data = self.load_portfolio_history(sid, start, end)
            merged["tick_history"].update(data["tick_history"])
            merged["value_history"].update(data["value_history"])
            merged["cash_history"].update(data["cash_history"])
            merged["signals_history"].update(data["signals_history"])
        return merged

    def load_orders_multi(
        self,
        session_ids: list[str],
        start: datetime = None,
        end: datetime = None,
    ) -> dict:
        """Load and merge orders from multiple sessions."""
        merged = {
            "all_orders": {},
            "filled_orders_by_id": {},
            "pending_orders_by_id": {},
        }
        for sid in session_ids:
            data = self.load_orders(sid, start, end)
            merged["all_orders"].update(data["all_orders"])
            merged["filled_orders_by_id"].update(data["filled_orders_by_id"])
            merged["pending_orders_by_id"].update(data["pending_orders_by_id"])
        return merged

    # ------------------------------------------------------------------ #
    #  AnalysisEngine integration
    # ------------------------------------------------------------------ #

    def create_analysis_engine(
        self,
        session_id: str,
        start: datetime = None,
        end: datetime = None,
    ):
        """Build an AnalysisEngine from stored session data.

        Creates lightweight proxy Portfolio and OrderManager objects
        populated with history from MongoDB, then returns an AnalysisEngine
        ready for run_full_analysis().

        Args:
            session_id: Session to load.
            start: Optional start datetime filter.
            end: Optional end datetime filter.

        Returns:
            AnalysisEngine instance.
        """
        return self._build_analysis_engine(
            [session_id], start, end
        )

    def create_analysis_engine_multi(
        self,
        session_ids: list[str],
        start: datetime = None,
        end: datetime = None,
    ):
        """Build an AnalysisEngine from multiple sessions combined.

        Args:
            session_ids: List of session IDs to combine.
            start: Optional start datetime filter.
            end: Optional end datetime filter.

        Returns:
            AnalysisEngine instance.
        """
        return self._build_analysis_engine(
            session_ids, start, end
        )

    def _build_analysis_engine(
        self,
        session_ids: list[str],
        start: datetime = None,
        end: datetime = None,
    ):
        """Internal helper to build AnalysisEngine from one or more sessions."""
        from trading.analysis.analysis_engine import AnalysisEngine

        # Load data
        if len(session_ids) == 1:
            pf_data = self.load_portfolio_history(session_ids[0], start, end)
            order_data = self.load_orders(session_ids[0], start, end)
        else:
            pf_data = self.load_portfolio_history_multi(session_ids, start, end)
            order_data = self.load_orders_multi(session_ids, start, end)

        # Build proxy portfolio
        portfolio = _PortfolioShell(pf_data)

        # Build proxy order manager
        om = _OrderManagerShell(order_data)

        return AnalysisEngine(portfolio, om)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _time_filter(start: datetime = None, end: datetime = None) -> dict:
        """Build a MongoDB timestamp range filter."""
        f = {}
        if start:
            f["$gte"] = start
        if end:
            f["$lte"] = end
        return f if f else {}


class _PortfolioShell:
    """Lightweight proxy that satisfies AnalysisEngine's Portfolio interface.

    Only populates the history dicts that AnalysisEngine reads. Not a real
    Portfolio — cannot process signals or manage orders.
    """

    def __init__(self, pf_data: dict):
        self.keep_history = True
        self.tick_history = pf_data["tick_history"]
        self.value_history = pf_data["value_history"]
        self.cash_history = pf_data["cash_history"]
        self.signals_history = pf_data["signals_history"]

        # AnalysisEngine reads these for metrics
        if self.value_history:
            sorted_ts = sorted(self.value_history.keys())
            self.total_value = self.value_history[sorted_ts[-1]]
            self.cash = self.cash_history[sorted_ts[-1]]
        else:
            self.total_value = 0.0
            self.cash = 0.0

        self.positions = {}
        self.om = None


class _OrderManagerShell:
    """Lightweight proxy that satisfies AnalysisEngine's OrderManager interface.

    Only populates the order dicts that AnalysisEngine reads.
    """

    def __init__(self, order_data: dict):
        self._all_orders = order_data["all_orders"]
        self._filled_orders_by_id = order_data["filled_orders_by_id"]
        self._pending_orders_by_id = order_data["pending_orders_by_id"]

    @property
    def all_orders(self):
        return self._all_orders

    @property
    def filled_orders_by_id(self):
        return self._filled_orders_by_id

    @property
    def pending_orders_by_id(self):
        return self._pending_orders_by_id
