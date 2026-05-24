"""
Base classes for execution engines.

Engines are the top-level orchestrators that drive the trading pipeline.
They coordinate four components — DataProvider, Algorithm, Portfolio, and
OrderManager — through a tick-based loop where each tick of market data
flows through the pipeline:

    DataProvider -> Algorithm -> Portfolio -> OrderManager

Two base classes are provided:

    BaseEngine:
        Synchronous engine for backtesting. Subclasses implement run() to
        iterate over data and on_tick() to process each tick through the
        algo -> portfolio -> OM pipeline.

    AsyncEngine(BaseEngine):
        Asynchronous engine for live/streaming data. Receives ticks via
        an asyncio.Queue from async data handlers (e.g. WebSocket callbacks)
        and processes them in a background thread to avoid blocking the
        event loop.

Implementations:
    - BacktestingEngine:         Iterates over DataProvider ticks synchronously
    - AlpacaRealTimeEngine:      Streams live data from Alpaca WebSocket
    - TickAggregationPassthroughEngine: Aggregates ticks before processing
"""
from typing import final
from trading.core.classes import PriceData, TickResults
from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.om.order_manager import OrderManager
from trading.data_providers.data_provider import DataProvider
from abc import ABC, abstractmethod
import asyncio
import signal
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class BaseEngine(ABC):
    """
    Abstract base class for all execution engines.

    Provides the common interface and component wiring for engines that
    drive the trading pipeline. Subclasses implement run(), on_tick(),
    and finalize() to define how data flows through the system.

    Subclassing notes:
        - Implement run() as the main loop or orchestration entrypoint.
          For backtesting, this typically iterates over DataProvider ticks.
          For live engines, this starts the data stream.
        - Implement on_tick(tick) to process a single tick through the
          algo -> portfolio -> OM pipeline. The standard pattern is:
              signals = self.al.on_data(tick)
              return self.pf.process_market_signals_for_tick(signals, tick)
        - Implement finalize() for end-of-run cleanup (analysis, reporting).
        - All four components (dp, al, om, pf) are optional — engines can
          operate in reduced modes (e.g. data streaming without trading).

    Warm-up:
        Call warm_up(data_provider) before the main run loop to pre-populate
        the algorithm's price history from historical data. This prevents a
        cold-start period where the algorithm has empty history deques and
        cannot generate meaningful signals.

        The method iterates the data provider, collects all ticks, and
        passes them to Algorithm.warm_up(). Not @final — engines can
        override to add extra warmup behaviour (e.g. warming up portfolio
        state or indicators).

        AlpacaRealTimeEngine calls warm_up() automatically during _connect()
        when a "warmup" section is present in the engine config. It defaults
        the data provider's "limit" to the algorithm's history_length so
        exactly the right number of bars are fetched.

    State persistence:
        When a ``state_store`` section is present in the engine config and
        ``enabled`` is True, the engine creates a TradingStateStore and
        writes tick data, portfolio snapshots, signals, and orders to
        MongoDB after each tick via ``_persist_tick()``.

    SIGINT handling:
        By default engines preserve normal ``Ctrl-C`` behavior so
        long-running backtests and HPO jobs can be interrupted cleanly.
        Set the global ``debug_on_sigint: true`` flag in ``config.yaml``
        only when you explicitly want ``Ctrl-C`` to open the debug REPL
        instead.

        Subclasses call ``_persist_tick(tick, signals, tick_results)`` at
        the end of their ``on_tick()`` implementation. ``_init_state_store()``
        is called during ``__init__`` and creates or resumes a session.

    Attributes:
        cfg (dict): Engine configuration dict.
        dp (DataProvider): Data source. Provides tick data via iterate().
            May be None for engines that receive data externally (e.g.
            WebSocket streams).
        al (Algorithm): Trading algorithm. Converts ticks into signals
            via on_data().
        pf (Portfolio): Portfolio manager. Converts signals into orders
            via process_market_signals_for_tick().
        om (OrderManager): Order execution backend. Submits and tracks
            orders. Typically set on the Portfolio before engine starts.
        state_store (TradingStateStore | None): MongoDB persistence layer.
            None when state_store is not configured or disabled.
        session_id (str | None): Current session ID in the state store.

    Example:
        class MyEngine(BaseEngine):
            def run(self):
                for tick in self.dp.iterate():
                    self.on_tick(tick)
                self.finalize()

            def on_tick(self, tick):
                signals = self.al.on_data(tick)
                return self.pf.process_market_signals_for_tick(signals, tick)

            def finalize(self):
                print(f"Final value: ${self.pf.total_value:,.2f}")
    """
    dp = DataProvider
    al = Algorithm
    pf = Portfolio
    om = OrderManager

    def __init__(
            self, cfg:dict= None, dp: DataProvider = None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):
        """
        Initialize the engine with its pipeline components.

        Args:
            cfg: Engine configuration dict. Contents are engine-specific
                (e.g. experiment names, run parameters). Defaults to
                empty dict if None.
            dp: DataProvider for sourcing tick data. Required for
                backtesting engines, optional for live engines that
                receive data externally.
            al: Algorithm instance for generating signals from tick data.
            om: OrderManager instance for order execution. Usually also
                passed to the Portfolio constructor.
            pf: Portfolio instance for converting signals to orders and
                tracking positions/cash.
        """
        if cfg is None: cfg = {}
        self.cfg = cfg
        self.dp = dp
        self.al = al
        self.om = om
        self.pf = pf
        self.state_store = None
        self.session_id = None
        self._debug_requested = False
        self._debug_repl = None
        self._install_debug_handler()
        self._init_state_store()
        logger.debug(f"Engine initialized: {self.__class__.__name__} al={type(al).__name__ if al else None} pf={type(pf).__name__ if pf else None} om={type(om).__name__ if om else None}")

    def _install_debug_handler(self):
        """Optionally replace SIGINT with a handler that enters the debug REPL."""
        from utils.config_manager import ConfigManager

        if not ConfigManager().get("debug_on_sigint", False):
            return

        def _handler(signum, frame):
            self._debug_requested = True
        try:
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            pass  # can't install from a non-main thread; debug mode unavailable

    def _check_debug(self):
        """Enter the debug REPL if Ctrl-C was pressed since the last tick."""
        if not self._debug_requested:
            return
        self._debug_requested = False
        if self._debug_repl is None:
            from utils.debug_repl import DebugREPL
            self._debug_repl = DebugREPL()
            self._debug_repl.attach(al=self.al, pf=self.pf, om=self.om, engine=self)
        self._debug_repl.enter()
        if self._debug_repl._quit:
            raise SystemExit(0)

    def get_progress_snapshot(self):
        """Return a JSON-serializable progress snapshot for the debug REPL."""
        return None

    def _init_state_store(self):
        """Initialize TradingStateStore from config if enabled."""
        ss_cfg = self.cfg.get("state_store")
        if ss_cfg is None:
            # Fall back to root config.yaml via singleton ConfigManager
            from utils.config_manager import ConfigManager
            ss_cfg = ConfigManager().get("state_store") or {}
        if not ss_cfg.get("enabled", False):
            return

        try:
            from utils.trading_state_store import TradingStateStore
        except ImportError:
            logger.warning("pymongo not installed — state_store disabled")
            return

        uri = ss_cfg.get("connection_uri", "mongodb://localhost:27017")
        db = ss_cfg.get("database", "trading")
        self.state_store = TradingStateStore(connection_uri=uri, database=db)

        # Build algo/portfolio metadata
        algo_portfolio_meta = {
            "algorithm_class": f"{type(self.al).__module__}.{type(self.al).__name__}" if self.al else None,
            "algorithm_config": self.al.cfg if self.al else None,
            "portfolio_class": f"{type(self.pf).__module__}.{type(self.pf).__name__}" if self.pf else None,
            "portfolio_config": self.pf.cfg if self.pf else None,
        }

        # Create or resume session
        session_id = ss_cfg.get("session_id")
        if session_id and self.state_store.get_session(session_id):
            self.session_id = session_id
            self.state_store.update_session(session_id, {
                f"metadata.{k}": v for k, v in algo_portfolio_meta.items()
            })
            logger.info(f"Resumed state_store session {session_id[:8]}...")
        else:
            metadata = ss_cfg.get("metadata", {})
            metadata.update(algo_portfolio_meta)
            self.session_id = self.state_store.create_session(
                session_id=session_id,
                name=ss_cfg.get("session_name", ""),
                account_id=ss_cfg.get("account_id", ""),
                metadata=metadata,
            )
            logger.info(f"Created state_store session {self.session_id[:8]}...")

    def _persist_tick(
        self,
        tick: list[PriceData],
        signals: list = None,
        tick_results: 'TickResults' = None,
    ):
        """Write tick data to the state store (if enabled).

        Called by subclass on_tick() implementations after processing.
        Saves tick data, portfolio snapshot, signals, and upserts all orders.

        Args:
            tick: Current tick PriceData list.
            signals: Signals generated this tick (may be empty/None).
            tick_results: TickResults from portfolio processing.
        """
        if self.state_store is None or self.session_id is None:
            return

        timestamp = tick[0].timestamp if tick else None
        if timestamp is None:
            return

        try:
            self.state_store.save_tick(
                self.session_id,
                timestamp,
                tick,
                self.pf.cash if self.pf else 0.0,
                self.pf.total_value if self.pf else 0.0,
                signals,
                self.pf.positions if self.pf else {},
            )
            if self.om:
                self.state_store.save_orders(self.session_id, self.om.all_orders)
        except Exception:
            logger.exception("Failed to persist tick to state_store")

    def warm_up(self, data_provider: DataProvider):
        """Pre-populate the algorithm's price history from a DataProvider.

        Call before the main run loop so the algorithm has sufficient
        history from tick one. Not @final — engines can override if they
        need extra warmup logic (e.g. also warming up portfolio state).

        Args:
            data_provider: A configured DataProvider instance.
                load_data() is called automatically by iterate() if needed.
        """
        if self.al is None:
            logger.warning("warm_up called but no algorithm is set")
            return
        ticks = list(data_provider.iterate())
        self.al.warm_up(ticks)

    @abstractmethod
    def run(self):
        """
        Run the engine's main loop.

        Override this in your subclass. For backtesting engines, this
        typically iterates over self.dp.iterate() and calls on_tick()
        for each tick. For live engines, this starts the data stream
        and blocks until stopped.
        """
        raise NotImplementedError()

    @abstractmethod
    def on_tick(self, tick: list[PriceData]) -> TickResults:
        """
        Process a single tick of market data through the pipeline.

        Override this in your subclass. The standard implementation is:
            signals = self.al.on_data(tick)
            return self.pf.process_market_signals_for_tick(signals, tick)

        Args:
            tick: List of PriceData objects for the current tick (one
                per symbol in the data stream).

        Returns:
            TickResults containing any orders generated this tick.
        """
        raise NotImplementedError()

    @abstractmethod
    def finalize(self):
        """
        Finalize the engine after the main loop completes.

        Override this in your subclass for end-of-run tasks such as
        running AnalysisEngine, logging results to MLflow, or printing
        summary reports. Called after run() completes or when the engine
        is stopped.
        """
        raise NotImplementedError()


class AsyncEngine(BaseEngine):
    """
    Base class for asynchronous/streaming execution engines.

    Solves the async-sync boundary problem: WebSocket callbacks (async)
    need to invoke on_tick() (sync, potentially slow — runs algo +
    portfolio + broker API calls) without blocking the event loop.

    Architecture:
        async callback  -->  submit_tick(tick)  -->  asyncio.Queue
                                                         |
                                                    _worker() loop
                                                         |
                                                  asyncio.to_thread(on_tick)
                                                         |
                                                  algo -> portfolio -> OM

    Key design decisions:
        - Ticks are enqueued via submit_tick() (non-blocking, called from
          async data handlers like WebSocket callbacks).
        - A single _worker task pulls ticks from the queue and runs
          on_tick() in a background thread via asyncio.to_thread(),
          keeping the event loop free for more incoming data.
        - The queue serializes tick processing, preventing race conditions
          on shared portfolio/position state.
        - Exceptions in on_tick() are logged but don't crash the worker,
          allowing the engine to continue processing subsequent ticks.

    Subclassing notes:
        - Implement on_tick(tick) — sync method, same pattern as
          BacktestingEngine (algo -> portfolio pipeline).
        - Implement _connect() — async method that starts your data
          stream (e.g. WebSocket). This method should block for the
          lifetime of the stream (start() awaits it, keeping the event
          loop alive).
        - Implement _disconnect() — async method that tears down the
          data stream.
        - Call await self.submit_tick(tick) from your async data handlers
          to enqueue ticks for processing.

    Lifecycle:
        # Sync entry point (blocks until stream ends):
        engine.run()

        # Or async entry point:
        await engine.start()    # spawns worker, calls _connect()
        ...                     # data flows via submit_tick -> on_tick
        await engine.stop()     # calls _disconnect(), drains queue

    Important:
        _connect() must block (or await a long-running coroutine) for the
        engine to stay alive. If _connect() returns immediately, start()
        completes, asyncio.run() exits, and the program terminates. For
        WebSocket-based engines, await the stream's async run method
        inside _connect().

    Attributes:
        _queue (asyncio.Queue): Internal tick queue connecting async data
            handlers to the sync on_tick() processing.
        _worker_task (asyncio.Task): The background task running _worker().
            Created by start(), canceled by stop().

    Example:
        class MyStreamEngine(AsyncEngine):
            async def _connect(self):
                self.ws = WebSocketClient(self.cfg["url"])
                self.ws.on_message(self._handle_message)
                await self.ws.run_forever()  # blocks

            async def _disconnect(self):
                await self.ws.close()

            async def _handle_message(self, msg):
                tick = [parse_tick(msg)]
                await self.submit_tick(tick)

            def on_tick(self, tick):
                signals = self.al.on_data(tick)
                return self.pf.process_market_signals_for_tick(signals, tick)
    """

    def __init__(self, cfg, dp, al, om, pf):
        """
        Initialize the async engine with a tick queue and worker slot.

        Args:
            cfg: Engine configuration dict.
            dp: DataProvider (optional, often None for streaming engines).
            al: Algorithm instance.
            om: OrderManager instance.
            pf: Portfolio instance.
        """
        super().__init__(cfg, dp, al, om, pf)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._debug_task: asyncio.Task | None = None
        self._debug_exit_exc: BaseException | None = None

    @final
    async def start(self):
        """
        Start the async engine.

        Spawns the _worker task (pulls ticks from queue, runs on_tick in
        a thread), then awaits _connect() which should block for the
        lifetime of the data stream.
        """
        logger.debug("Starting async engine: spawning worker task")
        self._worker_task = asyncio.create_task(self._worker())
        self._debug_task = asyncio.create_task(self._debug_monitor())
        logger.debug("Connecting to data source")
        try:
            await self._connect()
        finally:
            if self._debug_task is not None:
                self._debug_task.cancel()
                self._debug_task = None
        if self._debug_exit_exc is not None:
            exc = self._debug_exit_exc
            self._debug_exit_exc = None
            raise exc

    @final
    def run(self):
        """
        Sync entry point. Creates an event loop and runs start().

        Blocks until the data stream ends or the engine is stopped.
    This is the typical entry point for scripts and launchers.
        """
        asyncio.run(self.start())

    @final
    def finalize(self):
        """
        Sync entry point for shutdown. Runs stop() in an event loop.

        Disconnects from the data source, drains remaining ticks in the
        queue, and cancels the worker task.
        """
        asyncio.run(self.stop())

    @final
    async def stop(self):
        """
        Stop the async engine gracefully.

        Disconnects from the data source, waits for all queued ticks
        to finish processing, then cancels the worker task.
        """
        logger.debug("Stopping async engine: disconnecting")
        await self._disconnect()
        logger.debug(f"Draining queue ({self._queue.qsize()} items remaining)")
        await self._queue.join()
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        if self._debug_task is not None:
            self._debug_task.cancel()
            self._debug_task = None
        logger.debug("Worker task canceled, engine stopped")

    async def submit_tick(self, tick: list[PriceData]):
        """
        Enqueue a tick for processing.

        Non-blocking and safe to call from async data handlers (e.g.
        WebSocket callbacks). The tick will be picked up by _worker()
        and processed via on_tick() in a background thread.

        Args:
            tick: List of PriceData objects for this tick (one per symbol).
        """
        symbols = [p.symbol for p in tick] if tick else []
        logger.debug(f"Tick enqueued: {symbols} queue_size={self._queue.qsize()}")
        await self._queue.put(tick)

    @abstractmethod
    def on_tick(self, tick: list[PriceData]) -> TickResults:
        """
        Process a single tick through the trading pipeline.

        Override this in your subclass. This is a sync method even in
        AsyncEngine — it runs in a background thread via
        asyncio.to_thread() so it doesn't block the event loop.

        The standard implementation is:
            signals = self.al.on_data(tick)
            return self.pf.process_market_signals_for_tick(signals, tick)

        Args:
            tick: List of PriceData objects for the current tick.

        Returns:
            TickResults containing any orders generated this tick.
        """
        raise NotImplementedError()

    async def _worker(self):
        """
        Internal tick processing loop.

        Runs as an asyncio.Task created by start(). Pulls ticks from the
        queue one at a time and runs on_tick() in a background thread via
        asyncio.to_thread(). This serializes tick processing (no two ticks
        are processed concurrently), preventing race conditions on shared
        portfolio/position state.

        Exceptions in on_tick() are logged but do not crash the worker.
        """
        logger.debug("Worker started, waiting for ticks")
        while True:
            tick = await self._queue.get()
            try:
                symbols = [p.symbol for p in tick] if tick else []
                logger.debug(f"Worker processing tick: {symbols}")
                result = await asyncio.to_thread(self.on_tick, tick)
                if result and result.orders:
                    logger.debug(f"Tick produced {len(result.orders)} orders")
            except Exception:
                logger.exception("on_tick raised an exception")
            finally:
                self._queue.task_done()
            # Enter the debug REPL between ticks if Ctrl-C was pressed.
            # Runs in a thread so the event loop stays alive for WebSocket
            # keepalives while the user is at the prompt.
            if self._debug_requested:
                await asyncio.to_thread(self._check_debug)

    async def _debug_monitor(self):
        """Poll for debug requests even when no ticks are flowing."""
        while True:
            try:
                if self._debug_requested:
                    await asyncio.to_thread(self._check_debug)
            except SystemExit as exc:
                self._debug_exit_exc = exc
                await self._disconnect()
                return
            await asyncio.sleep(0.1)

    @abstractmethod
    async def _connect(self):
        """
        Connect to the data source and start receiving data.

        Override this in your subclass. This method should:
        1. Set up the data stream (e.g. WebSocket connection)
        2. Register async handlers that call submit_tick()
        3. Block for the lifetime of the stream (e.g. await the
           stream's run_forever method)

        Important: This method must block (not return immediately) or
        the engine will shut down. start() awaits this method, and if
        it returns, asyncio.run() exits.

        Example:
            async def _connect(self):
                self.stream = WebSocket(self.cfg["url"])
                self.stream.on_bar(self._handle_bar)
                await self.stream._run_forever()  # blocks
        """
        raise NotImplementedError

    @abstractmethod
    async def _disconnect(self):
        """
        Disconnect from the data source and stop receiving data.

        Override this in your subclass. Clean up the data stream
        connection. Called by stop() before draining the queue.

        Example:
            async def _disconnect(self):
                self.stream.stop()
        """
        raise NotImplementedError
