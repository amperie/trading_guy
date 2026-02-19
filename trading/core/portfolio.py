"""
Base class for portfolio management.

The portfolio sits between the algorithm (which produces signals) and the
order manager (which executes orders). Each tick, the engine calls
process_market_signals_for_tick() which:

    1. Polls the OrderManager for status changes on pending orders
    2. Updates cash and positions from any newly filled orders
    3. Optionally syncs cash/positions from the broker (every N ticks)
    4. Delegates to the subclass strategy via process_tick_market_signals_logic()
    5. Submits resulting orders to the OrderManager
    6. Recalculates total portfolio value
    7. Records history (if enabled)

Data flow:
    Engine -> process_market_signals_for_tick(signals, tick)
        -> OM.update_pending_orders()        (check for fills)
        -> _update_pf_from_changed_orders()  (adjust cash/positions)
        -> _sync_from_broker()               (periodic, if enabled)
        -> process_tick_market_signals_logic() (subclass strategy)
        -> OM.submit_orders()                (send new orders)
        -> _update_pf_value()                (recalc total value)
        -> record history
        -> return TickResults

Broker synchronization:
    When sync_with_broker is enabled in the portfolio config, the
    portfolio periodically queries the OrderManager's broker backend
    for the true account state (cash and positions) and overwrites
    local values. This corrects drift from asynchronous order fills
    and recovers positions from previous trading sessions. The sync
    runs every sync_interval ticks (default 60) and also at engine
    startup via _sync_from_broker(). For backtesting, the sync is
    a no-op since BacktestingOM.get_broker_account_state() returns None.
"""
from trading.core.classes import MarketSignal, Order, PriceData, Position, OrderStatus, OrderAction, OrderType
from trading.core.classes import TickResults
from trading.core.om.order_manager import OrderManager
from utils.logger import Logger
from abc import ABC, abstractmethod
from typing import final
from datetime import datetime

from utils.utils import find_pricedata_in_list

logger = Logger().get_logger(__name__)


class Portfolio(ABC):
    """
    Base class for portfolio logic and order generation.

    Manages cash, positions, and portfolio value. Coordinates with the
    OrderManager to submit and track orders. Subclasses implement
    process_tick_market_signals_logic() to define how signals are
    converted into orders.

    Subclassing notes:
        - Implement process_tick_market_signals_logic(signals, tick) to
          convert MarketSignals into Orders.
        - Do NOT override process_market_signals_for_tick(); it is @final
          and manages the full tick lifecycle (pending order updates, broker
          sync, order submission, portfolio value tracking, history recording).
        - Use self.cash and self.positions for position sizing decisions.
        - Use self.om (OrderManager) for order creation helpers if needed.
        - Return a TickResults(orders=[...]) from your strategy method.
          Zero-quantity orders are filtered out automatically.

    Attributes:
        cfg (dict): Portfolio configuration dict.
        om (OrderManager): The order manager used to submit and track orders.
        cash (float): Current available cash. Updated automatically on
            buy/sell fills and broker syncs.
        positions (dict[str, Position]): Current holdings keyed by symbol.
            Updated automatically on buy/sell fills and broker syncs.
        total_value (float): Cash + market value of all positions. Recalculated
            each tick using closing prices from the current tick data.
        keep_history (bool): When True, tick_history, value_history,
            cash_history, and signals_history are populated each tick.
        sync_with_broker (bool): When True, periodically sync cash and
            positions from the broker via OrderManager.get_broker_account_state().
            Configured via cfg["sync_with_broker"]. Default: False.
        tick_history (dict[datetime, list[PriceData]]): Raw tick data per
            timestamp. Only populated when keep_history is True.
        value_history (dict[datetime, float]): Portfolio total value per
            timestamp. Only populated when keep_history is True.
        cash_history (dict[datetime, float]): Cash balance per timestamp.
            Only populated when keep_history is True.
        signals_history (dict[datetime, list[MarketSignal]]): Signals received
            per timestamp. Only populated when keep_history is True and
            signals were present for that tick.
        previous_price (dict[str, float]): Last known closing price per symbol.
            Used as fallback when a position's symbol is missing from the
            current tick data.

    Order processing flow:
        When process_market_signals_for_tick() receives a filled order:
        - MARKET BUY:  cash decreases, position quantity increases
        - MARKET SELL: cash increases, position quantity decreases
        - BRACKET (PENDING_SALE): initial BUY is applied, order stays pending
          waiting for stop-loss or profit-taker to trigger
        - BRACKET (FILLED): exit leg (SOLD_ORDER) is applied as a sell
        - CANCELED: no cash/position changes, order marked as processed

    Broker sync flow:
        When sync_with_broker is True, every sync_interval ticks:
        - Calls OM.get_broker_account_state() for current broker state
        - Overwrites self.cash with broker cash
        - Adds/updates/removes positions to match broker holdings
        - Logs all changes at INFO level

    Example:
        class MyPortfolio(Portfolio):
            def process_tick_market_signals_logic(
                    self, signals, tick) -> TickResults:
                orders = []
                for signal in signals:
                    if signal.type == SignalType.BUY:
                        pd = find_pricedata_in_list(signal.symbol, tick)
                        qty = int(self.cash / pd.close)
                        if qty > 0:
                            order = Order.create_market_order(
                                signal.symbol, OrderAction.BUY, qty, 0.0, tick)
                            orders.append(order)
                return TickResults(orders=orders)

        om = BacktestingOM()
        pf = MyPortfolio(cfg={"symbol": "AAPL"}, order_manager=om,
                         starting_cash=100000, keep_history=True)

        # With broker sync (for live trading):
        om = AlpacaOrderManager(alpaca_cfg)
        pf = MyPortfolio(
            cfg={"symbol": "AAPL", "cash": 100000,
                 "sync_with_broker": True, "sync_interval": 30},
            order_manager=om)
    """

    def __init__(
            self, cfg: dict = None,
            order_manager: OrderManager = None,
            starting_cash: float = 0.0,
            starting_positions: dict[str, Position] = None,
            keep_history: bool = False):
        """
        Initialize the portfolio with cash, positions, and configuration.

        Args:
            cfg: Portfolio configuration dict. Recognized keys:
                - cash (float): Initial cash balance. Overrides starting_cash.
                - keep_history (bool): Record tick/value/cash/signal history.
                    Overrides the keep_history parameter.
                - sync_with_broker (bool): Enable periodic broker state
                    synchronization. When True, cash and positions are
                    overwritten from the broker every sync_interval ticks.
                    Default: False. Has no effect when the OrderManager's
                    get_broker_account_state() returns None (e.g. backtesting).
                - sync_interval (int): Number of ticks between broker syncs.
                    Default: 60. Only used when sync_with_broker is True.
                All other keys are available to the subclass via self.cfg.
            order_manager: The OrderManager instance for submitting and
                tracking orders. Stored as self.om.
            starting_cash: Initial cash balance. Overridden by cfg["cash"]
                if present.
            starting_positions: Initial holdings. Dict mapping symbol string
                to Position objects. Defaults to empty dict.
            keep_history: Whether to record tick, value, cash, and signal
                history each tick. Overridden by cfg["keep_history"] if
                present. Required for AnalysisEngine to work.
        """
        self.cfg = cfg if cfg is not None else {}
        self.om = order_manager
        self.cash: float = cfg["cash"] if "cash" in cfg else starting_cash
        self.positions: dict[str, Position] = {} if starting_positions is None else starting_positions
        self.total_value: float = 0.0
        self.keep_history = cfg['keep_history'] if 'keep_history' in cfg else keep_history
        self.previous_price = {}
        # Broker sync settings
        self.sync_with_broker: bool = self.cfg.get("sync_with_broker", False)
        self._sync_interval: int = self.cfg.get("sync_interval", 60)
        self._order_sync_limit: int = self.cfg.get("order_sync_limit", 0)
        self._tick_count: int = 0
        # History data structures
        self.tick_history: dict[datetime, list[PriceData]] = {}
        self.value_history: dict[datetime, float] = {}
        self.cash_history: dict[datetime, float] = {}
        self.signals_history: dict[datetime, list[MarketSignal]] = {}

        logger.info(f"Portfolio initialized - Cash: ${self.cash:,.2f}, Positions: {len(starting_positions or {})}, History: {keep_history}")

    @final
    def _update_pf_value(self, current_tick: list[PriceData]) -> float:
        """
        Recalculate total portfolio value from cash and current positions.

        Uses closing prices from current_tick for each held symbol. If a
        symbol is missing from the tick, falls back to the last known price
        stored in self.previous_price. Raises ValueError if no price has
        ever been seen for a held position.

        Args:
            current_tick: List of PriceData for the current tick.

        Returns:
            The updated total portfolio value (cash + positions).

        Raises:
            ValueError: If a position has no price data and no previous
                price is available.
        """
        logger.debug(f"Updating portfolio value - Cash: ${self.cash:,.2f}, Positions: {len(self.positions)}")
        retval = self.cash
        for p in self.positions.keys():
            sp = find_pricedata_in_list(p, current_tick)
            if sp is None and p not in self.previous_price:
                # TODO: what to do if a position doesn't have price data?
                logger.error(f"No price data found for {p}")
                raise ValueError(f"Price data for position {p} not found")
            if sp is None:
                price = self.previous_price[p]
            else:
                price = sp.close
            # Store price for next time in case tick doesn't contain it
            self.previous_price[p] = price
            retval += self.positions[p].quantity * price
        self.total_value = retval
        logger.debug(f"Portfolio value updated - Total: ${retval:,.2f}")
        return retval

    @final
    def _update_pf_buy(self, order: Order):
        """
        Apply a filled BUY order to the portfolio.

        Deducts order cost (cash + tx_cost) from self.cash and adds
        the purchased quantity to self.positions. Creates a new Position
        if the symbol is not already held.

        Args:
            order: A filled Order with action=BUY. Must have price,
                quantity, cash, and tx_cost populated.
        """
        if self.cash - order.cash - order.tx_cost < 0:
            logger.error(f"Insufficient funds for BUY - Cash: ${self.cash:,.2f}, Required: ${order.cash + order.tx_cost:,.2f}")

        self.cash = self.cash - order.cash - order.tx_cost
        if order.symbol in self.positions:
            self.positions[order.symbol].quantity += order.quantity
        else:
            self.positions[order.symbol] = Position(order.symbol, order.quantity)
        order.processed_by_portfolio = True

        logger.info(f"BUY executed - {order.symbol}: {order.quantity} @ ${order.price:.2f} (Cost: ${order.cash + order.tx_cost:,.2f})")
        logger.debug(f"Cash after BUY: ${self.cash:,.2f}, Position: {self.positions[order.symbol].quantity}")

    @final
    def _update_pf_sell(self, order: Order):
        """
        Apply a filled SELL order to the portfolio.

        Adds order proceeds (cash - tx_cost) to self.cash and reduces
        the position quantity. Raises if the symbol is not held or if
        the sell quantity exceeds the current position.

        Args:
            order: A filled Order with action=SELL. Must have price,
                quantity, cash, and tx_cost populated.

        Raises:
            Exception: If no position exists for the symbol or if the
                sell quantity exceeds the held quantity.
        """
        if order.symbol not in self.positions:
            logger.error(f"Cannot SELL {order.symbol} - No position exists, cancelling order")
            raise Exception(f"Cannot SELL {order.symbol} - No position exists")
        if self.positions[order.symbol].quantity < order.quantity:
            logger.error(f"SELL {order.symbol} - Insufficient quantity (Have: {self.positions[order.symbol].quantity}, Need: {order.quantity})")
            raise Exception("SELL {order.symbol} - Insufficient quantity")

        self.cash = self.cash + order.cash - order.tx_cost
        self.positions[order.symbol].quantity -= order.quantity
        order.processed_by_portfolio = True

        logger.info(f"SELL executed - {order.symbol}: {order.quantity} @ ${order.price:.2f} (Proceeds: ${order.cash - order.tx_cost:,.2f})")
        logger.debug(f"Cash after SELL: ${self.cash:,.2f}, Remaining position: {self.positions[order.symbol].quantity}")

    @final
    def _update_pf_from_order(self, order_id: str) -> Order:
        """
        Apply a single order's status change to the portfolio.

        Looks up the order from the OrderManager by ID, inspects its type
        and status, and applies the appropriate cash/position update:

        - MARKET + FILLED: applies buy or sell
        - MARKET + CANCELED: marks as processed, no changes
        - BRACKET + PENDING_SALE: applies initial buy, keeps order pending
        - BRACKET + FILLED: applies exit sell from SOLD_ORDER child

        Skips orders already marked as processed_by_portfolio.

        Args:
            order_id: The order ID to look up and process.

        Returns:
            The processed Order object.

        Raises:
            NotImplementedError: If the order type/status combination is
                not handled.
        """
        # go through all the types of Orders and OrderStatus to see what to do
        order = self.om.all_orders[order_id]
        status = order.status

        logger.debug(f"Processing order {order_id[:8]}... - Type: {order.type}, Status: {status}, Action: {order.action if hasattr(order, 'action') else 'N/A'}")

        if order.processed_by_portfolio:
            logger.debug(f"Order {order_id[:8]}... already processed, skipping")
            return order

        if order.type == OrderType.MARKET:
            if status == OrderStatus.FILLED:
                if order.action == OrderAction.BUY:
                    self._update_pf_buy(order)
                    logger.info(f"MARKET BUY order {order.symbol}: {order.quantity} @ ${order.price:.2f}")
                    return order
                elif order.action == OrderAction.SELL:
                    logger.info(f"MARKET SELL order {order.symbol}: {order.quantity} @ ${order.price:.2f}")
                    self._update_pf_sell(order)
                    return order
                else:
                    raise NotImplementedError(f"MARKET order {order.action} action not implemented")
            elif status == OrderStatus.CANCELED:
                logger.info(f"MARKET CANCELED order {order.symbol}")
                # Do nothing
                order.processed_by_portfolio = True
                return order
            else:
                raise NotImplementedError(f"MARKET order with unimplemented status {status}: {order}")

        elif order.type == OrderType.BRACKET:
            if status == OrderStatus.PENDING_SALE:
                # Initial buy is done, adjust the portfolio
                self._update_pf_buy(order)
                # Set the processed_by_portfolio flag back to False since it's still pending
                order.processed_by_portfolio = False
                logger.info(
                    f"BRACKET order {order.symbol} - Initial BUY filled, awaiting STOP/PROFIT trigger "
                    f"at ${order.get_child_order("STOP").price:.2f}///${order.get_child_order("PROFIT").price:.2f}"
                )
                return order
            elif status == OrderStatus.FILLED:
                # Sale happened, update from the SOLD_ORDER
                so = order.SOLD_ORDER
                self._update_pf_sell(so)
                exit_type = "STOP-LOSS" if so.type == OrderType.STOP_LOSS else "PROFIT-TAKER" if so.type == OrderType.PROFIT_TAKER else "MANUAL"
                logger.info(
                    f"BRACKET order {order.symbol} - Exited via {exit_type} @ ${so.price:.2f}///"
                    f"${(so.price - order.price) * so.quantity:.2f} net"
                )
                return order
            else:
                raise NotImplementedError(f"BRACKET order with unimplemented status {status}: {order}")

        else:
            raise NotImplementedError(f"Unimplemented order type {order.type}: {order}")

    @final
    def _update_pf_from_changed_orders(self, changed_order_ids: list[str]) -> list[Order]:
        """
        Apply multiple order status changes to the portfolio.

        Iterates through a list of order IDs that have changed status
        since the last tick and calls _update_pf_from_order() for each.

        Args:
            changed_order_ids: List of order ID strings to process.

        Returns:
            List of the processed Order objects.
        """
        ret_val = []
        if changed_order_ids:
            logger.debug(f"Updating portfolio from {len(changed_order_ids)} changed orders")
        for order_id in changed_order_ids:
            ro = self._update_pf_from_order(order_id)
            ret_val.append(ro)
        return ret_val

    def _sync_from_broker(self):
        """Synchronize local cash and positions from the broker.

        Calls the OrderManager's get_broker_account_state(). If the backend
        does not support it (returns None), this is a no-op. Otherwise,
        overwrites local cash and positions with broker values.
        """
        if self.om is None:
            return
        state = self.om.get_broker_account_state()
        if state is None:
            return

        old_cash = self.cash
        new_cash = state["cash"]
        self.cash = new_cash

        broker_positions = state.get("positions", {})
        old_symbols = set(self.positions.keys())
        new_symbols = set(broker_positions.keys())

        # Update or add positions
        for symbol, qty in broker_positions.items():
            if symbol in self.positions:
                old_qty = self.positions[symbol].quantity
                if old_qty != qty:
                    logger.info(f"Broker sync: {symbol} position {old_qty} -> {qty}")
                self.positions[symbol].quantity = qty
            else:
                self.positions[symbol] = Position(symbol, qty)
                logger.info(f"Broker sync: new position {symbol} qty={qty}")

        # Remove positions no longer held at broker
        for symbol in old_symbols - new_symbols:
            logger.info(f"Broker sync: removing position {symbol} (no longer held at broker)")
            del self.positions[symbol]

        logger.info(f"Broker sync complete - Cash: ${old_cash:,.2f} -> ${new_cash:,.2f}, Positions: {len(self.positions)}")

    @final
    def process_market_signals_for_tick(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        """
        Main tick processing entry point. Called by the engine each tick.

        Orchestrates the full tick lifecycle:
            1. Polls OrderManager for pending order status changes
            2. Applies fills/cancels to cash and positions
            3. Periodically syncs cash/positions from broker (if enabled)
            4. Calls subclass strategy (process_tick_market_signals_logic)
            5. Filters out zero-quantity orders
            6. Submits new orders to OrderManager
            7. Applies any immediately-filled orders
            8. Recalculates total portfolio value
            9. Records history (if keep_history is True)

        Step 3 runs every sync_interval ticks when sync_with_broker is True.
        It calls _sync_from_broker() which queries the OrderManager's broker
        backend and overwrites local cash and positions. For backtesting
        backends this is a no-op.

        Do NOT override this method. Implement your strategy in
        process_tick_market_signals_logic() instead.

        Args:
            signals: List of MarketSignal objects from the algorithm for
                this tick. May be empty if the algorithm produced no signals.
            tick: List of PriceData objects for the current tick (one per
                symbol in the data stream).

        Returns:
            TickResults containing the list of new orders submitted this tick.
        """
        if len(signals) > 0:
            logger.debug(f"Tick {tick[0].timestamp}: Processing {len(signals)} signals - Symbols: {[s.symbol for s in signals]}")

        # Before processing new signals, update all pending orders, portfolio value and positions
        # get the list of order IDs that changed status since last tick
        changed_orders = self.om.update_pending_orders(tick, self.positions, self.cash)

        if len(changed_orders) > 0:
            logger.debug(f"Pending orders updated: {len(changed_orders)} orders changed status")

        # Update the portfolio to reflect orders that changed status
        processed_orders = self._update_pf_from_changed_orders(changed_orders)

        # Periodically sync cash/positions from broker (no-op for backtesting)
        self._tick_count += 1
        if self.sync_with_broker and self._tick_count % self._sync_interval == 0:
            self._sync_from_broker()
            if self._order_sync_limit > 0:
                self.om.sync_orders_from_broker(limit=self._order_sync_limit)

        # Call the market signals processing logic and get list of new orders
        all_orders_tr = self.process_tick_market_signals_logic(
            signals, tick
        )
        all_orders = all_orders_tr.orders

        # Remove 0 quantity orders
        orders = [o for o in all_orders if o.quantity > 0]

        if len(all_orders) - len(orders) > 0:
            logger.debug(f"Filtered out {len(all_orders) - len(orders)} zero-quantity orders")

        # Submit orders to Order Manager
        submitted_orders = self.om.submit_orders(orders, tick, self.positions, self.cash)

        if len(submitted_orders) > 0:
            logger.debug(f"Submitted {len(submitted_orders)} new orders to OrderManager")

        # Process any new orders that were filled immediately
        filled_orders = [o.order_id for o in submitted_orders if o.status in {OrderStatus.FILLED, OrderStatus.PENDING_SALE}]
        if filled_orders:
            logger.debug(f"{len(filled_orders)} orders filled immediately on submission")
        self._update_pf_from_changed_orders(filled_orders)

        # Update portfolio value based on current tick, positions and cash
        self._update_pf_value(tick)

        # Log when an order is happening
        if len(orders) > 0:
            logger.info(
                f"Tick {tick[0].timestamp} - Value: ${self.total_value:,.2f}, "
                f"Cash: ${self.cash:,.2f}, Positions: {len(self.positions)}, "
                f"Orders: {len(orders)}")

        # Store the history
        if self.keep_history:
            self.tick_history[tick[0].timestamp] = tick
            self.cash_history[tick[0].timestamp] = self.cash
            self.value_history[tick[0].timestamp] = self.total_value
            if len(signals) > 0:
                self.signals_history[tick[0].timestamp] = signals

        # Return the new orders that were created
        ret_val = TickResults(orders=orders)
        return ret_val

    @abstractmethod
    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        """
        Portfolio decision logic. Override this in your subclass.

        Called once per tick after pending orders have been updated and
        their fills applied. Use self.cash and self.positions to make
        sizing decisions. Return orders to submit; zero-quantity orders
        are filtered out automatically.

        Args:
            signals: List of MarketSignal objects from the algorithm.
                May be empty if no signals were generated this tick.
            tick: List of PriceData objects for the current tick (one
                per symbol in the data stream).

        Returns:
            TickResults containing a list of Order objects to submit.
            Return TickResults(orders=[]) if no orders should be placed.
        """
        raise NotImplementedError("process_market_signals_logic needs to be overridden")
