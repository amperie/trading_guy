"""
Base class for order execution backends.

The OrderManager sits between the Portfolio (which decides what to trade)
and the execution backend (broker API, backtesting simulator, etc.). It
maintains three internal order dictionaries and provides a public API that
the Portfolio calls, delegating actual execution to four abstract backend
hooks that subclasses implement.

Architecture:
    Portfolio -> submit_order(order)
        -> _submit_order_to_backend(order)   [subclass hook]
        -> track in _all_orders + _pending_orders or _filled_orders

    Portfolio -> update_pending_orders(tick)
        -> _update_orders_statuses_from_backend(pending)  [subclass hook]
        -> move completed orders from _pending to _filled
        -> return list of changed order IDs

    Portfolio -> _sync_from_broker()
        -> get_broker_account_state()  [optional override]
        -> overwrite local cash/positions with broker values

Order lifecycle:
    PENDING -> FILLED         (market orders, immediate fill)
    PENDING -> PENDING_SALE   (bracket entry filled, awaiting exit)
    PENDING_SALE -> FILLED    (bracket exit triggered: stop-loss or profit-taker)
    PENDING -> CANCELED       (order canceled before fill)

Implementations:
    - BacktestingOM:       Instant fills, simulated bracket logic
    - AlpacaOrderManager:  Live execution via Alpaca Trading API
"""
from trading.core.classes import Order, PriceData, OrderStatus, Position, BracketOrder, OrderType
from utils.utils import trim_dictionary
from abc import ABC, abstractmethod
from typing import final, Union
from utils.logger import Logger

logger = Logger().get_logger(__name__)

class OrderManager(ABC):
    """
    Base class for order execution backends.

    Manages three internal order dictionaries and provides a public API
    for the Portfolio to submit, track, and cancel orders. Subclasses
    implement four backend hooks for actual order execution.

    Subclassing notes:
        - Implement the four abstract backend hooks:
            * _submit_order_to_backend(): Execute/send an order.
            * _update_order_status_from_backend(): Refresh one order.
            * _update_orders_statuses_from_backend(): Refresh all pending orders.
            * _cancel_order(): Cancel a pending order.
        - Optionally override get_broker_account_state() to enable
          broker state synchronization. The default returns None (no-op).
          Live broker backends (e.g. AlpacaOrderManager) should return
          a dict with "cash" (float) and "positions" (dict[str, int]).
        - Do NOT override public methods (submit_order, submit_orders,
          update_pending_orders, update_order_status, cancel_all_pending_orders).
          They are @final and manage internal bookkeeping before/after
          calling your backend hooks.
        - Update Order fields in-place (status, price, executed_datetime,
          cash, quantity, platform_id) and return the same order instance.
        - For bracket orders, set order.SOLD_ORDER to the child order that
          completed the exit (stop-loss or profit-taker) when the bracket
          is fully filled.

    Attributes:
        cfg (dict): Backend configuration dict passed on init.
        filled_orders_by_id (dict): Read-only property. Orders that have
            reached a terminal state (FILLED or CANCELED). Keyed by order_id.
        pending_orders_by_id (dict): Read-only property. Orders still
            awaiting fill or exit trigger. Keyed by order_id.
        all_orders (dict): Read-only property. Every order ever submitted.
            Keyed by order_id.

    Internal bookkeeping:
        When an order is submitted:
        - Added to _all_orders
        - If immediately FILLED/CANCELED -> added to _filled_orders_by_id
        - Otherwise -> added to _pending_orders_by_id

        When a pending order's status changes to FILLED/CANCELED:
        - Moved from _pending_orders_by_id to _filled_orders_by_id
        - For BRACKET orders that are FILLED, the SOLD_ORDER child is
          also registered as its own entry in _filled and _all dicts
          (needed by AnalysisEngine for trade extraction)

    Example:
        class MyBrokerOM(OrderManager):
            def _submit_order_to_backend(self, order, current_tick,
                                          positions, pf_cash):
                # Send to broker API
                response = broker.place_order(order.symbol, order.quantity)
                order.platform_id = response.id
                order.status = OrderStatus.PENDING
                return order

            def _update_order_status_from_backend(self, order, current_tick,
                                                   positions, pf_cash):
                response = broker.get_order(order.platform_id)
                if response.status == "filled":
                    order.status = OrderStatus.FILLED
                    order.price = response.fill_price
                return order

            def _update_orders_statuses_from_backend(self, orders,
                                                      current_tick,
                                                      positions, pf_cash):
                changed = []
                for order in orders.values():
                    prev = order.status
                    self._update_order_status_from_backend(order)
                    if order.status != prev:
                        changed.append(order.order_id)
                return changed

            def _cancel_order(self, order_id):
                order = self._all_orders[order_id]
                broker.cancel(order.platform_id)
                order.status = OrderStatus.CANCELED
                return order
    """

    def __init__(self, cfg: dict=None):
        """
        Initialize the order manager with empty order tracking dicts.

        Args:
            cfg: Backend configuration dict. Contents depend on the
                subclass (e.g. API keys for live brokers, or empty
                for backtesting). Stored as self.cfg.
        """
        self.cfg = cfg or {}
        self._filled_orders_by_id: dict[str, Union[BracketOrder, Order]] = {}
        self._pending_orders_by_id: dict[str, Union[BracketOrder, Order]] = {}
        self._all_orders: dict[str, Union[BracketOrder, Order]] = {}

    @property
    def filled_orders_by_id(self) -> dict[str, Union[BracketOrder, Order]]:
        """Orders that have reached a terminal state (FILLED or CANCELED)."""
        return self._filled_orders_by_id

    @property
    def pending_orders_by_id(self) -> dict[str, Union[BracketOrder, Order]]:
        """Orders still awaiting fill or exit trigger (PENDING or PENDING_SALE)."""
        return self._pending_orders_by_id

    @property
    def all_orders(self) -> dict[str, Union[BracketOrder, Order]]:
        """Every order ever submitted, regardless of current status."""
        return self._all_orders

    @final
    def submit_order(
            self, order: Order, current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        """
        Submit a single order for execution.

        Validates the order hasn't been submitted before, delegates to the
        backend via _submit_order_to_backend(), then files the order into
        the appropriate internal dict based on its resulting status.

        Args:
            order: The Order (or BracketOrder) to submit. Must have a
                unique order_id.
            current_tick: Current tick data, passed to the backend for
                price-dependent execution (e.g. backtesting fill logic).
            positions: Current portfolio positions, passed to the backend
                for validation (e.g. ensuring shares exist for sells).
            pf_cash: Current portfolio cash balance.

        Returns:
            The submitted Order with status updated by the backend.

        Raises:
            ValueError: If an order with the same order_id already exists.
        """
        if order.order_id in self._all_orders:
            raise ValueError(f"Order already exists {order}\n\nExisting Order: {self._all_orders[order.order_id]}")

        logger.debug(f"Submitting order: {order.order_id} {order.type.name} {order.action.name} {order.symbol} qty={order.quantity}")

        # Put the order into the backend first
        so = self._submit_order_to_backend(order, current_tick, positions, pf_cash)

        self._all_orders[so.order_id] = so
        if so.status in {OrderStatus.FILLED, OrderStatus.CANCELED}:
            self._filled_orders_by_id[so.order_id] = so
            logger.debug(f"Order immediately {so.status.name}: {so.order_id} price={so.price} qty={so.quantity}")
        else:
            self._pending_orders_by_id[so.order_id] = so
            logger.debug(f"Order pending: {so.order_id} status={so.status.name}")

        return so

    @final
    def submit_orders(
            self, orders: list[Order], current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> list[Order]:
        """
        Submit multiple orders for execution.

        Convenience wrapper that calls submit_order() for each order
        in sequence.

        Args:
            orders: List of Order objects to submit.
            current_tick: Current tick data, passed through to each submission.
            positions: Current portfolio positions.
            pf_cash: Current portfolio cash balance.

        Returns:
            List of submitted Order objects with statuses updated.
        """
        ret_val = []
        for order in orders:
            so = self.submit_order(order, current_tick, positions, pf_cash)
            ret_val.append(so)
        return ret_val


    @final
    def _update_order_lists(self, order: Union[BracketOrder, Order]):
        """
        Move a single order from pending to filled if its status is terminal.

        Called after a backend status refresh. If the order has transitioned
        to FILLED or CANCELED, it is moved from _pending_orders_by_id to
        _filled_orders_by_id. For filled BRACKET orders, the SOLD_ORDER
        child is also registered as its own entry in both _filled and _all
        dicts for downstream analysis.

        Args:
            order: The order to check and potentially move.
        """
        if (order.status in {OrderStatus.FILLED, OrderStatus.CANCELED}
                and order.order_id in self._pending_orders_by_id):
            logger.debug(f"Order {order.status.name}: {order.order_id} {order.symbol} price={order.price} qty={order.quantity}")
            self._filled_orders_by_id[order.order_id] = order
            del self._pending_orders_by_id[order.order_id]

            # If order is a FILLED BRACKET also break off the SOLD child order
            # and register it as its own order for later analysis reasons
            if order.type == OrderType.BRACKET and order.status == OrderStatus.FILLED:
                so = order.SOLD_ORDER
                if so is not None:
                    logger.debug(f"Bracket exit via {so.type.name}: {so.order_id} {so.symbol} price={so.price} qty={so.quantity}")
                    self._filled_orders_by_id[so.order_id] = so
                    self._all_orders[so.order_id] = so
                else:
                    logger.error(f"BRACKET FILLED with no SOLD_ORDER: {order.order_id}")

    @final
    def _update_all_order_lists(self):
        """
        Scan all pending orders and move any that are now terminal.

        Iterates through _pending_orders_by_id, moves FILLED/CANCELED
        orders to _filled_orders_by_id, and registers BRACKET exit
        child orders. Uses trim_dictionary() to remove moved orders
        from _pending_orders_by_id.

        Returns:
            The trimmed pending orders dict (same reference as
            self._pending_orders_by_id).
        """
        orders = []
        for order in self._pending_orders_by_id.values():
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED}:
                orders.append(order.order_id)
                self._filled_orders_by_id[order.order_id] = order

                # If order is a FILLED BRACKET also break off the SOLD child order
                # and register it as its own order for later analysis reasons
                if order.type == OrderType.BRACKET and order.status == OrderStatus.FILLED:
                    so = order.SOLD_ORDER
                    if so is not None:
                        self._filled_orders_by_id[so.order_id] = so
                        self._all_orders[so.order_id] = so
                    else:
                        logger.error(f"BRACKET FILLED with no SOLD_ORDER: {order.order_id}")

        return trim_dictionary(self._pending_orders_by_id, orders)

    @final
    def update_pending_orders(
            self, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> list[str]:
        """
        Poll the backend for status changes on all pending orders.

        Called by the Portfolio at the start of each tick to detect fills,
        cancellations, or bracket exit triggers. Delegates to the backend
        via _update_orders_statuses_from_backend(), then reconciles internal
        order lists.

        Args:
            current_tick: Current tick data, passed to the backend for
                price-dependent status checks (e.g. bracket stop/profit
                trigger evaluation in backtesting).
            positions: Current portfolio positions.
            pf_cash: Current portfolio cash balance.

        Returns:
            List of order_id strings for orders whose status changed.
        """
        logger.debug(f"Updating {len(self._pending_orders_by_id)} pending orders")
        updated_orders = self._update_orders_statuses_from_backend(
            self._pending_orders_by_id, current_tick, positions, pf_cash
        )
        if updated_orders:
            logger.debug(f"Orders changed status: {updated_orders}")
        self._update_all_order_lists()
        return updated_orders

    @final
    def update_order_status(
            self, order: Order, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        """
        Refresh the status of a single order from the backend.

        Delegates to _update_order_status_from_backend(), then reconciles
        internal order lists if the status changed to a terminal state.

        Args:
            order: The order to refresh.
            current_tick: Current tick data.
            positions: Current portfolio positions.
            pf_cash: Current portfolio cash balance.

        Returns:
            The updated Order object.
        """
        ret_val = self._update_order_status_from_backend(
            order, current_tick, positions, pf_cash
        )
        self._update_order_lists(ret_val)
        return ret_val

    @final
    def cancel_all_pending_orders(self, order_type: OrderType=None) -> list[str]:
        """
        Cancel all pending orders, optionally filtered by order type.

        Iterates through pending orders, calls _cancel_order() on each
        matching order, and reconciles internal lists.

        Args:
            order_type: If provided, only cancel orders of this type
                (e.g. OrderType.BRACKET). If None, cancel all pending
                orders regardless of type.

        Returns:
            List of order_id strings for orders that were canceled.
        """
        ret_val = []
        keys = list(self._pending_orders_by_id.keys())
        for order_id in keys:
            # If order_type is None it means to cancel all types
            # If it's set to a type, only cancel orders of that type
            order = self.all_orders[order_id]
            if order_type is None or order.type == order_type:
                self._cancel_order(order_id)
                self._update_all_order_lists()
                ret_val.append(order_id)
                logger.info(f"Canceled {order.type} order {order_id}")
        return ret_val

    def sync_orders_from_broker(self, limit: int = 100) -> dict | None:
        """Pull orders from broker into local dicts. Returns None if not supported."""
        return None

    def get_broker_account_state(self) -> dict | None:
        """Return broker cash/positions, or None if not supported.

        Subclasses backed by a real broker (e.g. Alpaca) should override
        this to query the broker for the current account state.

        Returns:
            A dict with at least "cash" (float) and "positions"
            (dict[str, int] mapping symbol to quantity), or None if
            the backend does not support account queries (e.g. backtesting).
        """
        return None

    @abstractmethod
    def _update_order_status_from_backend(
            self, order: Order, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        """
        Refresh a single order's status from the execution backend.

        Override this in your subclass. Query the backend for the current
        state of the order and update the Order object in-place.

        Responsibilities:
            - Query backend for the order state.
            - Update order.status to reflect the current state.
            - If filled: populate order.price, order.executed_datetime,
              order.cash, order.quantity, and order.platform_id.
            - For bracket orders: handle the PENDING -> PENDING_SALE ->
              FILLED lifecycle and set order.SOLD_ORDER on final fill.
            - Return the same order instance (modified in-place).

        Args:
            order: The order to refresh. Modify in-place.
            current_tick: Current tick data (used by backtesting backends
                for price-dependent logic like bracket triggers).
            positions: Current portfolio positions.
            pf_cash: Current portfolio cash balance.

        Returns:
            The updated Order object (same instance as input).
        """
        raise NotImplementedError

    @abstractmethod
    def _update_orders_statuses_from_backend(
            self, orders: dict[str, Order], current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> list[str]:
        """
        Refresh statuses for all pending orders from the execution backend.

        Override this in your subclass. Typically iterates through the
        orders dict and calls _update_order_status_from_backend() for each,
        tracking which orders changed status.

        Responsibilities:
            - Poll backend for all provided pending orders.
            - Update each order in-place.
            - Return a list of order_ids whose status changed.

        Args:
            orders: Dict of pending orders keyed by order_id. These are
                references to the same objects in _pending_orders_by_id.
            current_tick: Current tick data.
            positions: Current portfolio positions.
            pf_cash: Current portfolio cash balance.

        Returns:
            List of order_id strings for orders whose status changed.
        """
        raise NotImplementedError

    @abstractmethod
    def _submit_order_to_backend(
            self, order: Order, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        """
        Submit an order to the execution backend.

        Override this in your subclass. Send the order to the broker or
        simulate execution for backtesting.

        Responsibilities:
            - Submit the order to the backend.
            - Set order.status (PENDING, FILLED, PENDING_SALE, etc.).
            - Set order.platform_id if the backend assigns one.
            - If filled immediately: populate order.price, order.quantity,
              order.cash, and order.executed_datetime.
            - Return the same order instance (modified in-place).

        Args:
            order: The order to submit. Modify in-place.
            current_tick: Current tick data (used by backtesting backends
                for immediate fill price).
            positions: Current portfolio positions.
            pf_cash: Current portfolio cash balance.

        Returns:
            The submitted Order object (same instance as input).
        """
        raise NotImplementedError

    @abstractmethod
    def _cancel_order(self, order_id: str) -> Order:
        """
        Cancel a pending order in the execution backend.

        Override this in your subclass. Set the order's status to
        CANCELED after successfully canceling with the backend.

        Args:
            order_id: The ID of the order to cancel. The order can be
                looked up via self._all_orders[order_id].

        Returns:
            The canceled Order object with status set to CANCELED.
        """
        raise NotImplementedError
