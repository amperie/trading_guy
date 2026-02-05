"""
Portfolio class
- Stores configuration about the portfolio strategy and priorities for each security
- Tracks what the portfolio comprises
- Takes signals as input and adjusts the portfolio
Interfaces:
- Initialize
- create_orders: takes signal inputs and creates a list of orders
    - Keeps a history of orders and signals
    - Has an OrderManager member to handle the orders
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

    Subclassing notes:
    - Implement process_tick_market_signals_logic to turn signals + tick into orders.
    - Do not override process_market_signals_for_tick; it manages order updates,
      order submission, and portfolio value tracking.
    - Use self.cash / self.positions for sizing, and self.om for order creation.
    - If keep_history is enabled, history is recorded automatically.
    """

    def __init__(
            self, cfg: dict = None,
            order_manager: OrderManager = None,
            starting_cash: float = 0.0,
            starting_positions: dict[str, Position] = None,
            keep_history: bool = False):

        self.cfg = cfg if cfg is not None else {}
        self.om = order_manager
        self.cash: float = cfg["cash"] if "cash" in cfg else starting_cash
        self.positions: dict[str, Position] = {} if starting_positions is None else starting_positions
        self.total_value: float = 0.0
        self.keep_history = cfg['keep_history'] if 'keep_history' in cfg else keep_history
        self.previous_price = {}
        # History data structures
        self.tick_history: dict[datetime, list[PriceData]] = {}
        self.value_history: dict[datetime, float] = {}
        self.cash_history: dict[datetime, float] = {}
        self.signals_history: dict[datetime, list[MarketSignal]] = {}

        logger.info(f"Portfolio initialized - Cash: ${self.cash:,.2f}, Positions: {len(starting_positions or {})}, History: {keep_history}")

    @final
    def _update_pf_value(self, current_tick: list[PriceData]) -> float:
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
        # go through all the types of Orders and OrderStatus to see what to do
        order = self.om.all_orders[order_id]
        status = order.status

        logger.debug(f"Processing order {order_id[:8]}... - Type: {order.type}, Status: {status}, Action: {order.action if hasattr(order, 'action') else 'N/A'}")

        if order.processed_by_portfolio:
            # Order has already been processed, do nothing
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
        Goes through a list of order_ids that have changed since the last tick and
        updates the potfolio correspondingly. Returns a list of the Orders that were updated
        """
        ret_val = []
        for order_id in changed_order_ids:
            ro = self._update_pf_from_order(order_id)
            ret_val.append(ro)
        return ret_val

    @final
    def process_market_signals_for_tick(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        """
        Rebalance portfolio based on signals
        :param signals:
        :param tick:
        :return:
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
        Portfolio decision logic.

        Args:
            signals: MarketSignal list for this tick.
            tick: PriceData list for this tick.

        Returns:
            TickResults with any orders to submit.
        """
        raise NotImplementedError("process_market_signals_logic needs to be overridden")
