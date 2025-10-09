"""
Portfolio class
- Stores configuration about the portfolio strategy and priorities for each security
- Tracks what the portfolio is comprised of
- Takes signals as input and adjusts the portfolio
Interfaces:
- Initialize
- create_orders: takes signal inputs and creates a list of orders
    - Keeps a history of orders and signals
    - Has an OrderManager member to handle the orders
"""
from core.classes import MarketSignal, Order, PriceData, Position, OrderStatus, OrderAction
from core.order_manager import OrderManager
from abc import ABC, abstractmethod, final
from datetime import datetime

from utils.utils import find_pricedata_in_list


class Portfolio(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.order_manager = None
        self.signals: list[MarketSignal] = []
        self.orders:list[Order] = []
        self.cash: float = 0.0
        self.positions: dict[str, Position] = {}
        self.total_value: float = 0.0
        self.keep_history = cfg['keep_history']
        self.tick_history: dict[datetime, list[PriceData]] = {}
        self.value_history: dict[datetime, float] = {}
        self.cash_history: dict[datetime, float] = {}
        self.orders_by_id: dict[str, Order] = {}
        self.pending_orders_by_id: dict[str, Order] = {}

    @final
    def set_order_manager(self, order_manager: OrderManager):
        self.order_manager = order_manager

    @final
    def _process_filled_orders(self, orders: list[Order]):
        for order in orders:
            # Only process filled orders that haven't been processed already
            if order.status != OrderStatus.FILLED or order.processed_by_portfolio:
                continue
            # TODO: logic to catch if an order can't be executed because of low cash
            if order.action == OrderAction.BUY:
                cash_change = order.cash + order.tx_cost
                # update positions
                self.positions[order.symbol].quantity += order.quantity
                self.cash -= cash_change
                order.processed_by_portfolio = True
                self.orders_by_id[order.id] = order
            # TODO: logic to catch if an order needs to be adjusted to a smaller quantity
            if order.action == OrderAction.SELL:
                cash_change = order.cash - order.tx_cost
                self.positions[order.symbol].quantity -= order.quantity
                self.cash += cash_change
                order.processed_by_portfolio = True
                self.orders_by_id[order.id] = order


    @final
    def _store_orders(self, orders: list[Order]):
        self.orders.extend(orders)
        for order in orders:
            self.orders_by_id[order.id] = order
            if order.status == OrderStatus.FILLED:
                # Remove it from pending orders dict if it's there
                if order.id in self.pending_orders_by_id:
                    self._close_pending_order(order)
            else:
                # If it's not filled store it in pending orders dict
                self.pending_orders_by_id[order.id] = order


    @final
    def _close_pending_order(self, order: Order):
        if order.id in self.pending_orders_by_id:
            del self.pending_orders_by_id[order.id]
            self.orders_by_id[order.id] = order

    @final
    def _process_pending_order(self, order: Order, current_tick: list[PriceData]) -> Order:
        # TODO: Logic to process orders based on their type
        pass

    @final
    def _process_pending_orders(self, current_tick: list[PriceData]):
        """See if any pending orders require processing"""
        for order in self.pending_orders_by_id.values():
            self._process_pending_order(order, current_tick)

    @final
    def _update_pf_value(self, current_tick: list[PriceData]) -> float:
        retval = self.cash
        for p in self.positions.keys():
            sp = find_pricedata_in_list(p, current_tick)
            if sp is None:
                # TODO: what to do if a position doesn't have price data?
                pass
            retval += self.positions[p].quantity * sp.close
        self.total_value = retval
        return retval


    @final
    def process_market_signals(
            self, signals: list[MarketSignal],
            tick: list[PriceData])-> list[Order]:
        """
        Rebalance portfolio based on signals
        :param signals:
        :param tick:
        :return:
        """
        # Before processing new signals, update all pending orders
        self._process_pending_orders(tick)

        # Process new signals, process and store the orders
        orders = self.process_market_signals_logic(
            signals, tick
        )
        # Store all the orders that came back
        self._store_orders(orders)
        # Process the filled orders immediately
        self._process_filled_orders(orders)

        # Update portfolio value
        self._update_pf_value(tick)

        # Store the history
        if self.keep_history:
            self.tick_history[tick[0].timestamp] = tick
            self.cash_history[tick[0].timestamp] = self.cash
            self.value_history[tick[0].timestamp] = self.total_value

        # Return the new orders that were created
        return orders

    @abstractmethod
    def process_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData])-> list[Order]:
        raise NotImplementedError("process_market_signals_logic needs to be overriden")