"""
Handles execution of orders in backtesting or real time modes.
All this class does is execute the orders specified.
The portfolio class has all the logic built into it to track orders,
make sure they are able to execute, etc...
Interfaces:
get_orders_status: get status of orders that may take cycles to execute
execute_order(s): takes orders as input and executes them (backtesting or real time)
"""
from core.classes import Order, PriceData, OrderStatus, Position, BracketOrder
from utils.utils import trim_dictionary
from abc import ABC, abstractmethod
from typing import final, Union


class OrderManager(ABC):

    def __init__(self, cfg: dict=None):
        self.cfg = cfg or {}
        self._filled_orders_by_id: dict[str, Order] = {}
        self._pending_orders_by_id: dict[str, Order] = {}
        self._all_orders: dict[str, Order] = {}

    @property
    def filled_orders_by_id(self) -> dict[str, Union[BracketOrder, Order]]:
        return self._filled_orders_by_id

    @property
    def pending_orders_by_id(self) -> dict[str, Union[BracketOrder, Order]]:
        return self._pending_orders_by_id

    @property
    def all_orders(self) -> dict[str, Union[BracketOrder, Order]]:
        return self._all_orders

    @final
    def submit_order(
            self, order: Order, current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        if order.order_id in self._all_orders:
            raise ValueError(f"Order already exists {order}\n\nExisting Order: {self._all_orders[order.order_id]}")

        # Put the order into the backend first
        so = self._submit_order_to_backend(order, current_tick, positions, pf_cash)

        self._all_orders[so.order_id] = so
        if so.status in {OrderStatus.FILLED, OrderStatus.CANCELED}:
            self._filled_orders_by_id[so.order_id] = so
        else:
            self._pending_orders_by_id[so.order_id] = so

        return so

    @final
    def submit_orders(
            self, orders: list[Order], current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> list[Order]:
        ret_val = []
        for order in orders:
            so = self.submit_order(order, current_tick, positions, pf_cash)
            ret_val.append(so)
        return ret_val


    @final
    def _update_order_lists(self, order: Order):
        if (order.status in {OrderStatus.FILLED, OrderStatus.CANCELED}
                and order.order_id in self._pending_orders_by_id):
            self._filled_orders_by_id[order.order_id] = order
            del self._pending_orders_by_id[order.order_id]

    @final
    def _update_all_order_lists(self):
        orders = []
        for order in self._pending_orders_by_id.values():
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED}:
                orders.append(order.order_id)
                self._filled_orders_by_id[order.order_id] = order
        return trim_dictionary(self._pending_orders_by_id, orders)

    @final
    def update_pending_orders(
            self, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> list[str]:
        """
        Method to query the backend and get updated status for all pending orders
        Returns a list of order_ids that have changed status
        """
        updated_orders = self._update_orders_statuses_from_backend(
            self._pending_orders_by_id, current_tick, positions, pf_cash
        )
        self._update_all_order_lists()
        return updated_orders

    @final
    def update_order_status(
            self, order: Order, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        ret_val = self._update_order_status_from_backend(
            order, current_tick, positions, pf_cash
        )
        self._update_order_lists(ret_val)
        return ret_val

    @abstractmethod
    def _update_order_status_from_backend(
            self, order: Order, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        """
        Updates the status of one order from the backend and returns the updated order
        """
        raise NotImplementedError

    @abstractmethod
    def _update_orders_statuses_from_backend(
            self, orders: dict[str, Order], current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> list[str]:
        """
        Gets order statuses from backend, updates orders in the dictionary and returns a list
        of order_ids that changed status
        """
        raise NotImplementedError

    @abstractmethod
    def _submit_order_to_backend(
            self, order: Order, current_tick: list[PriceData]=None,
            positions: dict[str,Position]=None, pf_cash: float=0.0) -> Order:
        """
        Override this for specific backends
        """
        raise NotImplementedError