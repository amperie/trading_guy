"""
Handles execution of orders in backtesting or real time modes.
All this class does is execute the orders specified.
The portfolio class has all the logic built into it to track orders,
make sure they are able to execute, etc...
Interfaces:
get_orders_status: get status of orders that may take cycles to execute
execute_order(s): takes orders as input and executes them (backtesting or real time)
"""
from core.classes import Order, PriceData
from abc import ABC, abstractmethod


class OrderManager(ABC):

    def __init__(self, cfg: dict=None):
        self.cfg = cfg or {}

    @abstractmethod
    def buy(self, ticker: str, shares: int, tick: list[PriceData]) -> Order:
        pass

    @abstractmethod
    def sell(self, ticker:str, shares: int, tick: list[PriceData]) -> Order:
        pass

    @abstractmethod
    def update_order_statuses(self, orders: list[Order]):
        """
        Updates the status of orders in the list from the backend (alpaca, etc...)
        Args:
            orders: Orders to update

        Returns:
            list of orders that were updated
        """
        pass

    @abstractmethod
    def update_all_orders(self) -> list[Order]:
        """
        Updates all orders in orders list
        Returns:
            returns the orders that were updated from the backend (alpaca, etc...)
        """
        pass

    @abstractmethod
    def get_order_status(self, order: Order) -> Order:
        """
        Looks up the order status for the order and returns an updated version of it
        Args:
            order: order to get status for

        Returns:
            Order: order that was updated
        """
        pass

