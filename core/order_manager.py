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
from typing import final


class OrderManager(ABC):

    def __init__(self, cfg: dict=None):
        self.cfg = cfg or {}

    @abstractmethod
    def buy(self, symbol: str, shares: int, tick: list[PriceData]) -> Order:
        raise(NotImplementedError())

    @abstractmethod
    def sell(self, symbol:str, shares: int, tick: list[PriceData]) -> Order:
        raise(NotImplementedError())

    @abstractmethod
    def bracket_order(
            self, symbol:str, shares: int, profit_price: float, stop_price: float,
            tick: list[PriceData]) -> Order:
        raise(NotImplementedError())

    @final
    def update_all_orders(self, orders: list[Order], tick: list[PriceData]) -> list[Order]:
        """
        Updates all orders in orders list
        Args:
            orders: orders to get status for
            tick: current tick
        Returns:
            returns the orders that were updated from the backend (alpaca, etc...)
        """
        ret_val = [self.get_order_status(o, tick) for o in orders]
        return ret_val

    @abstractmethod
    def get_order_status(self, order: Order, tick: list[PriceData]) -> Order:
        """
        Looks up the order status for the order and returns an updated version of it
        Handles the logic of turning a pending order into a filled order
        Also needs to know how to handle different order types
        Args:
            order: order to get status for
            tick: current tick

        Returns:
            Order: order that was updated
        """
        raise(NotImplementedError())

