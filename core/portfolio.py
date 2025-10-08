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
from core.classes import MarketSignal, Order
from core.order_manager import OrderManager


class Portfolio:
    def __init__(self, cfg: dict=None):
        pass

    def set_order_manager(self, order_manager: OrderManager):
        self.order_manager = order_manager

    def process_market_signals(
            self, signals: list[MarketSignal])-> list[Order]:
        """
        Rebalance portfolio based on signals
        :param signals:
        :return:
        """
        pass