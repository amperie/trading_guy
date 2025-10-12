"""
Used for backtesting. Fulfills all orders instantly
"""
from datetime import datetime

from core.order_manager import OrderManager
from core.classes import Order, OrderStatus, OrderType, OrderAction, PriceData
from utils.utils import find_pricedata_in_list

class PerfectOm(OrderManager):

    def buy(self, symbol: str, quantity: int, tick: list[PriceData]) -> Order:
        pd = find_pricedata_in_list(symbol, tick)
        retval = Order(
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol=symbol,
            price=pd.close,
            quantity=quantity,
            cash=pd.close * quantity,
            tx_cost=0,
            status=OrderStatus.FILLED,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
        )
        return retval

    def sell(self, symbol: str, quantity: int, tick: list[PriceData]) -> Order:

        pd = find_pricedata_in_list(symbol, tick)
        retval = Order(
            action=OrderAction.SELL,
            type=OrderType.MARKET,
            symbol=symbol,
            price=pd.close,
            quantity=quantity,
            cash=pd.close * quantity,
            tx_cost=0,
            status=OrderStatus.FILLED,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
        )
        return retval

    def update_all_orders(self) -> list[Order]:
        retval = []

    def update_order_statuses(self, orders: list[Order]):
        retval = []

    def get_order_status(self, order: Order) -> Order:
        order.status = OrderStatus.FILLED
        return order