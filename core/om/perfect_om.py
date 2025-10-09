"""
Used for backtesting. Fulfills all orders instantly
"""
from core.order_manager import OrderManager
from core.classes import Order, OrderStatus, OrderType, OrderAction
from utils.utils import find_pricedata_in_list

class PerfectOm(OrderManager):

    def buy(self, symbol: str, quantity: int, tick: list[PriceData]) -> Order:
        pd = find_pricedata_in_list(symbol, tick)
        retval = Order(
            OrderAction.BUY,
            OrderType.MARKET,
            symbol,
            pd.close,
            quantity,
            pd.close * quantity,
            0,
            OrderStatus.FILLED
        )

    def sell(self, symbol: str, quantity: int, tick: list[PriceData]) -> Order:

        pd = find_pricedata_in_list(symbol, tick)
        retval = Order(
            OrderAction.SELL,
            OrderType.MARKET,
            symbol,
            pd.close,
            quantity,
            pd.close * quantity,
            0,
            OrderStatus.FILLED
        )

    def update_all_orders(self) -> list[Order]:
        retval = []

    def update_order_statuses(self, orders: list[Order]):
        retval = []