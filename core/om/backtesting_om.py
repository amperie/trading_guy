"""
Used for backtesting. Fulfills all orders instantly
"""
from datetime import datetime

from core.order_manager import OrderManager
from core.classes import Order, OrderStatus, OrderType, OrderAction, PriceData
from utils.utils import find_pricedata_in_list

class BacktestingOM(OrderManager):

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
            placed_datetime=pd.timestamp,
            executed_datetime=pd.timestamp,
        )
        return retval

    def get_order_status(self, order: Order, tick: list[PriceData]) -> Order:
        if order.status == OrderStatus.FILLED or OrderStatus.CANCELED:
            return order # Nothing to do

        # Get price data
        pd = find_pricedata_in_list(order.symbol, tick)
        # Handle different order types
        if order.type == OrderType.MARKET:
            if order.status == OrderStatus.PENDING:
                # This should never run since market orders always get filled instantly
                order.status = OrderStatus.FILLED
                order.price = pd.close
                order.executed_datetime = pd.timestamp
                order.processed_by_portfolio = False
                return order

        if order.type == OrderType.BRACKET:
            if order.status == OrderStatus.PENDING:
                # Bracket order is pending the initial buy
                order.status = OrderStatus.PENDING_SALE
                order.price = pd.close
                order.cash = pd.close * order.quantity
                order.executed_datetime = pd.timestamp
                order.processed_by_portfolio = False
                return order
            if order.status == OrderStatus.PENDING_SALE:
                # Check if any child orders need to trigger
                so = order.child_orders_dict["STOP"]
                po = order.child_orders_dict["PROFIT_TAKER"]
                price = pd.close
                if price <= so.price:
                    # If current price is lower than stop order price, need to sell
                    order.status = OrderStatus.FILLED
                    so.status = OrderStatus.FILLED
                    po.status = OrderStatus.CANCELED
                    order.price = price
                    order.executed_datetime = pd.timestamp
                    order.processed_by_portfolio = False
                    so.price = price
                    so.cash = price * so.quantity
                    so.executed_datetime = pd.timestamp
                    so.processed_by_portfolio = False
                    return order
                if price >= po.price:
                    # If current price is higher than profit taking order, trigger the order
                    order.status = OrderStatus.FILLED
                    so.status = OrderStatus.CANCELED
                    po.status = OrderStatus.FILLED
                    order.price = price
                    order.executed_datetime = pd.timestamp
                    order.processed_by_portfolio = False
                    po.price = price
                    po.cash = price * po.quantity
                    po.executed_datetime = pd.timestamp
                    po.processed_by_portfolio = False
                    return order

        raise Exception(f"Order type {order.type} not implemented")