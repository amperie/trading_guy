"""
Used for backtesting. Fulfills all orders instantly
"""
from datetime import datetime

from core.order_manager import OrderManager
from core.classes import Order, OrderStatus, OrderType, OrderAction, PriceData
from utils.utils import find_pricedata_in_list

class BacktestingOM(OrderManager):

    def buy(
            self, symbol: str, quantity: int,
            tick: list[PriceData], tx_cost: float = 0.0) -> Order:
        pd = find_pricedata_in_list(symbol, tick)
        retval = Order(
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol=symbol,
            price=pd.close,
            quantity=quantity,
            cash=pd.close * quantity,
            tx_cost=tx_cost,
            status=OrderStatus.FILLED,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
        )
        return retval

    def sell(
            self, symbol: str, quantity: int,
            tick: list[PriceData], tx_cost: float = 0.0) -> Order:

        pd = find_pricedata_in_list(symbol, tick)
        retval = Order(
            action=OrderAction.SELL,
            type=OrderType.MARKET,
            symbol=symbol,
            price=pd.close,
            quantity=quantity,
            cash=pd.close * quantity,
            tx_cost=tx_cost,
            status=OrderStatus.FILLED,
            placed_datetime=pd.timestamp,
            executed_datetime=pd.timestamp,
        )
        return retval

    def bracket_order(
            self, symbol: str, quantity: int, profit_price_pct: float, stop_price_pct: float,
            tick: list[PriceData], tx_cost: float = 0.0) -> Order:

        pd = find_pricedata_in_list(symbol, tick)
        price = pd.close
        profit_price = price * (1.0 + profit_price_pct)
        stop_price = price * (1.0 - stop_price_pct)
        retval = Order.create_bracket_order(
            symbol, price, profit_price, stop_price, quantity, tx_cost
        )
        # Process the buy in this order before returning
        return self.get_order_status(retval, tick)

    def get_order_status(self, order: Order, tick: list[PriceData]) -> Order:
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
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
                so = order._child_orders_dict["STOP"]
                po = order._child_orders_dict["PROFIT_TAKER"]
                manual_sale = order._child_orders_dict["MANUAL_SALE"]
                manual_order = order._child_orders_dict["MANUAL_ORDER"]
                price = pd.close

                if manual_sale and manual_order is None:
                    # If there is a manual order to sell at market
                    # create and execute that order
                    mo = Order(
                        action=OrderAction.SELL,
                        type=OrderType.MARKET,
                        symbol=order.symbol,
                        price=pd.close,
                        quantity=order.quantity,
                        cash=0,
                        tx_cost=order.tx_cost,
                        status=OrderStatus.PENDING,
                        placed_datetime=pd.timestamp,
                        executed_datetime=datetime.now(),
                        parent_id=order.order_id,
                    )
                    # Add this manual order
                    order._child_orders_dict["MANUAL_ORDER"] = mo

                if manual_sale and manual_order is not None:
                    # Process any manual orders that are registered
                    mo = order._child_orders_dict["MANUAL_ORDER"]
                    mo = self.get_order_status(mo, tick)
                    if mo.status == OrderStatus.FILLED:
                        # If it got filled, process all the other orders
                        order.status = OrderStatus.FILLED
                        so.status = OrderStatus.CANCELED
                        po.status = OrderStatus.CANCELED
                        order.executed_datetime = pd.timestamp
                        order.processed_by_portfolio = False
                        order._child_orders_dict['SALE_ORDER'] = mo
                        return order

                if price <= so.price:
                    # If current price is lower than stop order price, need to sell
                    order.status = OrderStatus.FILLED
                    so.status = OrderStatus.FILLED
                    po.status = OrderStatus.CANCELED
                    order.executed_datetime = pd.timestamp
                    order.processed_by_portfolio = False
                    so.price = price
                    so.cash = price * so.quantity
                    so.executed_datetime = pd.timestamp
                    so.processed_by_portfolio = False
                    order._child_orders_dict["SALE_ORDER"] = so
                    return order
                if price >= po.price:
                    # If current price is higher than profit taking order, trigger the order
                    order.status = OrderStatus.FILLED
                    so.status = OrderStatus.CANCELED
                    po.status = OrderStatus.FILLED
                    order.executed_datetime = pd.timestamp
                    order.processed_by_portfolio = False
                    po.price = price
                    po.cash = price * po.quantity
                    po.executed_datetime = pd.timestamp
                    po.processed_by_portfolio = False
                    order._child_orders_dict["SALE_ORDER"] = po
                    return order

            # Sale didn't happen, return the original bracket order
            return order

        raise Exception(f"Order type {order.type} not implemented")