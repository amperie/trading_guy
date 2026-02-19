from typing import Union
from trading.core.classes import Order, PriceData, OrderType, OrderStatus, Position, OrderAction, BracketOrder
from trading.core.om.order_manager import OrderManager
from utils.utils import find_pricedata_in_list
from utils.logger import Logger

logger = Logger().get_logger(__name__)

def _process_market_order(
        order: Order, pd: PriceData, pf_cash: float, quantity: int,
        positions: dict[str, Position]=None) -> Order:

    if pd is None:
        return order

    if order.action == OrderAction.SELL:
        # When selling make sure you don't sell more than you have
        if positions is not None and order.symbol not in positions:
            position_quantity = 0
        else:
            position_quantity = positions[order.symbol].quantity
        tx_quantity = min(order.quantity, quantity, position_quantity)
    else:
        tx_quantity = min(order.quantity, int(pf_cash / pd.close))

    order.quantity = tx_quantity
    if tx_quantity <= 0:
        order.status = OrderStatus.CANCELED
    else:
        order.status = OrderStatus.FILLED
    order.price = pd.close
    order.cash = pd.close * order.quantity
    order.placed_datetime = pd.timestamp
    order.executed_datetime = pd.timestamp

    return order


class BacktestingOrderManager(OrderManager):

    def _cancel_order(self, order_id: str) -> Order:
        if order_id in self._pending_orders_by_id:
            self._pending_orders_by_id[order_id].status = OrderStatus.CANCELED
        return self._all_orders[order_id]

    def _update_order_status_from_backend(
            self, order: Union[BracketOrder, Order], current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float = 0.0) -> Order:
        """
        Checks whether orders can fill and returns the order
        """
        if current_tick is None:
            raise ValueError("Backtesting OM requires current_tick to be set")
        if order.status == OrderStatus.FILLED:
            # Nothing to do
            return order

        pd = find_pricedata_in_list(order.symbol, current_tick)
        if pd is None:
            logger.error(f"No price data found for {order.symbol}")
            return order

        if order.type == OrderType.MARKET:
            return _process_market_order(order, pd, pf_cash, order.quantity, positions)

        elif order.type == OrderType.BRACKET:
            # Process all options for a bracket order
            if order.status == OrderStatus.PENDING:
                # This should not happen as they should never be in PENDING state
                raise ValueError("BRACKET order should not be in PENDING state")
            elif order.status == OrderStatus.PENDING_SALE:
                # Order has been bought but sale hasn't triggered yet
                # Check to see if a sale should trigger now
                curr_price = pd.close
                so = order.get_child_order("STOP")
                po = order.get_child_order("PROFIT")
                manual_sale = order.MANUAL_SALE
                manual_order = order.get_child_order("MANUAL_ORDER")

                # If user triggered a manual sale, make a market order and fill it
                if manual_sale and manual_order is None:
                    # Create a manual order to sell right away
                    mo = Order.create_market_order(
                        order.symbol, OrderAction.SELL, order.quantity, order.tx_cost,
                        current_tick
                    )
                    mo = self._submit_order_to_backend(mo, current_tick, positions, pf_cash)
                    order.add_child_order("MANUAL_ORDER", mo)
                    order.SOLD_ORDER = mo
                    order.status = OrderStatus.FILLED
                    so.status = OrderStatus.CANCELED
                    po.status = OrderStatus.CANCELED
                    return order
                elif curr_price <= so.price:
                    # Trigger stop loss order. Sell
                    # Make sure to sell the right amount, check what positions we have
                    if positions is not None and order.symbol not in positions:
                        position_quantity = 0
                    else:
                        position_quantity = positions[order.symbol].quantity
                    tx_quantity = min(so.quantity, position_quantity)

                    order.status = OrderStatus.FILLED
                    so.status = OrderStatus.FILLED
                    po.status = OrderStatus.CANCELED
                    order.MANUAL_SALE = False
                    so.executed_datetime = pd.timestamp
                    so.price = curr_price
                    so.cash = curr_price * tx_quantity
                    so.quantity = tx_quantity
                    order.SOLD_ORDER = so
                    return order
                elif curr_price >= po.price:
                    # Trigger profit taking order. Sell high
                    # Make sure to sell the right amount, check what positions we have
                    if positions is not None and order.symbol not in positions:
                        position_quantity = 0
                    else:
                        position_quantity = positions[order.symbol].quantity
                    tx_quantity = min(so.quantity, position_quantity)

                    order.status = OrderStatus.FILLED
                    po.status = OrderStatus.FILLED
                    so.status = OrderStatus.CANCELED
                    order.MANUAL_SALE = False
                    po.executed_datetime = pd.timestamp
                    po.price = curr_price
                    po.cash = curr_price * tx_quantity
                    po.quantity = tx_quantity
                    order.SOLD_ORDER = po
                    return order
                else:
                    # No sale has gotten triggered, nothing to do
                    return order
            else:
                raise ValueError(f"Invalid order state: {order.status}")

        raise NotImplementedError(f"{order.type} type not implemented")

    def _update_orders_statuses_from_backend(
            self, orders: dict[str, Order],
            current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float = 0.0) -> list[str]:
        if current_tick is None:
            raise ValueError("Backtesting OM requires current_tick to be set")
        if positions is None:
            raise ValueError("Backtesting OM requires positions to be set")
        # Process all orders and create a return value list of just the ones that changed
        ret_val = []
        for order in orders.values():
            original_status = order.status
            order = self._update_order_status_from_backend(order, current_tick, positions, pf_cash)
            if order.status != original_status:
                ret_val.append(order.order_id)
        return ret_val


    def _submit_order_to_backend(
            self, order: Order, current_tick: list[PriceData] = None,
            positions: dict[str,Position]=None, pf_cash: float = 0.0) -> Order:

        if current_tick is None:
            raise ValueError("Backtesting OM requires current_tick to be set")
        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED}:
            # nothing to do
            return order

        pd = find_pricedata_in_list(order.symbol, current_tick)
        if order.symbol in positions:
            position = positions[order.symbol]
        else:
            position = Position(order.symbol, 0)

        # Go through all supported order types and process them
        if order.type == OrderType.MARKET:
            _process_market_order(order, pd, pf_cash, position.quantity, positions)

        elif order.type == OrderType.BRACKET:
            order.status = OrderStatus.PENDING_SALE
            order.quantity = min(order.quantity, int(pf_cash/pd.close))
            order.price = pd.close
            order.cash = pd.close * order.quantity
            order.executed_datetime = pd.timestamp

        else:
            raise ValueError(f"Unknown order type {order.type}")

        return order

