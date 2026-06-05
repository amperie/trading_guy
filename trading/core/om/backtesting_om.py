import datetime as _dt
from typing import Union
from trading.core.classes import Order, PriceData, OrderType, OrderStatus, Position, OrderAction, BracketOrder
from trading.core.om.order_manager import OrderManager
from utils.utils import find_pricedata_in_list
from utils.logger import Logger

logger = Logger().get_logger(__name__)

_MARKET_OPEN  = _dt.time(9, 30)
_MARKET_CLOSE = _dt.time(16, 0)


def _is_market_hours(ts) -> bool:
    """Return True if ts falls within regular market hours (09:30–16:00 ET)."""
    import pandas as pd
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return _MARKET_OPEN <= t.time() < _MARKET_CLOSE

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


def _position_quantity(symbol: str, positions: dict[str, Position] | None) -> int:
    if positions is None or symbol not in positions:
        return 0
    return positions[symbol].quantity


def _fill_bracket_exit(order: BracketOrder, exit_order: Order, other_order: Order, pd: PriceData, positions) -> Order:
    tx_quantity = min(exit_order.quantity, _position_quantity(order.symbol, positions))
    order.status = OrderStatus.FILLED
    exit_order.status = OrderStatus.FILLED
    other_order.status = OrderStatus.CANCELED
    order.MANUAL_SALE = False
    exit_order.executed_datetime = pd.timestamp
    exit_order.price = pd.close
    exit_order.cash = pd.close * tx_quantity
    exit_order.quantity = tx_quantity
    order.SOLD_ORDER = exit_order
    return order


class BacktestingOrderManager(OrderManager):

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        self._market_hours_only: bool = (self.cfg or {}).get("market_hours_only", False)

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

        if self._market_hours_only and pd is not None and not _is_market_hours(pd.timestamp):
            return order  # outside market hours — leave order pending
        if pd is None:
            logger.debug(f"No price data found for {order.symbol}")
            return order

        if order.type == OrderType.MARKET:
            return _process_market_order(order, pd, pf_cash, order.quantity, positions)

        elif order.type == OrderType.BRACKET:
            # Process all options for a bracket order
            if order.status == OrderStatus.PENDING:
                # Price data was unavailable when first submitted — complete submission now
                # (pd is guaranteed non-None here: the early-return above handles pd is None)
                actual_fill = pd.close
                order.status = OrderStatus.PENDING_SALE
                order.quantity = min(order.quantity, int(pf_cash / actual_fill))
                order.price = actual_fill
                order.cash = actual_fill * order.quantity
                order.executed_datetime = pd.timestamp
                # Rescale bracket child prices to be relative to the actual fill price,
                # correcting for any stale entry estimate used during order creation.
                intended = getattr(order, 'intended_entry_price', 0.0)
                if intended > 0 and actual_fill != intended:
                    scale = actual_fill / intended
                    so = order.get_child_order("STOP")
                    po = order.get_child_order("PROFIT")
                    if so and so.type == OrderType.TRAILING_STOP:
                        so.trail_hwm = actual_fill
                        so.update_trailing_stop(actual_fill)
                    elif so:
                        so.price = round(so.price * scale, 2)
                    if po:
                        po.price = round(po.price * scale, 2)
                else:
                    so = order.get_child_order("STOP")
                    if so and so.type == OrderType.TRAILING_STOP:
                        so.trail_hwm = actual_fill
                        so.update_trailing_stop(actual_fill)
                return order
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
                if so.type == OrderType.TRAILING_STOP:
                    so.update_trailing_stop(curr_price)

                if curr_price <= so.price:
                    # Trigger stop loss order. Sell
                    return _fill_bracket_exit(order, so, po, pd, positions)
                elif curr_price >= po.price:
                    # Trigger profit taking order. Sell high
                    return _fill_bracket_exit(order, po, so, pd, positions)
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
        working_positions = self._clone_positions_snapshot(positions)
        working_cash = pf_cash
        for order in orders.values():
            original_status = order.status
            order = self._update_order_status_from_backend(order, current_tick, working_positions, working_cash)
            if order.status != original_status:
                ret_val.append(order.order_id)
                working_positions, working_cash = self._apply_order_to_snapshot(order, working_positions, working_cash)
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

        if self._market_hours_only and pd is not None and not _is_market_hours(pd.timestamp):
            return order  # outside market hours — leave order pending

        if pd is None:
            logger.warning(f"No price data for {order.symbol} on this tick — order {order.order_id} will retry next tick")
            return order
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
            stop_order = order.get_child_order("STOP")
            if stop_order is not None and stop_order.type == OrderType.TRAILING_STOP:
                stop_order.trail_hwm = pd.close
                stop_order.update_trailing_stop(pd.close)

        else:
            raise ValueError(f"Unknown order type {order.type}")

        return order

