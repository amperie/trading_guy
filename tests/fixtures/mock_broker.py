"""
Mock broker for testing real-time order management
"""
from datetime import datetime
from typing import Dict, List
import uuid

from core.order_manager import OrderManager
from core.classes import Order, PriceData, OrderAction, OrderType, OrderStatus


class MockBroker(OrderManager):
    """
    Mock broker that simulates real-time order execution with delays

    Useful for testing RealTimeOrderManager implementations without
    connecting to actual broker APIs.
    """

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        self.orders_by_id: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}

        # Simulation settings
        self.fill_delay_ticks = cfg.get('fill_delay_ticks', 1) if cfg else 1
        self.fill_probability = cfg.get('fill_probability', 1.0) if cfg else 1.0
        self.tick_count = 0

    def buy(self, symbol: str, shares: int, tick: list[PriceData]) -> Order:
        """Create a buy order (starts as PENDING)"""
        from utils.utils import find_pricedata_in_list

        pd = find_pricedata_in_list(symbol, tick)

        order = Order(
            placed_datetime=datetime.now(),
            executed_datetime=None,
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol=symbol,
            price=pd.close,
            quantity=shares,
            cash=pd.close * shares,
            status=OrderStatus.PENDING,
            tx_cost=0.0
        )

        self.orders_by_id[order.order_id] = order
        self.pending_orders[order.order_id] = order

        return order

    def sell(self, symbol: str, shares: int, tick: list[PriceData]) -> Order:
        """Create a sell order (starts as PENDING)"""
        from utils.utils import find_pricedata_in_list

        pd = find_pricedata_in_list(symbol, tick)

        order = Order(
            placed_datetime=datetime.now(),
            executed_datetime=None,
            action=OrderAction.SELL,
            type=OrderType.MARKET,
            symbol=symbol,
            price=pd.close,
            quantity=shares,
            cash=pd.close * shares,
            status=OrderStatus.PENDING,
            tx_cost=0.0
        )

        self.orders_by_id[order.order_id] = order
        self.pending_orders[order.order_id] = order

        return order

    def update_order_statuses(self, orders: List[Order]) -> List[Order]:
        """Update status of specific orders"""
        updated = []
        for order in orders:
            if order.order_id in self.orders_by_id:
                updated_order = self._simulate_fill(self.orders_by_id[order.order_id])
                updated.append(updated_order)
        return updated

    def update_all_orders(self) -> List[Order]:
        """Update all pending orders"""
        self.tick_count += 1
        updated = []

        for order_id in list(self.pending_orders.keys()):
            order = self.pending_orders[order_id]
            updated_order = self._simulate_fill(order)

            if updated_order.status == OrderStatus.FILLED:
                updated.append(updated_order)
                del self.pending_orders[order_id]
                self.filled_orders[order_id] = updated_order

        return updated

    def get_order_status(self, order: Order) -> Order:
        """Get current status of an order"""
        if order.order_id in self.orders_by_id:
            return self.orders_by_id[order.order_id]
        return order

    def _simulate_fill(self, order: Order) -> Order:
        """Simulate order filling with delay"""
        import random

        if order.status == OrderStatus.FILLED:
            return order

        # Check if enough ticks have passed
        if self.tick_count >= self.fill_delay_ticks:
            # Probabilistic fill
            if random.random() <= self.fill_probability:
                order.status = OrderStatus.FILLED
                order.executed_datetime = datetime.now()
                self.orders_by_id[order.order_id] = order

        return order

    def cancel_order(self, order: Order) -> Order:
        """Cancel a pending order"""
        if order.order_id in self.pending_orders:
            order.status = OrderStatus.CANCELED
            del self.pending_orders[order.order_id]
            self.orders_by_id[order.order_id] = order

        return order

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return list(self.pending_orders.values())

    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders"""
        return list(self.filled_orders.values())

    def reset(self):
        """Reset the mock broker state"""
        self.orders_by_id.clear()
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.tick_count = 0
