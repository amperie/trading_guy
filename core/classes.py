"""
Required data passing classes
"""
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import uuid

from tests.scratch import Order


class SignalType(Enum):
    BUY = 1
    SELL = 2

class OrderType(Enum):
    MARKET = 1
    BRACKET = 2
    STOP_LOSS = 3
    PROFIT_TAKER = 4

class OrderAction(Enum):
    BUY = 1
    SELL = 2

class OrderStatus(Enum):
    PENDING = 0
    FILLED = 1
    CANCELED = 2

@dataclass
class PriceData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = None
    vwap: float = None
    exchange: float = None

    @classmethod
    def from_dict(cls, data: dict, timestamp: datetime=None):
        """Create a PriceData instance from a dictionary."""
        if timestamp is not None:
            data['timestamp'] = timestamp
        return cls(**data)

@dataclass
class Position:
    symbol: str
    quantity: int

@dataclass
class MarketSignal:
    type: SignalType
    symbol: str
    strength: int # 0-100 to indicate how strong the signal is

@dataclass
class Order:
    placed_datetime: datetime
    executed_datetime: datetime
    action: OrderAction
    type: OrderType
    symbol: str
    price: float
    quantity: int
    cash: float
    status: OrderStatus
    order_id: str = field(default_factory=lambda: f"local-{str(uuid.uuid4())}")
    tx_cost: float = 0.0
    parent_id: str = None
    child_orders: list[str] = field(default_factory=lambda: [])
    processed_by_portfolio: bool = False

    @staticmethod
    def create_bracket_order(
            symbol: str,
            price: float,
            high_sell_price: float,
            low_sell_price: float,
            quantity: int,
            tx_cost: float=0,
            ) -> list[Order]:
        """
        Helper function to create a bracket order. It creates three orders
        the main buy order and two orders to take profit or stop loss
        Args:
            symbol:
            price:
            quantity:
            tx_cost:
            high_sell_price:
            low_sell_price:

        Returns:

        """
        main_order = Order(
            action=OrderAction.BUY,
            type=OrderType.BRACKET,
            symbol=symbol,
            price=price,
            quantity=quantity,
            cash=price * quantity,
            tx_cost=tx_cost,
            status=OrderStatus.PENDING,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
        )

        stop_loss_order = Order(
            action=OrderAction.SELL,
            type=OrderType.STOP_LOSS,
            symbol=symbol,
            price=low_sell_price,
            quantity=quantity,
            cash=0,
            tx_cost=tx_cost,
            status=OrderStatus.PENDING,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
            parent_id=main_order.order_id,
        )

        profit_order = Order(
            action=OrderAction.SELL,
            type=OrderType.PROFIT_TAKER,
            symbol=symbol,
            price=high_sell_price,
            quantity=quantity,
            cash=0,
            tx_cost=tx_cost,
            status=OrderStatus.PENDING,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
            parent_id=main_order.order_id,
        )

        main_order.child_orders = [stop_loss_order.order_id, profit_order.order_id]
        return [main_order, stop_loss_order, profit_order]
