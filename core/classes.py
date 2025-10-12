"""
Required data passing classes
"""
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import uuid

class SignalType(Enum):
    BUY = 1
    SELL = 2

class OrderType(Enum):
    MARKET = 1
    STOP = 2
    STOP_LOSS = 3
    BRACKET = 4

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
