"""
Required data passing classes
"""
import datetime
from enum import Enum
from dataclasses import dataclass

class SignalType(Enum):
    BUY = 1
    SELL = 2

class OrderType(Enum):
    MARKET = 1
    STOP = 2
    STOP_LOSS = 3
    BRACKET = 4

class OrderStatus(Enum):
    PENDING = 0
    FILLED = 1

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
    vwp: float = None
    exchange: float = None


@dataclass
class MarketSignal():
    type: OrderType
    ticker: str

@dataclass
class Order():
    type: OrderType
    ticker: str
    status: OrderStatus