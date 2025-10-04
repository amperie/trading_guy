"""
Required data passing classes
"""
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

@dataclass
class MarketSignal():
    type: OrderType
    ticker: str

@dataclass
class Order():
    type: OrderType
    ticker: str