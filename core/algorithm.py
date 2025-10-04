"""
Base class for algorithms.

Interfaces:
 - Initialize (initializes)
 - on_data: gets called when new market data is available
 - 
"""
from abc import ABC, abstractmethod

from core.classes import MarketSignal


class Algorithm(ABC):
    @abstractmethod
    def initialize(self):
        pass
    @abstractmethod
    def on_data(self) -> list[MarketSignal]:
        pass