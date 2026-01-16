"""
Base class for algorithms.

Interfaces:
 - Initialize (initializes)
 - on_data: gets called when new market data is available
 - 
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, final, List

from trading.core.classes import MarketSignal, PriceData
from collections import defaultdict, deque


class Algorithm(ABC):

    default_cfg = {
        "history_length": 0,
        "full_history": False,
    }

    def __init__(self, cfg: Dict[str, Any]=None, history_length: int=0):

        super().__init__()
        if cfg is None:
            cfg = self.default_cfg
        else:
            cfg = {**self.default_cfg, **cfg}

        self.cfg = cfg
        self.history_length = history_length
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.price_data_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.full_history: List[Dict[str, PriceData]] = []

        if self.history_length > 0:
            # Initialize dequeues with maxlen for automatic size management
            self.price_history = defaultdict(lambda: deque(maxlen=self.history_length))
            self.price_data_history = defaultdict(lambda: deque(maxlen=self.history_length))
        else:
            self.price_history = {}

        self.full_history = []

    def _update_history(self, data: list[PriceData]):
        for pd in data:
            symbol = pd.symbol
            # Append to limited history
            if self.history_length > 0:
                self.price_history[symbol].append(pd.close)
                self.price_data_history[symbol].append(pd)
            # Append full history if configured to do so
        if "full_history" in self.cfg and self.cfg["full_history"] is True:
            self.full_history.append(data)

    @final
    def get_price_history(self):
        return self.price_history

    @final
    def get_price_data_history(self):
        return self.price_data_history

    @final
    def get_full_history(self):
        return self.full_history

    @final
    def on_data(self, data: list[PriceData]) -> list[MarketSignal]:
        # Add logic for history and other stuff here
        self._update_history(data)

        # Run the logic after
        return self.on_data_logic(data)

    @abstractmethod
    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        raise NotImplementedError("on_data_logic needs to be overriden")
