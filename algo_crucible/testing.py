from __future__ import annotations

from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData, SignalType


class BuyAndHoldAlgorithm(Algorithm):
    def __init__(self, cfg=None, history_length: int = 0):
        super().__init__(cfg, history_length)
        self._bought = set()

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []
        for bar in data:
            if bar.symbol not in self._bought:
                self._bought.add(bar.symbol)
                signals.append(MarketSignal(SignalType.BUY, bar.symbol, 100))
        return signals
