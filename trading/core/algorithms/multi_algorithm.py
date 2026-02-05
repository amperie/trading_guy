from typing import Dict, Any

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal


class MultiAlgorithm(Algorithm):

    def __init__(self, cfg: Dict[str, Any] = None, history_length: int = 0):

        super().__init__(cfg, history_length)
        self.algorithms = []

    def add_algorithm(self, algorithm: Algorithm):
        self.algorithms.append(algorithm)

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:

        return_signals = []
        for alg in self.algorithms:
            signals = alg.on_data(data)
            return_signals.extend(signals)

        return return_signals