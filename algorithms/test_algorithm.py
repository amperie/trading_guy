from typing import Dict

from core.algorithm import Algorithm
from core.classes import PriceData, MarketSignal


class TestAlgorithm(Algorithm):
    pass

    def on_data_logic(self, data: Dict[str,PriceData]) -> list[MarketSignal]:
        pass