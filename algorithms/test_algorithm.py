from typing import Dict
import random
from core.algorithm import Algorithm
from core.classes import PriceData, MarketSignal, SignalType


class TestAlgorithm(Algorithm):

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        r = random.randint(0,100)
        if r > 50:
            signal = SignalType.BUY
        else:
            signal = SignalType.SELL

        retval = [MarketSignal(signal, x.symbol, 100) for x in data]
        return retval
