from typing import Dict, Any
from core.algorithm import Algorithm
from core.classes import PriceData, MarketSignal, SignalType
from core.ta.analyzer import TechnicalAnalyzer
from utils.utils import get_symbols_in_list


momentum_default_params = {
    "macd_fastperiod": 12,
    "macd_slowperiod": 26,
    "macd_signalperiod": 9,
    "rsi_period": 14,
    "extra_history_period": 200,
}


class MacdRsiAlgorithm(Algorithm):

    def __init__(self, cfg: Dict[str, Any]=None, history_length: int=0):
        """
        This algorithm requires the right context passed in with these parameters:
        Defaults are:
            momentum_default_params = {
                "macd_fastperiod": 12,
                "macd_slowperiod": 26,
                "macd_signalperiod": 9,
                "rsi_period": 14,
                "extra_history_period": 200, To more accurately calculate moving averages
            }
        """
        if history_length != 0:
            history = history_length
        elif "extra_history_period" in cfg:
            history = cfg["extra_history_period"] + cfg["macd_slowperiod"]
        else:
            history = cfg["macd_slowperiod"] * 2


        super().__init__(cfg, history)
        self.macd_fastperiod = cfg["macd_fastperiod"]
        self.macd_slowperiod = cfg["macd_slowperiod"]
        self.macd_signalperiod = cfg["macd_signalperiod"]
        self.rsi_period = cfg["rsi_period"]
        self.ta = TechnicalAnalyzer()

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """
        Iterates through all symbols in the data and generates a market signal for each
        if appropriate
        """
        ret_val = []
        for pd in data:
            symbol = pd.symbol
            macd = self.ta.calculate_macd(
                self.price_history[symbol], self.macd_slowperiod, self.macd_fastperiod,
                self.macd_signalperiod, True
            )
            rsi = self.ta.calculate_rsi(self.price_history[symbol], self.rsi_period)
            if rsi is None or macd is None or macd.last_macd is None:
                # Not enough data yet
                continue
            if rsi.rsi > 40 and macd.histogram * macd.last_macd.histogram < 0:
                if macd.histogram > 0:
                    ret = MarketSignal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        strength=100, # TODO: make this dynamic based on MACD and RSI
                        metadata ={"rsi": rsi, "macd": macd, "macd_last": macd.last_macd},
                    )
                    ret_val.append(ret)
                else:
                    ret = MarketSignal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        strength=100,
                        metadata ={"rsi": rsi, "macd": macd, "macd_last": macd.last_macd},
                    )
                    ret_val.append(ret)

        return ret_val

