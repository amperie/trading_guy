from typing import Dict, Any
from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType
from trading.ta.analyzer import TechnicalAnalyzer

momentum_default_params = {
    "macd_fastperiod": 12,
    "macd_slowperiod": 26,
    "macd_signalperiod": 9,
    "rsi_period": 14,
    "extra_history_period": 200,
}

# Feature weights and thresholds from correlation analysis
#
# HOW TO UPDATE THESE VALUES:
# 1. Run backtesting/signals_orders_analysis.ipynb to completion
# 2. Execute Section 12 to generate feature weights
# 3. Copy the FEATURE_WEIGHTS dictionary from the output
# 4. Replace the dictionary below
#
FEATURE_WEIGHTS = {
    # Feature name: (correlation_coefficient, winner_median, loser_median, min_val, max_val)
    'signal_macd_last.histogram': (0.0421, -0.0029, -0.0032, -0.0293, -0.0000),
    'entry_hour': (0.0315, 15.0000, 15.0000, 0.0000, 23.0000),
    'signal_macd.histogram': (0.0244, 0.0024, 0.0026, 0.0000, 0.0533),
    'stop_loss_price': (0.0142, 56.5488, 54.9501, 27.8388, 119.5200),
    'profit_target_price': (0.0135, 57.7500, 56.3479, 28.4500, 121.9373),
    'entry_price': (0.0106, 57.1200, 55.7900, 28.1200, 120.7300),
    'signal_macd_last.macd': (0.0015, -0.1286, -0.1084, -2.2797, 0.9101),
    'signal_macd.macd': (0.0014, -0.1240, -0.1018, -2.2504, 0.9105),
    'signal_macd.signal': (0.0011, -0.1271, -0.1054, -2.2529, 0.9102),
    'signal_macd_last.signal': (0.0011, -0.1271, -0.1055, -2.2531, 0.9102),
}
SIGNAL_THRESHOLD = 60


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
                # Calculate dynamic signal strength based on features
                strength = 100

                if macd.histogram > 0 and strength >= SIGNAL_THRESHOLD:
                    ret = MarketSignal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        strength=strength,
                        metadata ={"rsi": rsi, "macd": macd, "macd_last": macd.last_macd},
                    )
                    ret_val.append(ret)
                elif macd.histogram < 0:
                    ret = MarketSignal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        strength=strength,
                        metadata ={"rsi": rsi, "macd": macd, "macd_last": macd.last_macd},
                    )
                    ret_val.append(ret)

        return ret_val

    def get_indicator_snapshot(self, data: list[PriceData] | None = None) -> dict | None:
        snapshot = {}
        source = data or []
        for pd in source:
            symbol = pd.symbol
            macd = self.ta.calculate_macd(
                self.price_history[symbol],
                self.macd_slowperiod,
                self.macd_fastperiod,
                self.macd_signalperiod,
                True,
            )
            rsi = self.ta.calculate_rsi(self.price_history[symbol], self.rsi_period)
            if rsi is None or macd is None or macd.last_macd is None:
                continue
            snapshot[symbol] = {
                "rsi": rsi,
                "macd": macd,
                "macd_last": macd.last_macd,
            }
        return snapshot or None

    def reconfigure(self, new_params: dict) -> None:
        super().reconfigure(new_params)
        for attr in ("macd_fastperiod", "macd_slowperiod", "macd_signalperiod", "rsi_period"):
            if attr in new_params:
                setattr(self, attr, new_params[attr])
