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
    'macd_histogram': (0.15, 0.5, -0.2, -2.0, 2.0),  # Example values - update from analysis
    'rsi': (0.10, 55.0, 45.0, 0.0, 100.0),
    'macd_divergence': (0.12, 0.3, -0.1, -1.5, 1.5),
    'rsi_distance_neutral': (-0.08, 8.0, 15.0, 0.0, 50.0),
    'macd_histogram_slope': (0.18, 0.1, -0.05, -1.0, 1.0),
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

        # Load custom feature weights if provided in config
        self.feature_weights = cfg.get("feature_weights", FEATURE_WEIGHTS)

    def calculate_signal_strength(self, rsi, macd, macd_last) -> float:
        """
        Calculate signal strength (0-100) based on feature correlations with winning trades.

        Uses correlation-based scoring:
        1. Extract features from indicators
        2. Normalize to 0-1 scale using min-max scaling
        3. Weight by correlation coefficient
        4. Combine and scale to 0-100

        Args:
            rsi: RSI indicator object
            macd: MACD indicator object
            macd_last: Previous MACD indicator object

        Returns:
            Signal strength between 0 (very weak) and 100 (very strong)
        """
        # Extract features (same as in analysis notebook)
        features = {}

        # Direct indicator values
        if rsi:
            features['rsi'] = rsi.rsi
            features['rsi_distance_neutral'] = abs(rsi.rsi - 50)

        if macd:
            features['macd_histogram'] = macd.histogram
            features['macd_divergence'] = macd.macd - macd.signal

            # Engineered features
            if macd_last:
                features['macd_histogram_slope'] = macd.histogram - macd_last.histogram

        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0

        for feat_name, (correlation, winner_med, loser_med, min_val, max_val) in self.feature_weights.items():
            if feat_name not in features:
                continue

            feat_value = features[feat_name]

            # Normalize to 0-1 scale using min-max scaling
            if max_val > min_val:
                normalized = (feat_value - min_val) / (max_val - min_val)
                normalized = max(0.0, min(1.0, normalized))  # Clip to [0, 1]
            else:
                normalized = 0.5  # Default if range is invalid

            # If negative correlation, invert the score (lower is better)
            if correlation < 0:
                normalized = 1.0 - normalized

            # Weight by absolute correlation strength
            weight = abs(correlation)
            total_score += normalized * weight
            total_weight += weight

        # Calculate final strength (0-100)
        if total_weight > 0:
            strength = (total_score / total_weight) * 100
        else:
            strength = 50.0  # Default neutral strength if no features available

        # Ensure bounds
        strength = max(0.0, min(100.0, strength))

        return strength

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
                strength = self.calculate_signal_strength(rsi, macd, macd.last_macd)

                if macd.histogram > 0:
                    ret = MarketSignal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        strength=strength,
                        metadata ={"rsi": rsi, "macd": macd, "macd_last": macd.last_macd},
                    )
                    ret_val.append(ret)
                else:
                    ret = MarketSignal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        strength=strength,
                        metadata ={"rsi": rsi, "macd": macd, "macd_last": macd.last_macd},
                    )
                    ret_val.append(ret)

        return ret_val

