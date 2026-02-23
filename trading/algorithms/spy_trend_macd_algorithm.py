from typing import Dict, Any

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType
from utils.logger import Logger
from utils.utils import find_pricedata_in_list

logger = Logger().get_logger(__name__)


class SpyTrendMACDAlgorithm(Algorithm):
    """
    Uses SPY MACD trend to pick UPRO (bullish) or SPXU (bearish).

    MACD (Moving Average Convergence Divergence) indicator:
    - MACD Line = Fast EMA - Slow EMA
    - Signal Line = EMA of MACD Line
    - Histogram = MACD Line - Signal Line

    Trading Logic:
    - MACD Line > Signal Line → Bullish → BUY UPRO
    - MACD Line < Signal Line → Bearish → BUY SPXU

    Emits a BUY signal for the target symbol when the regime flips.
    """

    default_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "macd_fast_period": 12,      # Fast EMA period
        "macd_slow_period": 26,      # Slow EMA period
        "macd_signal_period": 9,     # Signal line EMA period
        "strength_scale": 20.0,      # Multiplier for signal strength calculation
    }

    def __init__(self, cfg: Dict[str, Any] = None, history_length: int = 0):
        cfg = {**self.default_cfg, **(cfg or {})}

        # Calculate required history length based on MACD parameters
        # Need enough history for slow EMA + signal EMA
        if history_length <= 0:
            history_length = cfg["macd_slow_period"] + cfg["macd_signal_period"]

        super().__init__(cfg, history_length)

        self.spy_symbol = cfg["spy_symbol"]
        self.upro_symbol = cfg["upro_symbol"]
        self.spxu_symbol = cfg["spxu_symbol"]
        self.macd_fast_period = int(cfg["macd_fast_period"])
        self.macd_slow_period = int(cfg["macd_slow_period"])
        self.macd_signal_period = int(cfg["macd_signal_period"])
        self.strength_scale = cfg["strength_scale"]

        self._last_target = None
        self._macd_history = []  # Store MACD line values for signal line calculation

    def _ema(self, values, window: int) -> float | None:
        """
        Calculate Exponential Moving Average.

        Args:
            values: Price history (deque or list)
            window: EMA period

        Returns:
            EMA value or None if insufficient data
        """
        if len(values) < window:
            return None

        # Convert to list for indexing
        prices = list(values)

        # Smoothing factor: k = 2 / (N + 1)
        k = 2.0 / (window + 1)

        # Initial EMA = SMA of first 'window' values
        ema = sum(prices[:window]) / window

        # Apply EMA formula to remaining values
        for i in range(window, len(prices)):
            ema = prices[i] * k + ema * (1 - k)

        return ema

    def _calculate_macd(self, history) -> tuple[float, float, float] | None:
        """
        Calculate MACD indicator components.

        Args:
            history: Price history (deque or list)

        Returns:
            Tuple of (macd_line, signal_line, histogram) or None if insufficient data
        """
        # Need enough history for slow EMA
        if len(history) < self.macd_slow_period:
            return None

        # Calculate MACD line = Fast EMA - Slow EMA
        fast_ema = self._ema(history, self.macd_fast_period)
        slow_ema = self._ema(history, self.macd_slow_period)

        if fast_ema is None or slow_ema is None:
            return None

        macd_line = fast_ema - slow_ema

        # Store MACD line value for signal line calculation
        self._macd_history.append(macd_line)

        # Keep only necessary history for signal line calculation
        max_history = self.macd_signal_period * 2
        if len(self._macd_history) > max_history:
            self._macd_history = self._macd_history[-max_history:]

        # Calculate Signal line = EMA of MACD line
        if len(self._macd_history) < self.macd_signal_period:
            return None

        signal_line = self._ema(self._macd_history, self.macd_signal_period)

        if signal_line is None:
            return None

        # Calculate histogram = MACD line - Signal line
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        pd = find_pricedata_in_list(self.spy_symbol, data)
        if pd is None:
            return []

        history = self.price_history.get(self.spy_symbol)
        if history is None:
            return []

        # Calculate MACD components
        macd_result = self._calculate_macd(history)
        if macd_result is None:
            return []

        macd_line, signal_line, histogram = macd_result

        # Determine target based on MACD crossover
        # MACD > Signal → Bullish → UPRO
        # MACD < Signal → Bearish → SPXU
        target = self.upro_symbol if macd_line >= signal_line else self.spxu_symbol

        # Only emit signal when target changes (regime flip)
        if target == self._last_target:
            return []

        # Calculate signal strength based on histogram magnitude
        # Larger histogram = stronger signal
        histogram_abs = abs(histogram)

        # Normalize histogram to percentage of current price
        price_pct = (histogram_abs / pd.close) * 100.0

        # Scale to 1-100 range
        strength = min(100, int(price_pct * self.strength_scale))
        if strength == 0:
            strength = 1

        self._last_target = target
        return [
            MarketSignal(
                type=SignalType.BUY,
                symbol=target,
                strength=strength,
                metadata={
                    "spy_price": pd.close,
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "histogram": histogram,
                    "histogram_pct": price_pct,
                    "target": target,
                },
            )
        ]

    def reconfigure(self, new_params: dict) -> None:
        super().reconfigure(new_params)
        for attr in ("macd_fast_period", "macd_slow_period", "macd_signal_period"):
            if attr in new_params:
                setattr(self, attr, int(new_params[attr]))
        if "strength_scale" in new_params:
            self.strength_scale = new_params["strength_scale"]
