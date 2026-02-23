from typing import Dict, Any

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType
from utils.utils import find_pricedata_in_list


class SpyTrendSwitchAlgorithm(Algorithm):
    """
    Uses SPY trend to pick UPRO (uptrend) or SPXU (downtrend).
    Emits a BUY signal for the target symbol when the regime flips.
    """

    default_cfg = {
        "spy_symbol": "SPY",
        "upro_symbol": "UPRO",
        "spxu_symbol": "SPXU",
        "fast_window": 10,
        "slow_window": 30,
        "strength_scale": 20.0,
    }

    def __init__(self, cfg: Dict[str, Any] = None, history_length: int = 0):
        cfg = {**self.default_cfg, **(cfg or {})}
        if history_length <= 0:
            history_length = cfg["slow_window"]
        super().__init__(cfg, history_length)

        self.spy_symbol = cfg["spy_symbol"]
        self.upro_symbol = cfg["upro_symbol"]
        self.spxu_symbol = cfg["spxu_symbol"]
        self.fast_window = cfg["fast_window"]
        self.slow_window = cfg["slow_window"]
        self.strength_scale = cfg["strength_scale"]

        self._last_target = None

    def _sma(self, values, window: int) -> float | None:
        if len(values) < window:
            return None
        recent = list(values)[-window:]
        return sum(recent) / window

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

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        pd = find_pricedata_in_list(self.spy_symbol, data)
        if pd is None:
            return []

        history = self.price_history.get(self.spy_symbol)
        if history is None or len(history) < self.slow_window:
            return []

        fast = self._sma(history, self.fast_window)
        slow = self._sma(history, self.slow_window)
        if fast is None or slow is None or slow == 0:
            return []

        target = self.upro_symbol if fast >= slow else self.spxu_symbol
        if target == self._last_target:
            return []

        diff_pct = abs(fast - slow) / slow * 100.0
        strength = min(100, int(diff_pct * self.strength_scale))
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
                    "spy_fast_sma": fast,
                    "spy_slow_sma": slow,
                    "diff_pct": diff_pct,
                    "target": target,
                },
            )
        ]

    def reconfigure(self, new_params: dict) -> None:
        super().reconfigure(new_params)
        for attr in ("fast_window", "slow_window"):
            if attr in new_params:
                setattr(self, attr, int(new_params[attr]))
        if "strength_scale" in new_params:
            self.strength_scale = new_params["strength_scale"]
