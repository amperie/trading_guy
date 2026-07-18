from collections import deque

from trading.core.classes import MarketSignal, PriceData, SignalType
from trading.promoted.volatility_filtered_rsi_mean_reversion_backtest_c5f0894c.VolatilityFilteredRsiMeanReversionAlgorithm import (
    VolatilityFilteredRsiMeanReversionAlgorithm,
)


class VolatilityFilteredRsiMeanReversionAlgorithmV2(VolatilityFilteredRsiMeanReversionAlgorithm):
    """Tom RSI with stable volatility gating and one signal per RSI excursion."""

    def __init__(self, cfg: dict | None = None, history_length: int = 0):
        super().__init__(cfg or {}, history_length)
        self._refresh_v2_config()
        self._buy_ready: dict[str, bool] = {}
        self._sell_ready: dict[str, bool] = {}

    def _refresh_v2_config(self) -> None:
        regime = self.cfg.get("regime_detection", {})
        legacy_level = float(regime.get("atr_percentile_level", 50))
        self.atr_percentile_low = float(regime.get("atr_percentile_low", 0))
        self.atr_percentile_high = float(regime.get("atr_percentile_high", legacy_level))
        self.require_full_atr_window = bool(regime.get("require_full_atr_window", True))

        rsi = self.cfg.get("rsi_config", {})
        self.rsi_buy_rearm_threshold = float(
            rsi.get("buy_rearm_threshold", self.rsi_oversold_threshold + 10)
        )
        self.rsi_sell_rearm_threshold = float(
            rsi.get("sell_rearm_threshold", self.rsi_overbought_threshold - 10)
        )

    @property
    def required_warmup_bars(self) -> int:
        return max(
            super().required_warmup_bars,
            self.ma_long_period + self.atr_percentile_window - 1,
        )

    @staticmethod
    def _percentile_linear(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def _detect_two_bar_reversal_up(self, closes: deque) -> bool:
        n = max(2, self.reversal_bar_lookback)
        if len(closes) < n + 1:
            return False
        prior = list(closes)[-(n + 1):-1]
        return all(current < previous for previous, current in zip(prior, prior[1:])) and closes[-1] > closes[-2]

    def _detect_two_bar_reversal_down(self, closes: deque) -> bool:
        n = max(2, self.reversal_bar_lookback)
        if len(closes) < n + 1:
            return False
        prior = list(closes)[-(n + 1):-1]
        return all(current > previous for previous, current in zip(prior, prior[1:])) and closes[-1] < closes[-2]

    def _volatility_is_eligible(self, symbol: str, normalized_atr: float) -> tuple[bool, float, float]:
        history = self.atr_history.setdefault(symbol, deque(maxlen=self.atr_percentile_window))
        history.append(normalized_atr)
        if self.require_full_atr_window and len(history) < self.atr_percentile_window:
            return False, 0.0, 0.0
        values = list(history)
        low = self._percentile_linear(values, self.atr_percentile_low)
        high = self._percentile_linear(values, self.atr_percentile_high)
        if low is None or high is None:
            return False, 0.0, 0.0
        return low <= normalized_atr <= high, low, high

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []
        for bar in data:
            symbol = bar.symbol
            closes = self.price_history.get(symbol)
            bars = self.price_data_history.get(symbol)
            if closes is None or bars is None or len(closes) < self.ma_long_period:
                continue

            highs = deque(item.high for item in bars)
            lows = deque(item.low for item in bars)
            atr = self._calculate_atr(highs, lows, closes, self.atr_period)
            if atr is None or bar.close <= 0:
                continue
            normalized_atr = atr / bar.close
            vol_ok, atr_low, atr_high = self._volatility_is_eligible(symbol, normalized_atr)

            short_ma = self._calculate_ma(closes, self.ma_short_period)
            long_ma = self._calculate_ma(closes, self.ma_long_period)
            rsi = self._calculate_rsi(closes, self.rsi_period)
            if short_ma is None or long_ma in (None, 0) or rsi is None:
                continue

            if rsi >= self.rsi_buy_rearm_threshold:
                self._buy_ready[symbol] = True
            if rsi <= self.rsi_sell_rearm_threshold:
                self._sell_ready[symbol] = True
            self._buy_ready.setdefault(symbol, True)
            self._sell_ready.setdefault(symbol, True)

            ranging = abs((short_ma - long_ma) / long_ma) <= self.ma_proximity_tolerance
            signal_type = None
            if (
                ranging and vol_ok and rsi < self.rsi_oversold_threshold
                and self._buy_ready[symbol] and self._detect_two_bar_reversal_up(closes)
            ):
                signal_type = SignalType.BUY
                self._buy_ready[symbol] = False
                distance = (self.rsi_oversold_threshold - rsi) / max(self.rsi_oversold_threshold, 1)
            elif (
                rsi > self.rsi_overbought_threshold
                and self._sell_ready[symbol] and self._detect_two_bar_reversal_down(closes)
            ):
                # Exits are deliberately not blocked by entry-regime or volatility filters.
                signal_type = SignalType.SELL
                self._sell_ready[symbol] = False
                distance = (rsi - self.rsi_overbought_threshold) / max(100 - self.rsi_overbought_threshold, 1)
            else:
                continue

            strength = round(max(50.0, min(100.0, 50.0 + 50.0 * distance)), 2)
            signals.append(MarketSignal(signal_type, symbol, strength, {
                "regime": "RANGING" if ranging else "NON_RANGING",
                "rsi_value": round(rsi, 2),
                "ma_short": round(short_ma, 4),
                "ma_long": round(long_ma, 4),
                "atr": round(atr, 6),
                "normalized_atr": round(normalized_atr, 8),
                "atr_percentile_low_value": round(atr_low, 8),
                "atr_percentile_high_value": round(atr_high, 8),
                "price_reversal_confirmed": True,
            }))
        return signals

    def reconfigure(self, new_params: dict) -> None:
        old_window = self.atr_percentile_window
        super().reconfigure(new_params)
        self._refresh_v2_config()
        if self.atr_percentile_window != old_window:
            self.atr_history = {
                symbol: deque(values, maxlen=self.atr_percentile_window)
                for symbol, values in self.atr_history.items()
            }
