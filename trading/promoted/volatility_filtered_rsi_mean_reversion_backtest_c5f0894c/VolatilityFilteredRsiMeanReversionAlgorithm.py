from collections import deque
from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType


class VolatilityFilteredRsiMeanReversionAlgorithm(Algorithm):
    """
    Mean-reversion algorithm that emits BUY/SELL signals only when:
    1. Price oscillates near the 50/200 MA (ranging regime)
    2. Volatility (ATR) is below the 50th percentile
    3. RSI crosses oversold/overbought thresholds
    4. Two-bar price reversal pattern confirmed
    """

    def __init__(self, cfg: dict | None = None, history_length: int = 0):
        super().__init__(cfg, history_length or 250)

        self.symbol = self.cfg.get("symbol", "SPY")

        # Regime detection config
        regime_cfg = self.cfg.get("regime_detection", {})
        self.ma_short_period = regime_cfg.get("ma_short_period", 50)
        self.ma_long_period = regime_cfg.get("ma_long_period", 200)
        self.ma_proximity_tolerance = regime_cfg.get("ma_proximity_tolerance", 0.02)
        self.atr_period = regime_cfg.get("atr_period", 14)
        self.atr_percentile_window = regime_cfg.get("atr_percentile_window", 20)
        self.atr_percentile_level = regime_cfg.get("atr_percentile_level", 50)

        # RSI config
        rsi_cfg = self.cfg.get("rsi_config", {})
        self.rsi_period = rsi_cfg.get("rsi_period", 14)
        self.rsi_oversold_threshold = rsi_cfg.get("rsi_oversold_threshold", 30)
        self.rsi_overbought_threshold = rsi_cfg.get("rsi_overbought_threshold", 70)

        # Price confirmation config
        price_cfg = self.cfg.get("price_confirmation", {})
        self.reversal_bar_lookback = price_cfg.get("reversal_bar_lookback", 2)

        # Internal ATR history for percentile calculation
        self.atr_history = {}

    def _calculate_atr(self, highs: deque, lows: deque, closes: deque, period: int) -> float:
        """Calculate Average True Range."""
        if len(closes) < period:
            return None

        tr_values = []
        for i in range(-period, 0):
            high = highs[i]
            low = lows[i]
            close_prev = closes[i - 1] if i > -len(closes) else closes[0]

            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            tr_values.append(tr)

        return sum(tr_values) / period if tr_values else None

    def _calculate_rsi(self, closes: deque, period: int) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(closes) < period + 1:
            return None

        gains = []
        losses = []

        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            if delta > 0:
                gains.append(delta)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(delta))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _calculate_ma(self, closes: deque, period: int) -> float:
        """Calculate simple moving average."""
        if len(closes) < period:
            return None
        recent = list(closes)[-period:]
        return sum(recent) / period

    def _percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return None
        sorted_vals = sorted(values)
        index = int((percentile / 100.0) * (len(sorted_vals) - 1))
        return sorted_vals[index]

    def _detect_two_bar_reversal_up(self, closes: deque) -> bool:
        """
        Detect upward two-bar reversal:
        closes[-2] > closes[-3] (down bar) and closes[-1] > closes[-2] (up bar bounce).
        """
        if len(closes) < 3:
            return False
        # closes[-1] is current, closes[-2] is previous, closes[-3] is two-bar-back
        down_bar = closes[-2] < closes[-3]
        up_bar = closes[-1] > closes[-2]
        return down_bar and up_bar

    def _detect_two_bar_reversal_down(self, closes: deque) -> bool:
        """
        Detect downward two-bar reversal:
        closes[-2] < closes[-3] (up bar) and closes[-1] < closes[-2] (down bar bounce).
        """
        if len(closes) < 3:
            return False
        up_bar = closes[-2] > closes[-3]
        down_bar = closes[-1] < closes[-2]
        return up_bar and down_bar

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """
        Generate mean-reversion signals filtered by volatility regime and RSI.
        """
        signals = []

        for price_data in data:
            symbol = price_data.symbol

            # Get rolling history
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.ma_long_period:
                continue

            closes = self.price_history[symbol]
            if symbol not in self.price_data_history or len(self.price_data_history[symbol]) < self.atr_period:
                continue

            price_datas = self.price_data_history[symbol]

            # Step 1: Compute moving averages
            ma_50 = self._calculate_ma(closes, self.ma_short_period)
            ma_200 = self._calculate_ma(closes, self.ma_long_period)

            if ma_50 is None or ma_200 is None:
                continue

            # Step 2: Detect ranging regime
            ma_diff_pct = (ma_50 - ma_200) / ma_200
            is_ranging = abs(ma_diff_pct) <= self.ma_proximity_tolerance

            if not is_ranging:
                continue

            # Step 3: Compute ATR and volatility percentile
            highs = deque(pd.high for pd in price_datas)
            lows = deque(pd.low for pd in price_datas)

            current_atr = self._calculate_atr(highs, lows, closes, self.atr_period)

            if current_atr is None:
                continue

            # Maintain ATR history for percentile calculation
            if symbol not in self.atr_history:
                self.atr_history[symbol] = deque(maxlen=self.atr_percentile_window)
            self.atr_history[symbol].append(current_atr)

            # Step 4: Apply volatility filter
            atr_percentile = self._percentile(list(self.atr_history[symbol]), self.atr_percentile_level)

            if atr_percentile is None:
                continue

            volatility_is_low = current_atr < atr_percentile

            if not (is_ranging and volatility_is_low):
                continue

            # Step 5: Compute RSI
            rsi = self._calculate_rsi(closes, self.rsi_period)

            if rsi is None:
                continue

            # Step 6: Detect two-bar price reversals
            reversal_up = self._detect_two_bar_reversal_up(closes)
            reversal_down = self._detect_two_bar_reversal_down(closes)

            # Step 7 & 8: Emit signals based on RSI and reversal pattern
            signal = None
            strength = None

            if (rsi < self.rsi_oversold_threshold and reversal_up):
                # Calculate strength: distance from midline (50) normalized to 0.5-1.0
                rsi_distance = (self.rsi_oversold_threshold - rsi) / self.rsi_oversold_threshold
                strength = max(0.5, min(1.0, 0.5 + rsi_distance * 0.5))
                signal = SignalType.BUY

            elif (rsi > self.rsi_overbought_threshold and reversal_down):
                # Calculate strength: distance from midline (50) normalized to 0.5-1.0
                rsi_distance = (rsi - self.rsi_overbought_threshold) / (100 - self.rsi_overbought_threshold)
                strength = max(0.5, min(1.0, 0.5 + rsi_distance * 0.5))
                signal = SignalType.SELL

            if signal is not None:
                metadata = {
                    "regime": "RANGING",
                    "volatility_filtered": volatility_is_low,
                    "rsi_value": round(rsi, 2),
                    "ma_50": round(ma_50, 2),
                    "ma_200": round(ma_200, 2),
                    "atr_percentile_50": round(atr_percentile, 4),
                    "price_reversal_confirmed": True,
                }

                market_signal = MarketSignal(
                    type=signal,
                    symbol=symbol,
                    strength=round(strength, 2),
                    metadata=metadata,
                )
                signals.append(market_signal)

        return signals


if __name__ == "__main__":
    # Smoke test: instantiate and run synthetic ticks
    cfg = {
        "symbol": "SPY",
        "regime_detection": {
            "ma_short_period": 50,
            "ma_long_period": 200,
            "ma_proximity_tolerance": 0.02,
            "atr_period": 14,
            "atr_percentile_window": 20,
            "atr_percentile_level": 50,
        },
        "rsi_config": {
            "rsi_period": 14,
            "rsi_oversold_threshold": 30,
            "rsi_overbought_threshold": 70,
        },
        "price_confirmation": {
            "reversal_bar_lookback": 2,
        },
    }

    algo = VolatilityFilteredRsiMeanReversionAlgorithm(cfg, history_length=250)

    # Generate synthetic price data
    synthetic_data = []
    base_price = 400.0
    for i in range(260):
        price = base_price + (i % 50) * 0.5 - 12.5
        high = price + 1.0
        low = price - 1.0
        synthetic_data.append(
            PriceData(
                symbol="SPY",
                timestamp=1000000 + i,
                open=price,
                high=high,
                low=low,
                close=price,
                volume=1000000,
            )
        )

    # Feed ticks to algorithm
    for tick_data in synthetic_data:
        signals = algo.on_data([tick_data])
        if signals:
            for sig in signals:
                print(
                    f"Signal: {sig.type.name} {sig.symbol} strength={sig.strength} "
                    f"rsi={sig.metadata.get('rsi_value')} "
                    f"ma_diff={((sig.metadata.get('ma_50', 0) - sig.metadata.get('ma_200', 0)) / sig.metadata.get('ma_200', 1) * 100):.2f}%"
                )

    print("Smoke test completed successfully")