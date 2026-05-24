from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType
from typing import Dict, Any, Deque
from collections import deque


class AdxMomentumFilterEmaAlgorithm(Algorithm):
    """
    ADX-filtered momentum strategy using EMA crossovers with regime-aware parameters.
    
    Strategy:
    1. Calculate 14-bar ADX to measure trend strength
    2. Classify trend regime (STRONG, MODERATE, or RANGING)
    3. Select EMA periods based on regime (strong: 8/40/100, moderate: 12/40/150)
    4. Generate BUY/SELL signals on fast/slow EMA crossovers
    5. Confirm signals with mid-EMA filter and ADX gate (>25)
    6. Emit signals with strength scaled by ADX
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, history_length: int = 0):
        super().__init__(cfg, history_length or 300)
        
        # Extract configuration
        self.symbol = self.cfg.get("symbol", "SPY")
        
        # ADX configuration
        self.adx_period = self.cfg.get("adx_config", {}).get("adx_period", 14)
        self.trend_threshold = self.cfg.get("adx_config", {}).get("trend_threshold", 25)
        self.strong_trend_threshold = self.cfg.get("adx_config", {}).get("strong_trend_threshold", 35)
        self.weak_trend_lower_bound = self.cfg.get("adx_config", {}).get("weak_trend_lower_bound", 20)
        
        # Strong trend EMA parameters
        strong_cfg = self.cfg.get("strong_trend_ema_params", {})
        self.strong_fast_ema = strong_cfg.get("fast_ema_period", 8)
        self.strong_mid_ema = strong_cfg.get("mid_ema_period", 40)
        self.strong_slow_ema = strong_cfg.get("slow_ema_period", 100)
        
        # Moderate trend EMA parameters
        moderate_cfg = self.cfg.get("moderate_trend_ema_params", {})
        self.moderate_fast_ema = moderate_cfg.get("fast_ema_period", 12)
        self.moderate_mid_ema = moderate_cfg.get("mid_ema_period", 40)
        self.moderate_slow_ema = moderate_cfg.get("slow_ema_period", 150)
        
        # Override required warmup to ensure we have enough data
        self._required_warmup_bars_override = max(
            self.adx_period * 3,
            self.strong_slow_ema,
            self.moderate_slow_ema
        )

    def _calculate_ema(self, closes: Deque[float], period: int) -> float | None:
        """Calculate EMA for a given period. Returns None if insufficient data."""
        if len(closes) < period:
            return None
        
        closes_list = list(closes)
        multiplier = 2.0 / (period + 1)
        
        # Initialize EMA with SMA of first `period` bars
        ema = sum(closes_list[:period]) / period
        
        # Update EMA for remaining bars
        for i in range(period, len(closes_list)):
            ema = closes_list[i] * multiplier + ema * (1 - multiplier)
        
        return ema

    def _calculate_adx(self, price_data_history: Deque[PriceData]) -> float | None:
        """Calculate 14-bar ADX from price data history."""
        if len(price_data_history) < self.adx_period + 1:
            return None
        
        data_list = list(price_data_history)
        tr_values = []
        plus_dm_values = []
        minus_dm_values = []
        
        # Calculate True Range and Directional Movements
        for i in range(1, len(data_list)):
            curr = data_list[i]
            prev = data_list[i - 1]
            
            # True Range
            high_low = curr.high - curr.low
            high_close = abs(curr.high - prev.close)
            low_close = abs(curr.low - prev.close)
            tr = max(high_low, high_close, low_close)
            tr_values.append(tr)
            
            # Directional Movements
            up_move = curr.high - prev.high
            down_move = prev.low - curr.low
            
            plus_dm = 0.0
            minus_dm = 0.0
            
            if up_move > down_move and up_move > 0:
                plus_dm = up_move
            if down_move > up_move and down_move > 0:
                minus_dm = down_move
            
            plus_dm_values.append(plus_dm)
            minus_dm_values.append(minus_dm)
        
        if len(tr_values) < self.adx_period:
            return None
        
        # Calculate +DI and -DI
        atr = sum(tr_values[-self.adx_period:]) / self.adx_period
        plus_dm_sum = sum(plus_dm_values[-self.adx_period:])
        minus_dm_sum = sum(minus_dm_values[-self.adx_period:])
        
        if atr == 0:
            return None
        
        plus_di = (plus_dm_sum / atr) * 100
        minus_di = (minus_dm_sum / atr) * 100
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return None
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        
        # Smooth DX over adx_period bars using SMA
        dx_values = []
        for i in range(self.adx_period, len(data_list)):
            segment = data_list[i - self.adx_period + 1:i + 1]
            seg_tr = []
            seg_pdm = []
            seg_mdm = []
            
            for j in range(1, len(segment)):
                curr = segment[j]
                prev = segment[j - 1]
                
                tr = max(curr.high - curr.low, 
                        abs(curr.high - prev.close), 
                        abs(curr.low - prev.close))
                seg_tr.append(tr)
                
                up = curr.high - prev.high
                down = prev.low - curr.low
                
                pdm = up if (up > down and up > 0) else 0.0
                mdm = down if (down > up and down > 0) else 0.0
                
                seg_pdm.append(pdm)
                seg_mdm.append(mdm)
            
            seg_atr = sum(seg_tr) / len(seg_tr) if seg_tr else 1.0
            seg_pdm_sum = sum(seg_pdm)
            seg_mdm_sum = sum(seg_mdm)
            
            seg_plus_di = (seg_pdm_sum / seg_atr * 100) if seg_atr > 0 else 0.0
            seg_minus_di = (seg_mdm_sum / seg_atr * 100) if seg_atr > 0 else 0.0
            
            seg_di_sum = seg_plus_di + seg_minus_di
            seg_dx = abs(seg_plus_di - seg_minus_di) / seg_di_sum * 100 if seg_di_sum > 0 else 0.0
            
            dx_values.append(seg_dx)
        
        # ADX is SMA of DX values
        if len(dx_values) >= self.adx_period:
            adx = sum(dx_values[-self.adx_period:]) / self.adx_period
            return adx
        
        return None

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """
        Generate trading signals based on ADX trend strength and EMA crossovers.
        """
        signals = []
        
        for price_data in data:
            symbol = price_data.symbol
            
            # Check if we have sufficient history
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.strong_slow_ema:
                continue
            
            if symbol not in self.price_data_history or len(self.price_data_history[symbol]) < self.adx_period + 1:
                continue
            
            # Step 1: Calculate ADX
            adx = self._calculate_adx(self.price_data_history[symbol])
            if adx is None:
                continue
            
            # Step 2: Classify trend strength
            if adx > self.strong_trend_threshold:
                regime = "STRONG_TREND"
                fast_period = self.strong_fast_ema
                mid_period = self.strong_mid_ema
                slow_period = self.strong_slow_ema
            elif adx > self.trend_threshold:
                regime = "MODERATE_TREND"
                fast_period = self.moderate_fast_ema
                mid_period = self.moderate_mid_ema
                slow_period = self.moderate_slow_ema
            else:
                regime = "RANGING"
                continue  # Skip signals in ranging regime
            
            # Step 4: Calculate three EMAs
            closes = self.price_history[symbol]
            fast_ema = self._calculate_ema(closes, fast_period)
            mid_ema = self._calculate_ema(closes, mid_period)
            slow_ema = self._calculate_ema(closes, slow_period)
            
            if fast_ema is None or mid_ema is None or slow_ema is None:
                continue
            
            # Get previous bar EMAs for crossover detection
            if len(closes) < slow_period + 1:
                continue
            
            prev_closes = deque(list(closes)[:-1])
            prev_fast_ema = self._calculate_ema(prev_closes, fast_period)
            prev_slow_ema = self._calculate_ema(prev_closes, slow_period)
            
            if prev_fast_ema is None or prev_slow_ema is None:
                continue
            
            current_close = price_data.close
            
            # Step 5: Detect fast/slow EMA crossover
            upward_crossover = prev_fast_ema <= prev_slow_ema and fast_ema > slow_ema
            downward_crossover = prev_fast_ema >= prev_slow_ema and fast_ema < slow_ema
            
            # Step 7 & 6 & 8: Apply ADX gate, mid-EMA filter, and emit signals
            if adx < self.trend_threshold:
                continue  # ADX gate: skip if trend not strong enough
            
            if upward_crossover and current_close > mid_ema:
                # BUY signal
                strength = min(1.0, max(0.6, adx / 40.0))
                signal = MarketSignal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    strength=strength,
                    metadata={
                        "adx_value": adx,
                        "trend_regime": regime,
                        "fast_ema": fast_ema,
                        "mid_ema": mid_ema,
                        "slow_ema": slow_ema,
                        "ema_periods_used": (fast_period, mid_period, slow_period),
                        "crossover_type": "upward"
                    }
                )
                signals.append(signal)
            
            elif downward_crossover and current_close < mid_ema:
                # SELL signal
                strength = min(1.0, max(0.6, adx / 40.0))
                signal = MarketSignal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    strength=strength,
                    metadata={
                        "adx_value": adx,
                        "trend_regime": regime,
                        "fast_ema": fast_ema,
                        "mid_ema": mid_ema,
                        "slow_ema": slow_ema,
                        "ema_periods_used": (fast_period, mid_period, slow_period),
                        "crossover_type": "downward"
                    }
                )
                signals.append(signal)
        
        return signals


if __name__ == "__main__":
    # Smoke test: instantiate and run with synthetic ticks
    cfg = {
        "symbol": "SPY",
        "adx_config": {
            "adx_period": 14,
            "trend_threshold": 25,
            "strong_trend_threshold": 35,
            "weak_trend_lower_bound": 20,
        },
        "strong_trend_ema_params": {
            "fast_ema_period": 8,
            "mid_ema_period": 40,
            "slow_ema_period": 100,
        },
        "moderate_trend_ema_params": {
            "fast_ema_period": 12,
            "mid_ema_period": 40,
            "slow_ema_period": 150,
        },
    }
    
    algo = AdxMomentumFilterEmaAlgorithm(cfg=cfg, history_length=300)
    
    # Create synthetic price data for testing
    synthetic_data = []
    base_price = 450.0
    for i in range(320):
        price = base_price + (i * 0.1) + (5.0 if i % 10 < 5 else -5.0)
        pd = PriceData(
            symbol="SPY",
            timestamp=i,
            open=price,
            high=price + 2.0,
            low=price - 2.0,
            close=price,
            volume=1000000
        )
        synthetic_data.append(pd)
    
    # Feed data through algorithm
    for tick in synthetic_data:
        signals = algo.on_data([tick])
        if signals:
            for sig in signals:
                print(f"Tick {tick.timestamp}: {sig.type.name} SPY @ strength={sig.strength:.2f}, "
                      f"ADX={sig.metadata.get('adx_value', 0):.2f}, "
                      f"regime={sig.metadata.get('trend_regime')}")
    
    print("Smoke test completed successfully.")