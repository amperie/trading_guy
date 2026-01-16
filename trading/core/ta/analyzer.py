from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Union, Optional
import talib as ta
import numpy as np


@dataclass
class MACD:
    slow_period: int
    fast_period: int
    signal_period: int
    slow_ema: float
    fast_ema: float
    macd: float
    signal: float
    histogram: float
    last_macd: MACD = None


@dataclass
class RSI:
    period: int
    rsi: float
    avg_gain: float = None
    avg_loss: float = None
    rs: float = None  # Relative Strength (None if avg_loss is 0)


@dataclass
class BollingerBands:
    period: int
    std_dev: float  # Number of standard deviations (typically 2)
    upper: float
    middle: float  # SMA
    lower: float
    bandwidth: float  # (upper - lower) / middle
    percent_b: float  # Where price is relative to bands: (price - lower) / (upper - lower)
    current_price: float


@dataclass
class Stochastic:
    fastk_period: int
    slowk_period: int
    slowd_period: int
    slowk: float  # %K line (slow)
    slowd: float  # %D line (signal)


@dataclass
class ATR:
    period: int
    atr: float  # Average True Range


@dataclass
class ADX:
    period: int
    adx: float  # Average Directional Index
    plus_di: float  # +DI (Positive Directional Indicator)
    minus_di: float  # -DI (Negative Directional Indicator)


@dataclass
class CCI:
    period: int
    cci: float  # Commodity Channel Index


@dataclass
class WilliamsR:
    period: int
    willr: float  # Williams %R


@dataclass
class EMA:
    period: int
    ema: float  # Exponential Moving Average


@dataclass
class SMA:
    period: int
    sma: float  # Simple Moving Average

class TechnicalAnalyzer:

    @staticmethod
    def calculate_macd(prices: deque[float], slow_period: int=26, fast_period: int=12,
                       signal_period: int=9, calculate_prev_macd: bool = False) -> Union[None, MACD]:
        """Calculate MACD, Signal, Histogram using TAlib"""

        if len(prices) < slow_period:
            return None

        pr = np.array(prices)
        macd = ta.MACD(
            pr, fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period, )

        # If MACD is NaN, return since we don't have enough data yet
        if np.isnan(macd[0][-1].item()).item():
            return None

        if calculate_prev_macd:
            # If previous MACD is NaN, return since we don't have enough data yet
            if np.isnan(macd[0][-2].item()).item():
                return None

            prev_macd = MACD(
                slow_period=slow_period,
                fast_period=fast_period,
                signal_period=signal_period,
                slow_ema=0,
                fast_ema=0,
                macd=macd[0][-2].item(),
                signal=macd[1][-2].item(),
                histogram=macd[2][-2].item(),
            )
        else:
            prev_macd = None

        ret_val = MACD(
            slow_period=slow_period,
            fast_period=fast_period,
            signal_period=signal_period,
            slow_ema=0,
            fast_ema=0,
            macd=macd[0][-1].item(),
            signal=macd[1][-1].item(),
            histogram=macd[2][-1].item(),
            last_macd=prev_macd
        )
        return ret_val

    @staticmethod
    def calculate_rsi(prices: deque[float], period: int = 14) -> Union[None, RSI]:

        if len(prices) < period + 1:  # Need period + 1 to calculate first change
            return None

        pr = np.array(prices)
        rsi = ta.RSI(pr, period)

        return RSI(
            period=period,
            rsi=rsi[-1].item(),
        )

    @staticmethod
    def calculate_bollinger_bands(
        prices: deque[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Union[None, BollingerBands]:
        """Calculate Bollinger Bands using TAlib"""

        if len(prices) < period:
            return None

        pr = np.array(prices)
        upper, middle, lower = ta.BBANDS(
            pr,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0  # SMA
        )

        # Check if we have valid data
        if np.isnan(upper[-1]):
            return None

        # Calculate bandwidth and %B
        current_price = prices[-1]
        bandwidth = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else 0
        percent_b = (current_price - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 0.5

        return BollingerBands(
            period=period,
            std_dev=std_dev,
            upper=upper[-1].item(),
            middle=middle[-1].item(),
            lower=lower[-1].item(),
            bandwidth=bandwidth,
            percent_b=percent_b,
            current_price=current_price
        )

    @staticmethod
    def calculate_stochastic(
        high: deque[float],
        low: deque[float],
        close: deque[float],
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3
    ) -> Union[None, Stochastic]:
        """Calculate Stochastic Oscillator using TAlib"""

        if len(close) < fastk_period:
            return None

        h = np.array(high)
        l = np.array(low)
        c = np.array(close)

        slowk, slowd = ta.STOCH(
            h, l, c,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,  # SMA
            slowd_period=slowd_period,
            slowd_matype=0   # SMA
        )

        # Check if we have valid data
        if np.isnan(slowk[-1]):
            return None

        return Stochastic(
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
            slowk=slowk[-1].item(),
            slowd=slowd[-1].item()
        )

    @staticmethod
    def calculate_atr(
        high: deque[float],
        low: deque[float],
        close: deque[float],
        period: int = 14
    ) -> Union[None, ATR]:
        """Calculate Average True Range using TAlib"""

        if len(close) < period:
            return None

        h = np.array(high)
        l = np.array(low)
        c = np.array(close)

        atr = ta.ATR(h, l, c, timeperiod=period)

        # Check if we have valid data
        if np.isnan(atr[-1]):
            return None

        return ATR(
            period=period,
            atr=atr[-1].item()
        )

    @staticmethod
    def calculate_adx(
        high: deque[float],
        low: deque[float],
        close: deque[float],
        period: int = 14
    ) -> Union[None, ADX]:
        """Calculate Average Directional Index using TAlib"""

        if len(close) < period:
            return None

        h = np.array(high)
        l = np.array(low)
        c = np.array(close)

        adx = ta.ADX(h, l, c, timeperiod=period)
        plus_di = ta.PLUS_DI(h, l, c, timeperiod=period)
        minus_di = ta.MINUS_DI(h, l, c, timeperiod=period)

        # Check if we have valid data
        if np.isnan(adx[-1]):
            return None

        return ADX(
            period=period,
            adx=adx[-1].item(),
            plus_di=plus_di[-1].item(),
            minus_di=minus_di[-1].item()
        )

    @staticmethod
    def calculate_cci(
        high: deque[float],
        low: deque[float],
        close: deque[float],
        period: int = 14
    ) -> Union[None, CCI]:
        """Calculate Commodity Channel Index using TAlib"""

        if len(close) < period:
            return None

        h = np.array(high)
        l = np.array(low)
        c = np.array(close)

        cci = ta.CCI(h, l, c, timeperiod=period)

        # Check if we have valid data
        if np.isnan(cci[-1]):
            return None

        return CCI(
            period=period,
            cci=cci[-1].item()
        )

    @staticmethod
    def calculate_willr(
        high: deque[float],
        low: deque[float],
        close: deque[float],
        period: int = 14
    ) -> Union[None, WilliamsR]:
        """Calculate Williams %R using TAlib"""

        if len(close) < period:
            return None

        h = np.array(high)
        l = np.array(low)
        c = np.array(close)

        willr = ta.WILLR(h, l, c, timeperiod=period)

        # Check if we have valid data
        if np.isnan(willr[-1]):
            return None

        return WilliamsR(
            period=period,
            willr=willr[-1].item()
        )

    @staticmethod
    def calculate_ema(
        prices: deque[float],
        period: int = 20
    ) -> Union[None, EMA]:
        """Calculate Exponential Moving Average using TAlib"""

        if len(prices) < period:
            return None

        pr = np.array(prices)
        ema = ta.EMA(pr, timeperiod=period)

        # Check if we have valid data
        if np.isnan(ema[-1]):
            return None

        return EMA(
            period=period,
            ema=ema[-1].item()
        )

    @staticmethod
    def calculate_sma(
        prices: deque[float],
        period: int = 20
    ) -> Union[None, SMA]:
        """Calculate Simple Moving Average using TAlib"""

        if len(prices) < period:
            return None

        pr = np.array(prices)
        sma = ta.SMA(pr, timeperiod=period)

        # Check if we have valid data
        if np.isnan(sma[-1]):
            return None

        return SMA(
            period=period,
            sma=sma[-1].item()
        )


class TechnicalAnalyzerOLD:

    @staticmethod
    def calculate_ema_series(prices: deque[float], period: int) -> Union[list[None], list[float]]:
        """Calculate EMA series"""
        if len(prices) < period:
            return [None] * len(prices)

        price_list = list(prices)

        multiplier = 2 / (period + 1)
        ema_values = [None] * (period - 1)

        ema = sum(price_list[:period]) / period
        ema_values.append(ema)

        for price in price_list[period:]:
            ema = (price - ema) * multiplier + ema
            ema_values.append(ema)

        return ema_values

    @staticmethod
    def calculate_ema(prices: deque[float], period: int) -> Union[None, float]:
        """
        Calculate EMA for a list of prices.
        Returns the current EMA value.
        """
        if len(prices) < period:
            return None  # Not enough data

        price_list = list(prices)
        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Start with SMA for first value
        ema = sum(price_list[:period]) / period

        # Apply EMA formula for remaining prices
        for price in price_list[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def calculate_macd(prices: deque[float], slow_period: int=26, fast_period: int=12,
                       signal_period: int=9, calculate_prev_macd: bool = False) -> Union[None, MACD]:

        """Calculate MACD, Signal, Histogram"""
        if len(prices) < slow_period:
          return None

        # Calculate EMAs
        ema_fast = TechnicalAnalyzer.calculate_ema(prices, fast_period)
        ema_slow = TechnicalAnalyzer.calculate_ema(prices, slow_period)

        if ema_fast is None or ema_slow is None:
          return None

        macd_line = ema_fast - ema_slow

        # Calculate MACD series for signal line
        ema_fast_series = TechnicalAnalyzer.calculate_ema_series(prices, fast_period)
        ema_slow_series = TechnicalAnalyzer.calculate_ema_series(prices, slow_period)

        macd_series = []
        for fast, slow in zip(ema_fast_series, ema_slow_series):
          if fast is not None and slow is not None:
              macd_series.append(fast - slow)

        if len(macd_series) < signal_period:
          return None

        signal_line = TechnicalAnalyzer.calculate_ema(macd_series, signal_period)
        histogram = macd_line - signal_line if signal_line is not None else None

        if calculate_prev_macd:
            price_list = list(prices)
            prev_macd = TechnicalAnalyzer.calculate_macd(price_list[0:-1], slow_period, fast_period, signal_period, False)
        else:
            prev_macd = None

        ret_val = MACD(
            slow_period=slow_period,
            fast_period=fast_period,
            signal_period=signal_period,
            slow_ema=ema_slow,
            fast_ema=ema_fast,
            macd=macd_line,
            signal=signal_line,
            histogram=histogram,
            last_macd=prev_macd
        )
        return ret_val

    @staticmethod
    def calculate_rsi(prices: deque[float], period: int = 14) -> Union[None, RSI]:
        """
        Calculate RSI (Relative Strength Index).

        RSI is a momentum oscillator that measures the speed and magnitude of
        price changes. It oscillates between 0 and 100, with values above 70
        typically considered overbought and below 30 oversold.

        Formula:
        1. Calculate price changes (deltas)
        2. Separate gains and losses
        3. Calculate average gain and loss using Wilder's smoothing
        4. RS = Average Gain / Average Loss
        5. RSI = 100 - (100 / (1 + RS))

        Args:
            prices: List of price values
            period: RSI period (default: 14)

        Returns:
            RSI dataclass with rsi, avg_gain, avg_loss, rs values, or None if insufficient data

        Example:
            >>> prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            ...           45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
            >>> rsi_data = TechnicalAnalyzer.calculate_rsi(prices, 14)
            >>> 0 <= rsi_data.rsi <= 100
            True
        """
        if len(prices) < period + 1:  # Need period + 1 to calculate first change
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]

        # Calculate initial average gain and loss (simple average)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Apply Wilder's smoothing for remaining periods
        # Wilder's smoothing: avg = ((prev_avg * (period - 1)) + current) / period
        for i in range(period, len(gains)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        # Calculate RS and RSI
        if avg_loss == 0:
            rs = None
            rsi = 100.0  # No losses mean RSI = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        return RSI(
            period=period,
            rsi=rsi,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            rs=rs
        )

    @staticmethod
    def calculate_rsi_series(prices: list[float], period: int = 14) -> list[Optional[RSI]]:
        """
        Calculate RSI for each point in the price series.

        Args:
            prices: List of price values
            period: RSI period (default: 14)

        Returns:
            List of RSI dataclass objects (None for points with insufficient data)

        Example:
            >>> prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            ...           45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
            >>> rsi_series = TechnicalAnalyzer.calculate_rsi_series(prices, 14)
            >>> len(rsi_series) == len(prices)
            True
        """
        if len(prices) < period + 1:
            return [None] * len(prices)

        rsi_values = [None] * period  # First 'period' values are None

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]

        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Calculate first RSI value
        if avg_loss == 0:
            rsi = 100.0
            rs = None
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_values.append(RSI(
            period=period,
            rsi=rsi,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            rs=rs
        ))

        # Apply Wilder's smoothing for remaining periods
        for i in range(period, len(gains)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

            if avg_loss == 0:
                rsi = 100.0
                rs = None
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

            rsi_values.append(RSI(
                period=period,
                rsi=rsi,
                avg_gain=avg_gain,
                avg_loss=avg_loss,
                rs=rs
            ))

        return rsi_values