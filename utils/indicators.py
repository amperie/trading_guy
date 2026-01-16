"""
Technical indicators for trading algorithms.

This module provides implementations of common technical indicators
including EMA (Exponential Moving Average), MACD (Moving Average
Convergence Divergence), and RSI (Relative Strength Index).

NOTE: This module provides backward-compatible wrapper functions.
The actual implementations are in core.ta.analyzer.TechnicalAnalyzer.
"""

from typing import Optional, Tuple
from trading.core.ta import TechnicalAnalyzer


def calculate_ema(prices: list, period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average for a list of prices.

    Args:
        prices: List of price values
        period: EMA period (e.g., 12, 26)

    Returns:
        Current EMA value, or None if insufficient data

    Example:
        >>> prices = [22.27, 22.19, 22.08, 22.17, 22.18]
        >>> ema = calculate_ema(prices, 5)
        >>> round(ema, 2)
        22.18
    """
    if len(prices) < period:
        return None

    # Calculate multiplier: 2 / (period + 1)
    multiplier = 2 / (period + 1)

    # Initialize with Simple Moving Average
    ema = sum(prices[:period]) / period

    # Apply EMA formula for remaining prices
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_ema_series(prices: list, period: int) -> list[Optional[float]]:
    """
    Calculate EMA for each point in the price series.

    Args:
        prices: List of price values
        period: EMA period

    Returns:
        List of EMA values (None for points with insufficient data)

    Example:
        >>> prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13]
        >>> ema_series = calculate_ema_series(prices, 3)
        >>> len(ema_series) == len(prices)
        True
    """
    if len(prices) < period:
        return [None] * len(prices)

    multiplier = 2 / (period + 1)
    ema_values = [None] * (period - 1)

    # Initialize with SMA
    ema = sum(prices[:period]) / period
    ema_values.append(ema)

    # Calculate EMA for remaining points
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
        ema_values.append(ema)

    return ema_values


def calculate_macd(
    prices: list,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD Components:
    - MACD Line = EMA(fast) - EMA(slow)
    - Signal Line = EMA(MACD Line, signal_period)
    - Histogram = MACD Line - Signal Line

    Args:
        prices: List of price values
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) or None if insufficient data.
        signal_line and histogram may be None if insufficient data for signal calculation.

    Example:
        >>> prices = [22.27, 22.19, 22.08, 22.17, 22.18] + [22.0] * 30
        >>> macd, signal, histogram = calculate_macd(prices, 12, 26, 9)
        >>> isinstance(macd, float)
        True
    """
    if len(prices) < slow_period:
        return None

    # Calculate fast and slow EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    if ema_fast is None or ema_slow is None:
        return None

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # To calculate signal line, we need the MACD series
    ema_fast_series = calculate_ema_series(prices, fast_period)
    ema_slow_series = calculate_ema_series(prices, slow_period)

    # Calculate MACD series
    macd_series = []
    for fast, slow in zip(ema_fast_series, ema_slow_series):
        if fast is not None and slow is not None:
            macd_series.append(fast - slow)

    # Calculate signal line (EMA of MACD)
    if len(macd_series) < signal_period:
        return (macd_line, None, None)

    signal_line = calculate_ema(macd_series, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line if signal_line is not None else None

    return (macd_line, signal_line, histogram)


def calculate_macd_series(
    prices: list,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """
    Calculate MACD series for all price points.

    Args:
        prices: List of price values
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        Tuple of three lists: (macd_series, signal_series, histogram_series)
    """
    # Calculate EMA series
    ema_fast_series = calculate_ema_series(prices, fast_period)
    ema_slow_series = calculate_ema_series(prices, slow_period)

    # Calculate MACD series
    macd_series = []
    for fast, slow in zip(ema_fast_series, ema_slow_series):
        if fast is not None and slow is not None:
            macd_series.append(fast - slow)
        else:
            macd_series.append(None)

    # Calculate signal line series
    signal_series = [None] * len(prices)
    histogram_series = [None] * len(prices)

    # Find first non-None MACD value
    valid_macd = [m for m in macd_series if m is not None]

    if len(valid_macd) >= signal_period:
        # Calculate signal line starting from where we have enough MACD data
        start_idx = next(i for i, m in enumerate(macd_series) if m is not None)
        signal_ema_series = calculate_ema_series(valid_macd, signal_period)

        # Map back to original indices
        for i, sig in enumerate(signal_ema_series):
            idx = start_idx + i
            if idx < len(signal_series):
                signal_series[idx] = sig
                if sig is not None and macd_series[idx] is not None:
                    histogram_series[idx] = macd_series[idx] - sig

    return (macd_series, signal_series, histogram_series)


def calculate_rsi(prices: list, period: int = 14) -> Optional[float]:
    """
    Calculate RSI (Relative Strength Index).

    NOTE: This is a wrapper function for backward compatibility.
    The actual implementation is in TechnicalAnalyzer.calculate_rsi().

    RSI is a momentum oscillator that measures the speed and magnitude of
    price changes. It oscillates between 0 and 100, with values above 70
    typically considered overbought and below 30 oversold.

    Args:
        prices: List of price values
        period: RSI period (default: 14)

    Returns:
        Current RSI value (0-100), or None if insufficient data

    Example:
        >>> prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        ...           45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        >>> rsi = calculate_rsi(prices, 14)
        >>> 0 <= rsi <= 100
        True
    """
    rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period)
    return rsi_data.rsi if rsi_data is not None else None


def calculate_rsi_series(prices: list, period: int = 14) -> list[Optional[float]]:
    """
    Calculate RSI for each point in the price series.

    NOTE: This is a wrapper function for backward compatibility.
    The actual implementation is in TechnicalAnalyzer.calculate_rsi_series().

    Args:
        prices: List of price values
        period: RSI period (default: 14)

    Returns:
        List of RSI values (None for points with insufficient data)

    Example:
        >>> prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        ...           45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        >>> rsi_series = calculate_rsi_series(prices, 14)
        >>> len(rsi_series) == len(prices)
        True
    """
    rsi_data_series = TechnicalAnalyzer.calculate_rsi_series(prices, period)
    # Extract just the RSI values for backward compatibility
    return [rsi_data.rsi if rsi_data is not None else None for rsi_data in rsi_data_series]
