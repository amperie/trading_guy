"""
Example usage of all technical indicators in TechnicalAnalyzer.

This script demonstrates how to calculate and use:
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)
- Williams %R

All indicators return dataclasses with complete information.
"""

from collections import deque
from core.ta import (
    TechnicalAnalyzer,
    MACD,
    RSI,
    BollingerBands,
    Stochastic,
    ATR,
    ADX,
    CCI,
    WilliamsR
)


def example_usage():
    """Demonstrate usage of all technical indicators."""

    # Sample price data (50 periods)
    prices = deque([100.0 + i * 0.5 + (i % 5) * 0.3 for i in range(50)])
    high = deque([p + 1.0 for p in prices])
    low = deque([p - 1.0 for p in prices])

    print("=" * 60)
    print("Technical Indicators Example")
    print("=" * 60)

    # 1. RSI (Relative Strength Index)
    print("\n1. RSI (Relative Strength Index)")
    print("-" * 60)
    rsi: RSI = TechnicalAnalyzer.calculate_rsi(prices, period=14)
    if rsi:
        print(f"RSI: {rsi.rsi:.2f}")
        if rsi.avg_gain is not None:
            print(f"Avg Gain: {rsi.avg_gain:.4f}")
        if rsi.avg_loss is not None:
            print(f"Avg Loss: {rsi.avg_loss:.4f}")
        print(f"Interpretation: {'Overbought' if rsi.rsi > 70 else 'Oversold' if rsi.rsi < 30 else 'Neutral'}")

    # 2. MACD (Moving Average Convergence Divergence)
    print("\n2. MACD (Moving Average Convergence Divergence)")
    print("-" * 60)
    macd: MACD = TechnicalAnalyzer.calculate_macd(prices, slow_period=26, fast_period=12, signal_period=9)
    if macd:
        print(f"MACD Line: {macd.macd:.4f}")
        print(f"Signal Line: {macd.signal:.4f}")
        print(f"Histogram: {macd.histogram:.4f}")
        print(f"Signal: {'Bullish' if macd.histogram > 0 else 'Bearish'}")

    # 3. Bollinger Bands
    print("\n3. Bollinger Bands")
    print("-" * 60)
    bb: BollingerBands = TechnicalAnalyzer.calculate_bollinger_bands(prices, period=20, std_dev=2.0)
    if bb:
        print(f"Upper Band: {bb.upper:.2f}")
        print(f"Middle Band (SMA): {bb.middle:.2f}")
        print(f"Lower Band: {bb.lower:.2f}")
        print(f"Current Price: {bb.current_price:.2f}")
        print(f"Bandwidth: {bb.bandwidth:.4f}")
        print(f"%B: {bb.percent_b:.4f} ({'Above upper' if bb.percent_b > 1 else 'Below lower' if bb.percent_b < 0 else 'Within bands'})")

    # 4. Stochastic Oscillator
    print("\n4. Stochastic Oscillator")
    print("-" * 60)
    stoch: Stochastic = TechnicalAnalyzer.calculate_stochastic(high, low, prices, fastk_period=14, slowk_period=3, slowd_period=3)
    if stoch:
        print(f"%K (Fast): {stoch.slowk:.2f}")
        print(f"%D (Slow): {stoch.slowd:.2f}")
        print(f"Interpretation: {'Overbought' if stoch.slowk > 80 else 'Oversold' if stoch.slowk < 20 else 'Neutral'}")
        print(f"Signal: {'Bullish crossover' if stoch.slowk > stoch.slowd else 'Bearish crossover'}")

    # 5. ATR (Average True Range)
    print("\n5. ATR (Average True Range)")
    print("-" * 60)
    atr: ATR = TechnicalAnalyzer.calculate_atr(high, low, prices, period=14)
    if atr:
        print(f"ATR: {atr.atr:.2f}")
        print(f"Volatility: {'High' if atr.atr > 2.0 else 'Low' if atr.atr < 1.0 else 'Moderate'}")

    # 6. ADX (Average Directional Index)
    print("\n6. ADX (Average Directional Index)")
    print("-" * 60)
    adx: ADX = TechnicalAnalyzer.calculate_adx(high, low, prices, period=14)
    if adx:
        print(f"ADX: {adx.adx:.2f}")
        print(f"+DI: {adx.plus_di:.2f}")
        print(f"-DI: {adx.minus_di:.2f}")
        print(f"Trend Strength: {'Strong' if adx.adx > 25 else 'Weak' if adx.adx < 20 else 'Moderate'}")
        print(f"Direction: {'Uptrend' if adx.plus_di > adx.minus_di else 'Downtrend'}")

    # 7. CCI (Commodity Channel Index)
    print("\n7. CCI (Commodity Channel Index)")
    print("-" * 60)
    cci: CCI = TechnicalAnalyzer.calculate_cci(high, low, prices, period=14)
    if cci:
        print(f"CCI: {cci.cci:.2f}")
        print(f"Interpretation: {'Overbought' if cci.cci > 100 else 'Oversold' if cci.cci < -100 else 'Neutral'}")

    # 8. Williams %R
    print("\n8. Williams %R")
    print("-" * 60)
    willr: WilliamsR = TechnicalAnalyzer.calculate_willr(high, low, prices, period=14)
    if willr:
        print(f"Williams %R: {willr.willr:.2f}")
        print(f"Interpretation: {'Overbought' if willr.willr > -20 else 'Oversold' if willr.willr < -80 else 'Neutral'}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def example_algorithm_usage():
    """Example of how to use indicators in a trading algorithm."""

    print("\n\nAlgorithm Usage Example")
    print("=" * 60)

    # Simulate price history (need at least 35 periods for MACD)
    price_history = deque([100.0 + i * 0.2 for i in range(40)], maxlen=50)
    high_history = deque([p + 0.5 for p in price_history], maxlen=50)
    low_history = deque([p - 0.5 for p in price_history], maxlen=50)

    # Calculate multiple indicators for decision making
    rsi = TechnicalAnalyzer.calculate_rsi(price_history, period=14)
    bb = TechnicalAnalyzer.calculate_bollinger_bands(price_history, period=20)
    macd = TechnicalAnalyzer.calculate_macd(price_history)
    adx = TechnicalAnalyzer.calculate_adx(high_history, low_history, price_history)

    # Simple trading logic using multiple indicators
    print("\nTrading Decision Logic:")
    print("-" * 60)

    if rsi and bb and macd and adx:
        print(f"RSI: {rsi.rsi:.2f} ({'Oversold' if rsi.rsi < 30 else 'Overbought' if rsi.rsi > 70 else 'Neutral'})")
        print(f"Price vs BB: %B = {bb.percent_b:.2f} ({'Below bands' if bb.percent_b < 0 else 'Above bands' if bb.percent_b > 1 else 'In bands'})")
        print(f"MACD Histogram: {macd.histogram:.4f} ({'Bullish' if macd.histogram > 0 else 'Bearish'})")
        print(f"ADX Trend: {adx.adx:.2f} ({'Strong' if adx.adx > 25 else 'Weak'})")

        # Buy signal: RSI oversold + price below lower band + bullish MACD + strong trend
        buy_signal = (
            rsi.rsi < 30 and
            bb.percent_b < 0.2 and
            macd.histogram > 0 and
            adx.adx > 25
        )

        # Sell signal: RSI overbought + price above upper band + bearish MACD
        sell_signal = (
            rsi.rsi > 70 and
            bb.percent_b > 0.8 and
            macd.histogram < 0
        )

        print(f"\nDecision: {'BUY' if buy_signal else 'SELL' if sell_signal else 'HOLD'}")
    else:
        print("Insufficient data for analysis")

    print("=" * 60)


if __name__ == "__main__":
    example_usage()
    example_algorithm_usage()
