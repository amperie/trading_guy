"""
Test EMA function in SpyTrendSwitchAlgorithm
"""
from trading.core.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm

# Create algorithm instance
alg = SpyTrendSwitchAlgorithm()

# Test data: simple price sequence
prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Test with window = 5
window = 5

sma = alg._sma(prices, window)
ema = alg._ema(prices, window)

print("Test EMA vs SMA")
print("=" * 60)
print(f"Price data: {prices}")
print(f"Window: {window}")
print()
print(f"SMA({window}): {sma:.4f}")
print(f"EMA({window}): {ema:.4f}")
print()

# EMA reacts faster to recent price changes
print("EMA characteristics:")
print("- Gives more weight to recent prices")
print("- Reacts faster to price changes than SMA")
print("- Smoothing factor k = 2/(N+1) = {:.4f}".format(2/(window+1)))
print()

# Test with longer sequence to show difference
prices_trend = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                115, 120, 125, 130]  # Strong uptrend at end

sma_trend = alg._sma(prices_trend, 10)
ema_trend = alg._ema(prices_trend, 10)

print("=" * 60)
print("Test with trending data (strong uptrend at end):")
print(f"Price data: {prices_trend}")
print(f"Window: 10")
print()
print(f"SMA(10): {sma_trend:.4f}")
print(f"EMA(10): {ema_trend:.4f}")
print()
print("Notice: EMA is higher than SMA because it reacts faster")
print("        to the recent strong uptrend (115, 120, 125, 130)")
print()

# Test edge cases
print("=" * 60)
print("Edge cases:")
print(f"Insufficient data (window=10, len=5): {alg._ema([1,2,3,4,5], 10)}")
print(f"Exact window size (window=5, len=5): {alg._ema([10,11,12,13,14], 5):.4f}")
print("=" * 60)
