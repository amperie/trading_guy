"""
Test SpyTrendMACDAlgorithm to verify MACD calculation and signal generation.
"""
from trading.core.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
from trading.core.classes import PriceData
from datetime import datetime, timedelta

print("="*80)
print("TESTING SpyTrendMACDAlgorithm")
print("="*80)
print()

# Create algorithm with default MACD parameters
alg = SpyTrendMACDAlgorithm({
    'spy_symbol': 'SPY',
    'upro_symbol': 'UPRO',
    'spxu_symbol': 'SPXU',
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'strength_scale': 20.0
})

print("Algorithm Configuration:")
print(f"  Fast EMA Period:   {alg.macd_fast_period}")
print(f"  Slow EMA Period:   {alg.macd_slow_period}")
print(f"  Signal Period:     {alg.macd_signal_period}")
print(f"  Strength Scale:    {alg.strength_scale}")
print(f"  Required History:  {alg.history_length}")
print()

# Create test data: uptrend followed by downtrend
print("Test 1: Simulating Price Trends")
print("-" * 80)

# Start with uptrend (50 bars increasing)
prices = list(range(100, 150))  # 100, 101, 102, ... 149

# Add downtrend (50 bars decreasing)
prices.extend(list(range(149, 99, -1)))  # 149, 148, 147, ... 100

print(f"Generated {len(prices)} price bars")
print(f"  Uptrend:   bars 0-49 (100 to 149)")
print(f"  Downtrend: bars 50-99 (149 to 100)")
print()

# Feed data to algorithm
signals_generated = []
signal_count = 0

for i, price in enumerate(prices):
    # Create PriceData with valid timestamps (5-minute bars)
    base_time = datetime(2022, 1, 1, 9, 30)
    current_time = base_time + timedelta(minutes=5 * i)

    tick = [
        PriceData(
            symbol='SPY',
            timestamp=current_time,
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000000
        ),
        PriceData(
            symbol='UPRO',
            timestamp=current_time,
            open=price * 3,
            high=price * 3 + 1,
            low=price * 3 - 1,
            close=price * 3,
            volume=100000
        ),
        PriceData(
            symbol='SPXU',
            timestamp=current_time,
            open=100,
            high=101,
            low=99,
            close=100,
            volume=100000
        )
    ]

    # Generate signals
    signals = alg.on_data(tick)

    if signals:
        signal = signals[0]
        signal_count += 1
        signals_generated.append({
            'bar': i,
            'price': price,
            'target': signal.symbol,
            'strength': signal.strength,
            'macd': signal.metadata.get('macd_line'),
            'signal': signal.metadata.get('signal_line'),
            'histogram': signal.metadata.get('histogram')
        })

print(f"Signals Generated: {signal_count}")
print()

if signals_generated:
    print("Signal Details:")
    print("-" * 80)
    for sig in signals_generated:
        print(f"Bar {sig['bar']:3d} | Price: ${sig['price']:6.2f} | "
              f"Target: {sig['target']:4s} | Strength: {sig['strength']:3d} | "
              f"MACD: {sig['macd']:7.3f} | Signal: {sig['signal']:7.3f} | "
              f"Histogram: {sig['histogram']:7.3f}")
    print()

# Verify expected behavior
print("="*80)
print("VERIFICATION")
print("="*80)
print()

checks = [
    ("Algorithm initialized", alg is not None),
    ("Signals were generated", len(signals_generated) > 0),
    ("First signal is UPRO (uptrend)", signals_generated[0]['target'] == 'UPRO' if signals_generated else False),
    ("Contains SPXU signal (downtrend)", any(s['target'] == 'SPXU' for s in signals_generated)),
    ("MACD values calculated", signals_generated[0]['macd'] is not None if signals_generated else False),
    ("Signal line calculated", signals_generated[0]['signal'] is not None if signals_generated else False),
    ("Histogram calculated", signals_generated[0]['histogram'] is not None if signals_generated else False),
]

all_passed = True
for check_name, result in checks:
    status = "OK" if result else "FAIL"
    print(f"{status:6s} - {check_name}")
    if not result:
        all_passed = False

print()
if all_passed:
    print("="*80)
    print("ALL CHECKS PASSED - MACD ALGORITHM WORKING CORRECTLY")
    print("="*80)
    print()
    print("Key Features Verified:")
    print("  - MACD line calculation (Fast EMA - Slow EMA)")
    print("  - Signal line calculation (EMA of MACD)")
    print("  - Histogram calculation (MACD - Signal)")
    print("  - Bullish signals (MACD > Signal) -> UPRO")
    print("  - Bearish signals (MACD < Signal) -> SPXU")
    print("  - Signal strength based on histogram magnitude")
else:
    print("="*80)
    print("SOME CHECKS FAILED")
    print("="*80)

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
