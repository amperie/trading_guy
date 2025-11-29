# RSI Refactoring Summary

## Overview

RSI calculation methods have been moved from standalone functions in `utils/indicators.py` to static methods in the `TechnicalAnalyzer` class (`core/ta/analyzer.py`).

## What Changed

### New Structure

**Before:**
```python
from utils.indicators import calculate_rsi, calculate_rsi_series

rsi = calculate_rsi(prices, period=14)  # Returns: float
```

**After (Recommended):**
```python
from core.ta.analyzer import TechnicalAnalyzer

rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period=14)  # Returns: RSI dataclass
rsi_value = rsi_data.rsi  # Access the RSI value
```

**Backward Compatible (Still Works):**
```python
from utils.indicators import calculate_rsi, calculate_rsi_series

rsi = calculate_rsi(prices, period=14)  # Still returns: float (wrapper)
```

### New RSI Dataclass

The `TechnicalAnalyzer.calculate_rsi()` method now returns an `RSI` dataclass with additional information:

```python
@dataclass
class RSI:
    period: int          # RSI period (e.g., 14)
    rsi: float          # RSI value (0-100)
    avg_gain: float     # Average gain over period
    avg_loss: float     # Average loss over period
    rs: Optional[float] # Relative Strength (avg_gain / avg_loss, None if avg_loss is 0)
```

## Benefits

### 1. **Consistent API**
All technical indicators (EMA, MACD, RSI) are now in the same class:
```python
from core.ta.analyzer import TechnicalAnalyzer

ema = TechnicalAnalyzer.calculate_ema(prices, period=20)
macd = TechnicalAnalyzer.calculate_macd(prices)
rsi = TechnicalAnalyzer.calculate_rsi(prices, period=14)
```

### 2. **Additional Data Available**
The RSI dataclass provides more information for advanced strategies:
```python
rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period=14)

print(f"RSI: {rsi_data.rsi}")
print(f"Average Gain: {rsi_data.avg_gain}")
print(f"Average Loss: {rsi_data.avg_loss}")
print(f"RS (Relative Strength): {rsi_data.rs}")
```

### 3. **Better Organization**
Technical analysis methods are grouped in a dedicated module (`core/ta/analyzer.py`) rather than scattered across utility files.

## Migration Guide

### Option 1: Use TechnicalAnalyzer Directly (Recommended)

**Old Code:**
```python
from utils.indicators import calculate_rsi

rsi = calculate_rsi(prices, period=14)
if rsi < 30:
    # Oversold
    pass
```

**New Code:**
```python
from core.ta.analyzer import TechnicalAnalyzer

rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period=14)
if rsi_data and rsi_data.rsi < 30:
    # Oversold
    pass
```

### Option 2: Continue Using Wrappers (No Changes Needed)

The `utils/indicators.py` functions still work and maintain backward compatibility:

```python
from utils.indicators import calculate_rsi, calculate_rsi_series

# These still work exactly as before
rsi = calculate_rsi(prices, period=14)  # Returns float
rsi_series = calculate_rsi_series(prices, period=14)  # Returns list[float]
```

## Advanced Usage Examples

### Example 1: Using RS for Signal Strength

```python
from core.ta.analyzer import TechnicalAnalyzer
from core.algorithm import Algorithm

class AdvancedRSIAlgorithm(Algorithm):
    def on_data_logic(self, data):
        signals = []

        for pd in data:
            prices = list(self.price_history[pd.symbol])
            rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period=14)

            if rsi_data is None:
                continue

            # Use RS to modulate signal strength
            if rsi_data.rsi < 30:
                # Stronger signal if RS is very low (extreme oversold)
                strength = 90 if rsi_data.rs and rsi_data.rs < 0.5 else 75
                signals.append(MarketSignal(
                    type=SignalType.BUY,
                    symbol=pd.symbol,
                    strength=strength
                ))

        return signals
```

### Example 2: Tracking Average Gain/Loss

```python
rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period=14)

# Monitor momentum by tracking average gains vs losses
if rsi_data.avg_gain > rsi_data.avg_loss * 2:
    print("Strong bullish momentum")
elif rsi_data.avg_loss > rsi_data.avg_gain * 2:
    print("Strong bearish momentum")
```

### Example 3: Using RSI Series with Dataclass

```python
from core.ta.analyzer import TechnicalAnalyzer

rsi_series = TechnicalAnalyzer.calculate_rsi_series(prices, period=14)

# Each element is an RSI dataclass (or None)
for i, rsi_data in enumerate(rsi_series):
    if rsi_data is not None:
        print(f"Index {i}: RSI={rsi_data.rsi:.2f}, RS={rsi_data.rs:.2f}")
```

## Files Modified

1. **`core/ta/analyzer.py`**
   - Added `RSI` dataclass
   - Added `TechnicalAnalyzer.calculate_rsi()` static method
   - Added `TechnicalAnalyzer.calculate_rsi_series()` static method

2. **`utils/indicators.py`**
   - `calculate_rsi()` now wraps `TechnicalAnalyzer.calculate_rsi()`
   - `calculate_rsi_series()` now wraps `TechnicalAnalyzer.calculate_rsi_series()`
   - Wrappers extract just the `.rsi` value for backward compatibility

3. **`examples/rsi_example.py`**
   - Updated to show both usage patterns
   - Added `RSIWithTechnicalAnalyzer` example class

## Testing

### All Tests Passing

**Original tests (50 tests):** ✅ All passing
```bash
pytest tests/unit/test_indicators.py -v
# 50 passed in 1.70s
```

**New TechnicalAnalyzer tests (16 tests):** ✅ All passing
```bash
pytest tests/unit/test_technical_analyzer.py -v
# 16 passed in 0.11s
```

**Total: 66 tests, all passing**

### Test Coverage

- ✅ RSI dataclass structure
- ✅ RSI values calculation
- ✅ RSI series calculation
- ✅ Backward compatibility wrappers
- ✅ Integration with existing indicator tests
- ✅ MACD and EMA in TechnicalAnalyzer

## Recommendations

1. **New Code:** Use `TechnicalAnalyzer.calculate_rsi()` directly to access the full RSI dataclass
2. **Existing Code:** No changes needed - wrappers maintain full backward compatibility
3. **Advanced Strategies:** Leverage `avg_gain`, `avg_loss`, and `rs` for more sophisticated logic

## Summary

- ✅ RSI methods moved to `TechnicalAnalyzer` class
- ✅ New `RSI` dataclass provides additional information
- ✅ Full backward compatibility maintained via wrappers
- ✅ All 66 tests passing (50 original + 16 new)
- ✅ Consistent API across all technical indicators
- ✅ Better code organization

No breaking changes - all existing code continues to work!
