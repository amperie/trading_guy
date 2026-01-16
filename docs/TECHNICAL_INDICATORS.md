# Technical Indicators Documentation

## Overview

The `TechnicalAnalyzer` class provides battle-tested implementations of common technical indicators using TAlib. Each indicator returns a dataclass with complete information including intermediate calculations.

## Available Indicators

### 1. EMA (Exponential Moving Average)
**Dataclass:** `EMA`

```python
from trading.core.ta import TechnicalAnalyzer, EMA

ema: EMA = TechnicalAnalyzer.calculate_ema(prices, period=20)
# Returns: EMA(period=20, ema=105.45)
```

**Fields:**
- `period`: EMA period
- `ema`: Exponential Moving Average value

**Interpretation:**
- EMA gives more weight to recent prices
- More responsive to price changes than SMA
- Common periods: 9, 12, 20, 26, 50, 100, 200
- Price > EMA: Bullish signal
- Price < EMA: Bearish signal

---

### 2. SMA (Simple Moving Average)
**Dataclass:** `SMA`

```python
from trading.core.ta import TechnicalAnalyzer, SMA

sma: SMA = TechnicalAnalyzer.calculate_sma(prices, period=20)
# Returns: SMA(period=20, sma=104.50)
```

**Fields:**
- `period`: SMA period
- `sma`: Simple Moving Average value

**Interpretation:**
- SMA treats all prices equally
- Less responsive to recent changes than EMA
- Common periods: 10, 20, 50, 100, 200
- Price > SMA: Bullish signal
- Price < SMA: Bearish signal
- Golden Cross: SMA(50) crosses above SMA(200) - bullish
- Death Cross: SMA(50) crosses below SMA(200) - bearish

---

### 3. RSI (Relative Strength Index)
**Dataclass:** `RSI`

```python
from trading.core.ta import TechnicalAnalyzer, RSI

rsi: RSI = TechnicalAnalyzer.calculate_rsi(prices, period=14)
# Returns: RSI(period=14, rsi=65.5, avg_gain=0.5, avg_loss=0.3, rs=1.67)
```

**Fields:**
- `period`: RSI period
- `rsi`: RSI value (0-100)
- `avg_gain`: Average gain (optional)
- `avg_loss`: Average loss (optional)
- `rs`: Relative strength (avg_gain / avg_loss, None if avg_loss = 0)

**Interpretation:**
- RSI > 70: Overbought
- RSI < 30: Oversold

---

### 4. MACD (Moving Average Convergence Divergence)
**Dataclass:** `MACD`

```python
from trading.core.ta import TechnicalAnalyzer, MACD

macd: MACD = TechnicalAnalyzer.calculate_macd(
    prices,
    slow_period=26,
    fast_period=12,
    signal_period=9
)
# Returns: MACD(macd=1.5, signal=1.2, histogram=0.3, ...)
```

**Fields:**
- `slow_period`, `fast_period`, `signal_period`: Periods used
- `slow_ema`, `fast_ema`: EMA values
- `macd`: MACD line (fast_ema - slow_ema)
- `signal`: Signal line (EMA of MACD)
- `histogram`: MACD - Signal
- `last_macd`: Previous MACD (optional)

**Interpretation:**
- Histogram > 0: Bullish
- Histogram < 0: Bearish
- MACD crosses above signal: Buy signal
- MACD crosses below signal: Sell signal

---

### 5. Bollinger Bands
**Dataclass:** `BollingerBands`

```python
from trading.core.ta import TechnicalAnalyzer, BollingerBands

bb: BollingerBands = TechnicalAnalyzer.calculate_bollinger_bands(
    prices,
    period=20,
    std_dev=2.0
)
# Returns: BollingerBands(upper=105.2, middle=100.0, lower=94.8, ...)
```

**Fields:**
- `period`: SMA period
- `std_dev`: Number of standard deviations (typically 2.0)
- `upper`: Upper band
- `middle`: Middle band (SMA)
- `lower`: Lower band
- `bandwidth`: (upper - lower) / middle
- `percent_b`: (price - lower) / (upper - lower)
- `current_price`: Current price

**Interpretation:**
- %B > 1: Price above upper band (overbought)
- %B < 0: Price below lower band (oversold)
- %B = 0.5: Price at middle band
- Bandwidth: Volatility measure (wider = more volatile)

---

### 6. Stochastic Oscillator
**Dataclass:** `Stochastic`

```python
from trading.core.ta import TechnicalAnalyzer, Stochastic

stoch: Stochastic = TechnicalAnalyzer.calculate_stochastic(
    high, low, close,
    fastk_period=14,
    slowk_period=3,
    slowd_period=3
)
# Returns: Stochastic(slowk=75.5, slowd=72.3, ...)
```

**Fields:**
- `fastk_period`, `slowk_period`, `slowd_period`: Periods used
- `slowk`: %K line (slow)
- `slowd`: %D line (signal)

**Interpretation:**
- %K > 80: Overbought
- %K < 20: Oversold
- %K crosses above %D: Buy signal
- %K crosses below %D: Sell signal

---

### 7. ATR (Average True Range)
**Dataclass:** `ATR`

```python
from trading.core.ta import TechnicalAnalyzer, ATR

atr: ATR = TechnicalAnalyzer.calculate_atr(high, low, close, period=14)
# Returns: ATR(period=14, atr=2.5)
```

**Fields:**
- `period`: ATR period
- `atr`: Average True Range value

**Interpretation:**
- Higher ATR: Higher volatility
- Lower ATR: Lower volatility
- Used for stop-loss placement (e.g., 2x ATR)

---

### 8. ADX (Average Directional Index)
**Dataclass:** `ADX`

```python
from trading.core.ta import TechnicalAnalyzer, ADX

adx: ADX = TechnicalAnalyzer.calculate_adx(high, low, close, period=14)
# Returns: ADX(adx=35.5, plus_di=28.5, minus_di=12.3, ...)
```

**Fields:**
- `period`: ADX period
- `adx`: Average Directional Index
- `plus_di`: +DI (Positive Directional Indicator)
- `minus_di`: -DI (Negative Directional Indicator)

**Interpretation:**
- ADX > 25: Strong trend
- ADX < 20: Weak trend
- +DI > -DI: Uptrend
- +DI < -DI: Downtrend

---

### 9. CCI (Commodity Channel Index)
**Dataclass:** `CCI`

```python
from trading.core.ta import TechnicalAnalyzer, CCI

cci: CCI = TechnicalAnalyzer.calculate_cci(high, low, close, period=14)
# Returns: CCI(period=14, cci=125.5)
```

**Fields:**
- `period`: CCI period
- `cci`: Commodity Channel Index value

**Interpretation:**
- CCI > 100: Overbought
- CCI < -100: Oversold
- Crosses above -100: Buy signal
- Crosses below +100: Sell signal

---

### 10. Williams %R
**Dataclass:** `WilliamsR`

```python
from trading.core.ta import TechnicalAnalyzer, WilliamsR

willr: WilliamsR = TechnicalAnalyzer.calculate_willr(high, low, close, period=14)
# Returns: WilliamsR(period=14, willr=-25.5)
```

**Fields:**
- `period`: Williams %R period
- `willr`: Williams %R value (-100 to 0)

**Interpretation:**
- %R > -20: Overbought
- %R < -80: Oversold
- Inverted scale (0 = overbought, -100 = oversold)

---

## Usage Examples

### Simple Usage

```python
from collections import deque
from trading.core.ta import TechnicalAnalyzer

# Price data
prices = deque([100.0, 101.5, 102.0, ...])
high = deque([101.0, 102.0, 103.0, ...])
low = deque([99.5, 100.5, 101.0, ...])

# Calculate moving averages
ema_20 = TechnicalAnalyzer.calculate_ema(prices, period=20)
sma_50 = TechnicalAnalyzer.calculate_sma(prices, period=50)
sma_200 = TechnicalAnalyzer.calculate_sma(prices, period=200)

# Calculate indicators
rsi = TechnicalAnalyzer.calculate_rsi(prices, period=14)
bb = TechnicalAnalyzer.calculate_bollinger_bands(prices, period=20)
macd = TechnicalAnalyzer.calculate_macd(prices)
stoch = TechnicalAnalyzer.calculate_stochastic(high, low, prices)
atr = TechnicalAnalyzer.calculate_atr(high, low, prices)
adx = TechnicalAnalyzer.calculate_adx(high, low, prices)
cci = TechnicalAnalyzer.calculate_cci(high, low, prices)
willr = TechnicalAnalyzer.calculate_willr(high, low, prices)

# Access values
if rsi:
    print(f"RSI: {rsi.rsi:.2f}")
if bb:
    print(f"Price vs Bands: %B = {bb.percent_b:.2f}")
```

### Moving Average Crossover Strategy

```python
from collections import deque
from trading.core.ta import TechnicalAnalyzer


def moving_average_crossover(prices, history_prices=None):
    """
    Classic moving average crossover strategy.

    - Golden Cross: SMA(50) crosses above SMA(200) = BUY
    - Death Cross: SMA(50) crosses below SMA(200) = SELL
    """

    # Calculate current moving averages
    sma_50 = TechnicalAnalyzer.calculate_sma(prices, period=50)
    sma_200 = TechnicalAnalyzer.calculate_sma(prices, period=200)

    if not sma_50 or not sma_200:
        return "HOLD"  # Insufficient data

    # Calculate previous moving averages for crossover detection
    if history_prices and len(history_prices) >= 200:
        prev_sma_50 = TechnicalAnalyzer.calculate_sma(history_prices, period=50)
        prev_sma_200 = TechnicalAnalyzer.calculate_sma(history_prices, period=200)

        # Golden Cross: SMA(50) crosses above SMA(200)
        if prev_sma_50.sma <= prev_sma_200.sma and sma_50.sma > sma_200.sma:
            return "BUY"

        # Death Cross: SMA(50) crosses below SMA(200)
        elif prev_sma_50.sma >= prev_sma_200.sma and sma_50.sma < sma_200.sma:
            return "SELL"

    # No crossover - check current position
    if sma_50.sma > sma_200.sma:
        return "HOLD_BULLISH"  # Uptrend
    else:
        return "HOLD_BEARISH"  # Downtrend


# Example with EMA crossover (more responsive)
def ema_crossover(prices, history_prices=None):
    """
    EMA crossover strategy (faster signals than SMA).

    - EMA(12) crosses above EMA(26) = BUY
    - EMA(12) crosses below EMA(26) = SELL
    """

    ema_12 = TechnicalAnalyzer.calculate_ema(prices, period=12)
    ema_26 = TechnicalAnalyzer.calculate_ema(prices, period=26)

    if not ema_12 or not ema_26:
        return "HOLD"

    if history_prices and len(history_prices) >= 26:
        prev_ema_12 = TechnicalAnalyzer.calculate_ema(history_prices, period=12)
        prev_ema_26 = TechnicalAnalyzer.calculate_ema(history_prices, period=26)

        # Bullish crossover
        if prev_ema_12.ema <= prev_ema_26.ema and ema_12.ema > ema_26.ema:
            return "BUY"

        # Bearish crossover
        elif prev_ema_12.ema >= prev_ema_26.ema and ema_12.ema < ema_26.ema:
            return "SELL"

    return "HOLD"
```

### Algorithm Usage

```python
from trading.core.algorithm import Algorithm
from trading.core.ta import TechnicalAnalyzer


class MyStrategy(Algorithm):
    def on_data_logic(self, data):
        signals = []

        for pd in data:
            # Get price history
            prices = self.price_history[pd.symbol]

            if len(prices) < 20:
                continue

            # Calculate multiple indicators
            rsi = TechnicalAnalyzer.calculate_rsi(prices, period=14)
            bb = TechnicalAnalyzer.calculate_bollinger_bands(prices, period=20)

            # Trading logic
            if rsi and bb:
                # Buy: Oversold + below lower band
                if rsi.rsi < 30 and bb.percent_b < 0.2:
                    signals.append(MarketSignal(
                        type=SignalType.BUY,
                        symbol=pd.symbol,
                        strength=75
                    ))

                # Sell: Overbought + above upper band
                elif rsi.rsi > 70 and bb.percent_b > 0.8:
                    signals.append(MarketSignal(
                        type=SignalType.SELL,
                        symbol=pd.symbol,
                        strength=75
                    ))

        return signals
```

### Multi-Indicator Strategy

```python
def generate_signal(prices, high, low):
    """Generate trading signal using multiple indicators."""

    # Calculate all indicators
    rsi = TechnicalAnalyzer.calculate_rsi(prices, period=14)
    bb = TechnicalAnalyzer.calculate_bollinger_bands(prices, period=20)
    macd = TechnicalAnalyzer.calculate_macd(prices)
    adx = TechnicalAnalyzer.calculate_adx(high, low, prices)

    if not all([rsi, bb, macd, adx]):
        return "HOLD"  # Insufficient data

    # Buy conditions
    buy_conditions = [
        rsi.rsi < 30,              # Oversold
        bb.percent_b < 0.2,        # Below lower band
        macd.histogram > 0,        # Bullish MACD
        adx.adx > 25,              # Strong trend
        adx.plus_di > adx.minus_di # Uptrend
    ]

    # Sell conditions
    sell_conditions = [
        rsi.rsi > 70,              # Overbought
        bb.percent_b > 0.8,        # Above upper band
        macd.histogram < 0,        # Bearish MACD
        adx.minus_di > adx.plus_di # Downtrend
    ]

    # Decision
    if sum(buy_conditions) >= 4:
        return "BUY"
    elif sum(sell_conditions) >= 3:
        return "SELL"
    else:
        return "HOLD"
```

## Data Requirements

| Indicator | Required Data | Minimum Periods |
|-----------|---------------|-----------------|
| EMA | Close | period |
| SMA | Close | period |
| RSI | Close | period + 1 |
| MACD | Close | slow_period + signal_period |
| Bollinger Bands | Close | period |
| Stochastic | High, Low, Close | fastk_period |
| ATR | High, Low, Close | period |
| ADX | High, Low, Close | period |
| CCI | High, Low, Close | period |
| Williams %R | High, Low, Close | period |

## Implementation Details

- All indicators use TAlib for calculations (battle-tested, fast)
- Returns `None` if insufficient data
- Returns dataclasses with complete information
- All values are extracted from numpy arrays using `.item()` for type safety
- NaN checks ensure valid data before returning

## See Also

- Full example: `examples/technical_indicators_example.py`
- Tests: `tests/unit/test_technical_analyzer.py`
- Backward compatibility wrappers: `utils/indicators.py`
