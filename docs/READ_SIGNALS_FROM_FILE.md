# ReadSignalsFromFile Algorithm

## Overview

The `ReadSignalsFromFile` algorithm loads trading signals from a CSV file and generates `MarketSignal` objects based on timestamps. This is useful for:

- Backtesting with pre-computed ML model predictions
- Using external signal sources
- Replaying historical trading decisions
- Testing strategies with known signals

## Implementation Details

### Key Features

✅ **CSV Loading**: Loads signals from CSV file in `__init__`
✅ **Timestamp Matching**: Matches current data timestamp to CSV rows
✅ **Configurable**: Support for custom timestamp column names
✅ **Threshold-based**: Converts numeric signals to BUY/SELL based on threshold
✅ **Fast Lookups**: Uses pandas index for O(1) timestamp lookups
✅ **Graceful Handling**: Returns empty list if timestamp not found

### Modified Methods

**`__init__(cfg, history_length=0)`:**
- Loads CSV file from `cfg['csv_path']`
- Converts timestamp column to pandas datetime
- Sets timestamp as dataframe index for fast lookups
- Stores signal column name (first non-timestamp column)

**`on_data_logic(data: list[PriceData]) -> list[MarketSignal]`:**
- Extracts timestamp from each `PriceData` object
- Looks up signal value in dataframe for that timestamp
- Converts signal value to `SignalType.BUY` or `SignalType.SELL` based on threshold
- Returns list of `MarketSignal` objects

## CSV Format

### Required Structure

```csv
timestamp,signal
2024-01-01 09:30:00,0.75
2024-01-01 09:35:00,0.85
2024-01-01 09:40:00,0.45
```

**Required Columns:**
- **Timestamp column**: Default name is `'timestamp'` (configurable)
- **Signal column**: Any numeric column (uses first column after timestamp)

**Timestamp Format:**
- Any format pandas can parse (e.g., `YYYY-MM-DD HH:MM:SS`)
- Can be timezone-aware or timezone-naive
- Must match the timestamps in your `PriceData` objects

**Signal Values:**
- Numeric values (typically 0.0 to 1.0 for probabilities)
- Values >= `threshold` → `SignalType.BUY`
- Values < `threshold` → `SignalType.SELL`
- Can use any range (e.g., -1 to 1, 0 to 100)

### Custom Column Names

```csv
datetime,prediction,confidence
2024-01-01 09:30:00,0.75,0.9
2024-01-01 09:35:00,0.85,0.95
```

Configure with:
```python
config = {
    'csv_path': 'signals.csv',
    'timestamp_column': 'datetime',  # Custom timestamp column name
    'threshold': 0.5
}
```

The algorithm will automatically use `'prediction'` as the signal column (first column after timestamp).

## Configuration

### Required Parameters

- **`csv_path`** (str): Path to CSV file containing signals
  - Example: `'data/ml_signals.csv'`
  - Can be absolute or relative path

### Optional Parameters

- **`threshold`** (float): Signal threshold for BUY vs SELL
  - Default: `0.5`
  - Signal >= threshold → BUY
  - Signal < threshold → SELL

- **`timestamp_column`** (str): Name of timestamp column in CSV
  - Default: `'timestamp'`
  - Use if your CSV has a different timestamp column name

### Example Configuration

```python
config = {
    'csv_path': 'data/ml/UPRO_signals.csv',
    'threshold': 0.6,  # More conservative (higher confidence for BUY)
    'timestamp_column': 'timestamp'  # Optional: defaults to 'timestamp'
}

algorithm = ReadSignalsFromFile(cfg=config)
```

## Usage Examples

### Basic Usage

```python
from algorithms.test_algorithm import ReadSignalsFromFile
from core.classes import PriceData

# Configure the algorithm
config = {
    'csv_path': 'examples/signals_example.csv',
    'threshold': 0.5
}

# Create algorithm instance
algorithm = ReadSignalsFromFile(cfg=config)

# Process price data
price_data = [
    PriceData(
        timestamp=datetime(2024, 1, 1, 9, 30, 0),
        symbol="AAPL",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000
    )
]

# Get signals
signals = algorithm.on_data_logic(price_data)

# signals[0].type will be BUY or SELL based on CSV value
```

### With Backtesting Simulator

```python
from engines.simulator import Simulator

# Update config.yaml:
"""
simulator:
  algorithm:
    algorithm: "algorithms.test_algorithm.ReadSignalsFromFile"
    csv_path: "data/ml/predictions.csv"
    threshold: 0.6
    history_length: 0
  # ... other config
"""

# Run backtest
sim = Simulator(cfg_section_to_use="simulator")
sim.run()
```

### Using ML Model Predictions

```python
# 1. Generate predictions from ML model
import pandas as pd
import xgboost as xgb

# Load your trained model
model = xgb.XGBClassifier()
model.load_model('models/best_model.json')

# Load test data
test_data = pd.read_csv('data/ml/UPRO_5min_ml_val.csv')

# Generate predictions
predictions = model.predict_proba(test_data[feature_cols])[:, 1]

# Create signals CSV
signals_df = pd.DataFrame({
    'timestamp': test_data['timestamp'],
    'signal': predictions
})
signals_df.to_csv('data/ml/UPRO_predictions.csv', index=False)

# 2. Use in backtesting
config = {
    'csv_path': 'data/ml/UPRO_predictions.csv',
    'threshold': 0.7  # Only trade on high-confidence predictions
}
algorithm = ReadSignalsFromFile(cfg=config)
```

### Multiple Signal Columns

If your CSV has multiple signal columns, the algorithm uses the **first column after timestamp**:

```csv
timestamp,ml_signal,ta_signal,combined
2024-01-01 09:30:00,0.75,0.80,0.78
```

This CSV will use `'ml_signal'` as the signal column. If you want to use a different column, reorder your CSV columns.

## Signal Value to MarketSignal Conversion

The algorithm converts numeric signal values to `MarketSignal` objects:

### BUY Signal (signal_value >= threshold)

```python
signal_type = SignalType.BUY
strength = int(min(signal_value * 100, 100))
```

Example:
- Signal value: `0.85`
- Threshold: `0.5`
- Result: `BUY` with strength `85`

### SELL Signal (signal_value < threshold)

```python
signal_type = SignalType.SELL
strength = int(min((1 - signal_value) * 100, 100))
```

Example:
- Signal value: `0.25`
- Threshold: `0.5`
- Result: `SELL` with strength `75` (1 - 0.25 = 0.75)

## Timestamp Matching

### Exact Matching

The algorithm uses **exact timestamp matching**:

```python
# CSV timestamp
2024-01-01 09:30:00

# Will match PriceData with:
timestamp=datetime(2024, 1, 1, 9, 30, 0)

# Will NOT match:
timestamp=datetime(2024, 1, 1, 9, 30, 1)  # Off by 1 second
```

### Missing Timestamps

If a timestamp is not found in the CSV:
- No signal is generated for that timestamp
- The algorithm returns an empty list `[]`
- No error is raised

### Best Practices

1. **Pre-filter CSV**: Only include timestamps that exist in your price data
2. **Align frequencies**: Match CSV frequency to price data (e.g., both 5-minute bars)
3. **Test first**: Verify timestamps match using the example script

## Integration with ML Pipeline

### Complete Workflow

```python
# Step 1: Train model and generate predictions
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load features
df = pd.read_csv('data/ml/UPRO_5min_ml_full.csv')
X = df[feature_cols]
y = df['target_L5_P10_H288']

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Generate predictions for full dataset
predictions = model.predict_proba(X)[:, 1]

# Step 2: Create signals CSV
signals_df = pd.DataFrame({
    'timestamp': df['timestamp'],
    'signal': predictions
})
signals_df.to_csv('data/ml/UPRO_backtest_signals.csv', index=False)

# Step 3: Run backtest with signals
from algorithms.test_algorithm import ReadSignalsFromFile
from engines.simulator import Simulator

config = {
    'csv_path': 'data/ml/UPRO_backtest_signals.csv',
    'threshold': 0.6  # Adjust based on precision/recall tradeoff
}
algorithm = ReadSignalsFromFile(cfg=config)

# Use in simulator
# (Update config.yaml with algorithm settings)
sim = Simulator(cfg_section_to_use="simulator")
sim.run()
```

## Troubleshooting

### Common Issues

**1. ValueError: csv_path must be specified**
- Solution: Add `csv_path` to config dictionary

**2. ValueError: Timestamp column 'timestamp' not found**
- Solution: Check CSV column names, specify correct `timestamp_column` in config

**3. No signals generated**
- Solution: Verify timestamps in CSV match timestamps in price data
- Use `print(algorithm.signals_df.index)` to see loaded timestamps

**4. KeyError when looking up timestamp**
- This is normal - means timestamp not in CSV
- Check if CSV covers the full date range of your price data

### Debug Tips

```python
# Check loaded signals
print(f"Loaded {len(algorithm.signals_df)} signals")
print(algorithm.signals_df.head())

# Check signal column name
print(f"Signal column: {algorithm.signal_column}")

# Check timestamp range
print(f"First timestamp: {algorithm.signals_df.index[0]}")
print(f"Last timestamp: {algorithm.signals_df.index[-1]}")

# Check for timestamp match
test_ts = datetime(2024, 1, 1, 9, 30, 0)
if test_ts in algorithm.signals_df.index:
    print(f"Found signal: {algorithm.signals_df.loc[test_ts, algorithm.signal_column]}")
else:
    print(f"Timestamp {test_ts} not found in CSV")
```

## Performance Notes

- **Fast lookups**: O(1) timestamp lookups using pandas index
- **Memory efficient**: CSV loaded once in `__init__`, not on every tick
- **Large files**: Tested with 100K+ rows, works well
- **Recommendation**: For very large files (1M+ rows), consider filtering to date range first

## See Also

- Example: `examples/read_signals_example.py`
- Sample CSV: `examples/signals_example.csv`
- ML Integration: `scratch/ml_scratch_feature_eng.ipynb`
- Simulator Documentation: `engines/simulator.py`
