"""
Example: Using ReadSignalsFromFile Algorithm

This example demonstrates how to use the ReadSignalsFromFile algorithm
to load trading signals from a CSV file and generate MarketSignals based
on timestamps.

CSV Format:
-----------
The CSV file should have:
- A timestamp column (default name: 'timestamp')
- A signal column (any name, will use the first column after timestamp)
- Signal values should be numeric (e.g., 0.0 to 1.0 for probabilities)

Example CSV:
timestamp,signal
2024-01-01 09:30:00,0.75
2024-01-01 09:35:00,0.85
2024-01-01 09:40:00,0.45

The algorithm will:
1. Load the CSV file in __init__
2. Convert timestamps to pandas datetime objects
3. On each tick, lookup the signal value for the current timestamp
4. Generate BUY signal if value >= threshold, SELL otherwise
"""

from trading.core.algorithms.test_algorithm import ReadSignalsFromFile
from trading.core.classes import PriceData
from datetime import datetime


def example_usage():
    """Demonstrate ReadSignalsFromFile usage."""

    print("=" * 70)
    print("ReadSignalsFromFile Algorithm Example")
    print("=" * 70)

    # Configuration for the algorithm
    config = {
        'csv_path': 'examples/signals_example.csv',  # Path to CSV file
        'threshold': 0.5,  # Signal values >= 0.5 = BUY, < 0.5 = SELL
        'timestamp_column': 'timestamp'  # Name of timestamp column (optional)
    }

    # Create the algorithm
    print("\n1. Loading signals from CSV...")
    algorithm = ReadSignalsFromFile(cfg=config)
    print(f"   Signal column: '{algorithm.signal_column}'")
    print(f"   Loaded {len(algorithm.signals_df)} signals")
    print(f"   Threshold: {algorithm.threshold}")

    # Show loaded data
    print("\n2. Loaded signal data:")
    print(algorithm.signals_df.head())

    # Simulate incoming price data with timestamps matching the CSV
    print("\n3. Processing price data with matching timestamps...")

    test_timestamps = [
        datetime(2024, 1, 1, 9, 30, 0),
        datetime(2024, 1, 1, 9, 35, 0),
        datetime(2024, 1, 1, 9, 40, 0),
        datetime(2024, 1, 1, 9, 45, 0),
        datetime(2024, 1, 1, 9, 50, 0),
    ]

    for ts in test_timestamps:
        # Create mock price data
        price_data = [
            PriceData(
                timestamp=ts,
                symbol="TEST",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000
            )
        ]

        # Get signals from the algorithm
        signals = algorithm.on_data_logic(price_data)

        # Display results
        if signals:
            signal = signals[0]
            signal_value = algorithm.signals_df.loc[ts, algorithm.signal_column]
            print(f"   {ts} | Signal Value: {signal_value:.2f} | "
                  f"Type: {signal.type.name:4s} | Strength: {signal.strength:3d}")
        else:
            print(f"   {ts} | No signal found")

    # Test with a timestamp NOT in the CSV
    print("\n4. Testing with timestamp not in CSV...")
    missing_ts = datetime(2024, 1, 1, 11, 0, 0)
    price_data = [
        PriceData(
            timestamp=missing_ts,
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000
        )
    ]
    signals = algorithm.on_data_logic(price_data)
    print(f"   {missing_ts} | Signals returned: {len(signals)} (expected 0)")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def example_with_custom_csv():
    """Example showing how to use a custom CSV format."""

    print("\n\n" + "=" * 70)
    print("Custom CSV Format Example")
    print("=" * 70)

    # Create a custom CSV with different column names
    import pandas as pd
    import os

    custom_csv_path = 'examples/custom_signals.csv'

    # Create custom data
    custom_data = pd.DataFrame({
        'datetime': [
            '2024-01-01 14:00:00',
            '2024-01-01 14:05:00',
            '2024-01-01 14:10:00',
        ],
        'prediction': [0.8, 0.3, 0.6]
    })
    custom_data.to_csv(custom_csv_path, index=False)
    print(f"\n1. Created custom CSV at: {custom_csv_path}")
    print(custom_data)

    # Use custom configuration
    config = {
        'csv_path': custom_csv_path,
        'threshold': 0.5,
        'timestamp_column': 'datetime'  # Specify custom timestamp column name
    }

    # Load algorithm
    print("\n2. Loading algorithm with custom timestamp column...")
    algorithm = ReadSignalsFromFile(cfg=config)
    print(f"   Signal column: '{algorithm.signal_column}'")
    print(f"   Loaded {len(algorithm.signals_df)} signals")

    # Test with custom data
    print("\n3. Testing with custom timestamps...")
    test_ts = datetime(2024, 1, 1, 14, 0, 0)
    price_data = [PriceData(
        timestamp=test_ts,
        symbol="TEST",
        open=100.0, high=101.0, low=99.0, close=100.5, volume=1000
    )]

    signals = algorithm.on_data_logic(price_data)
    if signals:
        signal = signals[0]
        print(f"   {test_ts} | Type: {signal.type.name} | Strength: {signal.strength}")

    # Cleanup
    os.remove(custom_csv_path)
    print(f"\n4. Cleaned up custom CSV")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_usage()
    example_with_custom_csv()
