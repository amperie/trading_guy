"""
Download test data for indicator validation.

This script downloads real market data from various sources
to validate indicator calculations.
"""

import pandas as pd
from pathlib import Path


def download_yfinance_data():
    """Download data using yfinance."""
    try:
        import yfinance as yf
        print("Downloading data using yfinance...")

        # Download multiple tickers for testing
        tickers = ["SPY", "AAPL", "MSFT", "TSLA"]
        data_dir = Path("data/test_indicators")
        data_dir.mkdir(exist_ok=True, parents=True)

        for ticker in tickers:
            print(f"  Downloading {ticker}...")
            stock = yf.Ticker(ticker)

            # Download different periods
            periods = {
                "6mo": "6_months",
                "1y": "1_year",
                "2y": "2_years"
            }

            for period, suffix in periods.items():
                df = stock.history(period=period)
                if not df.empty:
                    filename = data_dir / f"{ticker}_{suffix}.csv"
                    df.to_csv(filename)
                    print(f"    Saved {filename} ({len(df)} rows)")

        print(f"\n[OK] Downloaded data to {data_dir}")
        return True

    except ImportError:
        print("[ERROR] yfinance not installed. Install with: pip install yfinance")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading data: {e}")
        return False


def create_synthetic_data():
    """Create synthetic test data with known characteristics."""
    import numpy as np

    print("\nCreating synthetic test data...")
    data_dir = Path("data/test_indicators")
    data_dir.mkdir(exist_ok=True, parents=True)

    np.random.seed(42)

    # Test case 1: Uptrend
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    trend = np.linspace(100, 150, 200)
    noise = np.random.normal(0, 2, 200)
    prices = trend + noise

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "UPTREND",
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 200)
    })
    filename = data_dir / "synthetic_uptrend.csv"
    df.to_csv(filename, index=False)
    print(f"  Created {filename}")

    # Test case 2: Downtrend
    trend = np.linspace(150, 100, 200)
    noise = np.random.normal(0, 2, 200)
    prices = trend + noise

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "DOWNTREND",
        "open": prices * 1.01,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 200)
    })
    filename = data_dir / "synthetic_downtrend.csv"
    df.to_csv(filename, index=False)
    print(f"  Created {filename}")

    # Test case 3: Sideways/Range-bound
    base = 125
    prices = base + np.random.normal(0, 5, 200)

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "SIDEWAYS",
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 200)
    })
    filename = data_dir / "synthetic_sideways.csv"
    df.to_csv(filename, index=False)
    print(f"  Created {filename}")

    # Test case 4: High volatility
    returns = np.random.normal(0.002, 0.05, 200)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "VOLATILE",
        "open": pd.Series(prices[:-1]) * 0.99,
        "high": pd.Series(prices[:-1]) * 1.05,
        "low": pd.Series(prices[:-1]) * 0.95,
        "close": prices[:-1],
        "volume": np.random.randint(1000000, 5000000, 200)
    })
    filename = data_dir / "synthetic_volatile.csv"
    df.to_csv(filename, index=False)
    print(f"  Created {filename}")

    # Test case 5: Known EMA values for validation
    # Simple sequence for easy manual verification
    prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
              22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
              23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17]

    dates_short = pd.date_range(start="2023-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame({
        "timestamp": dates_short,
        "symbol": "KNOWN",
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000000] * len(prices)
    })
    filename = data_dir / "known_values.csv"
    df.to_csv(filename, index=False)
    print(f"  Created {filename}")

    print(f"\n[OK] Created synthetic data in {data_dir}")
    return True


def generate_expected_values():
    """Generate expected EMA/MACD values using pandas for validation."""
    import numpy as np

    print("\nGenerating expected values for validation...")

    # Known price sequence
    prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
              22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
              23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17]

    series = pd.Series(prices)

    # Calculate EMA values
    ema_10 = series.ewm(span=10, adjust=False).mean()
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()

    # Calculate MACD
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    print("\nExpected values for known_values.csv:")
    print(f"\nEMA(10) last value: {ema_10.iloc[-1]:.6f}")
    print(f"EMA(12) last value: {ema_12.iloc[-1]:.6f}")
    print(f"EMA(26) last value: {ema_26.iloc[-1]:.6f}")
    print(f"\nMACD line last value: {macd_line.iloc[-1]:.6f}")
    print(f"Signal line last value: {signal_line.iloc[-1]:.6f}")
    print(f"Histogram last value: {histogram.iloc[-1]:.6f}")

    # Save expected values
    data_dir = Path("data/test_indicators")
    data_dir.mkdir(exist_ok=True, parents=True)

    expected = pd.DataFrame({
        "price": prices,
        "ema_10": ema_10,
        "ema_12": ema_12,
        "ema_26": ema_26,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram
    })

    filename = data_dir / "expected_values.csv"
    expected.to_csv(filename, index=False)
    print(f"\n[OK] Saved expected values to {filename}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING TEST DATA FOR INDICATOR VALIDATION")
    print("=" * 60)

    success = True

    # Try to download real data
    if download_yfinance_data():
        print("\n[OK] Real market data downloaded successfully")
    else:
        print("\n[WARNING] Could not download real market data (will use synthetic only)")
        success = False

    # Create synthetic data
    if create_synthetic_data():
        print("\n[OK] Synthetic test data created successfully")
    else:
        print("\n[ERROR] Failed to create synthetic data")
        success = False

    # Generate expected values
    if generate_expected_values():
        print("\n[OK] Expected values generated successfully")
    else:
        print("\n[ERROR] Failed to generate expected values")
        success = False

    if success:
        print("\n" + "=" * 60)
        print("[OK] ALL TEST DATA READY")
        print("=" * 60)
        print("\nRun tests with: pytest tests/unit/test_indicators.py -v")
    else:
        print("\n" + "=" * 60)
        print("[WARNING] PARTIAL SUCCESS")
        print("=" * 60)
        print("\nSome test data may be missing, but tests should still run.")
        print("Install yfinance for real market data: pip install yfinance")
