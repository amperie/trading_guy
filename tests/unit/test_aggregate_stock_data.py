"""
Comprehensive tests for aggregate_stock_data function.
Tests OHLCV aggregation correctness and symbol separation.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.utils import aggregate_stock_data, aggregate_stock_data_custom


class TestAggregateStockData:
    """Test suite for stock data aggregation functions"""

    @pytest.fixture
    def sample_1min_data(self):
        """Create sample 1-minute data for testing

        Note: Starts at 09:26 so that with right-closed intervals (09:25, 09:30],
        (09:30, 09:35], etc., we get clean 5-minute bins with 5 rows each.
        """
        # Create 30 minutes of 1-minute data for 2 symbols (09:26-09:55)
        timestamps = pd.date_range('2024-01-01 09:26', periods=30, freq='1min')

        data = []
        for symbol in ['AAPL', 'GOOGL']:
            for i, ts in enumerate(timestamps):
                # Create predictable OHLCV data for testing
                base_price = 100 if symbol == 'AAPL' else 200
                data.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'open': base_price + i * 0.1,
                    'high': base_price + i * 0.1 + 0.5,
                    'low': base_price + i * 0.1 - 0.3,
                    'close': base_price + i * 0.1 + 0.2,
                    'volume': 1000 * (i + 1),
                    'trade_count': 10 * (i + 1)
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def real_data(self):
        """Load real data from CSV file"""
        try:
            df = pd.read_csv('data/SPXU_GDXU_UPRO_1min.csv')
            # Take a small subset for faster testing
            return df.head(1000)
        except FileNotFoundError:
            pytest.skip("Real data file not found")

    def test_basic_aggregation_5min(self, sample_1min_data):
        """Test basic 5-minute aggregation"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # Should have 2 symbols × 6 intervals (30 min / 5 min) = 12 rows
        assert len(result) == 12

        # Check symbols are present
        assert set(result['symbol'].unique()) == {'AAPL', 'GOOGL'}

        # Check all required columns exist
        assert all(col in result.columns for col in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

    def test_symbols_kept_separate(self, sample_1min_data):
        """Verify symbols are aggregated separately, not mixed"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # For each timestamp, we should have separate rows for each symbol
        for ts in result['timestamp'].unique():
            ts_data = result[result['timestamp'] == ts]
            symbols_at_ts = ts_data['symbol'].tolist()

            # Should have both symbols (or at least data is separated by symbol)
            # Each symbol appears only once per timestamp
            assert len(symbols_at_ts) == len(set(symbols_at_ts))

    def test_ohlc_aggregation_correctness(self, sample_1min_data):
        """Test that OHLC values are aggregated correctly"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # Test first interval for AAPL (rows 0-4 of AAPL data)
        aapl_first = result[(result['symbol'] == 'AAPL') & (result['timestamp'] == result['timestamp'].min())].iloc[0]

        # Get original AAPL data for first 5 minutes
        aapl_original = sample_1min_data[sample_1min_data['symbol'] == 'AAPL'].head(5)

        # Verify OHLC aggregation
        assert aapl_first['open'] == aapl_original['open'].iloc[0], "Open should be first value"
        assert aapl_first['high'] == aapl_original['high'].max(), "High should be max value"
        assert aapl_first['low'] == aapl_original['low'].min(), "Low should be min value"
        assert aapl_first['close'] == aapl_original['close'].iloc[-1], "Close should be last value"

    def test_volume_aggregation(self, sample_1min_data):
        """Test that volume is summed correctly"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # Test first interval for AAPL
        aapl_first = result[(result['symbol'] == 'AAPL') & (result['timestamp'] == result['timestamp'].min())].iloc[0]

        # Get original AAPL data for first 5 minutes
        aapl_original = sample_1min_data[sample_1min_data['symbol'] == 'AAPL'].head(5)

        # Volume should be sum of all 5 minutes
        expected_volume = aapl_original['volume'].sum()
        assert aapl_first['volume'] == expected_volume, f"Expected volume {expected_volume}, got {aapl_first['volume']}"

    def test_trade_count_aggregation(self, sample_1min_data):
        """Test that trade_count is summed correctly"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # Test first interval for GOOGL
        googl_first = result[(result['symbol'] == 'GOOGL') & (result['timestamp'] == result['timestamp'].min())].iloc[0]

        # Get original GOOGL data for first 5 minutes
        googl_original = sample_1min_data[sample_1min_data['symbol'] == 'GOOGL'].head(5)

        # trade_count should be sum
        expected_count = googl_original['trade_count'].sum()
        assert googl_first['trade_count'] == expected_count

    def test_different_intervals(self, sample_1min_data):
        """Test different aggregation intervals

        Data spans 09:26-09:55 (30 minutes). With right-closed intervals:
        - 5min: (09:25,09:30], (09:30,09:35], ..., (09:50,09:55] = 6 bins
        - 10min: (09:20,09:30], (09:30,09:40], (09:40,09:50], (09:50,10:00] = 4 bins (2 partial, 2 full)
        - 15min: (09:15,09:30], (09:30,09:45], (09:45,10:00] = 3 bins (2 partial, 1 full)
        - 30min: (09:00,09:30], (09:30,10:00] = 2 bins (both partial)
        """
        # Test 5-minute - 6 full bins per symbol
        result_5min = aggregate_stock_data(sample_1min_data, interval='5min')
        assert len(result_5min) == 12  # 2 symbols × 6 intervals

        # Test 10-minute - 4 bins per symbol (2 partial, 2 full)
        result_10min = aggregate_stock_data(sample_1min_data, interval='10min')
        assert len(result_10min) == 8  # 2 symbols × 4 intervals

        # Test 15-minute - 3 bins per symbol (2 partial, 1 full)
        result_15min = aggregate_stock_data(sample_1min_data, interval='15min')
        assert len(result_15min) == 6  # 2 symbols × 3 intervals

        # Test 30-minute - 2 bins per symbol (both partial)
        result_30min = aggregate_stock_data(sample_1min_data, interval='30min')
        assert len(result_30min) == 4  # 2 symbols × 2 intervals

    def test_timestamp_sorting(self, sample_1min_data):
        """Test that results are sorted by timestamp and symbol"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # Check timestamps are in ascending order
        timestamps = result['timestamp'].tolist()
        assert timestamps == sorted(timestamps), "Timestamps should be sorted"

    def test_single_symbol(self):
        """Test aggregation with only one symbol"""
        # Start at 09:26 so (09:25, 09:30] and (09:30, 09:35] are clean 5-minute bins
        timestamps = pd.date_range('2024-01-01 09:26', periods=10, freq='1min')
        data = [{
            'timestamp': ts,
            'symbol': 'AAPL',
            'open': 100 + i,
            'high': 101 + i,
            'low': 99 + i,
            'close': 100.5 + i,
            'volume': 1000
        } for i, ts in enumerate(timestamps)]

        df = pd.DataFrame(data)
        result = aggregate_stock_data(df, interval='5min')

        # Should have 2 intervals for single symbol (09:30 and 09:35)
        assert len(result) == 2
        assert result['symbol'].unique()[0] == 'AAPL'

    def test_custom_aggregation_rules(self):
        """Test custom aggregation function with custom rules"""
        # Start at 09:26 so (09:25, 09:30] contains indices 0-4
        timestamps = pd.date_range('2024-01-01 09:26', periods=10, freq='1min')
        data = [{
            'timestamp': ts,
            'symbol': 'AAPL',
            'open': 100,
            'high': 101,
            'low': 99,
            'close': 100,
            'volume': 1000,
            'custom_field': i * 10
        } for i, ts in enumerate(timestamps)]

        df = pd.DataFrame(data)

        # Custom rule: average the custom_field
        custom_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'custom_field': 'mean'
        }

        result = aggregate_stock_data_custom(df, interval='5min', agg_rules=custom_rules)

        # Check custom field was averaged in first interval (09:30)
        # With right-closed intervals (09:25, 09:30], this contains indices 0-4
        first_interval = result.iloc[0]
        expected_custom = np.mean([0, 10, 20, 30, 40])  # First 5 values (indices 0-4)
        assert first_interval['custom_field'] == expected_custom

    def test_real_data_sanity_checks(self, real_data):
        """Sanity checks on real data aggregation"""
        # Aggregate to 5 minutes
        result = aggregate_stock_data(real_data, interval='5min')

        # Result should have fewer rows than input
        assert len(result) < len(real_data)

        # Result should have same symbols as input
        assert set(result['symbol'].unique()).issubset(set(real_data['symbol'].unique()))

        # Volume in aggregated data should be <= sum of original volumes per symbol
        for symbol in result['symbol'].unique():
            orig_volume = real_data[real_data['symbol'] == symbol]['volume'].sum()
            agg_volume = result[result['symbol'] == symbol]['volume'].sum()
            # They should be equal or very close (accounting for dropped NaN rows)
            assert agg_volume <= orig_volume

    def test_real_data_ohlc_bounds(self, real_data):
        """Verify OHLC relationships in aggregated real data"""
        result = aggregate_stock_data(real_data, interval='5min')

        for _, row in result.iterrows():
            # High should be >= Low
            assert row['high'] >= row['low'], f"High {row['high']} should be >= Low {row['low']}"

            # High should be >= Open and Close
            assert row['high'] >= row['open'], f"High should be >= Open"
            assert row['high'] >= row['close'], f"High should be >= Close"

            # Low should be <= Open and Close
            assert row['low'] <= row['open'], f"Low should be <= Open"
            assert row['low'] <= row['close'], f"Low should be <= Close"

    def test_exact_calculation_verification(self, sample_1min_data):
        """Detailed verification of exact calculations for a specific interval

        Data spans 09:26-09:55 (indices 0-29).
        With 10-minute right-closed intervals:
        - Bin 0 (09:30): (09:20, 09:30] contains indices 0-4 (09:26-09:30, 5 minutes)
        - Bin 1 (09:40): (09:30, 09:40] contains indices 5-14 (09:31-09:40, 10 minutes)
        - Bin 2 (09:50): (09:40, 09:50] contains indices 15-24 (09:41-09:50, 10 minutes)
        - Bin 3 (10:00): (09:50, 10:00] contains indices 25-29 (09:51-09:55, 5 minutes)
        """
        result = aggregate_stock_data(sample_1min_data, interval='10min')

        # Get second interval for AAPL (09:40 bin, contains indices 5-14)
        aapl_second = result[(result['symbol'] == 'AAPL')].iloc[1]

        # Get original data for that interval
        aapl_data = sample_1min_data[sample_1min_data['symbol'] == 'AAPL']
        aapl_interval = aapl_data.iloc[5:15]  # Indices 5-14 (09:31-09:40)

        # Verify each calculation explicitly
        assert aapl_second['open'] == pytest.approx(aapl_interval['open'].iloc[0])
        assert aapl_second['high'] == pytest.approx(aapl_interval['high'].max())
        assert aapl_second['low'] == pytest.approx(aapl_interval['low'].min())
        assert aapl_second['close'] == pytest.approx(aapl_interval['close'].iloc[-1])
        assert aapl_second['volume'] == pytest.approx(aapl_interval['volume'].sum())
        assert aapl_second['trade_count'] == pytest.approx(aapl_interval['trade_count'].sum())

    def test_no_data_mixing_between_symbols(self, sample_1min_data):
        """Ensure no data is mixed between symbols during aggregation"""
        result = aggregate_stock_data(sample_1min_data, interval='5min')

        # Get first interval
        first_ts = result['timestamp'].min()
        aapl_row = result[(result['timestamp'] == first_ts) & (result['symbol'] == 'AAPL')].iloc[0]
        googl_row = result[(result['timestamp'] == first_ts) & (result['symbol'] == 'GOOGL')].iloc[0]

        # AAPL prices should be around 100, GOOGL around 200
        assert 90 < aapl_row['open'] < 110, "AAPL price should be around 100"
        assert 190 < googl_row['open'] < 210, "GOOGL price should be around 200"

        # They should be clearly different (not mixed)
        assert abs(aapl_row['open'] - googl_row['open']) > 50

    def test_empty_intervals_removed(self):
        """Test that empty intervals are removed from results"""
        # Create data with a gap - start at 09:26 and 10:01 for clean bins
        # First group: 09:26-09:30 will create bin (09:25, 09:30]
        # Second group: 10:01-10:05 will create bin (10:00, 10:05]
        timestamps1 = pd.date_range('2024-01-01 09:26', periods=5, freq='1min')
        timestamps2 = pd.date_range('2024-01-01 10:01', periods=5, freq='1min')
        timestamps = timestamps1.tolist() + timestamps2.tolist()

        data = [{
            'timestamp': ts,
            'symbol': 'AAPL',
            'open': 100,
            'high': 101,
            'low': 99,
            'close': 100,
            'volume': 1000
        } for ts in timestamps]

        df = pd.DataFrame(data)
        result = aggregate_stock_data(df, interval='5min')

        # Should only have 2 intervals (09:30 and 10:05), not the gap in between
        assert len(result) == 2


class TestAggregateStockDataIntegration:
    """Integration tests with real data file"""

    def test_full_file_aggregation_5min(self):
        """Test aggregating entire real data file to 5 minutes"""
        try:
            df = pd.read_csv('data/SPXU_GDXU_UPRO_1min.csv')
        except FileNotFoundError:
            pytest.skip("Real data file not found")

        print(f"\nOriginal data: {len(df)} rows")

        result = aggregate_stock_data(df, interval='5min')
        print(f"5-minute aggregated: {len(result)} rows")

        # Basic sanity checks
        assert len(result) < len(df)
        assert set(result['symbol'].unique()) == set(df['symbol'].unique())

        # Check data integrity
        for symbol in result['symbol'].unique():
            orig_df = df[df['symbol'] == symbol]
            agg_df = result[result['symbol'] == symbol]

            # Volume should be preserved (within rounding)
            orig_volume = orig_df['volume'].sum()
            agg_volume = agg_df['volume'].sum()
            assert abs(orig_volume - agg_volume) / orig_volume < 0.01  # Within 1%

    def test_full_file_aggregation_15min(self):
        """Test aggregating entire real data file to 15 minutes"""
        try:
            df = pd.read_csv('data/SPXU_GDXU_UPRO_1min.csv')
        except FileNotFoundError:
            pytest.skip("Real data file not found")

        result = aggregate_stock_data(df, interval='15min')
        print(f"\n15-minute aggregated: {len(result)} rows")

        assert len(result) < len(df)

    def test_full_file_aggregation_hourly(self):
        """Test aggregating entire real data file to hourly"""
        try:
            df = pd.read_csv('data/SPXU_GDXU_UPRO_1min.csv')
        except FileNotFoundError:
            pytest.skip("Real data file not found")

        result = aggregate_stock_data(df, interval='1h')
        print(f"\nHourly aggregated: {len(result)} rows")

        assert len(result) < len(df)

    def test_full_file_aggregation_daily(self):
        """Test aggregating entire real data file to daily"""
        try:
            df = pd.read_csv('data/SPXU_GDXU_UPRO_1min.csv')
        except FileNotFoundError:
            pytest.skip("Real data file not found")

        result = aggregate_stock_data(df, interval='1D')
        print(f"\nDaily aggregated: {len(result)} rows")

        assert len(result) < len(df)

        # Daily data should be much smaller
        assert len(result) < len(df) / 100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
