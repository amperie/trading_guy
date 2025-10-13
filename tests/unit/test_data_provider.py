"""
Unit tests for DataProvider implementations
"""
import pytest
import pandas as pd
from datetime import datetime, timezone

from data_providers.data_provider import DataProvider
from data_providers.test_data_provider import TestDataProvider
from core.classes import PriceData


class TestTestDataProvider:
    """Tests for TestDataProvider (CSV file reader)"""

    def test_test_data_provider_initialization(self, data_provider_config):
        """Test TestDataProvider initialization"""
        dp = TestDataProvider(data_provider_config)

        assert dp.cfg == data_provider_config
        assert dp.data is None

    def test_test_data_provider_load_data(self, data_provider_config):
        """Test loading data from CSV"""
        dp = TestDataProvider(data_provider_config)
        dp.load_data()

        assert dp.data is not None
        assert isinstance(dp.data, pd.DataFrame)
        assert len(dp.data) > 0

    def test_test_data_provider_get_data(self, data_provider_config):
        """Test get_data method"""
        dp = TestDataProvider(data_provider_config)
        dp.load_data()

        data = dp.get_data()

        assert data is not None
        assert isinstance(data, pd.DataFrame)

    def test_test_data_provider_iterate(self, data_provider_config):
        """Test iterate generator"""
        dp = TestDataProvider(data_provider_config)

        # Iterate should auto-load data
        ticks = list(dp.iterate())

        assert len(ticks) > 0
        assert all(isinstance(tick, list) for tick in ticks)
        assert all(isinstance(pd, PriceData) for tick in ticks for pd in tick)

    def test_test_data_provider_pricedata_structure(self, data_provider_config):
        """Test that PriceData objects have correct structure"""
        dp = TestDataProvider(data_provider_config)

        ticks = list(dp.iterate())
        first_tick = ticks[0]
        first_pd = first_tick[0]

        assert hasattr(first_pd, 'symbol')
        assert hasattr(first_pd, 'timestamp')
        assert hasattr(first_pd, 'open')
        assert hasattr(first_pd, 'high')
        assert hasattr(first_pd, 'low')
        assert hasattr(first_pd, 'close')
        assert hasattr(first_pd, 'volume')

    def test_test_data_provider_chronological_order(self, data_provider_config):
        """Test that data is returned in chronological order"""
        dp = TestDataProvider(data_provider_config)

        ticks = list(dp.iterate())

        # Check that timestamps are increasing
        timestamps = [tick[0].timestamp for tick in ticks]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]

    def test_test_data_provider_groups_by_timestamp(self, data_provider_config):
        """Test that data is grouped by timestamp"""
        dp = TestDataProvider(data_provider_config)

        ticks = list(dp.iterate())

        # Each tick should have all symbols for that timestamp
        for tick in ticks:
            # All PriceData in a tick should have the same timestamp
            timestamps = [pd.timestamp for pd in tick]
            assert len(set(timestamps)) == 1

    def test_test_data_provider_truncate_zero(self, sample_csv_data):
        """Test with truncate=0 (no truncation)"""
        cfg = {
            "provider": "data_providers.test_data_provider.TestDataProvider",
            "path": str(sample_csv_data),
            "truncate": 0
        }
        dp = TestDataProvider(cfg)

        ticks = list(dp.iterate())

        # Should have 3 timestamps (from fixture)
        assert len(ticks) == 3

    def test_test_data_provider_truncate_positive(self, sample_csv_data):
        """Test with truncate > 0"""
        cfg = {
            "provider": "data_providers.test_data_provider.TestDataProvider",
            "path": str(sample_csv_data),
            "truncate": 2
        }
        dp = TestDataProvider(cfg)
        dp.load_data()

        # Should only load first 2 rows
        assert len(dp.data) == 2

    def test_test_data_provider_relative_path(self, tmp_path):
        """Test that relative paths are resolved correctly"""
        # Create a CSV in the tmp directory
        data = {
            'symbol': ['AAPL', 'AAPL'],
            'timestamp': ['2024-01-15 10:00:00+00:00', '2024-01-15 10:01:00+00:00'],
            'open': [150.0, 151.0],
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [151.0, 152.0],
            'volume': [100000, 110000]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        cfg = {
            "provider": "data_providers.test_data_provider.TestDataProvider",
            "path": str(csv_path),
            "truncate": 0
        }

        # Should not raise
        dp = TestDataProvider(cfg)
        dp.load_data()

        assert dp.data is not None

    def test_test_data_provider_multiple_symbols_per_tick(self, sample_csv_data):
        """Test that multiple symbols are grouped in same tick"""
        cfg = {
            "provider": "data_providers.test_data_provider.TestDataProvider",
            "path": str(sample_csv_data),
            "truncate": 0
        }
        dp = TestDataProvider(cfg)

        ticks = list(dp.iterate())

        # Each tick should have 2 symbols (AAPL and GOOGL from fixture)
        for tick in ticks:
            assert len(tick) == 2
            symbols = {pd.symbol for pd in tick}
            assert symbols == {'AAPL', 'GOOGL'}


class TestDataProviderBase:
    """Tests for DataProvider abstract base class"""

    def test_data_provider_is_abstract(self):
        """Test that DataProvider cannot be instantiated directly"""
        # DataProvider has abstract method load_data
        with pytest.raises(TypeError):
            dp = DataProvider()

    def test_data_provider_requires_load_data(self):
        """Test that subclasses must implement load_data"""

        class IncompleteDP(DataProvider):
            """Missing load_data implementation"""
            pass

        with pytest.raises(TypeError):
            dp = IncompleteDP()

    def test_data_provider_complete_implementation(self):
        """Test that complete implementation works"""

        class CompleteDP(DataProvider):
            def load_data(self):
                self.data = pd.DataFrame({
                    'symbol': ['AAPL', 'AAPL'],
                    'timestamp': ['2024-01-15 10:00:00+00:00', '2024-01-15 10:01:00+00:00'],
                    'open': [150.0, 151.0],
                    'high': [151.0, 152.0],
                    'low': [149.0, 150.0],
                    'close': [151.0, 152.0],
                    'volume': [100000, 110000]
                })

        cfg = {
            "provider": "test",
            "path": "test.csv",
            "truncate": 0
        }

        # Should not raise
        dp = CompleteDP(cfg)
        assert dp.data is None

        dp.load_data()
        assert dp.data is not None

    def test_data_provider_iterate_auto_loads(self):
        """Test that iterate auto-loads data if not loaded"""

        class AutoLoadDP(DataProvider):
            def load_data(self):
                self.data = pd.DataFrame({
                    'symbol': ['AAPL'],
                    'timestamp': ['2024-01-15 10:00:00+00:00'],
                    'open': [150.0],
                    'high': [151.0],
                    'low': [149.0],
                    'close': [151.0],
                    'volume': [100000]
                })

        cfg = {
            "provider": "test",
            "path": "test.csv",
            "truncate": 0
        }

        dp = AutoLoadDP(cfg)
        assert dp.data is None

        # Iterate should load data automatically
        ticks = list(dp.iterate())

        assert len(ticks) == 1
        assert dp.data is not None

    def test_data_provider_config_merging(self):
        """Test that config is properly merged with defaults"""

        class TestDP(DataProvider):
            def load_data(self):
                self.data = pd.DataFrame()

        custom_cfg = {"path": "custom.csv"}
        dp = TestDP(custom_cfg)

        # Should have the custom config
        assert "path" in dp.cfg
