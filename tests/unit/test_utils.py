"""
Unit tests for utility functions
"""
import pytest
from datetime import datetime, timezone

from utils.utils import (
    instantiate_from_string,
    find_pricedata_in_list,
    find_marketsignal_in_list
)
from core.classes import PriceData, MarketSignal, SignalType
from data_providers.test_data_provider import TestDataProvider
from algorithms.test_algorithm import TestAlgorithm


class TestInstantiateFromString:
    """Tests for instantiate_from_string function"""

    def test_instantiate_simple_class(self):
        """Test instantiating a class without arguments"""
        instance = instantiate_from_string(
            "data_providers.test_data_provider.TestDataProvider"
        )

        assert isinstance(instance, TestDataProvider)

    def test_instantiate_with_kwargs(self, data_provider_config):
        """Test instantiating a class with keyword arguments"""
        instance = instantiate_from_string(
            "data_providers.test_data_provider.TestDataProvider",
            cfg=data_provider_config
        )

        assert isinstance(instance, TestDataProvider)
        assert instance.cfg == data_provider_config

    def test_instantiate_algorithm(self):
        """Test instantiating an algorithm"""
        instance = instantiate_from_string(
            "algorithms.test_algorithm.TestAlgorithm"
        )

        assert isinstance(instance, TestAlgorithm)

    def test_instantiate_with_config(self, algorithm_config):
        """Test instantiating with configuration"""
        instance = instantiate_from_string(
            "algorithms.test_algorithm.TestAlgorithm",
            cfg=algorithm_config
        )

        assert instance.history_length == 10
        assert instance.cfg['full_history'] is True

    def test_instantiate_invalid_path(self):
        """Test that invalid path raises error"""
        with pytest.raises(ModuleNotFoundError):
            instantiate_from_string("invalid.module.path.ClassName")

    def test_instantiate_invalid_class(self):
        """Test that invalid class name raises error"""
        with pytest.raises(AttributeError):
            instantiate_from_string("data_providers.test_data_provider.InvalidClass")

    def test_instantiate_with_positional_args(self):
        """Test instantiating with positional arguments"""
        # Note: Most of our classes use keyword args, but test the mechanism
        instance = instantiate_from_string(
            "algorithms.test_algorithm.TestAlgorithm",
            {"history_length": 5}
        )

        assert isinstance(instance, TestAlgorithm)


class TestFindPriceDataInList:
    """Tests for find_pricedata_in_list function"""

    def test_find_existing_symbol(self, sample_pricedata_list):
        """Test finding an existing symbol"""
        result = find_pricedata_in_list("AAPL", sample_pricedata_list)

        assert result is not None
        assert isinstance(result, PriceData)
        assert result.symbol == "AAPL"

    def test_find_nonexistent_symbol(self, sample_pricedata_list):
        """Test finding a symbol that doesn't exist"""
        result = find_pricedata_in_list("TSLA", sample_pricedata_list)

        assert result is None

    def test_find_in_empty_list(self):
        """Test finding in empty list"""
        result = find_pricedata_in_list("AAPL", [])

        assert result is None

    def test_find_multiple_symbols(self, sample_pricedata_list):
        """Test finding different symbols"""
        aapl = find_pricedata_in_list("AAPL", sample_pricedata_list)
        googl = find_pricedata_in_list("GOOGL", sample_pricedata_list)
        msft = find_pricedata_in_list("MSFT", sample_pricedata_list)

        assert aapl.symbol == "AAPL"
        assert googl.symbol == "GOOGL"
        assert msft.symbol == "MSFT"

    def test_find_returns_correct_data(self, sample_pricedata_list):
        """Test that found data has correct values"""
        result = find_pricedata_in_list("GOOGL", sample_pricedata_list)

        assert result.close == 2810.0
        assert result.volume == 500000.0

    def test_find_first_match_only(self, sample_timestamp):
        """Test that only first match is returned if duplicates exist"""
        duplicate_list = [
            PriceData("AAPL", sample_timestamp, 150.0, 151.0, 149.0, 151.0, 100000),
            PriceData("AAPL", sample_timestamp, 152.0, 153.0, 150.0, 152.0, 110000)
        ]

        result = find_pricedata_in_list("AAPL", duplicate_list)

        # Should return first match
        assert result.close == 151.0

    def test_find_case_sensitive(self, sample_pricedata_list):
        """Test that symbol matching is case-sensitive"""
        result = find_pricedata_in_list("aapl", sample_pricedata_list)

        # Should not find lowercase
        assert result is None


class TestFindMarketSignalInList:
    """Tests for find_marketsignal_in_list function"""

    def test_find_existing_signal(self, sample_market_signals):
        """Test finding an existing signal"""
        result = find_marketsignal_in_list("AAPL", sample_market_signals)

        assert result is not None
        assert isinstance(result, MarketSignal)
        assert result.symbol == "AAPL"

    def test_find_nonexistent_signal(self, sample_market_signals):
        """Test finding a signal that doesn't exist"""
        result = find_marketsignal_in_list("TSLA", sample_market_signals)

        assert result is None

    def test_find_in_empty_list(self):
        """Test finding in empty list"""
        result = find_marketsignal_in_list("AAPL", [])

        assert result is None

    def test_find_multiple_signals(self, sample_market_signals):
        """Test finding different signals"""
        aapl = find_marketsignal_in_list("AAPL", sample_market_signals)
        googl = find_marketsignal_in_list("GOOGL", sample_market_signals)
        msft = find_marketsignal_in_list("MSFT", sample_market_signals)

        assert aapl.symbol == "AAPL"
        assert googl.symbol == "GOOGL"
        assert msft.symbol == "MSFT"

    def test_find_returns_correct_signal(self, sample_market_signals):
        """Test that found signal has correct values"""
        result = find_marketsignal_in_list("AAPL", sample_market_signals)

        assert result.type == SignalType.BUY
        assert result.strength == 80

    def test_find_different_signal_types(self, sample_market_signals):
        """Test finding signals with different types"""
        buy_signal = find_marketsignal_in_list("AAPL", sample_market_signals)
        sell_signal = find_marketsignal_in_list("GOOGL", sample_market_signals)

        assert buy_signal.type == SignalType.BUY
        assert sell_signal.type == SignalType.SELL

    def test_find_first_match_only(self):
        """Test that only first match is returned if duplicates exist"""
        duplicate_list = [
            MarketSignal(SignalType.BUY, "AAPL", 80),
            MarketSignal(SignalType.SELL, "AAPL", 60)
        ]

        result = find_marketsignal_in_list("AAPL", duplicate_list)

        # Should return first match
        assert result.type == SignalType.BUY

    def test_find_case_sensitive(self, sample_market_signals):
        """Test that symbol matching is case-sensitive"""
        result = find_marketsignal_in_list("aapl", sample_market_signals)

        # Should not find lowercase
        assert result is None


class TestUtilityFunctionsIntegration:
    """Integration tests for utility functions"""

    def test_find_functions_consistency(
            self, sample_pricedata_list, sample_market_signals):
        """Test that find functions work consistently"""
        # Find PriceData
        pd = find_pricedata_in_list("AAPL", sample_pricedata_list)

        # Find corresponding MarketSignal
        signal = find_marketsignal_in_list("AAPL", sample_market_signals)

        # Both should find AAPL
        assert pd is not None
        assert signal is not None
        assert pd.symbol == signal.symbol == "AAPL"

    def test_instantiate_and_use(self, data_provider_config):
        """Test instantiating and using a class"""
        dp = instantiate_from_string(
            "data_providers.test_data_provider.TestDataProvider",
            cfg=data_provider_config
        )

        # Should be able to use the instance
        ticks = list(dp.iterate())

        assert len(ticks) > 0
        first_pd = ticks[0][0]

        # Should be able to find it
        result = find_pricedata_in_list(first_pd.symbol, ticks[0])
        assert result is not None
