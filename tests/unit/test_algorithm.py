"""
Unit tests for Algorithm base class
"""
import pytest
from collections import deque

from core.algorithm import Algorithm
from core.classes import PriceData, MarketSignal, SignalType


class MockAlgorithm(Algorithm):
    """Mock algorithm for testing"""

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """Simple mock logic - buy if price > 150"""
        signals = []
        for pd in data:
            if pd.close > 150:
                signals.append(MarketSignal(SignalType.BUY, pd.symbol, 100))
            else:
                signals.append(MarketSignal(SignalType.SELL, pd.symbol, 50))
        return signals


class TestAlgorithm:
    """Tests for Algorithm base class"""

    def test_algorithm_initialization_default(self):
        """Test algorithm with default config"""
        algo = MockAlgorithm()

        assert algo.history_length == 0
        assert algo.cfg['full_history'] is False
        assert algo.full_history == []

    def test_algorithm_initialization_with_config(self, algorithm_config):
        """Test algorithm with custom config"""
        algo = MockAlgorithm(algorithm_config)

        assert algo.history_length == 10
        assert algo.cfg['full_history'] is True

    def test_algorithm_history_tracking(self, sample_pricedata_list):
        """Test that algorithm tracks price history"""
        cfg = {"history_length": 5, "full_history": False}
        algo = MockAlgorithm(cfg)

        # Process first tick
        algo.on_data(sample_pricedata_list)

        # Check that history was updated
        assert "AAPL" in algo.price_history
        assert "GOOGL" in algo.price_history
        assert "MSFT" in algo.price_history

        assert len(algo.price_history["AAPL"]) == 1
        assert algo.price_history["AAPL"][0] == 151.0

    def test_algorithm_history_length_limit(self, sample_pricedata_list):
        """Test that history respects maxlen"""
        cfg = {"history_length": 3, "full_history": False}
        algo = MockAlgorithm(cfg)

        # Process multiple ticks
        for i in range(5):
            algo.on_data(sample_pricedata_list)

        # History should only keep last 3
        assert len(algo.price_history["AAPL"]) == 3
        assert isinstance(algo.price_history["AAPL"], deque)

    def test_algorithm_price_data_history(self, sample_pricedata_list):
        """Test that full PriceData objects are stored in history"""
        cfg = {"history_length": 5, "full_history": False}
        algo = MockAlgorithm(cfg)

        algo.on_data(sample_pricedata_list)

        assert "AAPL" in algo.price_data_history
        assert len(algo.price_data_history["AAPL"]) == 1
        assert isinstance(algo.price_data_history["AAPL"][0], PriceData)
        assert algo.price_data_history["AAPL"][0].symbol == "AAPL"

    def test_algorithm_full_history_disabled(self, sample_pricedata_list):
        """Test that full history is not stored when disabled"""
        cfg = {"history_length": 5, "full_history": False}
        algo = MockAlgorithm(cfg)

        algo.on_data(sample_pricedata_list)
        algo.on_data(sample_pricedata_list)

        assert len(algo.full_history) == 0

    def test_algorithm_full_history_enabled(self, sample_pricedata_list):
        """Test that full history is stored when enabled"""
        cfg = {"history_length": 5, "full_history": True}
        algo = MockAlgorithm(cfg)

        algo.on_data(sample_pricedata_list)
        algo.on_data(sample_pricedata_list)

        assert len(algo.full_history) == 2
        assert isinstance(algo.full_history[0], list)

    def test_algorithm_zero_history_length(self, sample_pricedata_list):
        """Test algorithm with no history tracking"""
        cfg = {"history_length": 0, "full_history": False}
        algo = MockAlgorithm(cfg)

        algo.on_data(sample_pricedata_list)

        # With history_length = 0, price_history should be empty dict
        assert algo.price_history == {}

    def test_algorithm_on_data_returns_signals(self, sample_pricedata_list):
        """Test that on_data returns market signals"""
        algo = MockAlgorithm()

        signals = algo.on_data(sample_pricedata_list)

        assert isinstance(signals, list)
        assert len(signals) == 3
        assert all(isinstance(s, MarketSignal) for s in signals)

    def test_algorithm_signal_logic(self, sample_pricedata_list):
        """Test that algorithm logic is executed correctly"""
        algo = MockAlgorithm()

        signals = algo.on_data(sample_pricedata_list)

        # AAPL close=151 > 150 -> BUY
        aapl_signal = next(s for s in signals if s.symbol == "AAPL")
        assert aapl_signal.type == SignalType.BUY
        assert aapl_signal.strength == 100

    def test_algorithm_multiple_symbols(self, sample_pricedata_list):
        """Test algorithm handles multiple symbols correctly"""
        cfg = {"history_length": 5, "full_history": False}
        algo = MockAlgorithm(cfg)

        # Process 3 ticks
        for _ in range(3):
            algo.on_data(sample_pricedata_list)

        # Each symbol should have 3 entries in history
        assert len(algo.price_history["AAPL"]) == 3
        assert len(algo.price_history["GOOGL"]) == 3
        assert len(algo.price_history["MSFT"]) == 3

    def test_algorithm_config_merging(self):
        """Test that custom config is merged with defaults"""
        custom_cfg = {"history_length": 20}
        algo = MockAlgorithm(custom_cfg)

        # Custom value should be used
        assert algo.history_length == 20
        # Default value should be preserved
        assert algo.cfg['full_history'] is False
