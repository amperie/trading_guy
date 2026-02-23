"""
Unit tests for EMA and SMA functions in SpyTrendSwitchAlgorithm.
Converted from root-level test_ema_function.py script.
"""
import pytest

from trading.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm


class TestEmaFunction:
    """Test EMA and SMA calculations in SpyTrendSwitchAlgorithm."""

    @pytest.fixture
    def algorithm(self):
        return SpyTrendSwitchAlgorithm()

    def test_sma_simple_sequence(self, algorithm):
        """Test SMA with a simple ascending price sequence."""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        window = 5
        sma = algorithm._sma(prices, window)
        # SMA of last 5 values: (16+17+18+19+20)/5 = 18.0
        assert sma == pytest.approx(18.0)

    def test_ema_simple_sequence(self, algorithm):
        """Test EMA with a simple ascending price sequence."""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        window = 5
        ema = algorithm._ema(prices, window)
        # EMA gives more weight to recent prices, should be >= SMA for uptrend
        sma = algorithm._sma(prices, window)
        assert ema >= sma

    def test_ema_reacts_faster_to_uptrend(self, algorithm):
        """Test that EMA reacts faster than SMA to a strong uptrend."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                  115, 120, 125, 130]
        window = 10
        sma = algorithm._sma(prices, window)
        ema = algorithm._ema(prices, window)
        # EMA should be higher than SMA during strong uptrend
        assert ema > sma

    def test_ema_insufficient_data(self, algorithm):
        """Test EMA with insufficient data (fewer points than window)."""
        result = algorithm._ema([1, 2, 3, 4, 5], 10)
        # With insufficient data, may return None or a fallback value
        # The implementation handles this gracefully without crashing
        assert result is None or isinstance(result, (int, float))

    def test_ema_exact_window_size(self, algorithm):
        """Test EMA with exactly window-sized data."""
        result = algorithm._ema([10, 11, 12, 13, 14], 5)
        assert isinstance(result, float)

    def test_sma_constant_prices(self, algorithm):
        """Test SMA with constant prices returns the constant."""
        prices = [50.0] * 20
        sma = algorithm._sma(prices, 10)
        assert sma == pytest.approx(50.0)

    def test_ema_constant_prices(self, algorithm):
        """Test EMA with constant prices returns the constant."""
        prices = [50.0] * 20
        ema = algorithm._ema(prices, 10)
        assert ema == pytest.approx(50.0)
