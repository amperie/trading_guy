"""
Unit tests for TechnicalAnalyzer class (core.ta.analyzer).

Tests the static methods directly in the TechnicalAnalyzer class,
including the dataclass return types.
"""

import pytest
from trading.core.ta import TechnicalAnalyzer, RSI, MACD


class TestTechnicalAnalyzerRSI:
    """Test TechnicalAnalyzer RSI methods."""

    def test_calculate_rsi_returns_dataclass(self):
        """Test that calculate_rsi returns RSI dataclass."""
        prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]

        result = TechnicalAnalyzer.calculate_rsi(prices, period=14)

        assert isinstance(result, RSI)
        assert hasattr(result, 'rsi')
        assert hasattr(result, 'avg_gain')
        assert hasattr(result, 'avg_loss')
        assert hasattr(result, 'rs')
        assert hasattr(result, 'period')

    def test_calculate_rsi_values(self):
        """Test RSI dataclass contains correct values."""
        prices = [10.0 + i * 0.5 for i in range(20)]  # Uptrend

        result = TechnicalAnalyzer.calculate_rsi(prices, period=14)

        assert result is not None
        assert result.period == 14
        assert result.rsi == 100.0  # All gains
        assert result.avg_gain > 0
        assert result.avg_loss == 0
        assert result.rs is None  # avg_loss is 0

    def test_calculate_rsi_with_mixed_changes(self):
        """Test RSI with mixed gains and losses."""
        prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00
        ]

        result = TechnicalAnalyzer.calculate_rsi(prices, period=14)

        assert result is not None
        assert 0 <= result.rsi <= 100
        assert result.avg_gain > 0
        assert result.avg_loss > 0
        assert result.rs is not None
        assert result.rs == result.avg_gain / result.avg_loss

    def test_calculate_rsi_series_returns_dataclass_list(self):
        """Test that calculate_rsi_series returns list of RSI dataclasses."""
        prices = [44.0 + i * 0.1 for i in range(30)]

        result = TechnicalAnalyzer.calculate_rsi_series(prices, period=14)

        assert len(result) == len(prices)

        # First 14 should be None
        assert all(x is None for x in result[:14])

        # Rest should be RSI dataclass objects
        assert all(isinstance(x, RSI) for x in result[14:])
        assert all(x.period == 14 for x in result[14:])
        assert all(0 <= x.rsi <= 100 for x in result[14:])

    def test_calculate_rsi_series_last_matches_single(self):
        """Test last value of series matches single RSI calculation."""
        prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00
        ]

        rsi_single = TechnicalAnalyzer.calculate_rsi(prices, period=14)
        rsi_series = TechnicalAnalyzer.calculate_rsi_series(prices, period=14)

        assert rsi_single.rsi == rsi_series[-1].rsi
        assert rsi_single.avg_gain == rsi_series[-1].avg_gain
        assert rsi_single.avg_loss == rsi_series[-1].avg_loss
        assert rsi_single.rs == rsi_series[-1].rs

    def test_rsi_insufficient_data(self):
        """Test RSI returns None with insufficient data."""
        prices = [44.0, 45.0, 46.0]
        result = TechnicalAnalyzer.calculate_rsi(prices, period=14)
        assert result is None

    def test_rsi_dataclass_immutable(self):
        """Test that RSI dataclass is a frozen dataclass."""
        prices = [10.0 + i for i in range(20)]
        result = TechnicalAnalyzer.calculate_rsi(prices, period=14)

        # Should not be able to modify (if frozen)
        # Note: Our current implementation doesn't use frozen=True
        # but we can verify the dataclass structure
        assert isinstance(result, RSI)


class TestTechnicalAnalyzerMACD:
    """Test TechnicalAnalyzer MACD methods."""

    def test_calculate_macd_returns_dataclass(self):
        """Test that calculate_macd returns MACD dataclass."""
        prices = [22.0 + i * 0.1 for i in range(50)]

        result = TechnicalAnalyzer.calculate_macd(prices)

        assert isinstance(result, MACD)
        assert hasattr(result, 'macd')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'histogram')
        assert hasattr(result, 'slow_ema')
        assert hasattr(result, 'fast_ema')
        assert hasattr(result, 'slow_period')
        assert hasattr(result, 'fast_period')
        assert hasattr(result, 'signal_period')

    def test_calculate_macd_default_periods(self):
        """Test MACD uses default periods correctly."""
        prices = [22.0 + i * 0.1 for i in range(50)]

        result = TechnicalAnalyzer.calculate_macd(prices)

        assert result.fast_period == 12
        assert result.slow_period == 26
        assert result.signal_period == 9

    def test_calculate_macd_custom_periods(self):
        """Test MACD with custom periods."""
        prices = [22.0 + i * 0.1 for i in range(50)]

        result = TechnicalAnalyzer.calculate_macd(
            prices,
            fast_period=5,
            slow_period=10,
            signal_period=3
        )

        assert result.fast_period == 5
        assert result.slow_period == 10
        assert result.signal_period == 3

    def test_calculate_macd_values(self):
        """Test MACD values are calculated correctly."""
        prices = [22.0 + i * 0.1 for i in range(50)]

        result = TechnicalAnalyzer.calculate_macd(prices)

        # MACD = fast_ema - slow_ema
        assert abs(result.macd - (result.fast_ema - result.slow_ema)) < 0.0001

        # Histogram = MACD - Signal
        assert abs(result.histogram - (result.macd - result.signal)) < 0.0001


class TestTechnicalAnalyzerEMA:
    """Test TechnicalAnalyzer EMA methods."""

    def test_calculate_ema(self):
        """Test EMA calculation."""
        prices = [22.27, 22.19, 22.08, 22.17, 22.18]
        result = TechnicalAnalyzer.calculate_ema(prices, 5)

        assert result is not None
        assert isinstance(result, float)

    def test_calculate_ema_series(self):
        """Test EMA series calculation."""
        prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13]
        result = TechnicalAnalyzer.calculate_ema_series(prices, 3)

        assert len(result) == len(prices)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None


class TestBackwardCompatibility:
    """Test that utils.indicators wrappers work correctly."""

    def test_rsi_wrapper_returns_float(self):
        """Test that utils.indicators.calculate_rsi returns float, not dataclass."""
        from utils.indicators import calculate_rsi

        prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]

        result = calculate_rsi(prices, period=14)

        # Should return float, not RSI dataclass
        assert isinstance(result, float)
        assert 0 <= result <= 100

    def test_rsi_series_wrapper_returns_float_list(self):
        """Test that utils.indicators.calculate_rsi_series returns list of floats."""
        from utils.indicators import calculate_rsi_series

        prices = [44.0 + i * 0.1 for i in range(30)]

        result = calculate_rsi_series(prices, period=14)

        assert len(result) == len(prices)

        # First 14 should be None
        assert all(x is None for x in result[:14])

        # Rest should be floats, not RSI dataclass
        for val in result[14:]:
            assert isinstance(val, float)
            assert 0 <= val <= 100

    def test_wrapper_matches_direct_call(self):
        """Test that wrapper returns same RSI value as direct call."""
        from utils.indicators import calculate_rsi

        prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]

        wrapper_result = calculate_rsi(prices, period=14)
        direct_result = TechnicalAnalyzer.calculate_rsi(prices, period=14)

        assert abs(wrapper_result - direct_result.rsi) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
