"""
Smoke tests for Ray Tune MACD hyperparameter optimization imports.
Converted from root-level test_ray_macd_bracket_config.py script.
"""
import pytest


class TestRayImports:
    """Verify Ray-related imports work correctly."""

    def test_spy_trend_macd_algorithm_import(self):
        """Test that SpyTrendMACDAlgorithm can be imported."""
        from trading.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
        assert SpyTrendMACDAlgorithm is not None

    def test_dual_symbol_switch_portfolio_import(self):
        """Test that DualSymbolSwitchPortfolio can be imported."""
        from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
        assert DualSymbolSwitchPortfolio is not None

    def test_run_launchers_import(self):
        """Test that run_launchers module can be imported."""
        try:
            from trading.launchers.run_launchers import run_ray_spy_trend_macd
            assert run_ray_spy_trend_macd is not None
        except ImportError:
            pytest.skip("run_launchers module not available or missing ray dependency")
