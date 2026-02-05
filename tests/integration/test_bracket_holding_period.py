"""
Integration test for DualSymbolSwitchPortfolio with bracket orders and holding period.
Converted from root-level test_bracket_holding_period.py script.
"""
import os
import pytest

from trading.core.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm
from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.core.om.backtesting_om import BacktestingOM

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'SPY_UPRO_SPXU_5min_sorted.csv')
DATA_EXISTS = os.path.exists(DATA_PATH)


@pytest.mark.integration
@pytest.mark.skipif(not DATA_EXISTS, reason="Data file SPY_UPRO_SPXU_5min_sorted.csv not available")
class TestBracketHoldingPeriod:
    """Test bracket orders with holding period in a full backtest."""

    @pytest.fixture
    def backtest_results(self):
        """Run a backtest with bracket orders and 2-hour holding period."""
        from trading.backtesting.run_backtest_ray import run_single_backtest
        from trading.analysis import analysis_engine

        om = BacktestingOM()
        alg = SpyTrendSwitchAlgorithm({
            'spy_symbol': 'SPY',
            'upro_symbol': 'UPRO',
            'spxu_symbol': 'SPXU',
            'fast_window': 22,
            'slow_window': 220,
            'strength_scale': 44.0
        })
        dp = TestDataProvider({
            'path': DATA_PATH,
            'truncate': 0,
            'start_date': '2022-01-03',
            'end_date': '2022-03-01'
        })
        pf = DualSymbolSwitchPortfolio({
            'upro_symbol': 'UPRO',
            'spxu_symbol': 'SPXU',
            'tx_cost': 0.0,
            'min_signal_strength': 0,
            'stop_pct': 5.0,
            'profit_pct': 10.0,
            'holding_period_hours': 2
        }, om, 1000.0, {}, True)

        backtest_cfg = {
            'experiment_name': 'Dual Symbol Bracket Orders',
            'run_name': 'Holding Period Test',
            'description': 'Testing bracket orders with 2-hour holding period',
            'starting_cash': 1000.0
        }

        # Patch run_full_analysis to skip MLflow logging
        original = analysis_engine.AnalysisEngine.run_full_analysis

        def patched(self, **kwargs):
            kwargs['log_to_mlflow'] = False
            return original(self, **kwargs)

        analysis_engine.AnalysisEngine.run_full_analysis = patched
        try:
            results = run_single_backtest(backtest_cfg, alg, pf, dp)
        finally:
            analysis_engine.AnalysisEngine.run_full_analysis = original

        return results, pf, om

    def test_backtest_completes(self, backtest_results):
        """Test that the backtest runs to completion."""
        results, pf, om = backtest_results
        assert results is not None
        assert results['metrics'] is not None

    def test_metrics_are_reasonable(self, backtest_results):
        """Test that metrics have reasonable values."""
        results, pf, om = backtest_results
        metrics = results['metrics']
        assert metrics.total_return_pct > -100  # Portfolio retained some value
        assert metrics.total_trades >= 0

    def test_bracket_orders_created(self, backtest_results):
        """Test that bracket orders were created during the backtest."""
        results, pf, om = backtest_results
        bracket_count = sum(
            1 for order_id in om.all_orders
            if hasattr(om.all_orders[order_id], 'type')
            and om.all_orders[order_id].type.name == 'BRACKET'
        )
        assert bracket_count > 0, "Should have created bracket orders"
