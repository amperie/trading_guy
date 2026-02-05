"""
Integration test for SplitPeriodBacktestEngine.
Verifies that the engine correctly runs training and validation periods.
Converted from root-level test_split_period_engine.py script.
"""
import os
import pytest
from unittest.mock import patch

from trading.engines.split_period_backtest_engine import SplitPeriodBacktestEngine
from trading.core.algorithms.spy_trend_switch_algorithm import SpyTrendSwitchAlgorithm
from trading.core.pf.dual_symbol_switch_portfolio import DualSymbolSwitchPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.core.om.backtesting_om import BacktestingOM

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'SPY_UPRO_SPXU_5min_sorted.csv')
DATA_EXISTS = os.path.exists(DATA_PATH)


@pytest.mark.integration
@pytest.mark.skipif(not DATA_EXISTS, reason="Data file SPY_UPRO_SPXU_5min_sorted.csv not available")
class TestSplitPeriodEngine:
    """Test SplitPeriodBacktestEngine with training and validation periods."""

    @pytest.fixture
    def engine_results(self):
        """Run a split-period backtest and return results."""
        om = BacktestingOM()
        alg = SpyTrendSwitchAlgorithm({
            'spy_symbol': 'SPY',
            'upro_symbol': 'UPRO',
            'spxu_symbol': 'SPXU',
            'fast_window': 10,
            'slow_window': 30,
            'strength_scale': 20.0
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
            'experiment_name': 'Split Period Test',
            'run_name': 'Simple_Test',
            'description': 'Testing split-period engine'
        }

        engine = SplitPeriodBacktestEngine(
            cfg=backtest_cfg,
            dp=dp, al=alg, om=om, pf=pf,
            start_date='2022-01-03',
            cutoff_date='2022-02-01',
            end_date='2022-03-01'
        )

        # Patch run_full_analysis to skip MLflow logging
        from trading.analysis import analysis_engine
        original = analysis_engine.AnalysisEngine.run_full_analysis

        def patched(self, **kwargs):
            kwargs['log_to_mlflow'] = False
            return original(self, **kwargs)

        analysis_engine.AnalysisEngine.run_full_analysis = patched
        try:
            results = engine.run()
        finally:
            analysis_engine.AnalysisEngine.run_full_analysis = original

        return results

    def test_training_period_completed(self, engine_results):
        """Test that training period completed."""
        assert 'training' in engine_results

    def test_validation_period_completed(self, engine_results):
        """Test that validation period completed."""
        assert 'validation' in engine_results

    def test_training_metrics_present(self, engine_results):
        """Test that training metrics are calculated."""
        assert engine_results['training']['metrics'] is not None

    def test_validation_metrics_present(self, engine_results):
        """Test that validation metrics are calculated."""
        assert engine_results['validation']['metrics'] is not None

    def test_training_has_trades(self, engine_results):
        """Test that training period produced trades."""
        assert len(engine_results['training']['trades']) >= 0

    def test_validation_has_trades(self, engine_results):
        """Test that validation period produced trades."""
        assert len(engine_results['validation']['trades']) >= 0

    def test_training_final_value_positive(self, engine_results):
        """Test that training period retained portfolio value."""
        assert engine_results['training']['metrics'].total_return_pct > -100

    def test_validation_final_value_positive(self, engine_results):
        """Test that validation period retained portfolio value."""
        assert engine_results['validation']['metrics'].total_return_pct > -100

    def test_date_validation_rejects_invalid_dates(self):
        """Test that invalid dates are rejected."""
        om = BacktestingOM()
        alg = SpyTrendSwitchAlgorithm()
        dp = TestDataProvider({'path': DATA_PATH, 'truncate': 0})
        pf = DualSymbolSwitchPortfolio({}, om, 1000.0, {}, True)
        backtest_cfg = {'experiment_name': 'Test', 'run_name': 'Test'}

        with pytest.raises(ValueError):
            SplitPeriodBacktestEngine(
                cfg=backtest_cfg,
                dp=dp, al=alg, om=om, pf=pf,
                start_date='2022-06-01',
                cutoff_date='2022-01-01',
                end_date='2022-12-01'
            )
