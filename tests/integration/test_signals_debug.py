"""
Integration test for signal generation and order creation flow.
Converted from root-level test_signals_debug.py script.
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
class TestSignalsDebug:
    """Test that the algorithm generates signals and the portfolio creates orders."""

    def test_signal_generation_and_order_creation(self):
        """Test end-to-end signal generation and order creation."""
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
            'start_date': '2022-01-03'
        })
        pf = DualSymbolSwitchPortfolio({
            'upro_symbol': 'UPRO',
            'spxu_symbol': 'SPXU',
            'tx_cost': 0.0,
            'min_signal_strength': 0,
            'stop_pct': 5.0,
            'profit_pct': 10.0,
            'holding_period_hours': 2
        }, om, 1000.0, {}, False)

        tick_count = 0
        signal_count = 0
        order_count = 0

        for tick in dp.iterate():
            tick_count += 1

            # Verify first tick has expected symbols
            if tick_count == 1:
                symbols = [pd.symbol for pd in tick]
                assert 'SPY' in symbols, "First tick should contain SPY"

            signals = alg.on_data(tick)

            if signals:
                signal_count += len(signals)

            tick_results = pf.process_market_signals_for_tick(signals, tick)

            if tick_results.orders:
                order_count += len(tick_results.orders)

            if signal_count > 0 and order_count > 5:
                break

            if tick_count >= 10000:
                break

        assert tick_count > 0, "Should have processed ticks"
        assert signal_count > 0, "Should have generated signals"
        assert order_count > 0, "Should have created orders"
        assert pf.cash >= 0, "Cash should be non-negative"
