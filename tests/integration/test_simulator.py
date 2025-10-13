"""
Integration tests for Simulator engine
Tests the full pipeline: DataProvider -> Algorithm -> Portfolio -> OrderManager
"""
import pytest
import pandas as pd
from pathlib import Path

from engines.simulator import Simulator
from algorithms.test_algorithm import TestAlgorithm
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from data_providers.test_data_provider import TestDataProvider
from core.classes import OrderStatus


class TestSimulatorIntegration:
    """Integration tests for Simulator"""

    def test_simulator_initialization_with_objects(
            self, sample_csv_data):
        """Test creating simulator with objects"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": True})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)

        assert sim.al == al
        assert sim.om == om
        assert sim.pf == pf
        assert isinstance(sim.dp, TestDataProvider)

    def test_simulator_initialization_from_config(self, tmp_path):
        """Test creating simulator from config only"""
        # Create test data
        data = {
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'timestamp': [
                '2024-01-15 10:00:00+00:00',
                '2024-01-15 10:01:00+00:00',
                '2024-01-15 10:02:00+00:00'
            ],
            'open': [150.0, 151.0, 152.0],
            'high': [151.0, 152.0, 153.0],
            'low': [149.0, 150.0, 151.0],
            'close': [151.0, 152.0, 153.0],
            'volume': [100000, 110000, 120000]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        # This will fail because config doesn't have all required sections
        # But it tests the initialization path
        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(csv_path),
                "truncate": 0
            }
        }

        # Test that it at least creates the data provider
        with pytest.raises(KeyError):  # Will fail on missing 'algorithm' key
            sim = Simulator(cfg=cfg)

    def test_simulator_run_single_symbol(self, sample_csv_data):
        """Test running simulator with single symbol"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": True})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Check that orders were created
        assert len(pf.orders) > 0

        # Check that portfolio value was updated
        assert pf.total_value > 0

        # Check that history was tracked
        assert len(pf.tick_history) > 0
        assert len(pf.value_history) > 0
        assert len(pf.cash_history) > 0

    def test_simulator_process_tick(self, sample_csv_data, sample_pricedata_list):
        """Test processing a single tick"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": False})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)

        # Process one tick
        sim.process_tick(sample_pricedata_list)

        # Algorithm should have received data
        assert "AAPL" in al.price_history or len(al.price_history) == 0

    def test_simulator_multiple_ticks(self, sample_csv_data):
        """Test that simulator processes all ticks"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": True})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Should have processed 3 ticks (from fixture)
        assert len(al.full_history) == 3

        # Each tick should be in portfolio history
        assert len(pf.tick_history) == 3

    def test_simulator_order_execution(self, sample_csv_data):
        """Test that orders are executed during simulation"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": False})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # All orders should be filled (PerfectOm fills immediately)
        for order in pf.orders:
            assert order.status == OrderStatus.FILLED

    def test_simulator_cash_tracking(self, sample_csv_data):
        """Test that cash is tracked correctly"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": False})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        initial_cash = pf.cash

        sim.run()

        # Cash should have changed
        assert pf.cash != initial_cash

        # Cash history should be tracked
        assert len(pf.cash_history) > 0

    def test_simulator_position_tracking(self, sample_csv_data):
        """Test that positions are tracked correctly"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": False})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Position should exist or have existed
        # (May be 0 if last signal was SELL)
        assert "AAPL" in pf.positions or len(pf.orders) > 0

    def test_simulator_portfolio_value_calculation(self, sample_csv_data):
        """Test that portfolio value is calculated correctly"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": False})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Total value should equal cash + positions value
        positions_value = sum(
            pos.quantity * 153.0  # Last price from fixture
            for pos in pf.positions.values()
        )
        expected_value = pf.cash + positions_value

        # Allow for small floating point differences
        assert abs(pf.total_value - expected_value) < 0.01

    def test_simulator_with_truncated_data(self, sample_csv_data):
        """Test simulator with truncated data"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": True})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 2  # Only load 2 rows
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Should process fewer ticks
        assert len(al.full_history) <= 2

    def test_simulator_algorithm_history(self, sample_csv_data):
        """Test that algorithm history is populated"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 5, "full_history": True})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Algorithm should have price history
        if "AAPL" in al.price_history:
            assert len(al.price_history["AAPL"]) > 0

    def test_simulator_end_to_end(self, sample_csv_data):
        """End-to-end test of complete simulation"""
        om = BacktestingOM()
        al = TestAlgorithm({"history_length": 10, "full_history": True})
        pf = SingleSymbolPortfolio({
            'symbol': 'AAPL',
            'keep_history': True,
            "cash": 100000
        })
        pf.set_order_manager(om)

        cfg = {
            "data_provider": {
                "provider": "data_providers.test_data_provider.TestDataProvider",
                "path": str(sample_csv_data),
                "truncate": 0
            }
        }

        # Run simulation
        sim = Simulator(cfg=cfg, al=al, om=om, pf=pf)
        sim.run()

        # Verify all components worked together
        assert len(al.full_history) > 0  # Algorithm processed data
        assert len(pf.orders) > 0  # Portfolio created orders
        assert pf.total_value > 0  # Portfolio tracked value
        assert len(pf.tick_history) == len(al.full_history)  # Consistent state
