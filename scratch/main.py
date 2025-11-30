import sys
from pathlib import Path
from datetime import datetime

from core.classes import OrderType, OrderStatus, OrderAction, Order, PriceData

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.backtest_engine import BacktestingEngine
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from algorithms.test_algorithm import TestAlgorithm
from data_providers.test_data_provider import TestDataProvider

def test_simulator():
    # Get absolute path to data file relative to project root
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")
    om = BacktestingOM()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({'symbol':'AAPL', 'keep_history': True, "cash": 1000})
    pf.set_order_manager(om)

    s = BacktestingEngine(om=om, al=al, pf=pf)
    s.run()

def test_om():
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "test_data.csv")
    om = BacktestingOM()
    o = Order(
            action=OrderAction.BUY,
            type=OrderType.MARKET,
            symbol="SPY",
            price=0.0,
            quantity=2,
            cash=0,
            tx_cost=0.0,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
        )
    tick = [PriceData("SPY", datetime.now(),10.0,10.0,10.0,10.0,5,5,5,5)]
    so = om.submit_order(o, tick, {}, 1000.0)

    pass

def test_indicators():
    from core.ta.analyzer import TechnicalAnalyzer

    data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0,
    ]

    a = TechnicalAnalyzer.calculate_ema(data, 2)
    a = TechnicalAnalyzer.calculate_macd(data, 15, 9, 3, True)
    print(a)

test_indicators()
