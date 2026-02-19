import sys
from pathlib import Path
from datetime import datetime

from trading.core.classes import OrderType, OrderAction, Order, PriceData, BracketOrder, Position, MarketSignal, \
    SignalType

# For Jupyter notebooks, use current working directory
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.core.algorithms.test_algorithm import TestAlgorithm
from trading.data_providers.test_data_provider import TestDataProvider


project_root = Path.cwd()
data_path = str(project_root / "data" / "test_data.csv")

def market_orders():
    om = BacktestingOrderManager()
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


    tick = [PriceData("SPY", datetime.now(),10.0,10.0,10.0,20.0,5,5,5,5)]
    o2 = Order(
            action=OrderAction.SELL,
            type=OrderType.MARKET,
            symbol="SPY",
            price=0.0,
            quantity=2,
            cash=0,
            tx_cost=0.0,
            placed_datetime=datetime.now(),
            executed_datetime=datetime.now(),
        )
    so2 = om.submit_order(o2, tick, {}, 1000.0)

    pass

def bracket_orders():
    om = BacktestingOrderManager()
    bo = BracketOrder.create_bracket_order(
        symbol="SPY",
        quantity=10,
        high_sell_price=12,
        low_sell_price=9,
    )
    tick = [PriceData("SPY", datetime.now(),10.0,10.0,10.0,10.0,5,5,5,5)]
    so = om.submit_order(bo, tick, {}, 1000.0)
    pos = Position(
        "SPY", so.quantity
    )

    tick = [PriceData("SPY", datetime.now(),10.0,10.0,10.0,20.0,5,5,5,5)]
    so2 = om.update_pending_orders(tick, {"SPY": pos}, 500)

    pass

def portfolio_orders():
    # Get absolute path to data file relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_path = str(project_root / "data" / "test_data.csv")

    om = BacktestingOrderManager()
    al = TestAlgorithm({"history_length": 10, "full_history": True})
    dp_cfg = {"path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({'symbol': 'AAPL'}, om, 1000.0, {}, True)

    # BracketOrder test - first buy
    tick = [PriceData("AAPL", datetime.now(),0.0,0.0,0.0,10.0,0,0,0,0)]
    signal = MarketSignal(SignalType.BUY, "AAPL", 100)
    pf.process_market_signals_for_tick([signal], tick)
    # Trigger a profit taker sale
    tick = [PriceData("AAPL", datetime.now(),0.0,0.0,0.0,12.0,0,0,0,0)]
    pf.process_market_signals_for_tick([], tick)

    # buy again
    tick = [PriceData("AAPL", datetime.now(),0.0,0.0,0.0,13.0,0,0,0,0)]
    signal = MarketSignal(SignalType.BUY, "AAPL", 100)
    pf.process_market_signals_for_tick([signal], tick)

    # this should have no effect since there's no money left to buy
    tick = [PriceData("AAPL", datetime.now(),0.0,0.0,0.0,6.0,0,0,0,0)]
    pf.process_market_signals_for_tick([signal], tick)

    # trigger stop loss
    tick = [PriceData("AAPL", datetime.now(),0.0,0.0,0.0,3.0,0,0,0,0)]
    pf.process_market_signals_for_tick([], tick)

    pass

portfolio_orders()