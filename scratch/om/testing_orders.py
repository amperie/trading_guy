import sys
from pathlib import Path
from datetime import datetime

from core.classes import OrderType, OrderStatus, OrderAction, Order, PriceData, BracketOrder, Position

# For Jupyter notebooks, use current working directory
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from engines.simulator import Simulator
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from algorithms.test_algorithm import TestAlgorithm
from data_providers.test_data_provider import TestDataProvider


project_root = Path.cwd()
data_path = str(project_root / "data" / "test_data.csv")

def test_market_orders():
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

def test_bracket_orders():
    om = BacktestingOM()
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

test_bracket_orders()