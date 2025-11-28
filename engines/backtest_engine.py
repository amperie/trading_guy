"""
Simulator for backtesting
Iterates and feeds data through the system as if it were coming from a real time source
"""

from core.algorithm import Algorithm
from core.portfolio import Portfolio
from core.order_manager import OrderManager
from data_providers.data_provider import DataProvider
from core.classes import PriceData, Order
from engines.base_engine import BaseEngine

class BacktestingEngine(BaseEngine):

    def __init__(
            self, cfg:dict= None, dp: DataProvider = None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):
        super().__init__(cfg=cfg, dp=dp, al=al, om=om, pf=pf)

    def run(self):
        """Run through the data, feed it into the algorithm
        Feed all the market signals into the portfolio for it to execute on signals
        Portfolio outputs a list of orders, feed those to the OrderManager
        """

        for tick in self.dp.iterate():
            self.on_tick(tick)

    def on_tick(self, tick: list[PriceData]) -> list[Order]:

        market_signals = self.al.on_data(tick)
        orders = self.pf.process_market_signals_for_tick(market_signals, tick)
        return orders
