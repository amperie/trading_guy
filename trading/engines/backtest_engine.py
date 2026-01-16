"""
Simulator for backtesting
Iterates and feeds data through the system as if it were coming from a real time source
"""

from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.order_manager import OrderManager
from trading.data_providers.data_provider import DataProvider
from trading.core.classes import PriceData, TickResults
from trading.engines.base_engine import BaseEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)

class BacktestingEngine(BaseEngine):

    def __init__(
            self, cfg:dict= None, dp: DataProvider = None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):
        super().__init__(cfg=cfg, dp=dp, al=al, om=om, pf=pf)

    def finalize(self):
        pass

    def run(self):
        """Run through the data, feed it into the algorithm
        Feed all the market signals into the portfolio for it to execute on signals
        Portfolio outputs a list of orders, feed those to the OrderManager
        """
        iters = 0
        length = self.dp.get_data_length()
        logger.info(f"Starting backtest for {length} timestamps")

        for tick in self.dp.iterate():
            iters+=1
            if iters % 500 == 0:
                if len(tick) > 0:
                    ts = tick[0].timestamp
                else:
                    ts = "None"
                logger.info(f"Running iteration {iters} of {length} for timestamp {ts}")

            self.on_tick(tick)

    def on_tick(self, tick: list[PriceData]) -> TickResults:

        market_signals = self.al.on_data(tick)
        ret_val = self.pf.process_market_signals_for_tick(market_signals, tick)
        return ret_val
