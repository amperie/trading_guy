from core.algorithm import Algorithm
from core.order_manager import OrderManager
from core.portfolio import Portfolio
from data_providers.data_provider import DataProvider
from core.classes import PriceData, TickResults, MarketSignal
from engines.base_engine import BaseEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)

class DataTrackingAlgorithm(Algorithm):

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        # Does nothing since all we're using this algorithm for is tracking the history of data
        return []

class DataSetBuildingEngine(BaseEngine):
    def __init__(
            self, cfg: dict = None, dp: DataProvider = None,
            al: Algorithm = None, om: OrderManager = None,
            pf: Portfolio = None
    ):
        """
        :param cfg: configuration - requires 'output_file' and "history_length'
        """
        super().__init__(cfg, dp, al, om, pf)
        self.output_file = cfg['output_file']
        self.al = DataTrackingAlgorithm(history_length=cfg['history_length'])

    def run(self):
        """Run through the data, feed it into the algorithm to keep the history
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

        self.al.on_data(tick)
        prices = self.al.get_price_history()

        return TickResults()


    def finalize(self):
        pass