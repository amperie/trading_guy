from abc import abstractmethod
from typing import Any, final, Union
import pandas as pandas
from datetime import datetime
from core.algorithm import Algorithm
from core.order_manager import OrderManager
from core.portfolio import Portfolio
from data_providers.data_provider import DataProvider
from core.classes import PriceData, TickResults, MarketSignal
from engines.base_engine import BaseEngine
from utils.logger import Logger
from collections import deque

logger = Logger().get_logger(__name__)

class DataTrackingAlgorithm(Algorithm):

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        # Does nothing since all we're using this algorithm for is tracking the history of data
        return []

class DataSetBuildingEngineBase(BaseEngine):
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
        self.dataset = []
        if self.dp is not None and self.dp.data is not None:
            self.data = self.dp.data.copy()
            self.data['timestamp'] = pandas.to_datetime(self.data['timestamp'])
        else:
            self.data = pandas.DataFrame()

    @final
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

        self.finalize()

    @final
    def on_tick(self, tick: list[PriceData]) -> TickResults:

        self.al.on_data(tick)
        prices = self.al.get_price_history()
        row = self.create_dataset_row(tick, prices, self.al)
        if row is not None and type(row) is dict:
            self.dataset.append(row)
        elif row is not None and type(row) is list:
            self.dataset.extend(row)

        return TickResults()

    def create_dataset_row(
            self, tick: list[PriceData], prices: dict[str, deque],
            al: Algorithm=None) -> Union[None, dict[str, Any], list[dict[str, Any]]]:

        return self.featurize(self.cfg, tick, prices, al, self.data)

    @staticmethod
    def featurize(
            cfg: dict[str, Any], tick: list[PriceData],
            prices: dict[str, deque], al: Algorithm = None,
            df: pandas.DataFrame=None) \
            -> Union[None, dict[str, Any], list[dict[str, Any]]]:

        raise NotImplementedError()

    @staticmethod
    def targetize( cfg: dict[str, Any],
            curr_price: float, curr_ts: datetime, data: pandas.DataFrame) -> int:
        raise NotImplementedError()

    @final
    def finalize(self):

        logger.info(f"Writing {len(self.dataset)} rows to {self.output_file}")
        df = pandas.DataFrame(self.dataset)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Dataset written successfully")