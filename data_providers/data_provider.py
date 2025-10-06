"""
DataProvider base class
Inherited classes will provide test data, backtesting data or real time data
"""

import pandas as pd
from abc import ABC, abstractmethod

from core.classes import PriceData
from utils.config_manager import ConfigManager

class DataProvider(ABC):

    def __init__(self, cfg: dict=None):
        cm = ConfigManager()
        if cfg is None:
            self.cfg = cm.get("data_provider")
        else:
            self.cfg = {**cm.get("data_provider"), **cfg}
        self.data = None

    @abstractmethod
    def load_data(self):
        pass

    def get_data(self):
        return self.data

    def iterate(self):
        if self.data is None:
            self.load_data()

        # Check how the data is sorted and iterate from older to newer
        if self.data['timestamp'].iat[0] > self.data['timestamp'].iat[1]:
            # data is sorted descending (later times at the top
            # Iterate backwards through the df
            for row in self.data[::-1].itertuples(index=False):
              yield PriceData(
                  symbol=row.symbol,
                  timestamp=row.timestamp,
                  open=row.open,
                  high=row.high,
                  low=row.low,
                  close=row.close,
                  volume=row.volume
              )
        else:
            # Iterate forward
            for row in self.data.itertuples(index=False):
              yield PriceData(
                  symbol=row.symbol,
                  timestamp=row.timestamp,
                  open=row.open,
                  high=row.high,
                  low=row.low,
                  close=row.close,
                  volume=row.volume
              )
