"""
DataProvider base class
Inherited classes will provide test data, backtesting data or real time data
"""

import pandas as pd
from abc import ABC, abstractmethod

from core.classes import PriceData
from utils.config_manager import ConfigManager

class DataProvider(ABC):

    data: pd.DataFrame

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

    def iterate(self) -> list[PriceData]:
        if self.data is None:
            self.load_data()

        # Get the range of date times to iterate through
        dts = self.data.get_level_values('timestamp').unique().sort_values()
        # Iterate through all the timestamps in the data
        for ts in dts:
            pt = ts.to_pydatetime()
            # Get the data for the timestamp in a dict
            ts_data = (self.data.xs(
                        timestamp, level='timestamp')
                       .reset_index()
                       .to_dict(orient='records')
                       )
            # turn it into a list of PriceData
            retval = [
                # Add the datetime back in
                PriceData().from_dict(tick, pt)
                for tick in ts_data
            ]

            yield retval
  
