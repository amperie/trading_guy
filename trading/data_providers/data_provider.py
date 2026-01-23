"""
DataProvider base class
Inherited classes will provide test data, backtesting data or real time data
"""
from typing import final, Generator

import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

# from IPython.utils.terminal import set_term_title
from trading.core.classes import PriceData
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

    @final
    def get_data(self):
        return self.data

    @final
    def get_data_length(self):
        return self.data['timestamp'].unique().shape[0]

    def iterate(self) -> Generator[list[PriceData], None, None]:
        """
        Yields lists of PriceData, one list per timestamp.
        Each yielded list contains PriceData for all symbols at that timestamp.
        """

        if self.data is None:
            self.load_data()

        tmp_data = self.data.set_index(['symbol',"timestamp"])
        # Get the range of date times to iterate through
        dts = self.data['timestamp'].unique()

        # Handle both string and Timestamp types
        def parse_timestamp(ts):
            """Convert timestamp to datetime for sorting"""
            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime()
            elif isinstance(ts, str):
                return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S%z')
            else:
                # Already a datetime
                return ts

        sdts = sorted(dts, key=parse_timestamp)

        # Iterate through all the timestamps in the data
        for ts in sdts:
            # Convert to datetime for PriceData
            pt = parse_timestamp(ts)

            # Get the data for the timestamp in a dict
            ts_data = (tmp_data.xs(
                        ts, level='timestamp')
                       .reset_index()
                       .to_dict(orient='records')
                       )
            # turn it into a list of PriceData
            retval = [
                # Add the datetime back in
                PriceData.from_dict(tick, pt)
                for tick in ts_data
            ]

            yield retval
  
