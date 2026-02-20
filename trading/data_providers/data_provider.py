"""
Base class for data providers.

DataProviders are the entry point for market data in the trading pipeline.
They load data from a source (CSV files, APIs, live feeds), store it as a
pandas DataFrame, and expose it to engines via the iterate() generator
which yields one list[PriceData] per timestamp.

Data flow:
    Source (CSV / API / WebSocket)
        -> load_data()          (subclass populates self.data DataFrame)
        -> iterate()            (yields list[PriceData] per timestamp)
        -> Engine.on_tick(tick) (processes each tick through the pipeline)

DataFrame contract:
    self.data must be a pandas DataFrame with at minimum these columns:
        - timestamp: datetime or string (parsed automatically)
        - symbol: str (e.g. "AAPL", "SPY")
        - open: float
        - high: float
        - low: float
        - close: float
        - volume: float
    Optional columns (passed through to PriceData if present):
        - trade_count: int
        - vwap: float
        - exchange: str

Implementations:
    - TestDataProvider:     Reads CSV files from disk
    - AlpacaDataProvider:   Pulls historical bars from the Alpaca API
"""
from typing import final, Generator

import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

from trading.core.classes import PriceData
from utils.config_manager import ConfigManager
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class DataProvider(ABC):
    """
    Abstract base class for all data providers.

    Loads market data into a pandas DataFrame and provides iteration
    over it as PriceData objects grouped by timestamp. The engine calls
    iterate() to drive the backtesting loop.

    Subclassing notes:
        - Implement load_data() to populate self.data with a DataFrame
          matching the column contract (see module docstring).
        - load_data() is called lazily by iterate() if self.data is None,
          or can be called eagerly in __init__().
        - Do NOT override iterate(), get_data(), or get_data_length();
          they are @final.
        - Configuration is read from config.yaml's "data_provider" section
          and merged with any cfg dict passed to __init__().

    Attributes:
        cfg (dict): Merged configuration dict. Base values come from
            config.yaml's "data_provider" section; any keys in the cfg
            parameter override them. Common keys:
            - path (str): File path for CSV-based providers.
            - truncate (int): Number of rows to limit (0 = all data).
            - start_date (str): Filter start date.
            - end_date (str): Filter end date.
        data (pd.DataFrame): The loaded market data. None until load_data()
            is called. Must have timestamp, symbol, open, high, low, close,
            and volume columns at minimum.

    Example:
        class MyDataProvider(DataProvider):
            def __init__(self, cfg=None):
                super().__init__(cfg)
                self.load_data()  # eager load

            def load_data(self):
                self.data = pd.read_csv(self.cfg["path"])

        dp = MyDataProvider({"path": "data/AAPL.csv", "truncate": 0})
        for tick in dp.iterate():
            print(tick[0].symbol, tick[0].close)
    """

    data: pd.DataFrame

    def __init__(self, cfg: dict=None):
        """
        Initialize the data provider with merged configuration.

        Reads the base "data_provider" section from config.yaml, then
        merges any provided cfg dict on top (cfg values take precedence).

        Args:
            cfg: Optional configuration dict to override config.yaml
                values. If None, uses config.yaml's "data_provider"
                section as-is.
        """
        cm = ConfigManager()
        if cfg is None:
            self.cfg = cm.get("data_provider")
        else:
            self.cfg = {**cm.get("data_provider"), **cfg}
        self.data = None

    @abstractmethod
    def load_data(self):
        """
        Load market data and store it in self.data.

        Override this in your subclass. Must populate self.data with a
        pandas DataFrame containing at minimum: timestamp, symbol, open,
        high, low, close, volume columns.

        Called lazily by iterate() if self.data is None, or can be called
        eagerly in __init__() for providers that load data upfront.

        The DataFrame does not need to be sorted — iterate() sorts by
        timestamp automatically.
        """
        pass

    @final
    def get_data(self) -> pd.DataFrame:
        """
        Return the raw DataFrame.

        Returns:
            The loaded pandas DataFrame, or None if load_data() has not
            been called yet.
        """
        return self.data

    @final
    def get_data_length(self) -> int:
        """
        Return the number of unique timestamps in the data.

        This represents the total number of ticks that iterate() will
        yield, which is the number of iterations the engine will perform.

        Returns:
            Count of unique timestamps in the DataFrame.
        """
        return self.data['timestamp'].unique().shape[0]

    def iterate(self) -> Generator[list[PriceData], None, None]:
        """
        Yield one list[PriceData] per timestamp, sorted chronologically.

        This is the primary interface used by engines to consume market
        data. Each yielded list contains PriceData objects for every
        symbol present at that timestamp, enabling multi-asset strategies
        to see all symbols in a single tick.

        Lazy-loads data via load_data() if self.data is None.

        Timestamp handling:
            - Supports pd.Timestamp, datetime, and string formats.
            - Strings are parsed as '%Y-%m-%d %H:%M:%S%z'.
            - All timestamps are sorted chronologically regardless of
              the original order in the DataFrame.

        Yields:
            list[PriceData]: One list per unique timestamp, containing
            a PriceData object for each symbol at that timestamp. Lists
            are yielded in chronological order.

        Example:
            dp = TestDataProvider({"path": "data/SPY_AAPL.csv"})
            for tick in dp.iterate():
                # tick might be [PriceData(SPY, ...), PriceData(AAPL, ...)]
                for pd in tick:
                    print(f"{pd.symbol}: {pd.close}")
        """
        if self.data is None:
            logger.debug("Data not loaded — calling load_data()")
            self.load_data()
            logger.info(f"Data loaded: {len(self.data)} rows, {self.data['timestamp'].nunique()} ticks, symbols={list(self.data['symbol'].unique())}")

        tmp_data = self.data.set_index(['symbol',"timestamp"])
        # Get the range of date times to iterate through
        dts = self.data['timestamp'].unique()

        # Handle both string and Timestamp types
        def parse_timestamp(ts):
            """Convert timestamp to datetime for sorting."""
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
