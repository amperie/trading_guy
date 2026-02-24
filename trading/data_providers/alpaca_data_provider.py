"""
Data provider that pulls historical bar data from the Alpaca API.
Enables backtesting with Alpaca data without manual CSV downloads.

Supports three query modes via load_data() / fetch_bars():
    1. Date range:   start_date + end_date  (all bars in the window)
    2. Recent bars:  limit (+ optional end_date)  (N most recent bars)
    3. Combined:     start_date + end_date + limit  (capped date range)

Mode 2 is particularly useful for algorithm warm-up: set limit to the
algorithm's history_length to fetch exactly enough bars to fill the
price_history deque without calculating a start date.

Config keys (cfg dict):
    api_key (str):      Alpaca API key (required)
    secret_key (str):   Alpaca secret key (required)
    symbols (list):     Symbols to fetch (required, e.g. ["AAPL", "SPY"])
    timeframe (str):    Bar size — "Minute", "Hour", "Day", "Week", "Month"
                        (default: "Minute")
    adjustment (str):   Price adjustment — "split", "raw", "all"
                        (default: "split")
    start_date (str):   Start of date range (optional)
    end_date (str):     End of date range (optional, defaults to now)
    limit (int):        Max bars to return (optional)
"""
import pandas as pd
from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading.data_providers.data_provider import DataProvider
from utils.logger import Logger

logger = Logger().get_logger(__name__)


# Map string names to TimeFrame objects
TIMEFRAME_MAP = {
    "Minute": TimeFrame.Minute,
    "Hour": TimeFrame.Hour,
    "Day": TimeFrame.Day,
    "Week": TimeFrame.Week,
    "Month": TimeFrame.Month,
}


class AlpacaDataProvider(DataProvider):

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)

        self.api_key = self.cfg["api_key"]
        self.secret_key = self.cfg["secret_key"]
        self.symbols = self.cfg["symbols"]

        timeframe_str = self.cfg.get("timeframe", "Minute")
        if timeframe_str not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unknown timeframe '{timeframe_str}'. "
                f"Valid options: {list(TIMEFRAME_MAP.keys())}"
            )
        self.timeframe = TIMEFRAME_MAP[timeframe_str]
        self.adjustment = self.cfg.get("adjustment", "split")

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        logger.debug(f"AlpacaDataProvider initialized symbols={self.symbols} timeframe={timeframe_str} adjustment={self.adjustment}")

    def load_data(self):
        start = self.cfg.get("start_date", None)
        end = self.cfg.get("end_date", None)
        limit = self.cfg.get("limit", None)

        if start is not None:
            start = pd.to_datetime(start).to_pydatetime()
        if end is not None:
            end = pd.to_datetime(end).to_pydatetime()

        logger.info(f"Fetching Alpaca bars: symbols={self.symbols} start={start} end={end} limit={limit}")
        self.data = self.fetch_bars(start=start, end=end, limit=limit)

    def fetch_bars(
        self,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Fetch historical bars from Alpaca and return a DataFrame
        matching the DataProvider contract.

        Args:
            start: Start datetime for the data range (optional).
            end: End datetime for the data range (optional, defaults
                to now on the Alpaca side).
            limit: Max number of bars to return (optional). Useful
                with end (no start) to get the N most recent bars.

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low,
            close, volume, trade_count, vwap
        """
        # When multiple symbols are requested, fetch each symbol individually.
        # The Alpaca API applies `limit` to the total response across all
        # symbols, not per-symbol, so a single multi-symbol call can starve
        # some tickers of data.  Per-symbol calls guarantee each gets the
        # full `limit`.
        if len(self.symbols) > 1:
            dfs = []
            for sym in self.symbols:
                req = StockBarsRequest(
                    symbol_or_symbols=[sym],
                    timeframe=self.timeframe,
                    start=start,
                    end=end,
                    limit=limit,
                    adjustment=self.adjustment,
                )
                sym_bars = self.client.get_stock_bars(req)
                sym_df = sym_bars.df
                if not sym_df.empty:
                    dfs.append(sym_df)
                else:
                    logger.warning(f"Alpaca returned no bars for {sym}")
            if dfs:
                df = pd.concat(dfs)
            else:
                df = pd.DataFrame()
        else:
            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbols,
                timeframe=self.timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=self.adjustment,
            )
            bars = self.client.get_stock_bars(request_params)
            df = bars.df

        if df.empty:
            logger.warning(f"Alpaca returned no bars for {self.symbols}")
            self.data = pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low",
                "close", "volume", "trade_count", "vwap",
            ])
            return self.data

        df = df.reset_index()
        logger.debug(f"Alpaca raw columns after reset_index: {list(df.columns)}")

        # Normalize timestamp column name (Alpaca SDK may vary)
        if "timestamp" not in df.columns:
            for candidate in ["Timestamp", "t", "datetime", "date"]:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "timestamp"})
                    break
            else:
                # Last resort: use the first datetime column
                dt_cols = df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]", "datetimetz"]).columns
                if len(dt_cols) > 0:
                    df = df.rename(columns={dt_cols[0]: "timestamp"})
                else:
                    raise KeyError(
                        f"Cannot find timestamp column in Alpaca response. "
                        f"Columns: {list(df.columns)}"
                    )

        # Normalize timestamps to EST (timezone-naive)
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

        # Ensure required columns exist
        expected_cols = [
            "timestamp", "symbol", "open", "high", "low",
            "close", "volume", "trade_count", "vwap",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        logger.info(f"Fetched {len(df)} bars for {self.symbols}: {df['timestamp'].min()} → {df['timestamp'].max()}")
        self.data = df
        return df
