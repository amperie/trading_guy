from collections import deque
from typing import Any, Union
from datetime import datetime
import pandas as pandas

from core.algorithm import Algorithm
from core.portfolio import Portfolio
from core.order_manager import OrderManager
from data_providers.data_provider import DataProvider
from core.classes import PriceData
from data_providers.test_data_provider import TestDataProvider
from engines.dataset_building_engine_base import DataSetBuildingEngineBase
from utils.utils import find_pricedata_in_list

class IndicatorLagDataSetEngine(DataSetBuildingEngineBase):

    """
    Requires extra config:
    symbol - symbol to use for building the dataset
    lag_values - how many lag values to include in the dataset
    """

    @staticmethod
    def featurize(
            cfg: dict[str, Any], tick: list[PriceData],
            prices: dict[str, deque], al: Algorithm = None,
            df: pandas.DataFrame=None) \
            -> Union[None, dict[str, Any], list[dict[str, Any]]]:

        symbol = cfg['symbol']
        lag_values = cfg['lag_values']
        loss_pct = cfg['loss_pct']
        profit_pct = cfg['profit_pct']

        pd = find_pricedata_in_list(symbol, tick)
        if pd is None:
            return None

        timestamp = pd.timestamp
        current_price = pd.close

        # Get historical prices (need lags + current)
        history = list(prices[symbol]) if symbol in prices else []

        if len(history) < lag_values-1:
            return None  # Not enough history yet

        # Create row with normalized prices
        row = {
            'timestamp': timestamp,
            'current_price': current_price,  # Current price normalized to itself = 1.0
        }

        # Add 9 lagged prices, normalized by current price
        for i in range(2, lag_values):
            lag_price = history[-i]  # -1 is most recent, -2 is one before, etc.
            row[f'price_lag_{i}'] = lag_price / current_price

        row['target'] = IndicatorLagDataSetEngine.targetize(cfg, current_price, timestamp, df)

        return row

    @staticmethod
    def targetize( cfg: dict[str, Any],
            curr_price: float, curr_ts: datetime, df: pandas.DataFrame) -> int:

        loss_pct = cfg['loss_pct']
        profit_pct = cfg['profit_pct']
        look_ahead_period = cfg['look_ahead_period']

        # Filter dataframe for rows after current timestamp
        future_df = df[df['timestamp'] > curr_ts]
        if len(future_df) == 0:
            # No lookahead data, return 0
            return 0
        # TODO: need to make sure it's sorted?
        # Get just the rows for the lookahead period
        lap = min(look_ahead_period, len(future_df))
        future_df = future_df.iloc[0:lap]
        prices = future_df['close'].values

        # Calculate highs and lows in the look ahead period and whether they were hit
        l, h = prices <= curr_price * (1.0 - loss_pct), prices >= curr_price * (1.0 + profit_pct)

        # profit price was not reached if no trues are in h
        if not h.any(): return 0
        # if low wasn't hit, then a high was at this point
        if not l.any(): return 1
        # If both were hit, which one was hit first?
        return 1 if h.argmax() < l.argmax() else 0

