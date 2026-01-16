from collections import deque
from typing import Any, Union
from datetime import datetime
import pandas as pandas

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData
from trading.engines.dataset_building_engine_base import DataSetBuildingEngineBase
from utils.utils import find_pricedata_in_list

class IndicatorLagDataSetEngine(DataSetBuildingEngineBase):

    """
    Requires extra config:
    symbol - symbol to use for building the dataset
    lag_values - how many lag values to include in the dataset
    """

    def __init__(
            self, cfg: dict = None, dp=None,
            al=None, om=None, pf=None
        ):
        super().__init__(cfg, dp, al, om, pf)
        # Convert timestamp column to datetime for comparisons
        self.data = self.dp.data.copy()
        self.data['timestamp'] = pandas.to_datetime(self.data['timestamp'])

        # Pre-compute all targets at once (vectorized)
        self._precompute_targets()

    def _precompute_targets(self):
        """Vectorized target calculation for entire dataset with multiple configurations"""
        closes = self.data['close'].values

        # Support single config or list of configs
        target_configs = self.cfg.get('target_configs')

        for config in target_configs:
            loss_pct = config['loss_pct']
            profit_pct = config['profit_pct']
            look_ahead = config['look_ahead_period']
            col_name = config.get('name', f'target_{int(loss_pct*100)}_{int(profit_pct*100)}_{look_ahead}')

            targets = []
            for i in range(len(closes)):
                end_idx = min(i + look_ahead + 1, len(closes))
                future_prices = closes[i+1:end_idx]

                if len(future_prices) == 0:
                    targets.append(0)
                    continue

                curr_price = closes[i]
                loss_hit = future_prices <= curr_price * (1.0 - loss_pct)
                profit_hit = future_prices >= curr_price * (1.0 + profit_pct)

                if not profit_hit.any():
                    targets.append(0)
                elif not loss_hit.any():
                    targets.append(1)
                else:
                    targets.append(1 if profit_hit.argmax() < loss_hit.argmax() else 0)

            self.data[col_name] = targets

    @staticmethod
    def featurize(
            cfg: dict[str, Any], tick: list[PriceData],
            prices: dict[str, deque], al: Algorithm = None,
            df: pandas.DataFrame=None) \
            -> Union[None, dict[str, Any], list[dict[str, Any]]]:

        symbol = cfg['symbol']
        lag_values = cfg['lag_values']

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

        # Add lagged prices, normalized by current price
        for i in range(2, lag_values):
            lag_price = history[-i]  # -1 is most recent, -2 is one before, etc.
            row[f'price_lag_{i}'] = lag_price / current_price

        # Look up pre-computed target(s) from dataframe
        target_row = df[df['timestamp'] == timestamp]
        if len(target_row) > 0:
            # Add all target columns (could be multiple)
            for col in df.columns:
                if col.startswith('target'):
                    row[col] = target_row[col].iloc[0]
        else:
            # Fallback if timestamp not found
            row['target'] = 0

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

