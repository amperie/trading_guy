import importlib
from typing import Any, Union

import pandas as pd

from core.classes import PriceData, MarketSignal


def instantiate_from_string(full_path: str, *args, **kwargs) -> Any:
    """
    Dynamically import and instantiate a class from a full dotted path.

    Args:
        full_path: Full dotted path to the class (e.g., 'data_providers.polygon_provider.PolygonProvider')
        *args: Positional arguments to pass to the class constructor
        **kwargs: Keyword arguments to pass to the class constructor

    Returns:
        An instance of the specified class

    Example:
        instance = instantiate_from_string(
            'data_providers.polygon_provider.PolygonProvider',
            api_key='xyz'
        )
    """
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(*args, **kwargs)


def find_pricedata_in_list(symbol: str, pds: list[PriceData]) -> PriceData:
    try:
        retval = next(x for x in pds if x.symbol == symbol)
    except StopIteration:
        retval = None
    return retval

def get_symbols_in_list(data: list[Union[MarketSignal, PriceData]]) -> list[str]:
    ret_val = []
    for item in data:
        symbol = item.symbol
        if not symbol in ret_val:
            ret_val.append(symbol)
    return ret_val

def find_marketsignal_in_list(symbol: str, pds: list[MarketSignal]) -> MarketSignal:
    try:
        retval = next(x for x in pds if x.symbol == symbol)
    except StopIteration:
        retval = None
    return retval

def trim_dictionary(dictionary: dict, keys_to_delete: list[str]) -> dict:
    for key in keys_to_delete:
        if key in dictionary:
            del dictionary[key]

    return dictionary


def aggregate_stock_data(
    df: pd.DataFrame,
    interval: str = '5min',
    timestamp_col: str = 'timestamp',
    symbol_col: str = 'symbol'
) -> pd.DataFrame:
    """
    Aggregate multi-symbol stock data to a specified time granularity.

    Each symbol is resampled independently to avoid mixing data across symbols.
    Properly handles OHLCV aggregation (Open: first, High: max, Low: min, Close: last, Volume: sum).

    Important: Timestamps represent the END of the interval (right edge).
    For example, aggregating 1-minute data from 9:26-9:30 to 5 minutes will produce
    a single row with timestamp 9:30 (not 9:26).

    Args:
        df: DataFrame with columns [timestamp, symbol, open, high, low, close, volume, ...]
        interval: Pandas resample frequency string (e.g., '5min', '15min', '1h', '1D')
        timestamp_col: Name of timestamp column (default: 'timestamp')
        symbol_col: Name of symbol column (default: 'symbol')

    Returns:
        DataFrame with aggregated data, maintaining same column structure

    Example:
        >>> df_1min = pd.read_csv('data/multi_symbol_1min.csv')
        >>> df_5min = aggregate_stock_data(df_1min, interval='5min')
        >>> # Timestamp 9:30 represents data from 9:26-9:30
        >>> df_15min = aggregate_stock_data(df_1min, interval='15min')
        >>> df_daily = aggregate_stock_data(df_1min, interval='1D')
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Define aggregation rules for OHLCV data
    agg_rules = {
        'open': 'first',      # First price in the interval
        'high': 'max',        # Highest price in the interval
        'low': 'min',         # Lowest price in the interval
        'close': 'last',      # Last price in the interval
        'volume': 'sum',      # Total volume in the interval
    }

    # Add other common columns if present
    if 'trade_count' in df.columns:
        agg_rules['trade_count'] = 'sum'  # Sum of all trades

    # Process each symbol separately
    aggregated_dfs = []

    for symbol in df[symbol_col].unique():
        # Filter data for this symbol
        symbol_df = df[df[symbol_col] == symbol].copy()

        # Set timestamp as index for resampling
        symbol_df = symbol_df.set_index(timestamp_col)

        # Resample to desired interval with appropriate aggregations
        # Only aggregate columns that exist in the dataframe
        # Use label='right' and closed='right' so timestamp represents the END of the interval
        # This means interval (1:25, 1:30] includes 1:26, 1:27, 1:28, 1:29, 1:30 (not 1:25)
        valid_agg_rules = {col: rule for col, rule in agg_rules.items() if col in symbol_df.columns}
        resampled = symbol_df.resample(interval, label='right', closed='right').agg(valid_agg_rules)

        # Remove rows with no data (check if OHLC columns are NaN)
        # We check 'close' as a proxy for whether there was actual data in this interval
        if 'close' in resampled.columns:
            resampled = resampled[resampled['close'].notna()]
        else:
            # Fallback to dropna if no close column
            resampled = resampled.dropna(how='all')

        # Add symbol column back
        resampled[symbol_col] = symbol

        # Reset index to make timestamp a column again
        resampled = resampled.reset_index()

        aggregated_dfs.append(resampled)

    # Combine all symbols back together
    result = pd.concat(aggregated_dfs, ignore_index=True)

    # Sort by timestamp and symbol for consistent output
    result = result.sort_values([timestamp_col, symbol_col]).reset_index(drop=True)

    return result


def aggregate_stock_data_custom(
    df: pd.DataFrame,
    interval: str = '5min',
    agg_rules: dict = None,
    timestamp_col: str = 'timestamp',
    symbol_col: str = 'symbol'
) -> pd.DataFrame:
    """
    Aggregate multi-symbol stock data with custom aggregation rules.

    Important: Timestamps represent the END of the interval (right edge).
    For example, a 5-minute interval ending at 9:30 represents data from 9:26-9:30.

    Args:
        df: DataFrame with multi-symbol stock data
        interval: Pandas resample frequency string
        agg_rules: Custom aggregation rules dict. If None, uses OHLCV defaults.
                   Example: {'open': 'first', 'close': 'last', 'volume': 'sum', 'custom_field': 'mean'}
        timestamp_col: Name of timestamp column
        symbol_col: Name of symbol column

    Returns:
        DataFrame with aggregated data

    Example:
        >>> custom_rules = {
        ...     'open': 'first',
        ...     'high': 'max',
        ...     'low': 'min',
        ...     'close': 'last',
        ...     'volume': 'sum',
        ...     'vwap': 'mean'
        ... }
        >>> df_custom = aggregate_stock_data_custom(df_1min, interval='5min', agg_rules=custom_rules)
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Use default OHLCV rules if not provided
    if agg_rules is None:
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }

    # Group by symbol and aggregate
    aggregated_dfs = []

    for symbol in df[symbol_col].unique():
        symbol_df = df[df[symbol_col] == symbol].copy()
        symbol_df = symbol_df.set_index(timestamp_col)

        # Only use aggregation rules for columns that exist
        valid_agg_rules = {col: rule for col, rule in agg_rules.items() if col in symbol_df.columns}

        # Use label='right' and closed='right' so timestamp represents the END of the interval
        # This means interval (1:25, 1:30] includes 1:26, 1:27, 1:28, 1:29, 1:30 (not 1:25)
        resampled = symbol_df.resample(interval, label='right', closed='right').agg(valid_agg_rules)

        # Remove rows with no data (check if OHLC columns are NaN)
        if 'close' in resampled.columns:
            resampled = resampled[resampled['close'].notna()]
        else:
            resampled = resampled.dropna(how='all')

        resampled[symbol_col] = symbol
        resampled = resampled.reset_index()

        aggregated_dfs.append(resampled)

    result = pd.concat(aggregated_dfs, ignore_index=True)
    result = result.sort_values([timestamp_col, symbol_col]).reset_index(drop=True)

    return result
