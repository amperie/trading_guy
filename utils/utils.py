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


def serialize_market_signal(signal: MarketSignal) -> dict:
    """
    Convert MarketSignal to JSON-serializable dictionary.

    Args:
        signal: MarketSignal object to serialize

    Returns:
        Dictionary with all signal data in JSON-serializable format

    Example:
        >>> signal = MarketSignal(type=SignalType.BUY, symbol="AAPL", strength=75, metadata={"reason": "crossover"})
        >>> data = serialize_market_signal(signal)
        >>> # {'type': 'BUY', 'symbol': 'AAPL', 'strength': 75, 'metadata': {'reason': 'crossover'}}
    """
    from core.classes import SignalType
    from datetime import datetime

    # Deep copy metadata and convert datetime objects to ISO strings
    metadata_copy = {}
    for key, value in signal.metadata.items():
        if isinstance(value, datetime):
            metadata_copy[key] = value.isoformat()
        else:
            metadata_copy[key] = value

    return {
        "type": signal.type.name,  # Convert enum to string (e.g., "BUY")
        "symbol": signal.symbol,
        "strength": signal.strength,
        "metadata": metadata_copy
    }


def deserialize_market_signal(data: dict) -> MarketSignal:
    """
    Convert dictionary back to MarketSignal object.

    Args:
        data: Dictionary with serialized signal data

    Returns:
        MarketSignal object

    Example:
        >>> data = {'type': 'BUY', 'symbol': 'AAPL', 'strength': 75, 'metadata': {'reason': 'crossover'}}
        >>> signal = deserialize_market_signal(data)
        >>> # MarketSignal(type=SignalType.BUY, symbol='AAPL', strength=75, metadata={'reason': 'crossover'})
    """
    from core.classes import SignalType
    from datetime import datetime

    # Reconstruct metadata, converting ISO strings back to datetime
    metadata = {}
    for key, value in data.get("metadata", {}).items():
        # Try to parse as datetime if it looks like ISO format
        if isinstance(value, str) and "T" in value:
            try:
                metadata[key] = datetime.fromisoformat(value)
            except ValueError:
                metadata[key] = value
        else:
            metadata[key] = value

    return MarketSignal(
        type=SignalType[data["type"]],  # Convert string back to enum
        symbol=data["symbol"],
        strength=data["strength"],
        metadata=metadata
    )


def serialize_signals_history(signals_history: list[list[MarketSignal]]) -> list[dict]:
    """
    Convert signals_history (list of lists) to JSON-serializable format.

    Preserves the tick grouping structure where each tick can have multiple signals.

    Args:
        signals_history: List where each element is a list of MarketSignal objects from one tick

    Returns:
        List of dictionaries, each containing tick_index and signals array

    Example:
        >>> signals_history = [
        ...     [MarketSignal(type=SignalType.BUY, symbol="AAPL", strength=75)],
        ...     [MarketSignal(type=SignalType.SELL, symbol="TSLA", strength=80)]
        ... ]
        >>> data = serialize_signals_history(signals_history)
        >>> # [
        >>> #   {'tick_index': 0, 'signals': [{'type': 'BUY', 'symbol': 'AAPL', ...}]},
        >>> #   {'tick_index': 1, 'signals': [{'type': 'SELL', 'symbol': 'TSLA', ...}]}
        >>> # ]
    """
    return [
        {
            "tick_index": idx,
            "signals": [serialize_market_signal(sig) for sig in tick_signals]
        }
        for idx, tick_signals in enumerate(signals_history)
    ]


def deserialize_signals_history(data: list[dict]) -> list[list[MarketSignal]]:
    """
    Convert JSON data back to signals_history format.

    Args:
        data: List of dictionaries with tick_index and signals arrays

    Returns:
        List of lists of MarketSignal objects, preserving tick grouping

    Example:
        >>> data = [
        ...     {'tick_index': 0, 'signals': [{'type': 'BUY', 'symbol': 'AAPL', 'strength': 75, 'metadata': {}}]},
        ...     {'tick_index': 1, 'signals': [{'type': 'SELL', 'symbol': 'TSLA', 'strength': 80, 'metadata': {}}]}
        ... ]
        >>> signals_history = deserialize_signals_history(data)
        >>> # [[MarketSignal(...)], [MarketSignal(...)]]
    """
    # Sort by tick_index to ensure correct order
    sorted_data = sorted(data, key=lambda x: x["tick_index"])

    return [
        [deserialize_market_signal(sig) for sig in tick["signals"]]
        for tick in sorted_data
    ]


def serialize_signals_history_flat(signals_history: list[list[MarketSignal]]) -> list[dict]:
    """
    Flatten signals_history to single list for easier analysis.

    This format is more convenient for loading into pandas DataFrames or
    analyzing in spreadsheet tools.

    Args:
        signals_history: List where each element is a list of MarketSignal objects from one tick

    Returns:
        Flattened list of signal dictionaries, each with tick_index added

    Example:
        >>> signals_history = [
        ...     [MarketSignal(type=SignalType.BUY, symbol="AAPL", strength=75)],
        ...     [MarketSignal(type=SignalType.SELL, symbol="TSLA", strength=80)]
        ... ]
        >>> data = serialize_signals_history_flat(signals_history)
        >>> # [
        >>> #   {'tick_index': 0, 'type': 'BUY', 'symbol': 'AAPL', 'strength': 75, 'metadata': {}},
        >>> #   {'tick_index': 1, 'type': 'SELL', 'symbol': 'TSLA', 'strength': 80, 'metadata': {}}
        >>> # ]
        >>> df = pd.DataFrame(data)  # Easy to analyze
    """
    flat_signals = []
    for tick_idx, tick_signals in enumerate(signals_history):
        for signal in tick_signals:
            sig_dict = serialize_market_signal(signal)
            sig_dict["tick_index"] = tick_idx
            flat_signals.append(sig_dict)
    return flat_signals
