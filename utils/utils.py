import copy
import importlib
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Union

import pandas as pd

from trading.core.classes import (
    PriceData, MarketSignal, Order, BracketOrder,
    OrderType, OrderAction, OrderStatus, OrderTimeInForce, SignalType,
)
from utils.logger import Logger

logger = Logger().get_logger(__name__)


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
    logger.debug(f"Loading class: {full_path}")
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


def set_nested_config_value(config: dict, key_path: str, value: Any) -> dict:
    """Set a config value using a dotted path, creating intermediate dicts."""
    parts = key_path.split(".")
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value
    return config


def merge_nested_config(base: dict, patch: dict) -> dict:
    """Deep-merge a patch dict into base config and return the mutated base."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_nested_config(base[key], value)
        else:
            base[key] = value
    return base


def build_tunable_patch(config: dict, param_keys: list[str]) -> dict:
    """Build a nested patch dict from sampled HPO config values."""
    patch: dict[str, Any] = {}
    for key in param_keys:
        if key in config:
            set_nested_config_value(patch, key, config[key])
    return patch


def apply_tunable_config(base_config: dict, sampled_config: dict, param_keys: list[str]) -> dict:
    """Return a deep-copied config with sampled HPO values applied."""
    merged = copy.deepcopy(base_config)
    patch = build_tunable_patch(sampled_config, param_keys)
    return merge_nested_config(merged, patch)


def parse_search_space(config_dict: dict) -> dict:
    """Convert YAML search_space config to Ray Tune distributions.

    Each entry should have a 'type' key and distribution-specific params:
        - randint: {type: randint, low: 5, high: 50}
        - uniform: {type: uniform, low: 1.0, high: 20.0}
        - choice: {type: choice, values: [1, 2, 3]}
        - loguniform: {type: loguniform, low: 1e-4, high: 1.0}

    Args:
        config_dict: Dictionary mapping parameter names to distribution specs.

    Returns:
        Dictionary mapping parameter names to Ray Tune sample objects.
    """
    from ray import tune

    space = {}
    for key, spec in config_dict.items():
        t = spec["type"]
        if t == "randint":
            space[key] = tune.randint(spec["low"], spec["high"])
        elif t == "uniform":
            space[key] = tune.uniform(spec["low"], spec["high"])
        elif t == "choice":
            space[key] = tune.choice(spec["values"])
        elif t == "loguniform":
            space[key] = tune.loguniform(spec["low"], spec["high"])
        else:
            raise ValueError(f"Unknown search space type '{t}' for key '{key}'")
    return space


def compute_warmup_start_date(warmup_bars: int, timeframe: str, reference_dt) -> object:
    """Compute the start datetime needed to cover warmup_bars bars of given timeframe.

    Args:
        warmup_bars: Number of bars needed.
        timeframe:   Alpaca timeframe string ("Minute", "Hour", "Day", "Week", "Month").
        reference_dt: Reference datetime to look back from.

    Returns:
        datetime — the earliest start date that should provide enough bars.
    """
    from datetime import timedelta

    MINUTES_PER_TRADING_DAY = 390
    if timeframe == "Minute":
        trading_days = (warmup_bars / MINUTES_PER_TRADING_DAY) + 1
    elif timeframe == "Hour":
        trading_days = (warmup_bars / 6.5) + 1
    elif timeframe == "Day":
        trading_days = warmup_bars + 1
    elif timeframe == "Week":
        trading_days = warmup_bars * 5 + 5
    elif timeframe == "Month":
        trading_days = warmup_bars * 22 + 22
    else:
        trading_days = warmup_bars
    calendar_days = int(trading_days * 7 / 5) + 5
    return reference_dt - timedelta(days=calendar_days)


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


def serialize_pricedata(pd_obj: PriceData) -> dict:
    """Convert a PriceData object to a dictionary suitable for MongoDB storage.

    Datetimes are kept as-is (pymongo handles them natively).

    Args:
        pd_obj: PriceData object to serialize.

    Returns:
        Dictionary with all PriceData fields.
    """
    return {
        "symbol": pd_obj.symbol,
        "timestamp": pd_obj.timestamp,
        "open": pd_obj.open,
        "high": pd_obj.high,
        "low": pd_obj.low,
        "close": pd_obj.close,
        "volume": pd_obj.volume,
        "trade_count": pd_obj.trade_count,
        "vwap": pd_obj.vwap,
        "exchange": pd_obj.exchange,
    }


def deserialize_pricedata(data: dict) -> PriceData:
    """Convert a dictionary back to a PriceData object.

    Args:
        data: Dictionary with PriceData fields.

    Returns:
        PriceData object.
    """
    return PriceData(
        symbol=data["symbol"],
        timestamp=data["timestamp"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        trade_count=data.get("trade_count"),
        vwap=data.get("vwap"),
        exchange=data.get("exchange"),
    )


def make_json_serializable(value: Any) -> Any:
    """Recursively convert common Python objects into JSON/Mongo-safe values."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return make_json_serializable(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: make_json_serializable(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return value


def restore_serialized_datetimes(value: Any) -> Any:
    """Recursively convert ISO datetime strings back into datetime objects."""
    if isinstance(value, dict):
        return {k: restore_serialized_datetimes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_serialized_datetimes(v) for v in value]
    if isinstance(value, str) and "T" in value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def serialize_order(order: Order) -> dict:
    """Convert an Order or BracketOrder to a dictionary suitable for MongoDB storage.

    Handles enum fields, child orders, and BracketOrder-specific fields.
    Datetimes are kept as-is (pymongo handles them natively).

    Args:
        order: Order or BracketOrder object to serialize.

    Returns:
        Dictionary with all order fields.
    """
    result = {
        "order_id": order.order_id,
        "placed_datetime": order.placed_datetime,
        "action": order.action.name,
        "type": order.type.name,
        "symbol": order.symbol,
        "price": order.price,
        "quantity": order.quantity,
        "cash": order.cash,
        "executed_datetime": order.executed_datetime,
        "time_in_force": order.time_in_force.name,
        "status": order.status.name,
        "tx_cost": order.tx_cost,
        "parent_id": order.parent_id,
        "platform_id": order.platform_id,
        "processed_by_portfolio": order.processed_by_portfolio,
        "is_bracket": isinstance(order, BracketOrder),
    }

    # Serialize child orders
    child_orders = {}
    for name in order._child_orders:
        child = order._child_orders_dict.get(name)
        if child is not None:
            child_orders[name] = serialize_order(child)
        else:
            child_orders[name] = None
    result["child_orders"] = child_orders if child_orders else {}

    # BracketOrder-specific fields
    if isinstance(order, BracketOrder):
        result["manual_sale"] = order.MANUAL_SALE
        if order.SOLD_ORDER is not None:
            result["sold_order_id"] = order.SOLD_ORDER.order_id
        else:
            result["sold_order_id"] = None

    return result


def deserialize_order(data: dict) -> Union[Order, BracketOrder]:
    """Convert a dictionary back to an Order or BracketOrder object.

    Args:
        data: Dictionary with serialized order data.

    Returns:
        Order or BracketOrder object with all fields restored.
    """
    is_bracket = data.get("is_bracket", False)

    if is_bracket:
        order = BracketOrder(
            placed_datetime=data["placed_datetime"],
            action=OrderAction[data["action"]],
            type=OrderType[data["type"]],
            symbol=data["symbol"],
            price=data["price"],
            quantity=data["quantity"],
            cash=data["cash"],
            executed_datetime=data.get("executed_datetime"),
            time_in_force=OrderTimeInForce[data["time_in_force"]],
            status=OrderStatus[data["status"]],
            order_id=data["order_id"],
            tx_cost=data.get("tx_cost", 0.0),
            parent_id=data.get("parent_id"),
            platform_id=data.get("platform_id"),
        )
        order.processed_by_portfolio = data.get("processed_by_portfolio", False)
        order.MANUAL_SALE = data.get("manual_sale", False)
    else:
        order = Order(
            placed_datetime=data["placed_datetime"],
            action=OrderAction[data["action"]],
            type=OrderType[data["type"]],
            symbol=data["symbol"],
            price=data["price"],
            quantity=data["quantity"],
            cash=data["cash"],
            executed_datetime=data.get("executed_datetime"),
            time_in_force=OrderTimeInForce[data["time_in_force"]],
            status=OrderStatus[data["status"]],
            order_id=data["order_id"],
            tx_cost=data.get("tx_cost", 0.0),
            parent_id=data.get("parent_id"),
            platform_id=data.get("platform_id"),
        )
        order.processed_by_portfolio = data.get("processed_by_portfolio", False)

    # Restore child orders
    child_orders_data = data.get("child_orders", {})
    for name, child_data in child_orders_data.items():
        if child_data is not None:
            child = deserialize_order(child_data)
            order.add_child_order(name, child)
        else:
            order.add_child_order(name, None)

    # Link SOLD_ORDER for BracketOrder
    if is_bracket:
        sold_order_id = data.get("sold_order_id")
        if sold_order_id is not None:
            # Find the sold order among child orders
            for name in order._child_orders:
                child = order._child_orders_dict.get(name)
                if child is not None and child.order_id == sold_order_id:
                    order.SOLD_ORDER = child
                    break

    return order


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
    return {
        "type": signal.type.name,  # Convert enum to string (e.g., "BUY")
        "symbol": signal.symbol,
        "strength": signal.strength,
        "metadata": make_json_serializable(signal.metadata),
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
    from trading.core.classes import SignalType
    return MarketSignal(
        type=SignalType[data["type"]],  # Convert string back to enum
        symbol=data["symbol"],
        strength=data["strength"],
        metadata=restore_serialized_datetimes(data.get("metadata", {})),
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
