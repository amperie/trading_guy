import importlib
from typing import Any, Union

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
