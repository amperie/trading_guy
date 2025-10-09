import importlib
from typing import Any

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
    retval = next(x for x in pds if x['symbol'] == symbol)

def find_marketsignal_in_list(symbol: str, pds: list[MarketSignal]) -> MarketSignal:
    retval = next(x for x in pds if x['symbol'] == symbol)