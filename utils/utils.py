import importlib
from typing import Any


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