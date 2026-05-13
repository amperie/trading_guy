from __future__ import annotations

import hashlib
import importlib
import importlib.util
import types
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from trading.config.models import ComponentConfig


class ComponentLoadError(ValueError):
    pass


def resolve_class_name(component: ComponentConfig) -> str:
    if component.class_name:
        return component.class_name
    return component.implementation.rsplit(".", 1)[-1]


def import_component_class(component: ComponentConfig):
    if component.source_url:
        return _import_remote_class(component)
    if component.source_path:
        return _import_path_class(component)
    return _import_local_class(component.implementation)


def instantiate_component(component: ComponentConfig, *args, **kwargs):
    cls = import_component_class(component)
    return cls(*args, **kwargs)


def _import_local_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _import_remote_class(component: ComponentConfig):
    source_url = component.source_url or ""
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ComponentLoadError(
            f"Remote component source_url must use http or https, got '{source_url}'."
        )

    module = _load_remote_module(source_url)
    class_name = resolve_class_name(component)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ComponentLoadError(
            f"Remote component '{source_url}' does not define class '{class_name}'."
        ) from exc


def _import_path_class(component: ComponentConfig):
    source_path = component.source_path or ""
    module = _load_path_module(source_path)
    class_name = resolve_class_name(component)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ComponentLoadError(
            f"Component file '{source_path}' does not define class '{class_name}'."
        ) from exc


@lru_cache(maxsize=128)
def _download_remote_source(source_url: str) -> str:
    with urlopen(source_url, timeout=30) as response:
        encoding = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(encoding)


@lru_cache(maxsize=128)
def _load_remote_module(source_url: str):
    source = _download_remote_source(source_url)
    module_name = f"trading_remote_{hashlib.sha256(source_url.encode('utf-8')).hexdigest()[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = source_url
    exec(compile(source, source_url, "exec"), module.__dict__)
    return module


@lru_cache(maxsize=128)
def _load_path_module(source_path: str):
    normalized = source_path.replace("\\", "/").strip()
    path = Path(normalized).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise ComponentLoadError(f"Component file not found: '{path}'")

    module_name = f"trading_file_{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ComponentLoadError(f"Could not load Python module from '{path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
