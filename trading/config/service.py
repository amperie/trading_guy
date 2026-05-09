from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from trading.config.component_registry import AllowExtraModel, KNOWN_COMPONENT_MODELS
from trading.config.models import ComponentConfig, ExperimentConfig
from utils.config_manager import ConfigManager
from utils.utils import instantiate_from_string


LEGACY_COMPONENT_KEYS = {
    "algorithm": "algorithm",
    "portfolio": "portfolio",
    "order_manager": "order_manager",
    "data_provider": "provider",
}

SENSITIVE_KEYS = {"api_key", "secret_key", "password", "token"}


@dataclass(slots=True)
class BuiltComponents:
    config: ExperimentConfig
    data_provider: Any | None
    algorithm: Any
    order_manager: Any
    portfolio: Any


class ExperimentDescriptor(BaseModel):
    mode: str
    execution_mode: str
    config_hash: str
    algorithm: str
    portfolio: str
    order_manager: str
    data_provider: str | None = None
    params: dict[str, Any]


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _normalize_component(section: dict[str, Any], legacy_key: str) -> dict[str, Any]:
    if "implementation" in section:
        params = copy.deepcopy(section.get("params") or {})
        passthrough = {
            k: copy.deepcopy(v)
            for k, v in section.items()
            if k not in {"implementation", "params"}
        }
        params.update(passthrough)
        return {"implementation": section["implementation"], "params": params}

    params = {k: copy.deepcopy(v) for k, v in section.items() if k != legacy_key}
    return {"implementation": section[legacy_key], "params": params}


def _config_model_for(implementation: str) -> type[BaseModel]:
    cls = _import_class(implementation)
    config_model = getattr(cls, "config_model", None)
    if callable(config_model):
        maybe_model = config_model()
        if isinstance(maybe_model, type) and issubclass(maybe_model, BaseModel):
            return maybe_model
    return KNOWN_COMPONENT_MODELS.get(implementation, AllowExtraModel)


def _import_class(dotted_path: str):
    import importlib

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _validate_component(component: ComponentConfig, role: str) -> ComponentConfig:
    model_cls = _config_model_for(component.implementation)
    validated = model_cls.model_validate(component.params)
    params = validated.model_dump(exclude_none=True)
    if role == "algorithm" and "history_length" in component.params:
        params["history_length"] = component.params["history_length"]
    return component.model_copy(update={"params": params})


def normalize_config_dict(raw_cfg: dict[str, Any]) -> ExperimentConfig:
    normalized = copy.deepcopy(raw_cfg)
    for section_name, legacy_key in LEGACY_COMPONENT_KEYS.items():
        if section_name in normalized and normalized[section_name] is not None:
            normalized[section_name] = _normalize_component(normalized[section_name], legacy_key)

    config = ExperimentConfig.model_validate(normalized)
    validated_components = {
        "algorithm": _validate_component(config.algorithm, "algorithm"),
        "portfolio": _validate_component(config.portfolio, "portfolio"),
        "order_manager": _validate_component(config.order_manager, "order_manager"),
        "data_provider": _validate_component(config.data_provider, "data_provider") if config.data_provider is not None else None,
    }
    return config.model_copy(update=validated_components)


def _flatten_dict(cfg: dict[str, Any], prefix: str = "config") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in SENSITIVE_KEYS:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, full_key))
        elif isinstance(value, list):
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                result[full_key[:250]] = tuple(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[full_key[:250]] = value
    return result


def load_experiment_config(config_path: str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    root_cfg = ConfigManager().config
    with open(config_path, "r") as handle:
        profile = yaml.safe_load(handle) or {}
    merged = _deep_merge(root_cfg, profile)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return normalize_config_dict(merged)


def _build_component(spec: ComponentConfig, role: str, order_manager: Any | None = None) -> Any:
    kwargs = dict(spec.params)
    if role == "algorithm":
        history_length = kwargs.pop("history_length", 0)
        return instantiate_from_string(spec.implementation, cfg=kwargs, history_length=history_length)
    if role == "portfolio":
        return instantiate_from_string(spec.implementation, cfg=kwargs, order_manager=order_manager)
    return instantiate_from_string(spec.implementation, cfg=kwargs) if kwargs else instantiate_from_string(spec.implementation)


def build_components(config: ExperimentConfig) -> BuiltComponents:
    order_manager = _build_component(config.order_manager, "order_manager")
    algorithm = _build_component(config.algorithm, "algorithm")
    portfolio = _build_component(config.portfolio, "portfolio", order_manager=order_manager)
    data_provider = None
    if config.data_provider is not None:
        data_provider = _build_component(config.data_provider, "data_provider")
    return BuiltComponents(
        config=config,
        data_provider=data_provider,
        algorithm=algorithm,
        order_manager=order_manager,
        portfolio=portfolio,
    )


class ExperimentService:
    @staticmethod
    def from_dict(raw_cfg: dict[str, Any]) -> ExperimentConfig:
        return normalize_config_dict(raw_cfg)

    @staticmethod
    def from_file(config_path: str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
        return load_experiment_config(config_path, overrides=overrides)

    @staticmethod
    def build(config: ExperimentConfig) -> BuiltComponents:
        return build_components(config)

    @staticmethod
    def describe(config: ExperimentConfig) -> ExperimentDescriptor:
        serializable = config.model_dump(mode="json", exclude_none=True)
        flat = _flatten_dict(serializable)
        config_hash = hashlib.sha256(
            json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return ExperimentDescriptor(
            mode=config.mode,
            execution_mode=config.execution_mode(),
            config_hash=config_hash,
            algorithm=config.algorithm.implementation,
            portfolio=config.portfolio.implementation,
            order_manager=config.order_manager.implementation,
            data_provider=config.data_provider.implementation if config.data_provider else None,
            params=flat,
        )


__all__ = [
    "BuiltComponents",
    "ExperimentDescriptor",
    "ExperimentService",
    "build_components",
    "load_experiment_config",
    "normalize_config_dict",
]
