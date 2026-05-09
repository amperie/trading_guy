from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading.config import BuiltComponents, ExperimentConfig, ExperimentDescriptor, ExperimentService


@dataclass(slots=True)
class ExperimentRequest:
    config: dict[str, Any] | None = None
    config_path: str | None = None
    overrides: dict[str, Any] | None = None


def load_experiment(request: ExperimentRequest) -> ExperimentConfig:
    if request.config_path:
        return ExperimentService.from_file(request.config_path, overrides=request.overrides)
    if request.config is None:
        raise ValueError("ExperimentRequest requires either config_path or config")
    if request.overrides:
        merged = dict(request.config)
        merged.update(request.overrides)
        return ExperimentService.from_dict(merged)
    return ExperimentService.from_dict(request.config)


def normalize_experiment(config: dict[str, Any]) -> ExperimentConfig:
    return ExperimentService.from_dict(config)


def build_runtime(config: ExperimentConfig) -> BuiltComponents:
    return ExperimentService.build(config)


def describe_experiment(config: ExperimentConfig) -> ExperimentDescriptor:
    return ExperimentService.describe(config)
