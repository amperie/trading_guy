from __future__ import annotations

from pydantic import BaseModel


class CustomAlgorithmConfig(BaseModel):
    lookback: int
    threshold: float = 1.0


class CustomAlgorithm:
    def __init__(self, cfg=None, history_length: int = 0):
        self.cfg = cfg or {}
        self.history_length = history_length

    @classmethod
    def config_model(cls):
        return CustomAlgorithmConfig


class CustomOrderManager:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}


class CustomPortfolio:
    def __init__(self, cfg=None, order_manager=None):
        self.cfg = cfg or {}
        self.order_manager = order_manager


class CustomDataProvider:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
