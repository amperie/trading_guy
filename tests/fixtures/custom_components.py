from __future__ import annotations

from pydantic import BaseModel
from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData, SignalType


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


class FixedBiasAlgorithmConfig(BaseModel):
    bias: float


class FixedBiasAlgorithm(Algorithm):
    def __init__(self, cfg=None, history_length: int = 0):
        super().__init__(cfg=cfg, history_length=history_length)
        self.bias = float((cfg or {}).get("bias", 0.0))

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        return []

    def get_bias(self, data: list[PriceData] | None = None) -> float:
        return self.bias

    @classmethod
    def config_model(cls):
        return FixedBiasAlgorithmConfig


class FixedSignalAlgorithmConfig(BaseModel):
    symbol: str
    signal_type: str = "BUY"
    strength: int = 100


class FixedSignalAlgorithm(Algorithm):
    def __init__(self, cfg=None, history_length: int = 0):
        super().__init__(cfg=cfg, history_length=history_length)
        cfg = cfg or {}
        self.symbol = cfg["symbol"]
        self.signal_type = SignalType[cfg.get("signal_type", "BUY").upper()]
        self.strength = int(cfg.get("strength", 100))

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        return [MarketSignal(type=self.signal_type, symbol=self.symbol, strength=self.strength)]

    @classmethod
    def config_model(cls):
        return FixedSignalAlgorithmConfig
