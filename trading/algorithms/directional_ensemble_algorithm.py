from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from trading.config.component_loader import instantiate_component
from trading.config.models import ComponentConfig
from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData, SignalType


class NestedAlgorithmSpec(BaseModel):
    implementation: str
    source_url: str | None = None
    source_path: str | None = None
    class_name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class DirectionalEnsembleAlgorithmParams(BaseModel):
    algorithm_a: NestedAlgorithmSpec
    algorithm_b: NestedAlgorithmSpec
    weight_a: float = 0.5
    weight_b: float = 0.5
    decision_threshold: float = 0.25
    target_symbol: str | None = None
    bullish_symbol: str | None = None
    bearish_symbol: str | None = None


class DirectionalEnsembleAlgorithm(Algorithm):
    """Combine two algorithms into one directional signal using continuous bias.

    Each child algorithm can optionally expose a continuous bias through
    Algorithm.get_bias() on a [-1, 1] scale. To preserve compatibility with
    legacy sparse-signal algorithms, the ensemble also falls back to deriving
    a bias from any signals emitted on the current tick.

    Emission modes:
    - target_symbol set:
        score >= threshold -> BUY target_symbol
        score <= -threshold -> SELL target_symbol
    - bullish_symbol / bearish_symbol set:
        score >= threshold -> BUY bullish_symbol
        score <= -threshold -> BUY bearish_symbol
    """

    @classmethod
    def config_model(cls):
        return DirectionalEnsembleAlgorithmParams

    def __init__(self, cfg: dict[str, Any] | None = None, history_length: int = 0):
        super().__init__(cfg or {}, history_length)
        validated = self.config_model().model_validate(self.cfg)
        self.cfg = validated.model_dump(exclude_none=True)

        self.algorithm_a = self._build_child(validated.algorithm_a)
        self.algorithm_b = self._build_child(validated.algorithm_b)
        self.weight_a = float(validated.weight_a)
        self.weight_b = float(validated.weight_b)
        self.decision_threshold = float(validated.decision_threshold)
        self.target_symbol = validated.target_symbol
        self.bullish_symbol = validated.bullish_symbol
        self.bearish_symbol = validated.bearish_symbol
        self._last_bias_a = 0.0
        self._last_bias_b = 0.0
        self._last_score = 0.0

        if self.target_symbol is None and (self.bullish_symbol is None or self.bearish_symbol is None):
            raise ValueError(
                "DirectionalEnsembleAlgorithm requires either target_symbol or "
                "both bullish_symbol and bearish_symbol."
            )

    def _build_child(self, spec: NestedAlgorithmSpec) -> Algorithm:
        params = dict(spec.params)
        history_length = int(params.pop("history_length", 0))
        component = ComponentConfig(
            implementation=spec.implementation,
            source_url=spec.source_url,
            source_path=spec.source_path,
            class_name=spec.class_name,
            params=params,
        )
        child = instantiate_component(component, cfg=params, history_length=history_length)
        if not isinstance(child, Algorithm):
            raise TypeError(
                f"DirectionalEnsembleAlgorithm child '{spec.implementation}' must inherit from Algorithm."
            )
        return child

    @staticmethod
    def _clamp_bias(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _normalize_weighted_score(self, weighted_sum: float) -> float:
        total_weight = abs(self.weight_a) + abs(self.weight_b)
        if total_weight <= 0:
            return 0.0
        return self._clamp_bias(weighted_sum / total_weight)

    def _signal_to_bias(self, signal: MarketSignal) -> float:
        magnitude = max(0.0, min(1.0, float(getattr(signal, "strength", 100)) / 100.0))

        if self.target_symbol is not None and signal.symbol == self.target_symbol:
            if signal.type == SignalType.BUY:
                return magnitude
            if signal.type == SignalType.SELL:
                return -magnitude

        if self.bullish_symbol is not None and signal.symbol == self.bullish_symbol:
            if signal.type == SignalType.BUY:
                return magnitude
            if signal.type == SignalType.SELL:
                return -magnitude

        if self.bearish_symbol is not None and signal.symbol == self.bearish_symbol:
            if signal.type == SignalType.BUY:
                return -magnitude
            if signal.type == SignalType.SELL:
                return magnitude

        return 0.0

    def _derive_child_bias(self, child: Algorithm, child_signals: list[MarketSignal], data: list[PriceData]) -> float:
        bias = self._clamp_bias(child.get_bias(data))
        if bias != 0.0:
            return bias

        derived = 0.0
        for signal in child_signals:
            signal_bias = self._signal_to_bias(signal)
            if abs(signal_bias) > abs(derived):
                derived = signal_bias
        return derived

    @property
    def required_warmup_bars(self) -> int:
        return max(
            self.history_length,
            self.algorithm_a.required_warmup_bars,
            self.algorithm_b.required_warmup_bars,
        )

    def get_bias(self, data: list[PriceData] | None = None) -> float:
        return self._last_score

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        child_a_signals = self.algorithm_a.on_data(data)
        child_b_signals = self.algorithm_b.on_data(data)

        bias_a = self._derive_child_bias(self.algorithm_a, child_a_signals, data)
        bias_b = self._derive_child_bias(self.algorithm_b, child_b_signals, data)
        score = self._normalize_weighted_score(self.weight_a * bias_a + self.weight_b * bias_b)
        self._last_bias_a = bias_a
        self._last_bias_b = bias_b
        self._last_score = score

        if abs(score) < self.decision_threshold:
            return []

        strength = max(1, min(100, int(round(abs(score) * 100))))
        metadata = {
            "algorithm_a_bias": self._last_bias_a,
            "algorithm_b_bias": self._last_bias_b,
            "ensemble_score": self._last_score,
            "decision_threshold": self.decision_threshold,
            "algorithm_a_class": self.algorithm_a.__class__.__name__,
            "algorithm_b_class": self.algorithm_b.__class__.__name__,
        }

        if score > 0:
            symbol = self.bullish_symbol or self.target_symbol
            signal_type = SignalType.BUY
        else:
            if self.bearish_symbol is not None:
                symbol = self.bearish_symbol
                signal_type = SignalType.BUY
            else:
                symbol = self.target_symbol
                signal_type = SignalType.SELL

        if symbol is None:
            return []

        return [MarketSignal(type=signal_type, symbol=symbol, strength=strength, metadata=metadata)]
