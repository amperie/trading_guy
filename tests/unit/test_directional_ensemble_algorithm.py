from __future__ import annotations

from datetime import datetime

import pytest

from trading.algorithms.directional_ensemble_algorithm import DirectionalEnsembleAlgorithm
from trading.algorithms.test_algorithm import TestAlgorithm as LegacyTestAlgorithm
from trading.config import ExperimentService
from trading.core.classes import PriceData, SignalType


def _tick(symbol: str = "SPY") -> list[PriceData]:
    return [
        PriceData(
            symbol=symbol,
            timestamp=datetime(2024, 1, 2, 9, 30),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000,
        )
    ]


def test_algorithm_default_bias_is_neutral():
    algorithm = LegacyTestAlgorithm({})
    assert algorithm.get_bias([]) == 0.0


def test_directional_ensemble_uses_child_continuous_biases():
    algorithm = DirectionalEnsembleAlgorithm(
        cfg={
            "algorithm_a": {
                "implementation": "tests.fixtures.custom_components.FixedBiasAlgorithm",
                "params": {"bias": 0.8},
            },
            "algorithm_b": {
                "implementation": "tests.fixtures.custom_components.FixedBiasAlgorithm",
                "params": {"bias": -0.2},
            },
            "weight_a": 0.75,
            "weight_b": 0.25,
            "decision_threshold": 0.3,
            "target_symbol": "SPY",
        }
    )

    signals = algorithm.on_data(_tick())

    assert len(signals) == 1
    signal = signals[0]
    assert signal.type == SignalType.BUY
    assert signal.symbol == "SPY"
    assert signal.strength == 55
    assert signal.metadata["algorithm_a_bias"] == pytest.approx(0.8)
    assert signal.metadata["algorithm_b_bias"] == pytest.approx(-0.2)
    assert signal.metadata["ensemble_score"] == pytest.approx(0.55)
    assert algorithm.get_bias() == pytest.approx(0.55)


def test_directional_ensemble_falls_back_to_legacy_child_signals():
    algorithm = DirectionalEnsembleAlgorithm(
        cfg={
            "algorithm_a": {
                "implementation": "tests.fixtures.custom_components.FixedSignalAlgorithm",
                "params": {"symbol": "UPRO", "signal_type": "BUY", "strength": 80},
            },
            "algorithm_b": {
                "implementation": "tests.fixtures.custom_components.FixedSignalAlgorithm",
                "params": {"symbol": "SPXU", "signal_type": "BUY", "strength": 20},
            },
            "weight_a": 0.5,
            "weight_b": 0.5,
            "decision_threshold": 0.25,
            "bullish_symbol": "UPRO",
            "bearish_symbol": "SPXU",
        }
    )

    signals = algorithm.on_data(_tick())

    assert len(signals) == 1
    signal = signals[0]
    assert signal.type == SignalType.BUY
    assert signal.symbol == "UPRO"
    assert signal.strength == 30
    assert signal.metadata["algorithm_a_bias"] == pytest.approx(0.8)
    assert signal.metadata["algorithm_b_bias"] == pytest.approx(-0.2)
    assert signal.metadata["ensemble_score"] == pytest.approx(0.3)


def test_experiment_service_builds_directional_ensemble_runtime():
    config = ExperimentService.from_dict(
        {
            "mode": "backtest",
            "algorithm": {
                "implementation": "trading.algorithms.directional_ensemble_algorithm.DirectionalEnsembleAlgorithm",
                "params": {
                    "algorithm_a": {
                        "implementation": "tests.fixtures.custom_components.FixedBiasAlgorithm",
                        "params": {"bias": 0.6},
                    },
                    "algorithm_b": {
                        "implementation": "tests.fixtures.custom_components.FixedBiasAlgorithm",
                        "params": {"bias": -0.1},
                    },
                    "weight_a": 0.6,
                    "weight_b": 0.4,
                    "decision_threshold": 0.2,
                    "target_symbol": "SPY",
                },
            },
            "portfolio": {
                "implementation": "tests.fixtures.custom_components.CustomPortfolio",
                "params": {},
            },
            "order_manager": {
                "implementation": "tests.fixtures.custom_components.CustomOrderManager",
                "params": {},
            },
            "data_provider": {
                "implementation": "tests.fixtures.custom_components.CustomDataProvider",
                "params": {},
            },
            "analysis": {"enabled": False, "log_to_mlflow": False},
            "state_store": {"enabled": False},
            "mlflow": {"enabled": False},
            "logging": {},
        }
    )

    built = ExperimentService.build(config)

    assert built.algorithm.__class__.__name__ == "DirectionalEnsembleAlgorithm"
    assert built.algorithm.algorithm_a.__class__.__name__ == "FixedBiasAlgorithm"
    assert built.algorithm.algorithm_b.__class__.__name__ == "FixedBiasAlgorithm"
    assert built.algorithm.target_symbol == "SPY"
