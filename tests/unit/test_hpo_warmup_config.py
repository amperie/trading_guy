from trading.commands.hpo import _build_minimal_warmup_dp_cfg
from trading.data_providers.alpaca_data_provider import AlpacaDataProvider


class _AlgoWithWarmup:
    def __init__(self, cfg=None, history_length=0):
        self.required_warmup_bars = int(history_length)


def test_build_minimal_warmup_dp_cfg_uses_explicit_window_for_alpaca():
    cfg = _build_minimal_warmup_dp_cfg(
        alg_cfg={"history_length": 490},
        validation_dp_cfg={"start_date": "2026-03-19"},
        training_dp_cfg={
            "symbols": ["UPRO"],
            "timeframe": "Minute",
            "start_date": "2025-01-01",
            "end_date": "2026-03-18 23:59:59.999999",
        },
        algorithm_class=_AlgoWithWarmup,
        data_provider_class=AlpacaDataProvider,
    )

    assert cfg is not None
    assert cfg["end_date"] == "2026-03-18 23:59:59.999999"
    assert cfg["limit"] == 490
    assert "start_date" in cfg
    assert cfg["start_date"] < cfg["end_date"]


def test_build_minimal_warmup_dp_cfg_uses_validation_start_when_training_end_missing():
    cfg = _build_minimal_warmup_dp_cfg(
        alg_cfg={"history_length": 10},
        validation_dp_cfg={"start_date": "2026-03-19"},
        training_dp_cfg={"timeframe": "Day"},
        algorithm_class=_AlgoWithWarmup,
        data_provider_class=AlpacaDataProvider,
    )

    assert cfg is not None
    assert cfg["end_date"] == "2026-03-18 23:59:59.999999"
    assert cfg["limit"] == 10
    assert cfg["start_date"] < cfg["end_date"]
