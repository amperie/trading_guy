from collections import deque
from datetime import datetime, timedelta

import yaml

from trading.algorithms.volatility_filtered_rsi_mean_reversion_v2 import (
    VolatilityFilteredRsiMeanReversionAlgorithmV2,
)
from trading.config import ExperimentService
from trading.core.classes import MarketSignal, PriceData, SignalType
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.risk_managed_single_symbol_portfolio import RiskManagedSingleSymbolPortfolio


def _bar(price=100.0, minute=0, hour=10):
    ts = datetime(2026, 1, 2, hour, 0) + timedelta(minutes=minute)
    return PriceData("UPRO", ts, price, price + 1, price - 1, price, 1000)


def _algo(**overrides):
    cfg = {
        "symbol": "UPRO",
        "regime_detection": {
            "ma_short_period": 3, "ma_long_period": 5, "ma_proximity_tolerance": .02,
            "atr_period": 3, "atr_percentile_window": 3,
            "atr_percentile_low": 0, "atr_percentile_high": 100,
            "require_full_atr_window": True,
        },
        "rsi_config": {
            "rsi_period": 3, "rsi_oversold_threshold": 30, "rsi_overbought_threshold": 70,
            "buy_rearm_threshold": 40, "sell_rearm_threshold": 60,
        },
        "price_confirmation": {"reversal_bar_lookback": 2},
    }
    cfg.update(overrides)
    return VolatilityFilteredRsiMeanReversionAlgorithmV2(cfg, history_length=10)


def _seed(algo, closes):
    bars = deque((_bar(price, i) for i, price in enumerate(closes)), maxlen=10)
    algo.price_history["UPRO"] = deque(closes, maxlen=10)
    algo.price_data_history["UPRO"] = bars
    return bars[-1]


def test_algorithm_requires_full_volatility_window(monkeypatch):
    algo = _algo()
    bar = _seed(algo, [100, 99, 98, 97, 98])
    monkeypatch.setattr(algo, "_calculate_ma", lambda *_: 100.0)
    monkeypatch.setattr(algo, "_calculate_atr", lambda *_: 1.0)
    monkeypatch.setattr(algo, "_calculate_rsi", lambda *_: 20.0)

    assert algo.on_data_logic([bar]) == []
    assert algo.on_data_logic([bar]) == []
    assert len(algo.on_data_logic([bar])) == 1


def test_algorithm_emits_once_until_rsi_rearms(monkeypatch):
    algo = _algo()
    bar = _seed(algo, [100, 99, 98, 97, 98])
    algo.atr_history["UPRO"] = deque([.01, .01, .01], maxlen=3)
    monkeypatch.setattr(algo, "_calculate_ma", lambda *_: 100.0)
    monkeypatch.setattr(algo, "_calculate_atr", lambda *_: 1.0)
    rsi = {"value": 20.0}
    monkeypatch.setattr(algo, "_calculate_rsi", lambda *_: rsi["value"])

    first = algo.on_data_logic([bar])
    assert len(first) == 1 and first[0].metadata["atr"] == 1.0
    assert algo.on_data_logic([bar]) == []
    rsi["value"] = 45.0
    assert algo.on_data_logic([bar]) == []
    rsi["value"] = 20.0
    assert len(algo.on_data_logic([bar])) == 1


def test_algorithm_uses_configured_reversal_lookback():
    algo = _algo(price_confirmation={"reversal_bar_lookback": 3})
    assert algo._detect_two_bar_reversal_up(deque([100, 99, 98, 99]))
    assert not algo._detect_two_bar_reversal_up(deque([100, 99, 100, 101]))


def test_algorithm_warmup_covers_long_ma_and_atr_percentile_history():
    algo = _algo(
        regime_detection={
            "ma_short_period": 20, "ma_long_period": 240,
            "ma_proximity_tolerance": .02, "atr_period": 14,
            "atr_percentile_window": 120, "atr_percentile_low": 20,
            "atr_percentile_high": 80, "require_full_atr_window": True,
        }
    )
    assert algo.required_warmup_bars == 359


def _portfolio(**overrides):
    cfg = {
        "symbol": "UPRO", "cash": 10_000, "max_exposure": .5,
        "risk_per_trade": .01, "atr_stop_multiple": 2,
        "profit_target_r_multiple": 1.5, "min_signal_strength": 0,
        "entry_start": "09:35", "entry_end": "15:30",
        "max_daily_loss_pct": 0, "max_drawdown_pct": 0,
    }
    cfg.update(overrides)
    return RiskManagedSingleSymbolPortfolio(cfg, BacktestingOrderManager(), keep_history=True)


def test_portfolio_caps_entry_by_risk_and_exposure():
    pf = _portfolio()
    signal = MarketSignal(SignalType.BUY, "UPRO", 100, {"atr": 2.0})
    result = pf.process_market_signals_for_tick([signal], [_bar()])

    assert result.orders[0].quantity == 25  # $100 risk / $4 stop
    assert pf.positions["UPRO"].quantity == 25
    bracket = result.orders[0]
    assert bracket.get_child_order("STOP").price == 96.0
    assert bracket.get_child_order("PROFIT").price == 106.0


def test_portfolio_honors_sell_and_starts_cooldown():
    pf = _portfolio(cooldown_minutes=60)
    buy = MarketSignal(SignalType.BUY, "UPRO", 100, {"atr": 2.0})
    pf.process_market_signals_for_tick([buy], [_bar()])
    sell = MarketSignal(SignalType.SELL, "UPRO", 100)
    result = pf.process_market_signals_for_tick([sell], [_bar(101, 1)])

    assert result.orders == []
    assert "UPRO" not in pf.positions
    assert pf._cooldown_until == _bar(101, 1).timestamp + timedelta(minutes=60)


def test_portfolio_blocks_entries_outside_liquidity_window():
    pf = _portfolio()
    signal = MarketSignal(SignalType.BUY, "UPRO", 100, {"atr": 2.0})
    assert pf.process_market_signals_for_tick([signal], [_bar(hour=9)]).orders == []


def test_portfolio_daily_loss_limit_liquidates():
    pf = _portfolio(max_daily_loss_pct=.005)
    buy = MarketSignal(SignalType.BUY, "UPRO", 100, {"atr": 2.0})
    pf.process_market_signals_for_tick([buy], [_bar()])

    result = pf.process_market_signals_for_tick([], [_bar(97, 1)])

    assert result.orders == []  # Active bracket is manually liquidated by the base lifecycle.
    assert "UPRO" not in pf.positions


def test_portfolio_confirms_drawdown_before_liquidating():
    pf = _portfolio(max_drawdown_pct=.005, drawdown_confirmation_bars=2)
    buy = MarketSignal(SignalType.BUY, "UPRO", 100, {"atr": 2.0})
    pf.process_market_signals_for_tick([buy], [_bar()])

    pf.process_market_signals_for_tick([], [_bar(97, 1)])
    assert "UPRO" in pf.positions
    pf.process_market_signals_for_tick([], [_bar(97, 2)])
    assert "UPRO" not in pf.positions


def test_v2_mongo_config_is_loadable():
    with open("configs/tom_rsi_mean_reversion_v2_mongo.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    experiment = ExperimentService.from_dict(cfg)
    assert experiment.data_provider.params["session_id"] == "tom_volatility_filtered_rsi_mean_reversion"
    assert experiment.algorithm.params["regime_detection"]["ma_long_period"] == 200
    assert experiment.portfolio.params["max_exposure"] == .5


def test_v2_hpo_configs_are_loadable():
    for path, samples in (
        ("configs/tom_rsi_v2_portfolio_hpo_split.yaml", 150),
        ("configs/tom_rsi_v2_algorithm_hpo_split.yaml", 200),
    ):
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        experiment = ExperimentService.from_dict(cfg)
        assert experiment.mode == "hpo"
        assert experiment.hpo.model_dump()["num_samples"] == samples
        assert experiment.state_store.enabled is False
