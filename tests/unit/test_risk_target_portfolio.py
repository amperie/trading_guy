from datetime import datetime, timedelta

import yaml

from trading.core.classes import MarketSignal, OrderAction, PriceData, SignalType
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.risk_target_portfolio import RiskTargetPortfolio


def _tick(price: float, minutes: int = 0) -> list[PriceData]:
    ts = datetime(2024, 1, 1, 10, 0) + timedelta(minutes=minutes)
    return [PriceData("SPY", ts, price, price, price, price, 1000)]


def test_risk_target_portfolio_buys_default_exposure_before_vol_history():
    om = BacktestingOrderManager()
    pf = RiskTargetPortfolio(
        {
            "symbol": "SPY",
            "default_exposure": 0.5,
            "max_exposure": 1.0,
            "target_volatility": 0.15,
            "volatility_lookback": 20,
        },
        om,
        10_000.0,
        {},
        True,
    )

    result = pf.process_market_signals_for_tick([MarketSignal(SignalType.BUY, "SPY", 100)], _tick(100.0))

    assert len(result.orders) == 1
    assert result.orders[0].action == OrderAction.BUY
    assert result.orders[0].quantity == 50
    assert pf.positions["SPY"].quantity == 50
    assert pf.cash == 5_000.0


def test_risk_target_portfolio_sell_signal_liquidates_position():
    om = BacktestingOrderManager()
    pf = RiskTargetPortfolio(
        {"symbol": "SPY", "default_exposure": 1.0, "max_exposure": 1.0},
        om,
        10_000.0,
        {},
        True,
    )
    pf.process_market_signals_for_tick([MarketSignal(SignalType.BUY, "SPY", 100)], _tick(100.0))

    result = pf.process_market_signals_for_tick([MarketSignal(SignalType.SELL, "SPY", 100)], _tick(110.0, 1))

    assert len(result.orders) == 1
    assert result.orders[0].action == OrderAction.SELL
    assert result.orders[0].quantity == 100
    assert "SPY" not in pf.positions
    assert pf.cash == 11_000.0


def test_risk_target_portfolio_drawdown_liquidates_and_halts_new_buys():
    om = BacktestingOrderManager()
    pf = RiskTargetPortfolio(
        {
            "symbol": "SPY",
            "default_exposure": 1.0,
            "max_exposure": 1.0,
            "drawdown_limit_pct": 0.10,
            "halt_on_drawdown": True,
        },
        om,
        10_000.0,
        {},
        True,
    )
    pf.process_market_signals_for_tick([MarketSignal(SignalType.BUY, "SPY", 100)], _tick(100.0))

    result = pf.process_market_signals_for_tick([], _tick(85.0, 1))

    assert len(result.orders) == 1
    assert result.orders[0].action == OrderAction.SELL
    assert "SPY" not in pf.positions
    assert pf.cash == 8_500.0
    assert pf._drawdown_halted is True

    second_result = pf.process_market_signals_for_tick([MarketSignal(SignalType.BUY, "SPY", 100)], _tick(85.0, 2))

    assert second_result.orders == []
    assert "SPY" not in pf.positions


def test_risk_target_portfolio_uses_volatility_to_reduce_exposure():
    om = BacktestingOrderManager()
    pf = RiskTargetPortfolio(
        {
            "symbol": "SPY",
            "default_exposure": 1.0,
            "max_exposure": 1.0,
            "target_volatility": 0.10,
            "volatility_lookback": 3,
            "annualization_factor": 252.0,
        },
        om,
        10_000.0,
        {},
        True,
    )
    for idx, price in enumerate([100.0, 110.0, 95.0, 115.0]):
        pf.process_market_signals_for_tick([], _tick(price, idx))

    result = pf.process_market_signals_for_tick([MarketSignal(SignalType.BUY, "SPY", 100)], _tick(115.0, 4))

    assert len(result.orders) == 1
    assert result.orders[0].action == OrderAction.BUY
    assert 0 < result.orders[0].quantity < int(10_000.0 / 115.0)
    assert pf._realized_volatility() > 0.10


def test_documented_risk_target_config_uses_loadable_portfolio_path():
    with open("configs/example_backtest_risk_target.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    portfolio_cfg = cfg["portfolio"]
    pf = RiskTargetPortfolio(portfolio_cfg, BacktestingOrderManager(), portfolio_cfg["cash"], {}, True)

    assert portfolio_cfg["portfolio"] == "trading.core.pf.risk_target_portfolio.RiskTargetPortfolio"
    assert pf.symbol == "SPY"
    assert pf.target_volatility == 0.15
    assert pf.annualization_factor == 19656.0
    assert pf.drawdown_limit_pct == 0.10
