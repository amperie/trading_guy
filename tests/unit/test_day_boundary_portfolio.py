from __future__ import annotations

from datetime import datetime

from trading.core.classes import OrderAction, PriceData
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.day_boundary_portfolio import DayBoundaryPortfolio


def _tick(price: float, ts: datetime) -> list[PriceData]:
    return [PriceData("SPY", ts, price, price, price, price, 1000)]


def _make_pf(flip_behavior: bool = False) -> DayBoundaryPortfolio:
    return DayBoundaryPortfolio(
        cfg={
            "symbol": "SPY",
            "cash": 10_000.0,
            "keep_history": True,
            "flip_behavior": flip_behavior,
            "tx_cost": 0.0,
        },
        order_manager=BacktestingOrderManager(),
    )


def test_default_buys_at_close_and_sells_next_open():
    pf = _make_pf()

    open_result = pf.process_market_signals_for_tick([], _tick(100.0, datetime(2024, 1, 2, 9, 30)))
    close_result = pf.process_market_signals_for_tick([], _tick(100.0, datetime(2024, 1, 2, 16, 0)))
    next_open_result = pf.process_market_signals_for_tick([], _tick(110.0, datetime(2024, 1, 3, 9, 30)))

    assert open_result.orders == []
    assert close_result.orders[0].action == OrderAction.BUY
    assert close_result.orders[0].quantity == 100
    assert next_open_result.orders[0].action == OrderAction.SELL
    assert next_open_result.orders[0].quantity == 100
    assert "SPY" not in pf.positions
    assert pf.cash == 11_000.0


def test_default_does_not_duplicate_close_buys():
    pf = _make_pf()

    first = pf.process_market_signals_for_tick([], _tick(100.0, datetime(2024, 1, 2, 16, 0)))
    second = pf.process_market_signals_for_tick([], _tick(101.0, datetime(2024, 1, 2, 16, 1)))

    assert len(first.orders) == 1
    assert second.orders == []
    assert pf.positions["SPY"].quantity == 100


def test_flipped_buys_at_open_and_sells_at_close():
    pf = _make_pf(flip_behavior=True)

    open_result = pf.process_market_signals_for_tick([], _tick(100.0, datetime(2024, 1, 2, 9, 30)))
    close_result = pf.process_market_signals_for_tick([], _tick(105.0, datetime(2024, 1, 2, 16, 0)))
    next_open_result = pf.process_market_signals_for_tick([], _tick(106.0, datetime(2024, 1, 3, 9, 30)))

    assert open_result.orders[0].action == OrderAction.BUY
    assert open_result.orders[0].quantity == 100
    assert close_result.orders[0].action == OrderAction.SELL
    assert close_result.orders[0].quantity == 100
    assert next_open_result.orders[0].action == OrderAction.BUY
    assert next_open_result.orders[0].quantity == int(10_500.0 / 106.0)


def test_flipped_does_not_buy_if_first_tick_is_at_close():
    pf = _make_pf(flip_behavior=True)

    result = pf.process_market_signals_for_tick([], _tick(100.0, datetime(2024, 1, 2, 16, 0)))

    assert result.orders == []
    assert "SPY" not in pf.positions
