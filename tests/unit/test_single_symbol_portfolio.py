from __future__ import annotations

from datetime import datetime

from trading.core.classes import MarketSignal, OrderType, PriceData, SignalType
from trading.core.om.backtesting_om import BacktestingOrderManager
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio


def _tick(symbol: str, price: float, ts: datetime) -> list[PriceData]:
    return [PriceData(symbol, ts, price, price, price, price, 1000)]


def _buy_signal(symbol: str) -> MarketSignal:
    return MarketSignal(SignalType.BUY, symbol, 100, metadata={})


def _make_pf() -> SingleSymbolPortfolio:
    om = BacktestingOrderManager(cfg={"market_hours_only": True})
    return SingleSymbolPortfolio(
        cfg={
            "symbol": "UPRO",
            "cash": 10000.0,
            "keep_history": True,
            "stop_pct": 1.5,
            "profit_pct": 2.0,
            "tx_cost": 0.0,
        },
        order_manager=om,
    )


def _make_trailing_pf() -> SingleSymbolPortfolio:
    om = BacktestingOrderManager(cfg={"market_hours_only": True})
    return SingleSymbolPortfolio(
        cfg={
            "symbol": "UPRO",
            "cash": 10000.0,
            "keep_history": True,
            "stop_pct": 1.5,
            "profit_pct": 2.0,
            "stop_type": "trailing",
            "trailing_stop_pct": 1.5,
            "tx_cost": 0.0,
        },
        order_manager=om,
    )


def _make_nested_trailing_pf() -> SingleSymbolPortfolio:
    om = BacktestingOrderManager(cfg={"market_hours_only": True})
    return SingleSymbolPortfolio(
        cfg={
            "symbol": "UPRO",
            "cash": 10000.0,
            "keep_history": True,
            "stop_pct": 25.0,
            "profit_pct": 50.0,
            "bracket": {
                "stop_type": "trailing",
                "stop_pct": 2.0,
                "trailing_stop_pct": 1.25,
                "profit_pct": 4.0,
            },
        },
        order_manager=om,
    )


def test_premarket_buy_signal_does_not_accumulate_brackets():
    pf = _make_pf()

    ts0 = datetime(2024, 1, 1, 8, 0)
    result1 = pf.process_market_signals_for_tick([_buy_signal("UPRO")], _tick("UPRO", 50.0, ts0))

    ts1 = datetime(2024, 1, 1, 8, 1)
    result2 = pf.process_market_signals_for_tick([_buy_signal("UPRO")], _tick("UPRO", 51.0, ts1))

    assert len(result1.orders) == 1
    assert len(result2.orders) == 0
    assert len(pf.om.pending_orders_by_id) == 1
    assert "UPRO" not in pf.positions


def test_opening_bell_fills_single_deferred_bracket_without_duplicate_entry():
    pf = _make_pf()

    pf.process_market_signals_for_tick([_buy_signal("UPRO")], _tick("UPRO", 50.0, datetime(2024, 1, 1, 8, 0)))
    pf.process_market_signals_for_tick([_buy_signal("UPRO")], _tick("UPRO", 51.0, datetime(2024, 1, 1, 8, 1)))

    result = pf.process_market_signals_for_tick([], _tick("UPRO", 52.0, datetime(2024, 1, 1, 9, 30)))

    assert len(result.orders) == 0
    assert len(pf.om.pending_orders_by_id) == 1
    assert pf.positions["UPRO"].quantity == int(10000.0 / 52.0)


def test_buy_signal_ignored_while_position_is_open():
    pf = _make_pf()

    pf.process_market_signals_for_tick([_buy_signal("UPRO")], _tick("UPRO", 50.0, datetime(2024, 1, 1, 9, 30)))
    qty_after_fill = pf.positions["UPRO"].quantity

    result = pf.process_market_signals_for_tick([_buy_signal("UPRO")], _tick("UPRO", 50.5, datetime(2024, 1, 1, 9, 31)))

    assert len(result.orders) == 0
    assert pf.positions["UPRO"].quantity == qty_after_fill
    assert len(pf.om.pending_orders_by_id) == 1


def test_trailing_stop_config_creates_trailing_bracket_child():
    pf = _make_trailing_pf()
    result = pf.process_market_signals_for_tick(
        [_buy_signal("UPRO")],
        _tick("UPRO", 50.0, datetime(2024, 1, 1, 9, 30)),
    )

    order = result.orders[0]
    stop = order.get_child_order("STOP")
    assert stop.type == OrderType.TRAILING_STOP
    assert stop.trail_percent == 1.5
    assert stop.price == 49.25


def test_nested_bracket_config_overrides_legacy_flat_keys():
    pf = _make_nested_trailing_pf()
    result = pf.process_market_signals_for_tick(
        [_buy_signal("UPRO")],
        _tick("UPRO", 50.0, datetime(2024, 1, 1, 9, 30)),
    )

    order = result.orders[0]
    stop = order.get_child_order("STOP")
    profit = order.get_child_order("PROFIT")
    assert stop.type == OrderType.TRAILING_STOP
    assert stop.trail_percent == 1.25
    assert stop.price == 49.38
    assert profit.price == 52.0
