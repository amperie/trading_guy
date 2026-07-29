from types import SimpleNamespace

import importlib
import pandas as pd


sa_app = importlib.import_module("web.session_analyzer.app")


class _FakeStore:
    def __init__(self):
        self.session = {"_id": "sess-1", "name": "Session 1", "account_id": "acct", "metadata": {"timeframe": "Minute"}}

    def get_session(self, session_id):
        return self.session if session_id == "sess-1" else None

    def load_equity_history(self, session_id):
        return {
            "value_history": {
                sa_app.datetime(2024, 1, 1, 9, 30): 1000.0,
                sa_app.datetime(2024, 1, 2, 9, 30): 1010.0,
                sa_app.datetime(2024, 1, 3, 9, 30): 1025.0,
            },
            "cash_history": {
                sa_app.datetime(2024, 1, 1, 9, 30): 500.0,
                sa_app.datetime(2024, 1, 2, 9, 30): 480.0,
                sa_app.datetime(2024, 1, 3, 9, 30): 470.0,
            },
        }

    def load_orders(self, session_id):
        status = SimpleNamespace(name="FILLED")
        order = SimpleNamespace(status=status)
        return {
            "all_orders": {"1": order},
            "filled_orders_by_id": {"1": order},
            "pending_orders_by_id": {},
        }

    def load_portfolio_history(self, session_id):
        return {
            "value_history": {
                sa_app.datetime(2024, 1, 1, 9, 30): 1000.0,
                sa_app.datetime(2024, 1, 2, 9, 30): 1010.0,
            },
            "cash_history": {
                sa_app.datetime(2024, 1, 1, 9, 30): 500.0,
                sa_app.datetime(2024, 1, 2, 9, 30): 480.0,
            },
            "tick_history": {
                sa_app.datetime(2024, 1, 1, 9, 30): [SimpleNamespace(symbol="AAPL", close=10.0, high=10.0, low=10.0)],
                sa_app.datetime(2024, 1, 2, 9, 30): [SimpleNamespace(symbol="AAPL", close=11.0, high=11.0, low=11.0)],
            },
            "signals_history": {
                sa_app.datetime(2024, 1, 2, 9, 30): [
                    SimpleNamespace(
                        symbol="AAPL",
                        type=SimpleNamespace(name="BUY"),
                        strength=1.0,
                        metadata={"rsi_value": 30.0},
                    )
                ]
            },
        }


def test_session_summary_route_returns_lightweight_payload(monkeypatch):
    monkeypatch.setattr(sa_app, "_get_state_store", lambda db=None: _FakeStore())

    def fail(*args, **kwargs):
        raise AssertionError("heavy analyzer should not run on summary route")

    monkeypatch.setattr(sa_app, "_build_analyzer_from_data", fail)
    monkeypatch.setattr(sa_app, "_fetch_benchmark_series", fail)

    client = sa_app.app.test_client()
    response = client.get("/api/session/sess-1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["portfolio"]["total_value"] == []
    assert payload["symbols"] == {}
    assert payload["signals"] == []
    assert payload["trades"] == []
    assert payload["orders"] is None
    assert payload["metrics"]["bars"] == 3


def test_session_details_route_returns_symbols_signals_and_trades(monkeypatch):
    monkeypatch.setattr(sa_app, "_get_state_store", lambda db=None: _FakeStore())
    monkeypatch.setattr(sa_app, "_fetch_benchmark_series", lambda value_history, metadata, session=None, symbol="SPY": ([], None))

    class _FakeAnalyzer:
        def extract_trades(self):
            return [
                SimpleNamespace(
                    symbol="AAPL",
                    entry_time=sa_app.datetime(2024, 1, 1, 9, 30),
                    exit_time=sa_app.datetime(2024, 1, 2, 9, 30),
                    entry_price=10.0,
                    exit_price=11.0,
                    quantity=1,
                    pnl=1.0,
                    pnl_pct=10.0,
                    duration=3600.0,
                    is_bracket=False,
                    bracket_exit_type=None,
                )
            ]

        def calculate_metrics(self):
            return {"total_return_pct": 2.5, "total_trades": 1}

        def calculate_benchmark_comparison(self):
            return {"_comparison": {"alpha": 1.0, "outperformance": True}}

    monkeypatch.setattr(sa_app, "_build_analyzer_from_data", lambda pf_data, order_data, session_metadata: _FakeAnalyzer())

    client = sa_app.app.test_client()
    response = client.get("/api/session/sess-1/details")
    assert response.status_code == 200
    payload = response.get_json()
    assert "AAPL" in payload["symbols"]
    assert payload["signals"][0]["symbol"] == "AAPL"
    assert payload["trades"][0]["symbol"] == "AAPL"
    assert payload["metrics"]["total_trades"] == 1


def test_spy_fallback_fetches_boundary_prices_from_alpaca(monkeypatch):
    calls = []

    class _FakeAlpacaProvider:
        def __init__(self, cfg):
            self.cfg = cfg
            calls.append(cfg)

        def load_data(self):
            pass

        def get_data(self):
            if self.cfg["sort"] == "asc":
                return pd.DataFrame([{"timestamp": sa_app.datetime(2024, 1, 1, 9, 30), "close": 100.0}])
            return pd.DataFrame([{"timestamp": sa_app.datetime(2024, 1, 3, 9, 30), "close": 105.0}])

    history = {
        sa_app.datetime(2024, 1, 1, 9, 30): 1000.0,
        sa_app.datetime(2024, 1, 3, 9, 30): 1100.0,
    }
    monkeypatch.setattr(sa_app, "_alpaca_credentials", lambda session=None: ("key", "secret"))
    monkeypatch.setattr(sa_app, "AlpacaDataProvider", _FakeAlpacaProvider)

    spy, error = sa_app._fetch_spy_series(history, {"timeframe": "Minute"}, {"account_id": "acct"})
    benchmark = sa_app._spy_comparison(sa_app._series_from_history(history), spy, {"total_return_pct": 10.0})

    assert error is None
    assert [call["sort"] for call in calls] == ["asc", "desc"]
    assert [point["y"] for point in spy] == [100.0, 105.0]
    assert benchmark["_comparison"]["benchmark_return_pct"] == 5.0
    assert benchmark["_comparison"]["alpha"] == 5.0


def test_session_summary_route_uses_requested_benchmark(monkeypatch):
    monkeypatch.setattr(sa_app, "_get_state_store", lambda db=None: _FakeStore())

    def fake_fetch(value_history, metadata, session=None, symbol="SPY"):
        raise AssertionError("summary route should not fetch benchmark data")

    monkeypatch.setattr(sa_app, "_fetch_benchmark_series", fake_fetch)

    client = sa_app.app.test_client()
    response = client.get("/api/session/sess-1?benchmark=qqq")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["benchmark_symbol"] == "QQQ"
    assert payload["symbols"] == {}
    assert payload["benchmark"] == {}


def test_session_indicators_route_is_separate(monkeypatch):
    monkeypatch.setattr(sa_app, "_get_state_store", lambda db=None: _FakeStore())
    monkeypatch.setattr(sa_app, "_reconstruct_indicator_series", lambda tick_history, session_metadata, max_points=0: {"_config": {}, "AAPL": {"rsi": [{"x": "2024-01-02T09:30:00", "y": 30.0}]}})

    client = sa_app.app.test_client()
    response = client.get("/api/session/sess-1/indicators")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["indicators"]["AAPL"]["rsi"][0]["y"] == 30.0
