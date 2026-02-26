import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
import math

from flask import Flask, jsonify, render_template

from utils.config_manager import ConfigManager
from utils.trading_state_store import TradingStateStore


app = Flask(__name__, template_folder="templates", static_folder="static")


def _get_state_store() -> TradingStateStore:
    cfg = ConfigManager().get("state_store", {}) or {}
    uri = os.getenv("MONGO_URI") or cfg.get("connection_uri", "mongodb://localhost:27017")
    db = os.getenv("MONGO_DB") or cfg.get("database", "trading")
    return TradingStateStore(connection_uri=uri, database=db)


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().isoformat()
    except Exception:
        pass
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _series_from_history(value_history: dict) -> list[dict]:
    series = []
    for ts in sorted(value_history.keys()):
        series.append({"x": ts.isoformat(), "y": float(value_history[ts])})
    return series


def _symbol_series_from_ticks(tick_history: dict) -> dict:
    symbol_series = {}
    for ts in sorted(tick_history.keys()):
        for pd in tick_history[ts]:
            symbol_series.setdefault(pd.symbol, []).append(
                {"x": ts.isoformat(), "y": float(pd.close)}
            )
    return symbol_series


def _trades_payload(trades: list) -> list[dict]:
    payload = []
    for trade in trades:
        payload.append(
            {
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "quantity": int(trade.quantity),
                "pnl": float(trade.pnl),
                "pnl_pct": float(trade.pnl_pct),
                "duration_hours": float(trade.duration) / 3600.0,
                "is_bracket": bool(trade.is_bracket),
                "bracket_exit_type": trade.bracket_exit_type,
            }
        )
    return payload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session/<session_id>")
def session_data(session_id: str):
    try:
        store = _get_state_store()
        session = store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        pf_data = store.load_portfolio_history(session_id)
        order_data = store.load_orders(session_id)
        engine = store.create_analysis_engine(session_id)

        trades = []
        metrics = {}
        benchmark = {}

        try:
            trades = engine.extract_trades()
        except Exception as exc:
            benchmark["_trades_error"] = str(exc)

        try:
            metrics = engine.calculate_metrics()
        except Exception as exc:
            benchmark["_metrics_error"] = str(exc)

        try:
            benchmark.update(engine.calculate_benchmark_comparison())
        except Exception as exc:
            benchmark["_benchmark_error"] = str(exc)

        filled_orders = list(order_data["filled_orders_by_id"].values())
        canceled_count = 0
        for order in filled_orders:
            status_name = getattr(order.status, "name", str(order.status))
            if status_name == "CANCELED":
                canceled_count += 1

        order_summary = {
            "total": len(order_data["all_orders"]),
            "filled": len(order_data["filled_orders_by_id"]),
            "pending": len(order_data["pending_orders_by_id"]),
            "canceled": canceled_count,
        }

        payload = {
            "session": _json_safe(session),
            "portfolio": {
                "total_value": _series_from_history(pf_data["value_history"]),
                "cash": _series_from_history(pf_data["cash_history"]),
            },
            "symbols": _symbol_series_from_ticks(pf_data["tick_history"]),
            "metrics": _json_safe(metrics),
            "benchmark": _json_safe(benchmark),
            "trades": _trades_payload(trades),
            "orders": order_summary,
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
