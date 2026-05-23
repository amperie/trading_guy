import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
import math
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request

from utils.config_manager import ConfigManager
from utils.trading_state_store import TradingStateStore
from trading.analysis.portfolio_analyzer import PortfolioAnalyzer


app = Flask(__name__, template_folder="templates", static_folder="static")


def _get_mongo_params(db: str = None) -> tuple[str, str]:
    cfg = ConfigManager().get("state_store", {}) or {}
    uri = os.getenv("MONGO_URI") or cfg.get("connection_uri", "mongodb://localhost:27017")
    resolved_db = db or os.getenv("MONGO_DB") or cfg.get("database", "trading_hp")
    return uri, resolved_db


def _get_state_store(db: str = None) -> TradingStateStore:
    uri, resolved_db = _get_mongo_params(db)
    return TradingStateStore(connection_uri=uri, database=resolved_db)


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


def _signal_payload(signals_history: dict, tick_history: dict) -> list[dict]:
    price_lookup = {}
    for ts, tick in tick_history.items():
        for pd in tick:
            price_lookup[(ts, pd.symbol)] = float(pd.close)

    payload = []
    for ts in sorted(signals_history.keys()):
        for sig in signals_history[ts]:
            payload.append(
                {
                    "x": ts.isoformat(),
                    "symbol": sig.symbol,
                    "type": getattr(sig.type, "name", str(sig.type)),
                    "strength": float(sig.strength) if sig.strength is not None else None,
                    "price": price_lookup.get((ts, sig.symbol)),
                    "metadata": _json_safe(sig.metadata or {}),
                }
            )
    return payload


def _calc_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    index = int((percentile / 100.0) * (len(sorted_vals) - 1))
    return float(sorted_vals[index])


def _reconstruct_indicator_series(tick_history: dict, session_metadata: dict) -> dict:
    algo_cfg = (session_metadata or {}).get("algorithm_config") or {}
    regime_cfg = algo_cfg.get("regime_detection") or {}
    rsi_cfg = algo_cfg.get("rsi_config") or {}

    ma_short_period = int(round(float(regime_cfg.get("ma_short_period", 50))))
    ma_long_period = int(round(float(regime_cfg.get("ma_long_period", 200))))
    atr_period = int(round(float(regime_cfg.get("atr_period", 14))))
    atr_percentile_window = int(round(float(regime_cfg.get("atr_percentile_window", 20))))
    atr_percentile_level = float(regime_cfg.get("atr_percentile_level", 50))
    rsi_period = int(round(float(rsi_cfg.get("rsi_period", 14))))

    rows_by_symbol: dict[str, list[tuple[datetime, object]]] = defaultdict(list)
    for ts in sorted(tick_history.keys()):
        for pd in tick_history[ts]:
            rows_by_symbol[pd.symbol].append((ts, pd))

    indicators = {
        "_config": {
            "ma_short_period": ma_short_period,
            "ma_long_period": ma_long_period,
            "atr_period": atr_period,
            "atr_percentile_window": atr_percentile_window,
            "atr_percentile_level": atr_percentile_level,
            "rsi_period": rsi_period,
            "rsi_oversold_threshold": rsi_cfg.get("rsi_oversold_threshold"),
            "rsi_overbought_threshold": rsi_cfg.get("rsi_overbought_threshold"),
        }
    }

    for symbol, rows in rows_by_symbol.items():
        closes: deque[float] = deque()
        highs: deque[float] = deque()
        lows: deque[float] = deque()
        atr_history: deque[float] = deque(maxlen=atr_percentile_window)

        symbol_payload = {
            "ma_short": [],
            "ma_long": [],
            "rsi": [],
            "atr": [],
            "atr_percentile": [],
        }

        for ts, pd in rows:
            closes.append(float(pd.close))
            highs.append(float(pd.high))
            lows.append(float(pd.low))

            x = ts.isoformat()

            if len(closes) >= ma_short_period:
                window = list(closes)[-ma_short_period:]
                symbol_payload["ma_short"].append({"x": x, "y": sum(window) / ma_short_period})

            if len(closes) >= ma_long_period:
                window = list(closes)[-ma_long_period:]
                symbol_payload["ma_long"].append({"x": x, "y": sum(window) / ma_long_period})

            if len(closes) >= rsi_period + 1:
                gains = []
                losses = []
                close_list = list(closes)
                for i in range(-rsi_period, 0):
                    delta = close_list[i] - close_list[i - 1]
                    if delta > 0:
                        gains.append(delta)
                        losses.append(0.0)
                    else:
                        gains.append(0.0)
                        losses.append(abs(delta))

                avg_gain = sum(gains) / rsi_period
                avg_loss = sum(losses) / rsi_period
                if avg_loss == 0:
                    rsi = 100.0 if avg_gain > 0 else 50.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                symbol_payload["rsi"].append({"x": x, "y": rsi})

            if len(closes) >= atr_period:
                tr_values = []
                close_list = list(closes)
                high_list = list(highs)
                low_list = list(lows)
                for i in range(-atr_period, 0):
                    high = high_list[i]
                    low = low_list[i]
                    close_prev = close_list[i - 1] if i > -len(close_list) else close_list[0]
                    tr_values.append(max(high - low, abs(high - close_prev), abs(low - close_prev)))

                current_atr = sum(tr_values) / atr_period if tr_values else None
                if current_atr is not None:
                    atr_history.append(current_atr)
                    symbol_payload["atr"].append({"x": x, "y": current_atr})
                    atr_pct = _calc_percentile(list(atr_history), atr_percentile_level)
                    if atr_pct is not None:
                        symbol_payload["atr_percentile"].append({"x": x, "y": atr_pct})

        indicators[symbol] = symbol_payload

    return indicators


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


@app.route("/api/sessions")
def list_sessions():
    try:
        db = request.args.get("db") or None
        store = _get_state_store(db=db)
        sessions = store.list_sessions()
        result = []
        for s in sessions:
            session_id = s.get("session_id") or str(s.get("_id", ""))
            name = s.get("name") or session_id
            created = s.get("created_at")
            result.append({
                "session_id": session_id,
                "name": name,
                "created_at": created.isoformat() if isinstance(created, datetime) else str(created or ""),
            })
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/session/<session_id>")
def session_data(session_id: str):
    try:
        db = request.args.get("db") or None
        store = _get_state_store(db=db)
        session = store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        pf_data = store.load_portfolio_history(session_id)
        order_data = store.load_orders(session_id)
        uri, resolved_db = _get_mongo_params(db)
        engine = PortfolioAnalyzer.from_mongodb(session_id, connection_uri=uri, database=resolved_db)

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
            "signals": _signal_payload(pf_data["signals_history"], pf_data["tick_history"]),
            "indicators": _reconstruct_indicator_series(
                pf_data["tick_history"],
                (session or {}).get("metadata") or {},
            ),
            "metrics": _json_safe(metrics),
            "benchmark": _json_safe(benchmark),
            "trades": _trades_payload(trades),
            "orders": order_summary,
            "errors": {k[1:]: v for k, v in benchmark.items() if k.startswith("_")},
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
