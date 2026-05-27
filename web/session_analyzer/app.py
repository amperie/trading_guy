import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
import math
from collections import defaultdict, deque
from types import SimpleNamespace

from flask import Flask, jsonify, render_template, request

from utils.config_manager import ConfigManager
from utils.trading_state_store import TradingStateStore
from trading.data_providers.alpaca_data_provider import AlpacaDataProvider, TIMEFRAME_MAP
from trading.analysis.portfolio_analyzer import PortfolioAnalyzer


app = Flask(__name__, template_folder="templates", static_folder="static")
SUMMARY_MAX_POINTS = 2000
DETAIL_MAX_POINTS = 1500
SIGNAL_MAX_POINTS = 2000


def _max_points_arg(name: str, default: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(100, min(value, 10000))


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


def _downsample_points(points: list[dict], max_points: int) -> list[dict]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, math.ceil((len(points) - 2) / max(1, max_points - 2)))
    sampled = [points[0], *points[1:-1:step], points[-1]]
    return sampled[: max_points - 1] + [points[-1]] if len(sampled) > max_points else sampled


def _downsample_series_map(series_map: dict[str, list[dict]], max_points: int) -> dict[str, list[dict]]:
    return {symbol: _downsample_points(series, max_points) for symbol, series in series_map.items()}


def _trading_days_from_history(value_history: dict) -> int:
    return len({ts.date() for ts in value_history.keys()})


def _metrics_from_equity(equity: list[dict]) -> dict:
    if not equity:
        return {}
    values = [float(pt["y"]) for pt in equity]
    first = values[0]
    last = values[-1]
    total_return = ((last - first) / first) * 100 if first else 0.0

    peak = -math.inf
    max_dd = 0.0
    returns = []
    for idx, value in enumerate(values):
        peak = max(peak, value)
        if peak:
            max_dd = min(max_dd, ((value - peak) / peak) * 100)
        if idx:
            prev = values[idx - 1]
            if prev:
                returns.append(((value - prev) / prev) * 100)

    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((r - mean) ** 2 for r in returns) / len(returns) if returns else 0.0
    std = math.sqrt(variance)
    downside = [r for r in returns if r < 0]
    downside_std = math.sqrt(sum(r * r for r in downside) / len(downside)) if downside else 0.0
    sharpe = (mean / std) * math.sqrt(252) if std else None
    sortino = (mean / downside_std) * math.sqrt(252) if downside_std else None

    start = datetime.fromisoformat(equity[0]["x"])
    end = datetime.fromisoformat(equity[-1]["x"])
    years = (end - start).total_seconds() / (365.25 * 24 * 3600) if end > start else 0.0
    annualized = ((last / first) ** (1 / years) - 1) * 100 if years > 0 and first > 0 and last > 0 else total_return
    calmar = annualized / abs(max_dd) if max_dd else None
    trading_days = len({pt["x"][:10] for pt in equity})

    return {
        "total_return_pct": total_return,
        "annualized_return": annualized,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd,
        "calmar_ratio": calmar,
        "volatility": std * math.sqrt(252),
        "win_rate": None,
        "profit_factor": None,
        "total_trades": None,
        "avg_trade_pnl": None,
        "trading_days": trading_days,
        "bars": len(equity),
    }


def _fetch_spy_series(value_history: dict, session_metadata: dict) -> tuple[list[dict], str | None]:
    if not value_history:
        return [], None
    cfg = ConfigManager()
    alpaca_cfg = cfg.get("alpaca", {}) or {}
    dp_cfg = cfg.get("data_provider", {}) or {}
    api_key = os.getenv("ALPACA_API_KEY") or alpaca_cfg.get("api_key") or dp_cfg.get("api_key")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or alpaca_cfg.get("secret_key") or dp_cfg.get("secret_key")
    if not api_key or not secret_key:
        return [], "Missing Alpaca credentials for SPY fallback"

    dates = sorted(value_history.keys())
    timeframe = ((session_metadata or {}).get("timeframe") or "Minute")
    if timeframe not in TIMEFRAME_MAP:
        timeframe = "Minute"
    try:
        provider = AlpacaDataProvider({
            "api_key": api_key,
            "secret_key": secret_key,
            "symbols": ["SPY"],
            "timeframe": timeframe,
            "start_date": dates[0].isoformat(sep=" "),
            "end_date": dates[-1].isoformat(sep=" "),
            "market_hours_only": True,
        })
        provider.load_data()
    except Exception as exc:
        return [], f"Failed to fetch SPY from Alpaca: {exc}"
    df = provider.get_data()
    if df is None or df.empty:
        return [], "Alpaca returned no SPY bars for session range"
    return [
        {"x": row.timestamp.isoformat(), "y": float(row.close)}
        for row in df.sort_values("timestamp").itertuples(index=False)
    ], None


def _spy_comparison(equity: list[dict], spy: list[dict], metrics: dict) -> dict:
    if not equity or not spy:
        return {}
    start = datetime.fromisoformat(equity[0]["x"])
    end = datetime.fromisoformat(equity[-1]["x"])
    spy_window = [
        item for item in spy
        if start <= datetime.fromisoformat(item["x"]) <= end
    ]
    if not spy_window:
        return {}
    spy_return = ((spy_window[-1]["y"] - spy_window[0]["y"]) / spy_window[0]["y"]) * 100
    portfolio_return = metrics.get("total_return_pct")
    if portfolio_return is None:
        first, last = equity[0]["y"], equity[-1]["y"]
        portfolio_return = ((last - first) / first) * 100
    return {
        "_comparison": {
            "portfolio_return_pct": portfolio_return,
            "benchmark_return_pct": spy_return,
            "alpha": portfolio_return - spy_return,
            "outperformance": portfolio_return > spy_return,
        }
    }


def _symbol_series_from_ticks(tick_history: dict, max_points: int = DETAIL_MAX_POINTS) -> dict:
    symbol_series = {}
    for ts in sorted(tick_history.keys()):
        for pd in tick_history[ts]:
            symbol_series.setdefault(pd.symbol, []).append(
                {"x": ts.isoformat(), "y": float(pd.close)}
            )
    return _downsample_series_map(symbol_series, max_points)


def _signal_payload(signals_history: dict, tick_history: dict, max_points: int = SIGNAL_MAX_POINTS) -> list[dict]:
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
    return _downsample_points(payload, max_points)


def _calc_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    index = int((percentile / 100.0) * (len(sorted_vals) - 1))
    return float(sorted_vals[index])


def _reconstruct_indicator_series(tick_history: dict, session_metadata: dict, max_points: int = DETAIL_MAX_POINTS) -> dict:
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

        indicators[symbol] = {
            key: _downsample_points(series, max_points)
            for key, series in symbol_payload.items()
        }

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


def _build_analyzer_from_data(pf_data: dict, order_data: dict, session_metadata: dict) -> PortfolioAnalyzer:
    sorted_ts = sorted(pf_data["value_history"]) if pf_data["value_history"] else []
    pf_shell = SimpleNamespace(
        keep_history=True,
        tick_history=pf_data["tick_history"],
        value_history=pf_data["value_history"],
        cash_history=pf_data["cash_history"],
        signals_history=pf_data["signals_history"],
        total_value=pf_data["value_history"][sorted_ts[-1]] if sorted_ts else 0.0,
        cash=pf_data["cash_history"][sorted_ts[-1]] if sorted_ts else 0.0,
        positions={},
    )
    om_shell = SimpleNamespace(
        filled_orders_by_id=order_data["filled_orders_by_id"],
        pending_orders_by_id=order_data["pending_orders_by_id"],
    )
    pf_shell.om = om_shell
    analyzer = PortfolioAnalyzer(pf_shell)
    analyzer._session_metadata = session_metadata or {}
    return analyzer


def _order_summary(order_data: dict) -> dict:
    canceled_count = sum(
        1
        for order in order_data["filled_orders_by_id"].values()
        if getattr(getattr(order, "status", None), "name", str(getattr(order, "status", ""))) == "CANCELED"
    )
    return {
        "total": len(order_data["all_orders"]),
        "filled": len(order_data["filled_orders_by_id"]),
        "pending": len(order_data["pending_orders_by_id"]),
        "canceled": canceled_count,
    }


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
def session_summary(session_id: str):
    try:
        db = request.args.get("db") or None
        store = _get_state_store(db=db)
        session = store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        history = store.load_equity_history(session_id)
        order_data = store.load_orders(session_id)
        equity_full = _series_from_history(history["value_history"])
        cash_full = _series_from_history(history["cash_history"])
        equity = _downsample_points(equity_full, _max_points_arg("points", SUMMARY_MAX_POINTS))
        cash = _downsample_points(cash_full, _max_points_arg("points", SUMMARY_MAX_POINTS))
        metrics = _metrics_from_equity(equity_full)

        payload = {
            "session": _json_safe(session),
            "portfolio": {
                "total_value": equity,
                "cash": cash,
            },
            "symbols": {},
            "signals": [],
            "indicators": {"_config": {}},
            "metrics": _json_safe(metrics),
            "benchmark": {},
            "trades": [],
            "orders": _order_summary(order_data),
            "errors": {},
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/session/<session_id>/details")
def session_details(session_id: str):
    try:
        db = request.args.get("db") or None
        store = _get_state_store(db=db)
        session = store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        pf_data = store.load_portfolio_history(session_id)
        order_data = store.load_orders(session_id)
        metadata = (session or {}).get("metadata") or {}
        analyzer = _build_analyzer_from_data(pf_data, order_data, metadata)
        point_limit = _max_points_arg("points", DETAIL_MAX_POINTS)
        signal_limit = _max_points_arg("signal_points", SIGNAL_MAX_POINTS)

        trades = []
        metrics = {}
        errors = {}
        try:
            trades = analyzer.extract_trades()
        except Exception as exc:
            errors["trades"] = str(exc)
        try:
            metrics = _json_safe(analyzer.calculate_metrics()) or {}
        except Exception as exc:
            errors["metrics"] = str(exc)

        symbols = _symbol_series_from_ticks(pf_data["tick_history"], max_points=point_limit)
        if "SPY" not in symbols:
            spy_series, spy_error = _fetch_spy_series(pf_data["value_history"], metadata)
            if spy_series:
                symbols["SPY"] = _downsample_points(spy_series, point_limit)
            elif spy_error:
                errors["spy"] = spy_error

        equity = _series_from_history(pf_data["value_history"])
        benchmark = _spy_comparison(equity, symbols.get("SPY", []), metrics)
        if not benchmark:
            try:
                benchmark = analyzer.calculate_benchmark_comparison()
            except Exception as exc:
                errors["benchmark"] = str(exc)

        payload = {
            "symbols": symbols,
            "signals": _signal_payload(pf_data["signals_history"], pf_data["tick_history"], max_points=signal_limit),
            "trades": _trades_payload(trades),
            "metrics": metrics,
            "benchmark": _json_safe(benchmark),
            "errors": errors,
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/session/<session_id>/indicators")
def session_indicators(session_id: str):
    try:
        db = request.args.get("db") or None
        store = _get_state_store(db=db)
        session = store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        pf_data = store.load_portfolio_history(session_id)
        indicators = _reconstruct_indicator_series(
            pf_data["tick_history"],
            ((session or {}).get("metadata") or {}),
            max_points=_max_points_arg("points", DETAIL_MAX_POINTS),
        )
        return jsonify({"indicators": indicators})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
