from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from algo_crucible.builders import build_components
from algo_crucible.models import Candidate
from trading.analysis.market_regime import classify_ticks


def load_frozen_candidate(run_dir: str | Path, candidate_id: str | None = None) -> Candidate:
    folder = _existing_dir(Path(run_dir), "stages/08_confirmation/frozen_candidates", "frozen_candidates")
    paths = sorted(folder.glob("candidate_*.json"))
    if candidate_id:
        paths = [folder / f"{candidate_id}.json"]
    if not paths or not paths[0].exists():
        raise FileNotFoundError("run_confirmation_stage must freeze a candidate before paper replay can run")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if not payload.get("tuning_locked"):
        raise ValueError("Frozen candidate is not marked tuning_locked")
    return Candidate.from_dict(payload["candidate"])


def replay_trace(resolved_cfg, candidate: Candidate, data_path: str | Path | None = None) -> list[dict[str, Any]]:
    workload = json.loads(json.dumps(resolved_cfg.workload))
    if data_path is not None:
        workload["data_provider"]["path"] = str(data_path)
    dp, al, om, pf = build_components(workload, candidate)
    ticks = list(dp.iterate())
    regime_cfg = candidate.algorithm_params.get("market_regime", {})
    regimes = classify_ticks(ticks, regime_cfg)
    rows = []
    for tick, tick_regimes in zip(ticks, regimes):
        ts = _timestamp(tick)
        filled_before = set(om.filled_orders_by_id)
        signals = al.on_data(tick)
        result = pf.process_market_signals_for_tick(signals, tick)
        filled_after = set(om.filled_orders_by_id)
        fills = [
            _normalize_order(om.filled_orders_by_id[order_id], include_status=True)
            for order_id in filled_after - filled_before
        ]
        rows.append({
            "timestamp": ts,
            "regimes": _normalize_regimes(tick_regimes),
            "signals": [_normalize_signal(signal) for signal in signals],
            "orders": [_normalize_order(order, include_status=False) for order in result.orders],
            "fills": sorted(fills, key=_execution_key),
        })
    return rows


def load_observed_trace(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv_rows(Path(path)):
        rows.append({
            "timestamp": _iso(row.get("timestamp")),
            "regimes": _decode_list(row.get("regimes") or row.get("regimes_json")),
            "signals": _decode_list(row.get("signals") or row.get("signals_json")),
            "orders": _decode_list(row.get("orders") or row.get("orders_json")),
            "fills": _decode_list(row.get("fills") or row.get("fills_json")),
        })
    return rows


def trace_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    frame = pd.DataFrame([{
        "timestamp": row["timestamp"],
        "regimes_json": json.dumps(row.get("regimes", []), sort_keys=True),
        "signals_json": json.dumps(row.get("signals", []), sort_keys=True),
        "orders_json": json.dumps(row.get("orders", []), sort_keys=True),
        "fills_json": json.dumps(row.get("fills", []), sort_keys=True),
    } for row in rows])
    return frame.to_csv(index=False)


def compare_traces(replay_rows: list[dict[str, Any]], observed_rows: list[dict[str, Any]], platform: dict[str, Any]) -> dict[str, Any]:
    cfg = platform.get("paper_replay", {})
    observed_by_ts = {row["timestamp"]: row for row in observed_rows}
    replay_by_ts = {row["timestamp"]: row for row in replay_rows}
    all_ts = sorted(set(observed_by_ts) | set(replay_by_ts))
    mismatches = []
    for ts in all_ts:
        replay = replay_by_ts.get(ts)
        observed = observed_by_ts.get(ts)
        if replay is None or observed is None:
            mismatches.append(_mismatch(ts, "timestamp_missing", replay, observed))
            continue
        for field in ("regimes", "signals", "orders"):
            if replay.get(field, []) != observed.get(field, []):
                mismatches.append(_mismatch(ts, f"{field}_mismatch", replay.get(field, []), observed.get(field, [])))
        fill_mismatch = _compare_fills(ts, replay.get("fills", []), observed.get("fills", []), float(cfg.get("fill_price_tolerance_pct", 0.01)))
        if fill_mismatch:
            mismatches.append(fill_mismatch)

    counts = {
        "timestamp_missing": sum(1 for row in mismatches if row["type"] == "timestamp_missing"),
        "regime_mismatch": sum(1 for row in mismatches if row["type"] == "regimes_mismatch"),
        "signal_mismatch": sum(1 for row in mismatches if row["type"] == "signals_mismatch"),
        "order_mismatch": sum(1 for row in mismatches if row["type"] == "orders_mismatch"),
        "fill_mismatch": sum(1 for row in mismatches if row["type"] == "fills_mismatch"),
    }
    passed = (
        counts["timestamp_missing"] <= int(cfg.get("max_missing_timestamps", 0))
        and counts["regime_mismatch"] <= int(cfg.get("max_regime_mismatches", 0))
        and counts["signal_mismatch"] <= int(cfg.get("max_signal_mismatches", 0))
        and counts["order_mismatch"] <= int(cfg.get("max_order_mismatches", 0))
        and counts["fill_mismatch"] <= int(cfg.get("max_fill_mismatches", 0))
    )
    return {
        "passed": passed,
        "failure_reason": "" if passed else "paper_replay_mismatch",
        "mismatches": mismatches,
        "metrics": {
            "paper_replay.rows_observed": float(len(observed_rows)),
            "paper_replay.rows_replayed": float(len(replay_rows)),
            "paper_replay.mismatches_total": float(len(mismatches)),
            "paper_replay.timestamp_missing": float(counts["timestamp_missing"]),
            "paper_replay.regime_mismatches": float(counts["regime_mismatch"]),
            "paper_replay.signal_mismatches": float(counts["signal_mismatch"]),
            "paper_replay.order_mismatches": float(counts["order_mismatch"]),
            "paper_replay.fill_mismatches": float(counts["fill_mismatch"]),
            "paper_replay.passed": 1.0 if passed else 0.0,
        },
    }


def mismatch_rows(mismatches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "timestamp": row["timestamp"],
        "type": row["type"],
        "replay": json.dumps(row["replay"], sort_keys=True, default=str),
        "observed": json.dumps(row["observed"], sort_keys=True, default=str),
    } for row in mismatches]


def _compare_fills(ts: str, replay: list[dict[str, Any]], observed: list[dict[str, Any]], tolerance_pct: float) -> dict[str, Any] | None:
    if not replay and not observed:
        return None
    replay = sorted(replay, key=_execution_key)
    observed = sorted(observed, key=_execution_key)
    if len(replay) != len(observed):
        return _mismatch(ts, "fills_mismatch", replay, observed)
    for left, right in zip(replay, observed):
        if {k: left.get(k) for k in ("symbol", "action", "type", "quantity")} != {k: right.get(k) for k in ("symbol", "action", "type", "quantity")}:
            return _mismatch(ts, "fills_mismatch", replay, observed)
        left_price = _num(left.get("price"))
        right_price = _num(right.get("price"))
        if left_price is not None and right_price is not None:
            denom = max(abs(right_price), 1e-9)
            if abs(left_price - right_price) / denom * 100.0 > tolerance_pct:
                return _mismatch(ts, "fills_mismatch", replay, observed)
    return None


def _normalize_regimes(regimes: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted([
        {"symbol": symbol, "regime": snapshot.composite_regime}
        for symbol, snapshot in regimes.items()
    ], key=lambda row: row["symbol"])


def _normalize_signal(signal) -> dict[str, Any]:
    return {
        "type": signal.type.name,
        "symbol": signal.symbol,
        "strength": int(signal.strength),
        "metadata": _stable_metadata(signal.metadata or {}),
    }


def _normalize_order(order, include_status: bool = True) -> dict[str, Any]:
    row = {
        "action": order.action.name,
        "type": order.type.name,
        "symbol": order.symbol,
        "quantity": int(order.quantity),
        "price": round(float(order.price or 0.0), 6),
    }
    if include_status:
        row["status"] = order.status.name
    return row


def _stable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    volatile_keys = {"order_id", "platform_id", "client_order_id"}
    return {
        key: value
        for key, value in sorted(metadata.items())
        if key not in volatile_keys
    }


def _execution_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("symbol", "")),
        str(row.get("action", "")),
        str(row.get("type", "")),
        int(row.get("quantity") or 0),
        float(row.get("price") or 0.0),
    )


def _timestamp(tick: list[Any]) -> str:
    return _iso(tick[0].timestamp) if tick else ""


def _iso(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).isoformat()


def _decode_list(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    decoded = json.loads(str(value))
    return decoded if isinstance(decoded, list) else []


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except EmptyDataError:
        return []


def _mismatch(ts: str, kind: str, replay: Any, observed: Any) -> dict[str, Any]:
    return {"timestamp": ts, "type": kind, "replay": replay, "observed": observed}


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _existing_dir(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = run_dir / relative_path
        if path.exists():
            return path
    return run_dir / relative_paths[0]
