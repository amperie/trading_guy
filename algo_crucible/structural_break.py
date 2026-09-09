from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


def load_structural_break_inputs(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    signal_rows = _read_csv_rows(run_dir / "stages/07_perturbation/summaries/perturbation_signal_forward_returns.csv")
    return signal_rows or _read_csv_rows(run_dir / "stages/07_perturbation/summaries/perturbation_return_stream.csv")


def analyze_structural_breaks(rows: list[dict[str, Any]], platform: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cfg = platform.get("structural_break_stability", {})
    if _has_signal_forward_returns(rows, cfg):
        return _analyze_rolling_ic(rows, cfg)

    min_observations = int(cfg.get("min_observations", 20))
    window = int(cfg.get("rolling_window", 10))
    max_concentration = _ratio_threshold(cfg.get("max_segment_return_concentration", 0.65))
    max_degradation = float(cfg.get("max_post_break_mean_degradation_pct", 50.0))
    min_post_break = float(cfg.get("min_post_break_mean_return_pct", -100.0))

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate_id, group in _group_returns(rows).items():
        returns = group["returns"]
        if len(returns) < min_observations:
            summary_rows.append(_summary_row(candidate_id, len(returns), False, "insufficient_observations", "return_concentration"))
            continue
        segments = _segments(returns, max(1, min(window, len(returns))))
        total_positive = sum(max(0.0, item["sum_return_pct"]) for item in segments)
        worst_break = _worst_break(segments)
        concentration = (
            max((max(0.0, item["sum_return_pct"]) for item in segments), default=0.0) / total_positive
            if total_positive > 0
            else 0.0
        )
        post_mean = worst_break.get("post_mean_return_pct")
        degradation = worst_break.get("mean_degradation_pct")
        accepted = (
            concentration <= max_concentration
            and post_mean is not None
            and post_mean >= min_post_break
            and degradation is not None
            and degradation <= max_degradation
        )
        reason = "" if accepted else _failure_reason(concentration, max_concentration, post_mean, min_post_break, degradation, max_degradation)
        detail_rows.extend({"candidate_id": candidate_id, **item} for item in segments)
        summary_rows.append({
            **_summary_row(candidate_id, len(returns), accepted, reason, "return_concentration"),
            "segment_count": len(segments),
            "rolling_window": window,
            "max_segment_return_concentration": concentration,
            "post_break_mean_return_pct": post_mean,
            "post_break_mean_degradation_pct": degradation,
            "break_after_segment": worst_break.get("break_after_segment"),
        })
    return {"window_rows": detail_rows, "summary_rows": summary_rows}


def structural_break_metrics(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    accepted = [row for row in summary_rows if row.get("accepted") is True]
    rejected = [row for row in summary_rows if row.get("accepted") is not True]
    concentrations = [_num(row.get("max_segment_return_concentration")) for row in summary_rows]
    degradations = [_num(row.get("post_break_mean_degradation_pct")) for row in summary_rows]
    ic_means = [_num(row.get("ic_mean")) for row in summary_rows]
    return {
        "structural_break.candidate_count": float(len(summary_rows)),
        "structural_break.accepted_candidates": float(len(accepted)),
        "structural_break.rejected_candidates": float(len(rejected)),
        "structural_break.worst_segment_concentration": max([v for v in concentrations if v is not None], default=0.0),
        "structural_break.worst_post_break_degradation_pct": max([v for v in degradations if v is not None], default=0.0),
        "structural_break.best_ic_mean": max([v for v in ic_means if v is not None], default=0.0),
    }


def _analyze_rolling_ic(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    signal_col = str(cfg.get("signal_column", "signal"))
    return_col = str(cfg.get("forward_return_column", "forward_return_pct"))
    windows = [int(value) for value in cfg.get("ic_windows", [63, 126]) if int(value) > 1]
    min_observations = int(cfg.get("min_observations", 20))
    min_ic_observations = int(cfg.get("min_ic_observations", 3))
    min_ic_mean = float(cfg.get("min_ic_mean", 0.0))
    min_ic_ir = float(cfg.get("min_ic_ir", 0.0))
    min_positive = _ratio_threshold(cfg.get("min_positive_ic_window_pct", 0.60))
    recent_count = int(cfg.get("recent_window_count", 3))
    recent_min_ic = float(cfg.get("recent_min_ic_mean", 0.0))
    min_post_break_ic = float(cfg.get("min_post_break_ic_mean", 0.0))
    max_break_degradation = float(cfg.get("max_post_break_ic_degradation", 0.50))

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate_id, group in _group_signal_rows(rows, signal_col, return_col).items():
        if len(group) < min_observations:
            summary_rows.append(_summary_row(candidate_id, len(group), False, "insufficient_observations", "rolling_ic"))
            continue
        candidate_windows = [row for window in windows for row in _rolling_ic_rows(candidate_id, group, window)]
        valid_ics = [value for value in (_num(row.get("ic")) for row in candidate_windows) if value is not None]
        detail_rows.extend(candidate_windows)
        if len(valid_ics) < min_ic_observations:
            summary_rows.append(_summary_row(candidate_id, len(group), False, "insufficient_ic_windows", "rolling_ic"))
            continue

        ic_mean = sum(valid_ics) / len(valid_ics)
        ic_std = _std(valid_ics)
        ic_ir = ic_mean / ic_std if ic_std > 0 else (float("inf") if ic_mean > 0 else 0.0)
        positive_pct = sum(1 for value in valid_ics if value > 0) / len(valid_ics)
        recent_mean = sum(valid_ics[-recent_count:]) / min(recent_count, len(valid_ics))
        break_row = _worst_ic_break(valid_ics)
        post_break_ic = break_row.get("post_break_ic_mean")
        degradation = break_row.get("ic_degradation")
        reasons = []
        if ic_mean < min_ic_mean:
            reasons.append("rolling_ic_mean_below_gate")
        if ic_ir < min_ic_ir:
            reasons.append("rolling_ic_ir_below_gate")
        if positive_pct < min_positive:
            reasons.append("positive_ic_window_rate_below_gate")
        if recent_mean < recent_min_ic:
            reasons.append("recent_ic_sign_flip")
        if post_break_ic is not None and post_break_ic < min_post_break_ic:
            reasons.append("post_break_ic_mean_below_gate")
        if degradation is not None and degradation > max_break_degradation:
            reasons.append("post_break_ic_degradation_above_gate")
        summary_rows.append({
            **_summary_row(candidate_id, len(group), not reasons, ",".join(reasons), "rolling_ic"),
            "ic_window_count": len(valid_ics),
            "ic_mean": ic_mean,
            "ic_ir": ic_ir,
            "positive_ic_window_pct": positive_pct * 100.0,
            "recent_ic_mean": recent_mean,
            "break_after_ic_window": break_row.get("break_after_ic_window"),
            "post_break_ic_mean": post_break_ic,
            "post_break_ic_degradation": degradation,
        })
    return {"window_rows": detail_rows, "summary_rows": summary_rows}


def _rolling_ic_rows(candidate_id: str, group: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    rows = []
    for end in range(window, len(group) + 1):
        chunk = group[end - window : end]
        rows.append({
            "candidate_id": candidate_id,
            "analysis_mode": "rolling_ic",
            "ic_window": window,
            "start_index": end - window,
            "end_index": end - 1,
            "start_timestamp": chunk[0].get("timestamp"),
            "end_timestamp": chunk[-1].get("timestamp"),
            "observation_count": len(chunk),
            "ic": _corr([row["signal"] for row in chunk], [row["forward_return_pct"] for row in chunk]),
        })
    return rows


def _worst_ic_break(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        return {"break_after_ic_window": None, "post_break_ic_mean": values[0] if values else None, "ic_degradation": 0.0}
    worst: dict[str, Any] = {"ic_degradation": -float("inf")}
    for idx in range(1, len(values)):
        pre_mean = sum(values[:idx]) / idx
        post_mean = sum(values[idx:]) / len(values[idx:])
        degradation = max(0.0, pre_mean - post_mean)
        if degradation > worst["ic_degradation"]:
            worst = {"break_after_ic_window": idx, "post_break_ic_mean": post_mean, "ic_degradation": degradation}
    return worst


def _segments(returns: list[float], window: int) -> list[dict[str, Any]]:
    rows = []
    for idx, start in enumerate(range(0, len(returns), window), start=1):
        chunk = returns[start : start + window]
        mean = sum(chunk) / len(chunk)
        vol = _std(chunk)
        rows.append({
            "analysis_mode": "return_concentration",
            "segment": idx,
            "start_index": start,
            "end_index": start + len(chunk) - 1,
            "observation_count": len(chunk),
            "sum_return_pct": sum(chunk),
            "mean_return_pct": mean,
            "sharpe_like": mean / vol if vol > 0 else 0.0,
        })
    return rows


def _worst_break(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if len(segments) < 2:
        return {"break_after_segment": None, "post_mean_return_pct": segments[0]["mean_return_pct"] if segments else None, "mean_degradation_pct": 0.0}
    worst: dict[str, Any] = {"mean_degradation_pct": -float("inf")}
    for idx in range(1, len(segments)):
        pre = segments[:idx]
        post = segments[idx:]
        pre_mean = _weighted_mean(pre)
        post_mean = _weighted_mean(post)
        degradation = max(0.0, pre_mean - post_mean)
        if degradation > worst["mean_degradation_pct"]:
            worst = {
                "break_after_segment": segments[idx - 1]["segment"],
                "post_mean_return_pct": post_mean,
                "mean_degradation_pct": degradation,
            }
    return worst


def _summary_row(candidate_id: str, observations: int, accepted: bool, reason: str, analysis_mode: str = "return_concentration") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "analysis_mode": analysis_mode,
        "observation_count": observations,
        "accepted": accepted,
        "failure_reason": reason,
    }


def _has_signal_forward_returns(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> bool:
    signal_col = str(cfg.get("signal_column", "signal"))
    return_col = str(cfg.get("forward_return_column", "forward_return_pct"))
    return any(_num(row.get(signal_col)) is not None and _num(row.get(return_col)) is not None for row in rows)


def _group_signal_rows(rows: list[dict[str, Any]], signal_col: str, return_col: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        signal = _num(row.get(signal_col))
        forward_return = _num(row.get(return_col))
        if candidate_id and signal is not None and forward_return is not None:
            grouped.setdefault(candidate_id, []).append({**row, "signal": signal, "forward_return_pct": forward_return})
    for group in grouped.values():
        group.sort(key=lambda row: (str(row.get("timestamp") or ""), str(row.get("window_id") or ""), int(_num(row.get("step")) or 0)))
    return grouped


def _failure_reason(concentration: float, max_concentration: float, post_mean: float | None, min_post_mean: float, degradation: float | None, max_degradation: float) -> str:
    if concentration > max_concentration:
        return "edge_concentrated_in_one_period"
    if post_mean is None or post_mean < min_post_mean:
        return "post_break_mean_return_below_gate"
    if degradation is None or degradation > max_degradation:
        return "post_break_degradation_above_gate"
    return "structural_break_unstable"


def _group_returns(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        value = _num(row.get("return_pct"))
        if not candidate_id or value is None:
            continue
        grouped.setdefault(candidate_id, {"returns": []})["returns"].append(value)
    return grouped


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except EmptyDataError:
        return []


def _weighted_mean(rows: list[dict[str, Any]]) -> float:
    count = sum(int(row["observation_count"]) for row in rows)
    return sum(float(row["mean_return_pct"]) * int(row["observation_count"]) for row in rows) / count if count else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _corr(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    denom = math.sqrt(left_var * right_var)
    if not denom:
        return None
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right)) / denom


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ratio_threshold(value: Any) -> float:
    threshold = float(value)
    return threshold / 100.0 if threshold > 1.0 else threshold
